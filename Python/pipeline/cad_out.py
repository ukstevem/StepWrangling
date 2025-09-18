# cad_out.py (merged & cleaned)

from pathlib import Path
import os, hashlib, tempfile
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import ezdxf

# OCC / pythonocc
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Message import Message_ProgressRange

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle
from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir


# Optional: DXF rendering helper (qsave may not exist)
try:
    from ezdxf.addons.drawing.matplotlib import qsave
except Exception:
    qsave = None


def _wire_to_circle_uv(face, wire, project_uv, tol_r=1e-5, tol_ang=1e-3):
    """
    If 'wire' is a full circle: return (cu, cv, radius), else None.
    Uses only OCC circle edges; multiple circle edges are allowed as long as
    they share the same center & radius and cover ~2π total angle.
    """
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    wx = BRepTools_WireExplorer(wire, face)

    centers = []
    radii = []
    ang_sum = 0.0
    saw_any = False
    two_pi = 2.0 * np.pi

    while wx.More():
        edge = wx.Current()
        c = BRepAdaptor_Curve(edge)
        if c.GetType() != GeomAbs_Circle:
            return None
        circ = c.Circle()  # gp_Circ
        centers.append(np.array([circ.Location().X(), circ.Location().Y(), circ.Location().Z()], float))
        radii.append(float(circ.Radius()))

        t0, t1 = float(c.FirstParameter()), float(c.LastParameter())
        ang = abs(t1 - t0)
        # normalize in case params wrap strangely
        while ang > two_pi:
            ang -= two_pi
        ang_sum += ang

        saw_any = True
        wx.Next()

    if not saw_any:
        return None

    # radius consistency
    if max(radii) - min(radii) > tol_r:
        return None

    # full circle?
    if abs(ang_sum - two_pi) > tol_ang:
        return None

    # center UV and radius
    C3 = np.mean(np.vstack(centers), axis=0)
    cu, cv = project_uv(C3)
    r = float(np.mean(radii))
    return (cu, cv, r)


# -------------------------
# STEP export
# -------------------------
def export_solid_to_step(solid, out_dir, filename):
    """
    Write `solid` to out_dir/filename.step using the 4-arg Transfer.
    """
    if solid is None:
        raise ValueError("export_solid_to_step: got None instead of a TopoDS_Shape")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_path = out_dir / f"{filename}.step"

    writer = STEPControl_Writer()
    prog = Message_ProgressRange()
    ok = writer.Transfer(solid, STEPControl_AsIs, True, prog)
    if ok != 1:
        raise RuntimeError(f"❌ STEP transfer failed for {filename!r} (status={ok})")

    status = writer.Write(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"❌ STEP write failed for {filename!r} (status={status})")

    print(f"✅ STEP saved to {step_path!r}")
    return step_path

# -------------------------
# Helpers: frames & projection
# -------------------------
def _frame_from_ax3(ax3):
    """
    Return (O, X, Y, Z) as unit np arrays from a gp_Ax3.
    If ax3 is None, return world frame.
    """
    if ax3 is None:
        O = np.zeros(3, float)
        X = np.array([1.0, 0.0, 0.0], float)
        Y = np.array([0.0, 1.0, 0.0], float)
        Z = np.array([0.0, 0.0, 1.0], float)
        return O, X, Y, Z

    O = np.array([ax3.Location().X(), ax3.Location().Y(), ax3.Location().Z()], dtype=float)
    X = np.array([ax3.XDirection().X(), ax3.XDirection().Y(), ax3.XDirection().Z()], dtype=float)
    Y = np.array([ax3.YDirection().X(), ax3.YDirection().Y(), ax3.YDirection().Z()], dtype=float)
    Z = np.array([ax3.Direction().X(),   ax3.Direction().Y(),   ax3.Direction().Z()], dtype=float)

    # normalize + enforce right-handed: cross(X,Y) ≈ Z
    X /= max(np.linalg.norm(X), 1e-12)
    Y /= max(np.linalg.norm(Y), 1e-12)
    Z /= max(np.linalg.norm(Z), 1e-12)
    if np.dot(np.cross(X, Y), Z) < 0:
        Y = np.cross(Z, X)
        Y /= max(np.linalg.norm(Y), 1e-12)
    return O, X, Y, Z

def _stable_plane_frame(face):
    """
    Deterministic UV frame for a planar face tied to global axes:
    - Z' = face normal pointing toward +Z
    - X' = projection of global +X into plane (fallback: +Y)
    - Y' = Z' × X' (right-handed)
    """
    s = BRepAdaptor_Surface(face)
    pln = s.Plane()

    O = np.array([pln.Location().X(), pln.Location().Y(), pln.Location().Z()])
    Z = np.array([pln.Axis().Direction().X(),
                  pln.Axis().Direction().Y(),
                  pln.Axis().Direction().Z()], dtype=float)
    Z /= max(np.linalg.norm(Z), 1e-12)
    if np.dot(Z, np.array([0.0, 0.0, 1.0])) < 0:
        Z = -Z

    def _proj(v):
        v = np.array(v, float)
        return v - np.dot(v, Z) * Z

    Xp = _proj([1.0, 0.0, 0.0])
    if np.linalg.norm(Xp) < 1e-9:
        Xp = _proj([0.0, 1.0, 0.0])
    Xp /= max(np.linalg.norm(Xp), 1e-12)
    Yp = np.cross(Z, Xp)
    Yp /= max(np.linalg.norm(Yp), 1e-12)
    return O, Xp, Yp, Z

def _project_uv_ax3(P3, O, X, Y):
    d = P3 - O
    return float(np.dot(d, X)), float(np.dot(d, Y))

def _project_uv(P3, O, Xp, Yp):
    d = P3 - O
    return float(np.dot(d, Xp)), float(np.dot(d, Yp))

# -------------------------
# Camera helper for thumbnails
# -------------------------
def _compute_ortho_camera(verts_local: np.ndarray, bounds, mode="iso",
                          iso_dir=(-1.0, -1.0, 1.0), margin=1.06, depth_pad=0.07):
    """
    Compute an orthographic camera that fits the part in screen space AND
    sets a safe clipping range along the view direction (no near-plane chops).
    Returns: dict with position, focal_point, up, parallel_scale, clipping=(near,far)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx, cy, cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)

    # View dir & 'up'
    if str(mode).lower() == "top":
        v = np.array([0.0, 0.0, 1.0], float)
        up_ref = np.array([0.0, 1.0, 0.0], float)
    else:
        v = np.array(iso_dir, float); v /= max(np.linalg.norm(v), 1e-12)
        up_ref = np.array([0.0, 1.0, 0.0], float)

    up = up_ref - np.dot(up_ref, v) * v
    if np.linalg.norm(up) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0], float)
        up = alt - np.dot(alt, v) * v
    up /= max(np.linalg.norm(up), 1e-12)
    s = np.cross(v, up); s /= max(np.linalg.norm(s), 1e-12)
    t = up

    # Screen fit (parallel_scale)
    P = verts_local
    Ps = P @ s
    Pt = P @ t
    width_screen  = float(Ps.max() - Ps.min()) if P.size else 1.0
    height_screen = float(Pt.max() - Pt.min()) if P.size else 1.0
    view_span = max(width_screen, height_screen)
    parallel_scale = 0.5 * view_span * float(margin)

    # Depth fit (camera distance & clipping range)
    C = np.array([cx, cy, cz], float)
    q = (P - C) @ v                        # depths of points relative to center along v
    q_min = float(q.min()) if P.size else -0.5
    q_max = float(q.max()) if P.size else  0.5
    depth_span = max(q_max - q_min, 1e-6)
    pad = max(depth_pad * depth_span, 1e-3)

    # Place camera in front of the frontmost point (along +v)
    d = q_max + pad
    cam_pos = C + d * v

    # Clipping distances from camera along DOP (positive): near just in front, far behind all points
    near = max(1e-3, pad * 0.5)
    far  = depth_span + pad * 1.5

    return {
        "position": tuple(cam_pos),
        "focal_point": (cx, cy, cz),
        "up": tuple(t),
        "parallel_scale": float(parallel_scale),
        "clipping": (float(near), float(far)),
    }

# -------------------------
# Thumbnails (camera selectable; ax3 optional for back-compat)
# -------------------------
def shape_to_thumbnail(solid, path, filename, ax3=None,
                       deflection=0.5, px=900,
                       camera="iso", iso_dir=(1.0, 1.0, 1.0), margin=1.06):
    """
    Render an orthographic thumbnail. If ax3 is given, we transform to that local frame.
    camera: "iso" (default) or "top"
    """
    BRepMesh_IncrementalMesh(solid, deflection)

    verts, faces = [], []
    topo = TopologyExplorer(solid)
    for face in topo.faces():
        tri = BRep_Tool.Triangulation(face, face.Location())
        if not tri:
            continue
        base = len(verts)
        for i in range(1, tri.NbNodes()+1):
            p = tri.Node(i)
            verts.append([p.X(), p.Y(), p.Z()])
        for i in range(1, tri.NbTriangles()+1):
            t = tri.Triangle(i); n1, n2, n3 = t.Get()
            faces.append([3, base+n1-1, base+n2-1, base+n3-1])
    if not verts or not faces:
        raise RuntimeError("❌ No triangulation data found in shape")

    # World → local (ax3 or world)
    O, X, Y, Z = _frame_from_ax3(ax3)
    R = np.column_stack([X, Y, Z])
    Rt = R.T
    V = np.array(verts, float)
    verts_local = (Rt @ (V.T - O.reshape(3,1))).T

    # Build mesh for PyVista
    faces_np = np.hstack(faces).astype(np.int32)
    mesh = pv.PolyData(verts_local, faces_np)

    # Camera
    cam = _compute_ortho_camera(verts_local, mesh.bounds, mode=camera, iso_dir=iso_dir, margin=margin)

    # 5) Render
    pl = pv.Plotter(off_screen=True, window_size=(px, px))
    pl.set_background("white")
    pl.add_mesh(mesh, color="lightsteelblue", show_edges=False)
    pl.enable_parallel_projection()              # orthographic (no skew)
    pl.camera.position = cam["position"]
    pl.camera.focal_point = cam["focal_point"]
    pl.camera.up = cam["up"]
    pl.camera.parallel_scale = cam["parallel_scale"]

    # ✅ Critical: set a safe clipping range; do NOT call reset afterwards
    if "clipping" in cam:
        pl.camera.clipping_range = cam["clipping"]

    out_png = Path(path) / f"{filename}.png"
    pl.show(screenshot=str(out_png))

    return out_png

from OCC.Core.BRepTools import BRepTools_WireExplorer
from math import hypot

def _ordered_uv_polyline(face, wire, project_uv, sampling_dist=1.0, samples_per_curve=None):
    """
    Return an ordered list of (u,v) along 'wire' so that successive points are contiguous.
    - Uses BRepTools_WireExplorer to iterate edges in wire order.
    - Reverses per-edge samples when needed to preserve continuity.
    - De-duplicates join points and closes the loop.
    """
    pts_uv = []
    prev = None

    wx = BRepTools_WireExplorer(wire, face)
    while wx.More():
        edge = wx.Current()
        c = BRepAdaptor_Curve(edge)

        # Build parameter samples
        ts = None
        try:
            disc = GCPnts_UniformAbscissa(c, float(sampling_dist))
            if disc.IsDone() and disc.NbPoints() >= 2:
                ts = [disc.Parameter(i) for i in range(1, disc.NbPoints()+1)]
        except Exception:
            pass
        if ts is None:
            t0, t1 = float(c.FirstParameter()), float(c.LastParameter())
            if c.GetType() == GeomAbs_Line:
                ts = [t0, t1]
            else:
                n = samples_per_curve or max(8, int(1 + abs(t1 - t0)/max(1e-6, sampling_dist)))
                ts = np.linspace(t0, t1, n)

        # Sample this edge in 3D → project to (u,v)
        edge_uv = []
        for t in ts:
            p = c.Value(float(t))
            P3 = np.array([p.X(), p.Y(), p.Z()], dtype=float)
            edge_uv.append(project_uv(P3))

        if not edge_uv:
            wx.Next()
            continue

        # Ensure continuity with previous point by reversing if needed
        if prev is not None:
            d_start = hypot(edge_uv[0][0] - prev[0], edge_uv[0][1] - prev[1])
            d_end   = hypot(edge_uv[-1][0] - prev[0], edge_uv[-1][1] - prev[1])
            if d_end < d_start:
                edge_uv.reverse()

            # drop duplicate join
            if hypot(edge_uv[0][0] - prev[0], edge_uv[0][1] - prev[1]) < 1e-8:
                edge_uv = edge_uv[1:]

        pts_uv.extend(edge_uv)
        prev = pts_uv[-1]
        wx.Next()

    # Close loop if needed
    if pts_uv:
        if hypot(pts_uv[0][0]-pts_uv[-1][0], pts_uv[0][1]-pts_uv[-1][1]) >= 1e-8:
            pts_uv.append(pts_uv[0])

    return pts_uv


# -------------------------
# DXF (plate outline)
# -------------------------
def generate_plate_dxf(aligned_solid,
                       filename,
                       sampling_dist: float = 1.0,
                       ax3=None,
                       put_bottom_left_at_origin: bool = True):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Largest planar face
    best_face, best_area = None, 0.0
    exp_f = TopExp_Explorer(aligned_solid, TopAbs_FACE)
    while exp_f.More():
        f = exp_f.Current()
        s = BRepAdaptor_Surface(f)
        if s.GetType() == 0:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(f, props)
            a = props.Mass()
            if a > best_area:
                best_area, best_face = a, f
        exp_f.Next()
    if best_face is None:
        raise RuntimeError("No planar face found on solid")

    # Projector
    if ax3 is not None:
        O, Xp, Yp, _ = _frame_from_ax3(ax3)
        project_uv = lambda P3: _project_uv_ax3(P3, O, Xp, Yp)
    else:
        O, Xp, Yp, _ = _stable_plane_frame(best_face)
        project_uv = lambda P3: _project_uv(P3, O, Xp, Yp)

    # Ordered polylines + circles per wire
    loops, circles = [], []
    wexp = TopExp_Explorer(best_face, TopAbs_WIRE)
    while wexp.More():
        wire = wexp.Current()

        # ✅ Try circle first
        as_circle = _wire_to_circle_uv(best_face, wire, project_uv)
        if as_circle is not None:
            circles.append(as_circle)  # (cu, cv, r)
        else:
            uv_pts = _ordered_uv_polyline(best_face, wire, project_uv, sampling_dist=sampling_dist)
            if uv_pts:
                loops.append(uv_pts)

        wexp.Next()

    # Optional BL shift
    if put_bottom_left_at_origin and (loops or circles):
        pts = [p for L in loops for p in L] + [(c[0], c[1]) for c in circles]
        allp = np.array(pts, float)
        shift = allp.min(axis=0)
        loops = [[(u - shift[0], v - shift[1]) for (u, v) in L] for L in loops]
        circles = [(cu - shift[0], cv - shift[1], r) for (cu, cv, r) in circles]

    # Write DXF
    doc = ezdxf.new("R2010")
    try:
        doc.header["$INSUNITS"] = 4  # mm
    except Exception:
        pass
    msp = doc.modelspace()
    for pts in loops:
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, format="xy", close=True)
    # ✅ real circles
    for (cu, cv, r) in circles:
        msp.add_circle((cu, cv), r)

    doc.saveas(str(filename))
    print(f"DXF written to {filename}")
    return filename

# -------------------------
# DXF render (matplotlib)
# -------------------------
from ezdxf.addons.drawing import matplotlib as ezdxf_renderer
def render_dxf_drawing(dxf_path, show_axes=False, output_png=None):
    """
    Load a DXF, render with matplotlib, and optionally save to PNG.
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_axes([0,0,1,1])
    ax.set_aspect('equal')

    ezdxf_renderer.draw_entities(msp, ax=ax)
    ax.autoscale()
    ax.axis('off' if not show_axes else 'on')

    if output_png:
        fig.savefig(str(output_png), dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Saved drawing thumbnail to {output_png}")
    else:
        plt.show()

# -------------------------
# BREP export (+ checksum)
# -------------------------
def export_solid_to_brep(solid, path, filename, algo="md5"):
    """
    Dump `solid` to path/filename.brep and return (checksum, brep_path).
    """
    dest_dir  = Path(path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{filename}.brep"

    fd, tmp_path = tempfile.mkstemp(suffix=".brep", dir=str(dest_dir))
    os.close(fd)

    try:
        if not breptools.Write(solid, str(tmp_path)):
            raise RuntimeError(f"❌ Failed to write BREP temp for {filename!r}")

        hasher = hashlib.new(algo)
        with open(tmp_path, "rb") as src, open(dest_file, "wb") as dst:
            for chunk in iter(lambda: src.read(8192), b""):
                dst.write(chunk)
                hasher.update(chunk)
        checksum = hasher.hexdigest()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"✅ BREP file written to {dest_file!r}")
    return checksum, dest_file

# -------------------------
# Mesh-fingerprint exporter
# -------------------------
def export_solid_to_brep_and_fingerprint(
    solid,
    out_dir,
    name,
    mesh_linear_deflection   = 1e-3,
    mesh_angular_deflection  = 0.1,
    mesh_rounding_precision  = 6,
    hash_algo                = "md5"
    ):
    """
    Write BREP and return a mesh-based fingerprint hash.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    brep_path = out_dir / f"{name}.brep"
    if brep_path.exists():
        brep_path.unlink()

    success = breptools.Write(solid, str(brep_path))
    if not success:
        raise RuntimeError(f"❌ Failed to write BREP to: {brep_path!r}")
    print(f"✅ BREP written to {brep_path!r}")

    BRepMesh_IncrementalMesh(
        solid,
        mesh_linear_deflection,
        False,
        mesh_angular_deflection,
        True
    )

    pts = []
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        tri  = BRep_Tool.Triangulation(face, face.Location())
        if tri:
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                pts.append((
                    round(p.X(), mesh_rounding_precision),
                    round(p.Y(), mesh_rounding_precision),
                    round(p.Z(), mesh_rounding_precision)
                ))
        exp.Next()
    if not pts:
        raise RuntimeError("⚠️ No mesh data found—check deflection settings!")

    unique_pts = sorted(set(pts))
    fmt = f"%.{mesh_rounding_precision}f,%.{mesh_rounding_precision}f,%.{mesh_rounding_precision}f;"
    buf = b"".join((fmt % pt).encode('ascii') for pt in unique_pts)

    h = hashlib.new(hash_algo)
    h.update(buf)
    fingerprint = h.hexdigest()
    return fingerprint

# -------------------------
# DXF fingerprint (Shapely)
# -------------------------
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
from shapely import affinity
from typing import Union, Tuple, Optional

def compute_dxf_fingerprint(dxf_path: Union[Path, str], tol: float = 0.01) -> str:
    """
    Quantize and hash a PCA/ax3-aligned DXF profile.
    """
    dxf_path = Path(dxf_path)
    doc = ezdxf.readfile(str(dxf_path))

    msp = doc.modelspace()

    segs = []

    # LINEs
    for line in msp.query('LINE'):
        start, end = line.dxf.start, line.dxf.end
        if start is None or end is None:
            continue
        try:
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
        except Exception:
            continue
        segs.append(LineString([(x1, y1), (x2, y2)]))

    # LWPOLYLINEs
    for pl in msp.query('LWPOLYLINE'):
        pts = list(pl.get_points('xy'))
        if not pts:
            continue
        closed = bool(pl.closed)
        for i in range(len(pts)-1):
            (x1, y1), (x2, y2) = pts[i], pts[i+1]
            segs.append(LineString([(float(x1), float(y1)), (float(x2), float(y2))]))
        if closed and len(pts) > 2:
            (x1, y1), (x2, y2) = pts[-1], pts[0]
            segs.append(LineString([(float(x1), float(y1)), (float(x2), float(y2))]))

    # ✅ CIRCLEs → chordize
    N = 64  # chord resolution for circles
    for ci in msp.query('CIRCLE'):
        cx, cy = float(ci.dxf.center[0]), float(ci.dxf.center[1])
        r = float(ci.dxf.radius)
        for k in range(N):
            a0 = 2*np.pi * k / N
            a1 = 2*np.pi * (k+1) / N
            x1, y1 = cx + r*np.cos(a0), cy + r*np.sin(a0)
            x2, y2 = cx + r*np.cos(a1), cy + r*np.sin(a1)
            segs.append(LineString([(x1, y1), (x2, y2)]))

    if not segs:
        raise RuntimeError("No LINE / LWPOLYLINE / CIRCLE geometry found in DXF for fingerprinting")


    merged = unary_union(segs)
    polys = list(polygonize(merged))
    if polys:
        poly = polys[0]
    else:
        poly = merged.convex_hull
        if poly.is_empty or not poly.exterior:
            raise RuntimeError("Cannot derive a loop for fingerprinting")

    cx, cy = poly.centroid.x, poly.centroid.y
    poly = affinity.translate(poly, xoff=-cx, yoff=-cy)
    minx, miny, maxx, maxy = poly.bounds
    max_dim = max(maxx - minx, maxy - miny)
    if max_dim == 0:
        raise RuntimeError("Degenerate profile with zero size")
    factor = 1.0 / max_dim
    poly = affinity.scale(poly, xfact=factor, yfact=factor, origin=(0,0))

    rounded = [(round(x/tol)*tol, round(y/tol)*tol) for x,y in poly.exterior.coords]
    ring = LineString(rounded)
    return hashlib.md5(ring.wkb).hexdigest()

# -------------------------
# 2D axes heuristic
# -------------------------
def _compute_edge_axes(segments, perp_tol=1e-3, share_tol=1e-6):
    """
    Given 2D segments [((x1,y1),(x2,y2)),…], return the first perpendicular pair.
    """
    info = []
    for p1, p2 in segments:
        P, Q = np.array(p1), np.array(p2)
        v = Q - P
        L = np.linalg.norm(v)
        if L < share_tol:
            continue
        info.append((L, v / L, P, Q))

    info.sort(key=lambda x: x[0], reverse=True)

    for Lx, vx, Ax, Bx in info:
        for Ly, vy, Py, Qy in info:
            if Ly >= Lx:
                continue
            if not (
                np.allclose(Py, Ax, atol=share_tol) or
                np.allclose(Qy, Ax, atol=share_tol) or
                np.allclose(Py, Bx, atol=share_tol) or
                np.allclose(Qy, Bx, atol=share_tol)
            ):
                continue
            if abs(np.dot(vx, vy)) <= perp_tol:
                if np.cross(vx, vy) < 0:
                    vy = -vy
                return vx, vy
    return None, None

# -------------------------
# Profile DXF (PCA or ax3 if provided)
# -------------------------
def export_profile_dxf_with_pca(
    shape,
    dxf_path: Union[Path, str],
    thumb_path: Optional[Union[Path, str]] = None,
    samples_per_curve: int = 16,
    fingerprint_tol: float = 0.5,
    ax3=None,
    canonicalize: bool = False,
) -> Tuple[str, Path, Path]:
    dxf_path = Path(dxf_path)

    # 1) Largest planar face
    best_face, best_area = None, 0.0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        surf = BRepAdaptor_Surface(f)
        if surf.GetType() == 0:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(f, props)
            a = props.Mass()
            if a > best_area:
                best_area, best_face = a, f
        exp.Next()
    if best_face is None:
        raise RuntimeError("No planar face found on shape")

    # ---- Path A: ax3 projection (no PCA) ----
    if ax3 is not None:
        O, Xp, Yp, _ = _frame_from_ax3(ax3)
        project_uv = lambda P3: _project_uv_ax3(P3, O, Xp, Yp)

        loops, circles = [], []
        wexp = TopExp_Explorer(best_face, TopAbs_WIRE)
        while wexp.More():
            wire = wexp.Current()
            as_circle = _wire_to_circle_uv(best_face, wire, project_uv)
            if as_circle is not None:
                circles.append(as_circle)  # (cu, cv, r)
            else:
                uv_pts = _ordered_uv_polyline(best_face, wire, project_uv, sampling_dist=1.0)
                if uv_pts:
                    loops.append(uv_pts)
            wexp.Next()

        # Canonicalize
        if canonicalize and (loops or circles):
            pts = [p for L in loops for p in L] + [(c[0], c[1]) for c in circles]
            A = np.array(pts, float)
            mn = A.min(axis=0)
            loops = [[(u - mn[0], v - mn[1]) for (u, v) in L] for L in loops]
            circles = [(cu - mn[0], cv - mn[1], r) for (cu, cv, r) in circles]

        doc = ezdxf.new(dxfversion="R2010")
        try:
            doc.header["$INSUNITS"] = 4  # millimeters
        except Exception:
            pass
        
        msp = doc.modelspace()
        for pts in loops:
            msp.add_lwpolyline(pts, format="xy", close=True)
        for (cu, cv, r) in circles:
            msp.add_circle((cu, cv), r)
        doc.saveas(str(dxf_path))

        # Thumbnail
        thumb_base = Path(thumb_path) if thumb_path else dxf_path.with_suffix('')
        thumbnail_path = thumb_base.with_suffix('.png')
        if qsave:
            try:
                qsave(msp, str(thumbnail_path), bg="#FFFFFF", fg="#000000")
            except Exception:
                _fallback_plot_segments(_loops_and_circles_to_segments(loops, circles), thumbnail_path)
        else:
            _fallback_plot_segments(_loops_and_circles_to_segments(loops, circles), thumbnail_path)

        try:
            fp = compute_dxf_fingerprint(dxf_path, tol=fingerprint_tol)
        except Exception:
            fp = ""
        return fp, dxf_path, thumbnail_path


    # ---- Path B: plane-projection + PCA orientation ----
    surf = BRepAdaptor_Surface(best_face)
    pln = surf.Plane()
    origin = np.array([pln.Location().X(), pln.Location().Y(), pln.Location().Z()], float)
    xdir3 = np.array([pln.XAxis().Direction().X(), pln.XAxis().Direction().Y(), pln.XAxis().Direction().Z()], float)
    ydir3 = np.array([pln.YAxis().Direction().X(), pln.YAxis().Direction().Y(), pln.YAxis().Direction().Z()], float)
    zdir3 = np.array([pln.Axis().Direction().X(),   pln.Axis().Direction().Y(),   pln.Axis().Direction().Z()], float)

    proj_plane = lambda P3: ((P3 - origin).dot(xdir3), (P3 - origin).dot(ydir3))

    raw_loops, raw_circles = [], []
    wexp = TopExp_Explorer(best_face, TopAbs_WIRE)
    while wexp.More():
        wire = wexp.Current()
        as_circle = _wire_to_circle_uv(best_face, wire, proj_plane)
        if as_circle is not None:
            raw_circles.append(as_circle)  # (cu, cv, r) in plane coords
        else:
            uv_pts = _ordered_uv_polyline(best_face, wire, proj_plane, sampling_dist=1.0)
            if uv_pts:
                raw_loops.append(uv_pts)
        wexp.Next()

    # PCA on all points (include circle samples so circles influence orientation)
    all_pts = [p for L in raw_loops for p in L]
    for (cu, cv, r) in raw_circles:
        for k in range(8):  # 8 samples around circle
            a = 2*np.pi * k / 8
            all_pts.append((cu + r*np.cos(a), cv + r*np.sin(a)))

    all_pts = np.array(all_pts, float)
    center2d = all_pts.mean(axis=0)
    cov2 = np.cov((all_pts - center2d).T)
    vals2, vecs2 = np.linalg.eigh(cov2)
    idx = np.argsort(vals2)[::-1]
    R = vecs2[:, idx].T
    if np.linalg.det(R) < 0:
        R[1, :] *= -1

    # Rotate everything
    loops = []
    for L in raw_loops:
        arr = np.array(L, float)
        arr2 = (R @ (arr - center2d).T).T
        loops.append([tuple(p) for p in arr2])

    circles = []
    for (cu, cv, r) in raw_circles:
        c2 = R @ (np.array([cu, cv]) - center2d)
        circles.append((float(c2[0]), float(c2[1]), float(r)))  # radius unchanged

    # Canonicalize
    if canonicalize and (loops or circles):
        pts2 = [p for L in loops for p in L] + [(c[0], c[1]) for c in circles]
        A = np.array(pts2, float)
        mn = A.min(axis=0)
        loops = [[(u - mn[0], v - mn[1]) for (u, v) in L] for L in loops]
        circles = [(cu - mn[0], cv - mn[1], r) for (cu, cv, r) in circles]

    # Write DXF
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for pts in loops:
        msp.add_lwpolyline(pts, format="xy", close=True)
    for (cu, cv, r) in circles:
        msp.add_circle((cu, cv), r)
    doc.saveas(str(dxf_path))

    # Thumbnail
    thumb_base = Path(thumb_path) if thumb_path else dxf_path.with_suffix('')
    thumbnail_path = thumb_base.with_suffix('.png')
    if qsave:
        try:
            qsave(msp, str(thumbnail_path), bg='#FFFFFF', fg='#000000')
        except Exception:
            _fallback_plot_segments(_loops_and_circles_to_segments(loops, circles), thumbnail_path)
    else:
        _fallback_plot_segments(_loops_and_circles_to_segments(loops, circles), thumbnail_path)

    fp = compute_dxf_fingerprint(dxf_path, tol=fingerprint_tol)
    return fp, dxf_path, thumbnail_path



def _fallback_plot_segments(segments, out_path):
    try:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        for (x1, y1), (x2, y2) in segments:
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8)
        ax.set_aspect('equal'); ax.axis('off')
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close(fig)
    except Exception:
        pass

def _loops_and_circles_to_segments(loops, circles, N=64):
    segs = []
    for pts in loops:
        for i in range(len(pts)-1):
            segs.append((pts[i], pts[i+1]))
    for (cx, cy, r) in circles:
        for k in range(N):
            a0 = 2*np.pi * k / N
            a1 = 2*np.pi * (k+1) / N
            p1 = (cx + r*np.cos(a0), cy + r*np.sin(a0))
            p2 = (cx + r*np.cos(a1), cy + r*np.sin(a1))
            segs.append((p1, p2))
    return segs
