from OCC.Core.BRepTools import breptools
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.GeomAbs import GeomAbs_C1
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from ezdxf.addons.drawing import matplotlib as ezdxf_renderer
from pathlib import Path
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

import os, hashlib
import numpy as np
import hashlib
import tempfile
import matplotlib.pyplot as plt
import os
import pyvista as pv
import ezdxf


from pathlib import Path
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect    import IFSelect_RetDone

from pathlib import Path
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect    import IFSelect_RetDone
from OCC.Core.Message     import Message_ProgressRange

def export_solid_to_step(solid, out_dir, filename):
    """
    Write `solid` to out_dir/filename.step using the 4-arg Transfer:
      Transfer(shape, STEPControl_AsIs, writeLegacyBRep, progress)
    Raises if transfer or write fails.
    """
    # make sure we actually got a shape
    if solid is None:
        raise ValueError("export_solid_to_step: got None instead of a TopoDS_Shape")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_path = out_dir / f"{filename}.step"

    writer = STEPControl_Writer()
    # optional: choose AP214
    # from OCC.Core.Interface import Interface_Static
    # Interface_Static.SetCVal("write.step.schema", "AP214")

    # — correctly call the 4-arg overload —
    prog = Message_ProgressRange()
    ok = writer.Transfer(solid, STEPControl_AsIs, True, prog)
    if ok != 1:
        raise RuntimeError(f"❌ STEP transfer failed for {filename!r} (status={ok})")

    status = writer.Write(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"❌ STEP write failed for {filename!r} (status={status})")

    print(f"✅ STEP saved to {step_path!r}")
    return step_path



def shape_to_thumbnail(solid, path, filename, deflection=0.5):
    # Mesh the shape
    BRepMesh_IncrementalMesh(solid, deflection)

    vertices = []
    faces = []

    topo = TopologyExplorer(solid)

    for face in topo.faces():
        triangulation = BRep_Tool.Triangulation(face, face.Location())
        if not triangulation:
            continue

        nb_nodes = triangulation.NbNodes()
        nb_triangles = triangulation.NbTriangles()

        offset = len(vertices)
        for i in range(1, nb_nodes + 1):
            pnt = triangulation.Node(i)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            faces.append([3, offset + n1 - 1, offset + n2 - 1, offset + n3 - 1])

    if not vertices or not faces:
        raise RuntimeError("❌ No triangulation data found in shape")

    pv_mesh = pv.PolyData(vertices, faces)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv_mesh, color="lightsteelblue", show_edges=False)
    plotter.set_background("white")
    plotter.camera_position = "iso"
    plotter.show(screenshot=Path(rf"{path}/{filename}.png"))

    print(f"🖼️ Thumbnail saved to {filename}")
    
    return(Path(rf"{path}/{filename}.png"))



import ezdxf
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa

def generate_plate_dxf(aligned_solid, filename, sampling_dist=1.0):
    """
    Extracts the largest planar face from `aligned_solid` (in the XY plane),
    samples each wire edge at ~`sampling_disbrepgprop.SurfacePropertiest` mm, and writes them as LWPolylines
    to DXF `filename`.
    """
    # 1) Find the largest face by area
    filename = Path(filename)
    best_face = None
    best_area = 0.0
    exp_f = TopExp_Explorer(aligned_solid, TopAbs_FACE)
    while exp_f.More():
        face = exp_f.Current()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        if area > best_area:
            best_area = area
            best_face = face
        exp_f.Next()

    if best_face is None:
        raise RuntimeError("No faces found on solid")

    # 2) Collect wires (outer contour + holes)
    wires_pts = []
    exp_w = TopExp_Explorer(best_face, TopAbs_WIRE)
    while exp_w.More():
        wire = exp_w.Current()
        pts = []
        exp_e = TopExp_Explorer(wire, TopAbs_EDGE)
        while exp_e.More():
            edge = exp_e.Current()
            # Instantiate curve adapter correctly
            curve = BRepAdaptor_Curve(edge)
            discretizer = GCPnts_UniformAbscissa(curve, sampling_dist)
            if discretizer.IsDone():
                for i in range(1, discretizer.NbPoints() + 1):
                    param = discretizer.Parameter(i)
                    p = curve.Value(param)
                    pts.append((p.X(), p.Y()))
            else:
                # fallback: start, mid, end
                p1 = curve.Value(curve.FirstParameter())
                p2 = curve.Value((curve.FirstParameter() + curve.LastParameter()) * 0.5)
                p3 = curve.Value(curve.LastParameter())
                pts.extend([(p1.X(), p1.Y()),
                            (p2.X(), p2.Y()),
                            (p3.X(), p3.Y())])
            exp_e.Next()
        # close the polyline
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        wires_pts.append(pts)
        exp_w.Next()

    # 3) Build and save DXF
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    for pts in wires_pts:
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, close=True)
    doc.saveas(filename)
    print(f"DXF written to {filename}")
    return filename

def render_dxf_drawing(dxf_path, show_axes=False, output_png=None):
    """
    Loads your DXF, renders it with matplotlib, and optionally
    saves the result to a PNG or displays it inline.
    """
    # 1) Read the DXF
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # 2) Set up a full‐figure axes
    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_axes([0,0,1,1])
    ax.set_aspect('equal')

    # 3) Draw all entities in modelspace
    ezdxf_renderer.draw_entities(msp, ax=ax)

    # 4) Tidy up
    ax.autoscale()
    ax.axis('off' if not show_axes else 'on')

    # 5) Output
    if output_png:
        fig.savefig(rf"c:\test_img.png", dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Saved drawing thumbnail to {output_png}")
    else:
        plt.show()


def export_solid_to_brep(solid, path, filename, algo="md5"):
    """
    Dump `solid` to path/filename.brep and return an MD5 (or whatever) checksum.
    Writes the temp file into the same directory so os.replace() never crosses drives.
    """
    dest_dir  = Path(path)
    dest_file = dest_dir / f"{filename}.brep"

    # 1) create a temp path *in the same directory* so os.replace won't fail
    fd, tmp_path = tempfile.mkstemp(suffix=".brep", dir=str(dest_dir))
    os.close(fd)

    # 2) write the BREP
    success = breptools.Write(solid, str(tmp_path))
    if not success:
        os.remove(tmp_path)
        raise RuntimeError(f"❌ Failed to write BREP to temp file for: {filename!r}")

    # 3) stream-copy + hash
    hasher = hashlib.new(algo)
    with open(tmp_path, "rb") as src, open(dest_file, "wb") as dst:
        for chunk in iter(lambda: src.read(8192), b""):
            dst.write(chunk)
            hasher.update(chunk)

    # 4) clean up
    os.remove(tmp_path)

    checksum = hasher.hexdigest()
    print(f"✅ BREP file written to {dest_file!r}")

    return checksum, dest_file


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
    1) Delete any existing out_dir/name.brep
    2) Write `solid` to out_dir/name.brep
    3) Build a fixed mesh, extract all mesh nodes, round & sort them
    4) Return a mesh-based fingerprint
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    brep_path = out_dir / f"{name}.brep"
    # 1) remove existing file if present
    if brep_path.exists():
        brep_path.unlink()

    # 2) write the B-Rep file
    success = breptools.Write(solid, str(brep_path))
    if not success:
        raise RuntimeError(f"❌ Failed to write BREP to: {brep_path!r}")
    print(f"✅ BREP written to {brep_path!r}")

    # 3) mesh the shape
    BRepMesh_IncrementalMesh(
        solid,
        mesh_linear_deflection,
        False,
        mesh_angular_deflection,
        True
    )

    # 4) collect & round all triangle nodes
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

    # 5) dedupe & sort
    unique_pts = sorted(set(pts))

    # 6) serialize & hash
    fmt = f"%.{mesh_rounding_precision}f,%.{mesh_rounding_precision}f,%.{mesh_rounding_precision}f;"
    buf = b"".join((fmt % pt).encode('ascii') for pt in unique_pts)

    h = hashlib.new(hash_algo)
    h.update(buf)
    fingerprint = h.hexdigest()

    return fingerprint

import math
import ezdxf
import numpy as np
from pathlib import Path

import hashlib
import numpy as np
import ezdxf
from pathlib import Path
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
from shapely import affinity
from typing import Union, Tuple, Optional
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.gp import gp_Vec

# Optional import for thumbnail generation via matplotlib
try:
    from ezdxf.addons.drawing.matplotlib import qsave
except ImportError:
    qsave = None


def compute_dxf_fingerprint(dxf_path: Union[Path, str], tol: float = 0.01) -> str:
    """
    Quantize and hash a PCA-aligned DXF profile:
      1) Load all LINE entities
      2) Merge into a geometry and extract the outer loop
         - Fallback: use convex hull if no loops are found
      3) Center at origin, scale longest dimension to 1.0
      4) Round coords to the given tolerance
      5) Hash the resulting geometry's WKB
    Returns an MD5 hex digest fingerprint.
    """
    dxf_path = Path(dxf_path)
    doc = ezdxf.readfile(str(dxf_path))

    # Extract all LINE segments
    lines = list(doc.modelspace().query('LINE'))
    segs = []
    for line in lines:
        start = line.dxf.start
        end = line.dxf.end
        if start is None or end is None:
            continue
        try:
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
        except Exception:
            continue
        segs.append(LineString([(x1, y1), (x2, y2)]))
    if not segs:
        raise RuntimeError("No LINE segments found in DXF for fingerprinting")
    merged = unary_union(segs)

    # Extract a closed loop, else use convex hull
    polys = list(polygonize(merged))
    if polys:
        poly = polys[0]
    else:
        poly = merged.convex_hull
        if poly.is_empty or not poly.exterior:
            raise RuntimeError("Cannot derive a loop for fingerprinting")

    # Normalize: center and scale
    cx, cy = poly.centroid.x, poly.centroid.y
    poly = affinity.translate(poly, xoff=-cx, yoff=-cy)
    minx, miny, maxx, maxy = poly.bounds
    max_dim = max(maxx - minx, maxy - miny)
    if max_dim == 0:
        raise RuntimeError("Degenerate profile with zero size")
    factor = 1.0 / max_dim
    poly = affinity.scale(poly, xfact=factor, yfact=factor, origin=(0,0))

    # Round coordinates
    rounded = [(round(x/tol)*tol, round(y/tol)*tol) for x,y in poly.exterior.coords]
    ring = LineString(rounded)

    # Hash WKB
    return hashlib.md5(ring.wkb).hexdigest()


# def export_profile_dxf_with_pca(shape,
#                                 dxf_path: Union[Path, str],
#                                 thumb_path: Optional[Union[Path, str]] = None,
#                                 samples_per_curve: int = 16,
#                                 fingerprint_tol: float = 0.01
#                                 ) -> Tuple[str, Path, Path]:
#     """
#     Export a PCA-aligned DXF profile, generate a thumbnail, and fingerprint it.

#     Args:
#       shape: OCC TopoDS_Shape to export
#       dxf_path: output DXF file path
#       thumb_path: optional base path for thumbnail (PNG). Defaults to same name as DXF.
#       samples_per_curve: number of points per curved edge
#       fingerprint_tol: rounding tolerance for fingerprint

#     Returns:
#       fingerprint (str): MD5 hash of quantized profile
#       dxf_path (Path): Path to saved DXF
#       thumbnail_path (Path): Path to saved PNG thumbnail
#     """
#     dxf_path = Path(dxf_path)
#     samples_per_curve = int(samples_per_curve)

#     # 1) Largest planar face
#     best_face, best_area = None, 0.0
#     exp = TopExp_Explorer(shape, TopAbs_FACE)
#     while exp.More():
#         f = exp.Current(); surf = BRepAdaptor_Surface(f)
#         if surf.GetType() != 0:
#             exp.Next(); continue
#         props = GProp_GProps(); brepgprop.SurfaceProperties(f, props)
#         area = props.Mass()
#         if area > best_area:
#             best_area, best_face = area, f
#         exp.Next()
#     if best_face is None:
#         raise RuntimeError("No planar face found on shape")

#     # 2) Boundary wires
#     analyser = ShapeAnalysis_FreeBounds(best_face)
#     bc = analyser.GetClosedWires(); wires = []
#     we = TopExp_Explorer(bc, TopAbs_WIRE)
#     while we.More(): wires.append(we.Current()); we.Next()
#     if not wires:
#         raise RuntimeError("No boundary wires found")

#     # 3) Sample & project
#     plane = BRepAdaptor_Surface(best_face).Plane()
#     all_pts, segments = [], []
#     for wire in wires:
#         ee = TopExp_Explorer(wire, TopAbs_EDGE)
#         while ee.More():
#             edge = ee.Current(); adaptor = BRepAdaptor_Curve(edge)
#             t0, t1 = adaptor.FirstParameter(), adaptor.LastParameter()
#             if adaptor.GetType() == GeomAbs_Line:
#                 params = [t0, t1]
#             else:
#                 params = np.linspace(float(t0), float(t1), samples_per_curve)
#             uv = []
#             for t in params:
#                 org = plane.Location()
#                 dx = gp_Vec(org, adaptor.Value(t)).Dot(gp_Vec(plane.XAxis().Direction()))
#                 dy = gp_Vec(org, adaptor.Value(t)).Dot(gp_Vec(plane.YAxis().Direction()))
#                 uv.append((dx, dy))
#             all_pts.extend(uv)
#             segments.extend(zip(uv[:-1], uv[1:]))
#             ee.Next()
#     if not segments:
#         raise RuntimeError("No segments generated from shape")

#     # 4) PCA rotation (right-handed)
#     pts = np.vstack(all_pts); center = pts.mean(axis=0)
#     cov = np.cov((pts-center).T)
#     vals, vecs = np.linalg.eigh(cov)
#     idx = np.argsort(vals)[::-1]
#     R = vecs[:,idx].T
#     if np.linalg.det(R) < 0:
#         R[1,:] *= -1

#     # 5) Rotate segments
#     rotated = []
#     for (x1,y1),(x2,y2) in segments:
#         p1 = R.dot(np.array([x1,y1]) - center)
#         p2 = R.dot(np.array([x2,y2]) - center)
#         rotated.append((tuple(p1), tuple(p2)))

#     # 6) Write DXF
#     doc = ezdxf.new(dxfversion="R2010"); msp = doc.modelspace()
#     for (u1,v1),(u2,v2) in rotated:
#         msp.add_line((u1,v1),(u2,v2))
#     doc.saveas(str(dxf_path))
#     # print(f"Wrote PCA-aligned DXF → {dxf_path}")

#     # 7) Thumbnail
#     thumb_base = Path(thumb_path) if thumb_path else dxf_path.with_suffix('')
#     thumbnail_path = thumb_base.with_suffix('.png')
#     def _manual(path):
#         try:
#             import matplotlib.pyplot as plt
#         except ImportError:
#             print("matplotlib not installed; cannot generate thumbnail")
#             return
#         fig, ax = plt.subplots(); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
#         for (x1,y1),(x2,y2) in rotated:
#             ax.plot([x1,x2],[y1,y2],'k-',linewidth=0.8)
#         ax.set_aspect('equal'); ax.axis('off')
#         fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
#         plt.close(fig)
#     if qsave:
#         try:
#             qsave(msp, str(thumbnail_path), bg='#FFFFFF', fg='#000000')
#             print(f"Saved thumbnail via qsave → {thumbnail_path}")
#         except Exception as e:
#             print(f"qsave failed: {e}, falling back to manual thumbnail")
#             _manual(thumbnail_path)
#     else:
#         print("ezdxf matplotlib extension not installed; using manual thumbnail")
#         _manual(thumbnail_path)

#     # 8) Fingerprint
#     fp = compute_dxf_fingerprint(dxf_path, tol=fingerprint_tol)
#     print(f"✅ DXF saved to {dxf_path}")
#     return fp, dxf_path, thumbnail_path




def _compute_edge_axes(segments, perp_tol=1e-3, share_tol=1e-6):
    """
    Given a list of 2D segments [((x1,y1),(x2,y2)),…], returns the first
    perpendicular pair found when scanning edges by descending length.
    Returns (x_axis, y_axis) as unit vectors, or (None, None) if none found.
    """
    info = []
    for p1, p2 in segments:
        P, Q = np.array(p1), np.array(p2)
        v = Q - P
        L = np.linalg.norm(v)
        if L < share_tol:
            continue
        info.append((L, v / L, P, Q))

    # sort by length descending
    info.sort(key=lambda x: x[0], reverse=True)

    # scan for first (longer edge, shorter perpendicular neighbor)
    for Lx, vx, Ax, Bx in info:
        for Ly, vy, Py, Qy in info:
            if Ly >= Lx:
                continue
            # must share an endpoint
            if not (
                np.allclose(Py, Ax, atol=share_tol) or
                np.allclose(Qy, Ax, atol=share_tol) or
                np.allclose(Py, Bx, atol=share_tol) or
                np.allclose(Qy, Bx, atol=share_tol)
            ):
                continue
            # check perpendicularity
            if abs(np.dot(vx, vy)) <= perp_tol:
                # enforce right-handed (2D cross-product > 0)
                if np.cross(vx, vy) < 0:
                    vy = -vy
                return vx, vy

    return None, None


def export_profile_dxf_with_pca(
    shape,
    dxf_path: Union[Path, str],
    thumb_path: Optional[Union[Path, str]] = None,
    samples_per_curve: int = 16,
    fingerprint_tol: float = 0.5
) -> Tuple[str, Path, Path]:
    dxf_path = Path(dxf_path)
    samples_per_curve = int(samples_per_curve)

    # 1) Largest planar face
    best_face, best_area = None, 0.0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        surf = BRepAdaptor_Surface(f)
        if surf.GetType() != 0:  # not a plane
            exp.Next()
            continue
        props = GProp_GProps()
        brepgprop.SurfaceProperties(f, props)
        area = props.Mass()
        if area > best_area:
            best_area, best_face = area, f
        exp.Next()

    if best_face is None:
        raise RuntimeError("No planar face found on shape")

    # 2–5) Sample only true in-plane edges using the face's normal
    surf = BRepAdaptor_Surface(best_face)
    pln = surf.Plane()
    origin = np.array([
        pln.Location().X(),
        pln.Location().Y(),
        pln.Location().Z()
    ])
    xdir3 = np.array([
        pln.XAxis().Direction().X(),
        pln.XAxis().Direction().Y(),
        pln.XAxis().Direction().Z()
    ])
    ydir3 = np.array([
        pln.YAxis().Direction().X(),
        pln.YAxis().Direction().Y(),
        pln.YAxis().Direction().Z()
    ])
    zdir3 = np.array([
        pln.Axis().Direction().X(),
        pln.Axis().Direction().Y(),
        pln.Axis().Direction().Z()
    ])  # face normal

    all_pts, segments = [], []
    we = TopExp_Explorer(best_face, TopAbs_WIRE)
    while we.More():
        wire = we.Current()
        ee = TopExp_Explorer(wire, TopAbs_EDGE)
        while ee.More():
            edge = ee.Current()
            adaptor = BRepAdaptor_Curve(edge)
            t0, t1 = adaptor.FirstParameter(), adaptor.LastParameter()
            if adaptor.GetType() == GeomAbs_Line:
                ts = [t0, t1]
            else:
                ts = np.linspace(float(t0), float(t1), samples_per_curve)

            # sample 3D points
            pts3d = []
            for t in ts:
                p = adaptor.Value(t)
                pts3d.append(np.array([p.X(), p.Y(), p.Z()]))

            # filter & project segments
            for P3, Q3 in zip(pts3d, pts3d[1:]):
                v3 = Q3 - P3
                L3 = np.linalg.norm(v3)
                if L3 < 1e-9:
                    continue
                # drop thickness edges
                if abs(np.dot(v3 / L3, zdir3)) > 1e-3:
                    continue

                # project into 2D face plane
                for Qp in (P3, Q3):
                    d = Qp - origin
                    all_pts.append((d.dot(xdir3), d.dot(ydir3)))
                segments.append((all_pts[-2], all_pts[-1]))
            ee.Next()
        we.Next()

    if not segments:
        raise RuntimeError("No in-plane segments found – check face sampling")

    # 6) Try longest+perpendicular rule
    x_axis, y_axis = _compute_edge_axes(segments)

    # 7) Fall back to in-plane PCA if no perp pair
    pts2 = np.vstack(all_pts)
    center2d = pts2.mean(axis=0)
    if x_axis is None:
        cov2 = np.cov((pts2 - center2d).T)
        vals2, vecs2 = np.linalg.eigh(cov2)
        idx2 = np.argsort(vals2)[::-1]
        R = vecs2[:, idx2].T
        if np.linalg.det(R) < 0:
            R[1, :] *= -1
    else:
        R = np.vstack([x_axis, y_axis])

    # 8) Rotate & write DXF
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for (u1, v1), (u2, v2) in segments:
        p1 = R.dot(np.array([u1, v1]) - center2d)
        p2 = R.dot(np.array([u2, v2]) - center2d)
        msp.add_line(tuple(p1), tuple(p2))
    doc.saveas(str(dxf_path))

    # 9) Thumbnail (manual + qsave fallback)
    thumb_base = Path(thumb_path) if thumb_path else dxf_path.with_suffix('')
    thumbnail_path = thumb_base.with_suffix('.png')

    def _manual(path):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        for (u1, v1), (u2, v2) in segments:
            p1 = R.dot(np.array([u1, v1]) - center2d)
            p2 = R.dot(np.array([u2, v2]) - center2d)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close(fig)

    if 'qsave' in globals() and qsave:
        try:
            qsave(msp, str(thumbnail_path), bg='#FFFFFF', fg='#000000')
        except Exception:
            _manual(thumbnail_path)
    else:
        _manual(thumbnail_path)

    # 10) Fingerprint
    fp = compute_dxf_fingerprint(dxf_path, tol=fingerprint_tol)

    return fp, dxf_path, thumbnail_path
