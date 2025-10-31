# geometry_utils.py — drop-in module
# Python 3.8+ compatible

from pathlib import Path
from typing import Optional, List, Dict, Any
import math
import numpy as np

# --- pythonOCC / OCCT imports ---
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import (
    TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_SOLID
)
from OCC.Core.TopoDS import topods, TopoDS_Shape

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from OCC.Core.gp import (
    gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax3, gp_Trsf, gp_Pln
)

from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)

from OCC.Core.BRepAdaptor import (
    BRepAdaptor_Curve, BRepAdaptor_Surface
)

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds


# ======================================================================================
# Diagnostics (optional): filled by compute_section_area() in robust mode
# ======================================================================================
CSA_DIAG: Dict[str, Any] = {}


# ======================================================================================
# Simple helpers
# ======================================================================================

# --- Hardened orientation helpers ---

import math
import numpy as np
from OCC.Core.gp import gp_Ax3, gp_Ax2, gp_Ax1, gp_Pnt, gp_Dir, gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

def _trsf_world_to_local(ax3: gp_Ax3) -> gp_Trsf:
    """Build world → local(ax3) transform; ax3 uses (origin, Z, X)."""
    world = gp_Ax3(gp_Pnt(0,0,0), gp_Dir(0,0,1), gp_Dir(1,0,0))
    t = gp_Trsf(); t.SetDisplacement(world, ax3)
    return t

def _bbox_world(shape):
    box = Bnd_Box(); brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)

def _dstv_min_corner_world(obb_geom):
    """Compute the DSTV min-corner (rear-bottom-left) point in world coords from OBB data."""
    c   = obb_geom["aligned_center"]
    ex  = float(obb_geom["aligned_extents"][0])
    ey  = float(obb_geom["aligned_extents"][1])
    ez  = float(obb_geom["aligned_extents"][2])
    dx  = gp_Vec(obb_geom["aligned_dir_x"]).Scaled(-0.5 * ex)
    dy  = gp_Vec(obb_geom["aligned_dir_y"]).Scaled(-0.5 * ey)
    dz  = gp_Vec(obb_geom["aligned_dir_z"]).Scaled(-0.5 * ez)
    return c.Translated(dx + dy + dz)

def _enforce_long_leg_on_Y_if_angle(shape, profile_type: str, dims: Dict|None):
    if profile_type != "L" or not dims: return shape, 0
    legY_nom = float(dims.get("leg_y", dims.get("height", 0.0)))
    legZ_nom = float(dims.get("leg_z", dims.get("width",  0.0)))
    want_Y, want_Z = (max(legY_nom, legZ_nom), min(legY_nom, legZ_nom))
    xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(shape)
    H, W = (ymax-ymin, zmax-zmin)
    tol = max(2.0, 0.01*max(want_Y, want_Z, 1.0))  # 2 mm or 1%
    def _close(a,b): return abs(a-b) <= tol
    if _close(H, want_Y) and _close(W, want_Z): return shape, 0
    sh90 = _rotX(shape, 1)
    xmin2,xmax2,ymin2,ymax2,zmin2,zmax2 = _bbox_local(sh90)
    H2,W2 = (ymax2-ymin2, zmax2-zmin2)
    if _close(H2, want_Y) and _close(W2, want_Z):
        return sh90, 1
    return shape, 0  # leave as-is; next stage will still pick heel at origin

def _to_origin_min_corner(shape):
    xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(shape)
    tr = gp_Trsf(); tr.SetTranslation(gp_Pnt(xmin,ymin,zmin), gp_Pnt(0,0,0))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

def _move_to_dstv_local(shape, obb_geom):
    """
    Translate shape so min-corner → (0,0,0) and rotate axes to world DSTV (Z up, X forward).
    Returns (shape_local, dstv_frame_world, (L,H,W)).
    """
    # 1) translate min-corner → origin
    min_corner = _dstv_min_corner_world(obb_geom)
    t_tr = gp_Trsf(); t_tr.SetTranslation(gp_Vec(min_corner, gp_Pnt(0,0,0)))
    sh_t = BRepBuilderAPI_Transform(shape, t_tr, True).Shape()

    # 2) rotate from OBB frame to world DSTV (Z up, X forward)
    obb_frame = gp_Ax3(gp_Pnt(0,0,0),
                       gp_Dir(obb_geom["aligned_dir_z"].XYZ()),
                       gp_Dir(obb_geom["aligned_dir_x"].XYZ()))
    dstv_world = gp_Ax3(gp_Pnt(0,0,0), gp_Dir(0,0,1), gp_Dir(1,0,0))
    t_rot = gp_Trsf(); t_rot.SetDisplacement(obb_frame, dstv_world)
    sh_local = BRepBuilderAPI_Transform(sh_t, t_rot, True).Shape()

    L, H, W = [float(v) for v in obb_geom["aligned_extents"]]
    return sh_local, dstv_world, (L, H, W)

import math
import numpy as np
from OCC.Core.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Ax1, gp_Trsf

def _section_yz_points(shape_local, x_pos: float, tol=1e-6):
    """
    Take a section at X = x_pos (DSTV-local), return list of (y,z) vertices from closed wires.
    """
    plane = gp_Pln(gp_Ax3(gp_Pnt(x_pos, 0, 0), gp_Dir(1, 0, 0)))  # normal along +X
    sec = BRepAlgoAPI_Section(shape_local, plane)
    sec.ComputePCurveOn1(True); sec.Approximation(True); sec.Build()
    if not sec.IsDone():
        return []

    # Keep only closed wires; drop loose edges
    fb = ShapeAnalysis_FreeBounds(sec.Shape(), tol, False, False)
    wires = []
    seq = fb.GetClosedWires()
    if seq and hasattr(seq, "Length"):
        for i in range(1, seq.Length() + 1):
            s = seq.Value(i)
            if s.ShapeType() == TopAbs_WIRE:
                wires.append(topods.Wire(s))

    pts = []
    for w in wires:
        exp = TopExp_Explorer(w, TopAbs_VERTEX)
        while exp.More():
            v = topods.Vertex(exp.Current())
            p = BRep_Tool.Pnt(v)  # already DSTV-local coords
            pts.append((p.Y(), p.Z()))
            exp.Next()
    return pts

def _rotate_180_about_world_x(shape):
    axisX = gp_Ax1(gp_Pnt(0,0,0), gp_Dir(1,0,0))
    r = gp_Trsf(); r.SetRotation(axisX, math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()

def canonicalize_channel_by_section(shape_local, L, H, W, tol_edge=2.0):
    """
    Decide flip from a mid-body section. Aim: more contour near z≈0 than z≈W.
    """
    # Pick a safe section away from ends
    x_pos = max(0.15*L, min(0.5*L, 0.25*L))
    yz = _section_yz_points(shape_local, x_pos)
    if not yz:
        # fallback: keep as-is
        return shape_local

    near_z0 = sum(1 for _, z in yz if abs(z - 0.0) <= tol_edge)
    near_zW = sum(1 for _, z in yz if abs(z - W)   <= tol_edge)

    # If the "toe" contour is closer to the far flange (z≈W), flip once
    if near_zW > near_z0:
        print("🔁 Section says channel toe is at z≈W → rotate 180° about X")
        shape_local = _rotate_180_about_world_x(shape_local)
    return shape_local

def canonicalize_angle_by_section(shape_local, L, H, W, tol_edge=2.0):
    """
    Aim: heel at (y≈0, z≈0). If most contour mass hugs y≈H/z≈W instead, flip once.
    """
    x_pos = max(0.15*L, min(0.5*L, 0.25*L))
    yz = _section_yz_points(shape_local, x_pos)
    if not yz:
        return shape_local

    near_y0 = sum(1 for y, _ in yz if abs(y - 0.0) <= tol_edge)
    near_yH = sum(1 for y, _ in yz if abs(y - H)   <= tol_edge)
    near_z0 = sum(1 for _, z in yz if abs(z - 0.0) <= tol_edge)
    near_zW = sum(1 for _, z in yz if abs(z - W)   <= tol_edge)

    # If far planes dominate, flip once about X
    far_score   = (near_yH > near_y0) + (near_zW > near_z0)
    near_score  = (near_y0 >= near_yH) + (near_z0 >= near_zW)
    if far_score > near_score:
        print("🔁 Section says angle heel near far planes → rotate 180° about X")
        shape_local = _rotate_180_about_world_x(shape_local)
    return shape_local


def _rotate_180_about_world_x(shape):
    axisX = gp_Ax1(gp_Pnt(0,0,0), gp_Dir(1,0,0))
    r = gp_Trsf(); r.SetRotation(axisX, math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()

def _canonicalize_channel(shape_local, H, W, tol=1.0):
    """
    Channel canonical: toe at z≈0, y spans [0,H]. If bbox closer to far-planes, flip once.
    """
    _, _, y0, y1, z0, z1 = _bbox_world(shape_local)
    dz_near = min(abs(z0-0.0), abs(z1-0.0))
    dz_far  = min(abs(z0-W),   abs(z1-W))
    dy_near = min(abs(y0-0.0), abs(y1-0.0))
    dy_far  = min(abs(y0-H),   abs(y1-H))
    if (dz_far + 1e-6 < dz_near - 1e-6) or (dy_far + 1e-6 < dy_near - 1e-6):
        print("🔁 Canonicalize Channel → rotate 180° about X")
        shape_local = _rotate_180_about_world_x(shape_local)
    return shape_local

def _canonicalize_angle(shape_local, H, W, tol=1.0):
    """
    Angle canonical: heel at (y≈0, z≈0). If bbox closer to far-planes, flip once.
    """
    _, _, y0, y1, z0, z1 = _bbox_world(shape_local)
    dy_near = min(abs(y0-0.0), abs(y1-0.0))
    dy_far  = min(abs(y0-H),   abs(y1-H))
    dz_near = min(abs(z0-0.0), abs(z1-0.0))
    dz_far  = min(abs(z0-W),   abs(z1-W))
    if (dy_far + 1e-6 < dy_near - 1e-6) or (dz_far + 1e-6 < dz_near - 1e-6):
        print("🔁 Canonicalize Angle → rotate 180° about X")
        shape_local = _rotate_180_about_world_x(shape_local)
    return shape_local


def _rotate_180_about_local_x(shape, ax3: gp_Ax3):
    """Rotate shape 180° about local X (length) axis of ax3."""
    axisX = gp_Ax1(ax3.Location(), ax3.XDirection())
    r = gp_Trsf(); r.SetRotation(axisX, math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()

def _enforce_canonical_channel(shape, frame_ax3: gp_Ax3, H: float, W: float):
    """
    Channel (U) canonical pose: toe at Z≈0 and Y spans [0,H].
    If bbox hugs Z≈W (far) or Y≈H (far), flip once about X.
    """
    ymin, ymax, zmin, zmax = _bbox_local(shape, frame_ax3)
    dz_near = min(abs(zmin-0.0), abs(zmax-0.0))
    dz_far  = min(abs(zmin-W),   abs(zmax-W))
    dy_near = min(abs(ymin-0.0), abs(ymax-0.0))
    dy_far  = min(abs(ymin-H),   abs(ymax-H))
    if (dz_far < dz_near) or (dy_far < dy_near):
        print("🔁 Canonicalize Channel: rotating 180° about local X")
        shape = _rotate_180_about_local_x(shape, frame_ax3)
    return shape

def _enforce_canonical_angle(shape, frame_ax3: gp_Ax3, H: float, W: float):
    """
    Angle (L) canonical pose: heel at (Y≈0, Z≈0).
    If bbox hugs Y≈H and/or Z≈W (far edges), flip once about X.
    """
    ymin, ymax, zmin, zmax = _bbox_local(shape, frame_ax3)
    dy_near = min(abs(ymin-0.0), abs(ymax-0.0))
    dy_far  = min(abs(ymin-H),   abs(ymax-H))
    dz_near = min(abs(zmin-0.0), abs(zmax-0.0))
    dz_far  = min(abs(zmin-W),   abs(zmax-W))
    if (dy_far < dy_near) or (dz_far < dz_near):
        print("🔁 Canonicalize Angle: rotating 180° about local X")
        shape = _rotate_180_about_local_x(shape, frame_ax3)
    return shape


def _wire_area_on_plane(plane: gp_Pln, wire) -> float:
    """Unsigned area of a single wire on a given plane."""
    face = BRepBuilderAPI_MakeFace(plane, wire).Face()
    gp = GProp_GProps()
    brepgprop.SurfaceProperties(face, gp)
    return abs(gp.Mass())


def count_solids_in_shape(shape: TopoDS_Shape) -> int:
    """Count standalone SOLIDs (detects multi-body STEP)."""
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _dir_to_np(d: gp_Dir) -> np.ndarray:
    return np.array([d.X(), d.Y(), d.Z()], dtype=float)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _cluster_dirs_weighted(dirs_areas: List[tuple], ang_tol_deg: float = 10.0) -> List[Dict[str, Any]]:
    """
    Cluster directions (with areas) by angular proximity. Folds ±n into the same cluster.
    Returns: [{'rep': unit_vec, 'area': float, 'count': int}, ...] sorted by area desc.
    """
    clusters: List[Dict[str, Any]] = []
    for nvec, a in dirs_areas:
        v = nvec.copy()
        # fold to a consistent hemisphere using a lexicographic rule
        if v[0] < 0 or (abs(v[0]) < 1e-12 and (v[1] < 0 or (abs(v[1]) < 1e-12 and v[2] < 0))):
            v = -v
        placed = False
        for c in clusters:
            rep = c['rep']
            if _angle_between(v, rep) <= ang_tol_deg or _angle_between(v, -rep) <= ang_tol_deg:
                c['rep'] = (c['rep'] * c['area'] + v * a) / (c['area'] + a)
                c['area'] += a
                c['count'] += 1
                placed = True
                break
        if not placed:
            clusters.append({'rep': v, 'area': float(a), 'count': 1})

    for c in clusters:
        nrm = np.linalg.norm(c['rep'])
        if nrm > 0:
            c['rep'] = c['rep'] / nrm
    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ======================================================================================
# Plate guard (alignment-free): multi-body + planar face normal clustering
# ======================================================================================

def plate_guard_faces(shape: TopoDS_Shape,
                      ang_tol_deg: float = 10.0,
                      dominant_min_frac: float = 0.55,
                      second_max_frac: float = 0.30,
                      big_cluster_frac: float = 0.10,
                      min_planar_frac: float = 0.70) -> (bool, Dict[str, Any]):
    """
    Alignment-free 'is this plausibly a PLATE?' using planar faces only.

    Accept as plate iff:
      - enough of the area is planar (>= min_planar_frac)
      - one dominant normal cluster holds >= dominant_min_frac of planar area
      - second cluster <= second_max_frac
      - no more than two 'big' clusters (>= big_cluster_frac each)
    """
    dirs_areas: List[tuple] = []
    total_area = 0.0
    total_planar_area = 0.0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = topods.Face(exp.Current())
        gp_props = GProp_GProps()
        brepgprop.SurfaceProperties(f, gp_props)
        a = float(abs(gp_props.Mass()))
        total_area += a

        srf = BRepAdaptor_Surface(f)
        # GeomAbs_Plane == 0
        if srf.GetType() == 0:
            n = srf.Plane().Axis().Direction()
            dirs_areas.append((_dir_to_np(n), a))
            total_planar_area += a
        exp.Next()

    if total_area <= 1e-12 or total_planar_area <= 1e-12:
        return False, {"reason": "no_area"}

    planar_frac = total_planar_area / total_area
    clusters = _cluster_dirs_weighted(dirs_areas, ang_tol_deg=ang_tol_deg)
    areas = [c['area'] for c in clusters]
    top = areas[0] if areas else 0.0
    second = areas[1] if len(areas) > 1 else 0.0
    top_frac = top / total_planar_area if total_planar_area > 0 else 0.0
    second_frac = second / total_planar_area if total_planar_area > 0 else 0.0
    big_clusters = sum(1 for a in areas if a / max(total_planar_area, 1e-12) >= big_cluster_frac)

    plausible = (
        planar_frac >= min_planar_frac and
        top_frac >= dominant_min_frac and
        second_frac <= second_max_frac and
        big_clusters <= 2
    )
    diag = {
        "planar_frac": float(planar_frac),
        "top_frac": float(top_frac),
        "second_frac": float(second_frac),
        "big_clusters": int(big_clusters),
        "n_clusters": len(clusters),
    }
    return plausible, diag


def plate_guard_heuristics(shape: TopoDS_Shape,
                           face_ang_tol_deg: float = 10.0,
                           dominant_min_frac: float = 0.55,
                           second_max_frac: float = 0.30,
                           big_cluster_frac: float = 0.10,
                           min_planar_frac: float = 0.70) -> (bool, Dict[str, Any]):
    """Decide if we should even ATTEMPT plate handling."""
    # 1) multi-body -> veto
    n_solids = count_solids_in_shape(shape)
    if n_solids > 1:
        return False, {"reason": "multi_body_shape", "n_solids": n_solids}

    # 2) face-based guard
    ok, diag_faces = plate_guard_faces(
        shape,
        ang_tol_deg=face_ang_tol_deg,
        dominant_min_frac=dominant_min_frac,
        second_max_frac=second_max_frac,
        big_cluster_frac=big_cluster_frac,
        min_planar_frac=min_planar_frac,
    )
    if not ok:
        diag_faces["reason"] = diag_faces.get("reason", "face_guard_veto")
        return False, diag_faces

    return True, {"reason": "pass_faces"}


# ======================================================================================
# Sectioning helpers
# ======================================================================================

def _extract_outer_and_holes(edges_shape: TopoDS_Shape, plane: gp_Pln, tol: float = 1e-6):
    """
    From section edges return:
      outer_wire (TopoDS_Wire or None),
      inner_wires (list of TopoDS_Wire),
      area_outer (float, outer loop gross area).
    """
    fb = ShapeAnalysis_FreeBounds(edges_shape, tol, False, False)
    seq = fb.GetClosedWires()
    if seq is None or not hasattr(seq, "Length") or seq.Length() == 0:
        return None, [], 0.0

    wires = []
    for i in range(1, seq.Length() + 1):
        s = seq.Value(i)
        if s.ShapeType() == TopAbs_WIRE:
            wires.append(topods.Wire(s))
    if not wires:
        return None, [], 0.0

    areas = [_wire_area_on_plane(plane, w) for w in wires]
    i_outer = int(max(range(len(areas)), key=lambda k: areas[k]))
    outer = wires[i_outer]
    inners = [w for j, w in enumerate(wires) if j != i_outer]
    return outer, inners, float(areas[i_outer])


# ======================================================================================
# Alignment, OBB & transforms
# ======================================================================================

def robust_align_solid_from_geometry(solid: TopoDS_Shape, tol: float = 1e-3):
    """
    Align using the longest edge on the largest planar face.
    Returns: (aligned_shape, trsf, to_cs, largest_face, dir_x, dir_y, dir_z)
    """
    # Step 1: largest planar face
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    largest_face = None
    max_area = 0.0
    while explorer.More():
        face = topods.Face(explorer.Current())
        gp_props = GProp_GProps()
        brepgprop.SurfaceProperties(face, gp_props)
        area = gp_props.Mass()
        if area > max_area:
            max_area = area
            largest_face = face
        explorer.Next()
    if not largest_face:
        raise RuntimeError("No planar face found.")

    # Step 2: longest edge on that face
    edge_exp = TopExp_Explorer(largest_face, TopAbs_EDGE)
    longest_vec = None
    max_length = 0.0
    while edge_exp.More():
        edge = topods.Edge(edge_exp.Current())
        curve = BRepAdaptor_Curve(edge)
        p1 = curve.Value(curve.FirstParameter())
        p2 = curve.Value(curve.LastParameter())
        vec = gp_Vec(p1, p2)
        length = vec.Magnitude()
        if length > max_length:
            max_length = length
            longest_vec = vec
        edge_exp.Next()
    if not longest_vec or longest_vec.Magnitude() < tol:
        raise RuntimeError("Failed to find valid longest edge.")

    dir_x = gp_Dir(longest_vec)

    # Step 3: face normal as Z
    surf_adapt = BRepAdaptor_Surface(largest_face)
    if surf_adapt.GetType() != 0:
        raise RuntimeError("Largest face is not planar.")
    dir_z = surf_adapt.Plane().Axis().Direction()

    # Step 4: orthonormalize
    dir_y = dir_z.Crossed(dir_x)
    dir_x = dir_y.Crossed(dir_z)

    # Step 5: build transform (gp_Ax3(origin, Z, X))
    origin = gp_Pnt(0, 0, 0)
    from_cs = gp_Ax3(origin, dir_z, dir_x)
    to_cs   = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))

    trsf = gp_Trsf()
    trsf.SetDisplacement(from_cs, to_cs)
    aligned = BRepBuilderAPI_Transform(solid, trsf, True).Shape()

    return aligned, trsf, to_cs, largest_face, dir_x, dir_y, dir_z


def ensure_right_handed(dir_x: gp_Dir, dir_y: gp_Dir, dir_z: gp_Dir):
    """Ensure right-handed coordinate system."""
    x = np.array([dir_x.X(), dir_x.Y(), dir_x.Z()])
    y = np.array([dir_y.X(), dir_y.Y(), dir_y.Z()])
    z = np.array([dir_z.X(), dir_z.Y(), dir_z.Z()])
    if np.dot(np.cross(x, y), z) < 0:
        print("⚠️ Detected left-handed system. Reversing Z to enforce right-handed convention.")
        dir_z = gp_Dir(-dir_z.X(), -dir_z.Y(), -dir_z.Z())
    return dir_x, dir_y, dir_z


def compute_obb_geometry(aligned_shape: TopoDS_Shape) -> Dict[str, Any]:
    """Axis-aligned bounding box geometry (post alignment)."""
    aabb = Bnd_Box()
    brepbndlib.Add(aligned_shape, aabb)
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = aabb.Get()
    except Exception:
        raise RuntimeError("Failed to extract bounding box from shape.")

    length = xmax - xmin
    height = ymax - ymin
    width  = zmax - zmin

    center = gp_Pnt(
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0
    )

    return {
        "aligned_extents": [length, height, width],
        "aligned_dir_x": gp_Dir(1, 0, 0),
        "aligned_dir_y": gp_Dir(0, 1, 0),
        "aligned_dir_z": gp_Dir(0, 0, 1),
        "aligned_center": center
    }


# ======================================================================================
# Volume / STEP IO
# ======================================================================================

def get_volume_from_step(step_path: Path) -> float:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STEP file: {step_path!r}")
    reader.TransferRoots()
    shape = reader.OneShape()
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def get_volume_from_shape(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


# ======================================================================================
# Cross-sectional area (robust)
# ======================================================================================

def compute_section_area(solid: TopoDS_Shape,
                         axis_dir: gp_Dir = gp_Dir(1, 0, 0),
                         sample_pos: Optional[float] = None,
                         tol: float = 1e-6) -> float:
    """
    GROSS cross-sectional area (outer boundary only).

    - If sample_pos is provided (offset from xmin), computes a single slice.
    - Else: samples >=5 sections along length, clusters areas by similarity (3%),
      and returns the median of the largest cluster (most common section).
    """
    # bounds
    bbox = Bnd_Box(); brepbndlib.Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    L = xmax - xmin
    if L <= 1e-12:
        return 0.0

    def _gross_area_at_x(x0_abs: float) -> Optional[float]:
        eps = max(1e-3, 1e-4 * L)  # avoid grazing ends
        x0 = max(xmin + eps, min(xmax - eps, x0_abs))
        plane = gp_Pln(gp_Ax3(gp_Pnt(x0, 0, 0), axis_dir))
        sec = BRepAlgoAPI_Section(solid, plane)
        sec.ComputePCurveOn1(True)
        sec.Approximation(True)
        # sec.SetFuzzyValue(tol)
        sec.Build()
        if not sec.IsDone():
            return None
        outer, _inners, area_outer = _extract_outer_and_holes(sec.Shape(), plane, tol=tol)
        if outer is None:
            return None
        return float(area_outer)

    # single-slice path
    if sample_pos is not None:
        x0_abs = xmin + float(sample_pos)
        a = _gross_area_at_x(x0_abs)
        if a is not None:
            return a
        # fallback
        gpv = GProp_GProps()
        brepgprop.VolumeProperties(solid, gpv)
        return gpv.Mass() / max(L, 1e-9)

    # robust multi-slice path
    n_samples = max(5, 21)   # force >= 5
    rel_end_margin = 0.06
    abs_min_margin = 15.0
    margin = max(rel_end_margin * L, abs_min_margin)
    x_lo = xmin + margin
    x_hi = xmax - margin
    if x_hi <= x_lo:
        x_lo = xmin + 0.2 * L
        x_hi = xmax - 0.2 * L

    vals: List[float] = []
    xs: List[float] = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        x0_abs = x_lo + t * (x_hi - x_lo)
        a = _gross_area_at_x(x0_abs)
        if a is not None:
            vals.append(a)
            xs.append(x0_abs)

    if not vals:
        gpv = GProp_GProps()
        brepgprop.VolumeProperties(solid, gpv)
        return gpv.Mass() / max(L, 1e-9)

    # cluster by 3% similarity; pick largest cluster; return its median
    import statistics
    area_rel_tol = 0.03
    clusters: List[Dict[str, Any]] = []  # {'ref': float, 'vals': List[float]}
    for a in sorted(vals):
        placed = False
        for c in clusters:
            ref = c['ref']
            if abs(a - ref) <= area_rel_tol * max(ref, 1e-9):
                c['vals'].append(a)
                c['ref'] = statistics.median(c['vals'])
                placed = True
                break
        if not placed:
            clusters.append({'ref': a, 'vals': [a]})
    clusters.sort(key=lambda c: len(c['vals']), reverse=True)
    core = clusters[0]['vals'] if clusters else vals
    med_core = float(statistics.median(core))

    # diagnostics
    CSA_DIAG.clear()
    CSA_DIAG.update({
        "mode": "robust_cluster",
        "n_total": len(vals),
        "n_core": len(core),
        "median_core": med_core,
        "min_core": float(min(core)),
        "max_core": float(max(core)),
        "rel_spread_core": float((max(core) - min(core)) / max(med_core, 1e-9)),
        "positions": xs,
        "area_rel_tol": area_rel_tol,
    })

    return med_core


def compute_average_section_area(solid: TopoDS_Shape,
                                 axis_dir: gp_Dir = gp_Dir(1, 0, 0),
                                 n_samples: int = 5,
                                 tol: float = 1e-6) -> float:
    """Even sampling average (legacy helper)."""
    bbox = Bnd_Box(); brepbndlib.Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    length = xmax - xmin
    if length <= tol:
        return 0.0

    areas: List[float] = []
    for i in range(n_samples):
        pos = (i + 0.5) * length / n_samples
        areas.append(compute_section_area(solid, axis_dir, sample_pos=pos, tol=tol))

    avg = sum(areas) / len(areas)
    diff = max(areas) - min(areas)
    if avg > 0 and diff > 0.01 * avg:
        print(f"Warning: section area varies by more than 1% "
              f"(min={min(areas):.6f}, max={max(areas):.6f})")
    return avg


# ======================================================================================
# Orientation helpers & DSTV alignment
# ======================================================================================

def swap_width_and_height_if_required(profile_match: Dict[str, Any], shape: TopoDS_Shape, obb_geom: Dict[str, Any]):
    """Rotate 90° about X if profile classification flagged a mismatch."""
    requires_swap = profile_match.get("Requires_rotation", False)
    if not requires_swap:
        return shape, obb_geom

    print("🔁 Swapping height/width axes to match profile classification")
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(obb_geom["aligned_center"], obb_geom["aligned_dir_x"]), math.pi / 2.0)
    shape_rotated = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    obb_geom = compute_obb_geometry(shape_rotated)
    return shape_rotated, obb_geom


def compute_dstv_origin(center: gp_Pnt, extents: List[float], dir_x: gp_Dir, dir_y: gp_Dir, dir_z: gp_Dir) -> gp_Pnt:
    """DSTV origin (rear-bottom-left) for an aligned OBB."""
    dx = -extents[0] / 2.0
    dy = -extents[1] / 2.0
    dz = -extents[2] / 2.0
    offset_vec = gp_Vec(dir_x).Scaled(dx) + gp_Vec(dir_y).Scaled(dy) + gp_Vec(dir_z).Scaled(dz)
    return center.Translated(offset_vec)


def align_obb_to_dstv_frame(shape: TopoDS_Shape, origin_local: gp_Pnt, dir_x: gp_Dir, dir_y: gp_Dir, dir_z: gp_Dir):
    """
    1) Translate origin_local to (0,0,0)
    2) Rotate axes to DSTV global (Z up, X forward)
    """
    # translate
    translate_vec = gp_Vec(origin_local, gp_Pnt(0, 0, 0))
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(translate_vec)
    shape_translated = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

    # rotate
    local_frame = gp_Ax3(gp_Pnt(0, 0, 0), dir_z, dir_x)  # Z = up, X = length
    dstv_frame  = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    trsf_rotate = gp_Trsf()
    trsf_rotate.SetDisplacement(local_frame, dstv_frame)
    shape_rotated = BRepBuilderAPI_Transform(shape_translated, trsf_rotate, True).Shape()

    return shape_rotated, trsf_rotate


def rotate_shape_around_axis(shape: TopoDS_Shape, center: gp_Pnt, axis_dir: gp_Dir, angle_rad: float):
    axis = gp_Ax1(center, axis_dir)
    trsf = gp_Trsf()
    trsf.SetRotation(axis, angle_rad)
    rotated = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    return rotated, trsf

# --- CG-based canonicalization in DSTV-local (min-corner at origin) ---

import math
from typing import Optional, Tuple
from OCC.Core.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir, gp_Ax1, gp_Trsf
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

# 0) basic bbox + translate to DSTV min-corner
def _bbox(shape) -> Tuple[float,float,float,float,float,float]:
    box = Bnd_Box(); brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)

def _to_dstv_local(shape):
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox(shape)
    t = gp_Trsf(); t.SetTranslation(gp_Pnt(xmin, ymin, zmin), gp_Pnt(0,0,0))
    return BRepBuilderAPI_Transform(shape, t, True).Shape(), (xmax-xmin, ymax-ymin, zmax-zmin)

def _rot_180_about_X(shape):
    r = gp_Trsf(); r.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(1,0,0)), math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()

# 1) take a mid-length section and get OUTER wire only
def _outer_wire_at_x(shape_local, x_pos: float, tol=1e-6):
    plane = gp_Pln(gp_Ax3(gp_Pnt(x_pos, 0, 0), gp_Dir(1,0,0)))  # normal +X
    sec = BRepAlgoAPI_Section(shape_local, plane)
    sec.ComputePCurveOn1(True); sec.Approximation(True); sec.Build()
    if not sec.IsDone():
        return None, plane

    fb = ShapeAnalysis_FreeBounds(sec.Shape(), tol, False, False)
    seq = fb.GetClosedWires()
    if not (seq and hasattr(seq, "Length") and seq.Length() > 0):
        return None, plane

    wires = []
    for i in range(1, seq.Length()+1):
        s = seq.Value(i)
        if s.ShapeType() == TopAbs_WIRE:
            wires.append(topods.Wire(s))
    if not wires:
        return None, plane

    # choose OUTER by max area on the plane
    def _wire_area(wire):
        face = BRepBuilderAPI_MakeFace(plane, wire).Face()
        g = GProp_GProps(); brepgprop.SurfaceProperties(face, g)
        return abs(g.Mass()), face

    areas = [(_wire_area(w), w) for w in wires]  # [( (area, face), wire ), ...]
    (area_max, face_max), wire_max = max(areas, key=lambda t: t[0][0])
    return wire_max, plane


# --- Fallback: get CG from a thin-slab boolean section (mitre-safe) ---
def _outer_cg_yz_slab(shape_local, x_pos: float, slab_thick: float = 2.0,
                      y_pad: float = 20.0, z_pad: float = 20.0) -> Optional[Tuple[float, float]]:
    """
    Fallback CG extractor: take a very thin X-aligned box around x_pos, boolean
    common with shape, then find the largest face whose normal ~ ±X.
    Return (y_c, z_c) in local (DSTV) coords.
    """
    from OCC.Core.gp import gp_Pnt, gp_Ax2
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    import math

    # Local bbox to size the slab in Y/Z
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox(shape_local)
    H = float(ymax - ymin)
    W = float(zmax - zmin)

    # Build a thin box centered at x_pos
    dx = float(slab_thick)
    dy = float(H + 2*y_pad)
    dz = float(W + 2*z_pad)

    # Box origin at (x_pos - dx/2, ymin - y_pad, zmin - z_pad)
    o = gp_Pnt(x_pos - 0.5*dx, ymin - y_pad, zmin - z_pad)
    box = BRepPrimAPI_MakeBox(o, dx, dy, dz).Shape()

    # Boolean common
    common = BRepAlgoAPI_Common(shape_local, box)
    common.Build()
    if not common.IsDone():
        return None
    sh = common.Shape()

    # Find faces with normal ≈ ±X and take largest by area
    best_area = -1.0
    best_face = None

    exp = TopExp_Explorer(sh, TopAbs_FACE)
    while exp.More():
        f = topods.Face(exp.Current())
        exp.Next()

        # quick normal check at mid-UV
        ad = BRepAdaptor_Surface(f)
        u0, u1 = ad.FirstUParameter(), ad.LastUParameter()
        v0, v1 = ad.FirstVParameter(), ad.LastVParameter()
        um = 0.5*(u0 + u1); vm = 0.5*(v0 + v1)

        pr = GeomLProp_SLProps(ad.Surface().Surface(), um, vm, 1, 1e-6)
        if not pr.IsNormalDefined():
            continue
        n = pr.Normal()
        nx, ny, nz = n.X(), n.Y(), n.Z()
        # how parallel to X?
        cos_to_x = abs(nx) / max(1e-12, math.sqrt(nx*nx + ny*ny + nz*nz))
        if cos_to_x < math.cos(math.radians(10.0)):
            continue

        # area
        g = GProp_GProps(); brepgprop.SurfaceProperties(f, g)
        area = abs(g.Mass())
        if area > best_area:
            best_area, best_face = area, f

    if best_face is None or best_area <= 0:
        return None

    # Centroid of that face → (Y,Z)
    g = GProp_GProps(); brepgprop.SurfaceProperties(best_face, g)
    c = g.CentreOfMass()
    return float(c.Y()), float(c.Z())

from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir, gp_Pln
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.ShapeFix import ShapeFix_Wire
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

# --- robust, miter-aware selection of a “fully formed” section x ---
def _cg_from_best_section(shape_local, *, xmin, xmax, ymin, ymax, zmin, zmax, L, H, W):
    """
    Return (y_c, z_c) at an X where the loop is fully formed (miters avoided).
    Scans a few X positions in the safe core of the part and picks the largest-area loop.
    """
    # safe distance from mitred ends
    guard = max(0.10 * L, 1.5 * W)      # e.g. 130 mm for a 1300 mm part, or 1.5*75 = 112.5 mm
    x_lo  = xmin + guard
    x_hi  = xmax - guard
    if not (x_hi > x_lo):
        return None  # part too short for a safe slice

    # make 7 candidates across the “core” (can tweak count if you like)
    xs = [x_lo + i * (x_hi - x_lo) / 6.0 for i in range(7)]

    best_area, best_face = 0.0, None

    # sectioning tolerances (mm)
    fuzzy = max(0.10, 0.002 * max(H, W))
    dx    = max(0.10, 0.001 * L)
    jitters = (0.0, +dx, -dx)

    # finite face spanning Y/Z bbox (+margin) improves robustness vs infinite plane
    def _plane_face_x_span(x_abs, margin=5.0):
        pln = gp_Pln(gp_Ax3(gp_Pnt(x_abs, 0.0, 0.0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)))
        yu, Yv = float(ymin - margin), float(ymax + margin)
        zu, Zv = float(zmin - margin), float(zmax + margin)
        return BRepBuilderAPI_MakeFace(pln, yu, Yv, zu, Zv).Face(), pln

    for x0 in xs:
        for j in jitters:
            x = x0 + j
            face_rect, pln = _plane_face_x_span(x, margin=max(2.0, 0.02 * max(H, W)))
            sec = BRepAlgoAPI_Section(shape_local, face_rect, True)
            sec.Approximation(True)
            sec.ComputePCurveOn1(True)
            sec.ComputePCurveOn2(True)
            sec.SetFuzzyValue(fuzzy)
            sec.Build()
            if not sec.IsDone():
                continue

            edges_comp = sec.Shape()

            # closed wires first
            fb = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, False, False)
            closed_comp = fb.GetClosedWires()

            # iterate wires in a compound
            def _iter_wires(comp):
                exp = TopExp_Explorer(comp, TopAbs_WIRE)
                while exp.More():
                    yield topods.Wire(exp.Current())
                    exp.Next()

            # evaluate closed wires
            for w in _iter_wires(closed_comp):
                face = BRepBuilderAPI_MakeFace(pln, w).Face()
                gp_props = GProp_GProps(); brepgprop.SurfaceProperties(face, gp_props)
                area = abs(gp_props.Mass())
                if area > best_area:
                    best_area, best_face = area, face

            # light attempt to fix opens if still nothing great
            if best_area < 1e-6:
                fb_open = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, True, False)
                for w in _iter_wires(fb_open.GetOpenWires()):
                    wf = ShapeFix_Wire(w); wf.SetMaxTolerance(fuzzy); wf.Perform()
                    face = BRepBuilderAPI_MakeFace(pln, wf.Wire()).Face()
                    gp_props = GProp_GProps(); brepgprop.SurfaceProperties(face, gp_props)
                    area = abs(gp_props.Mass())
                    if area > best_area:
                        best_area, best_face = area, face

            if best_face and best_area > 1e-6:
                # early return once we’ve got a solid, large loop
                g = GProp_GProps(); brepgprop.SurfaceProperties(best_face, g)
                c = g.CentreOfMass()
                return float(c.Y()), float(c.Z())

    # diagnostics (optional)
    print(f"[sec] no closed loop in core region [{x_lo:.1f},{x_hi:.1f}] (miters?)")
    return None

def _bbox_local(sh):
    box = Bnd_Box()
    brepbndlib.Add(sh, box)   # static method, no deprecation warning
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return xmin, xmax, ymin, ymax, zmin, zmax


def _count_wires(compound) -> int:
    n = 0
    exp = TopExp_Explorer(compound, TopAbs_WIRE)
    while exp.More():
        n += 1
        exp.Next()
    return n

def _iter_wires(compound):
    exp = TopExp_Explorer(compound, TopAbs_WIRE)
    while exp.More():
        yield topods.Wire(exp.Current())
        exp.Next()

def _outer_cg_yz_robust(shape_local, x_abs: float, *, L: float, H: float, W: float):
    """
    Return (y_c, z_c) of the OUTER loop at/near absolute X = x_abs (DSTV-local).
    Tries a sweep of Xs + jitter around each X, with fuzzy tolerances.
    """
    # Tolerances (mm) — slightly more generous than before
    fuzzy = max(0.10, 0.002 * max(H, W))   # e.g. 0.10–0.40 mm for your sizes
    dx    = max(0.10, 0.001 * L)           # e.g. 0.10–12.5 mm depending on L

    # 1) Sweep a few relative positions around the chosen mid-span X (0.25L by default)
    #    Keep everything well away from ends to avoid cope/bevel noise.
    rels   = (0.30, 0.35, 0.40, 0.45, 0.50)  # centered sweep
    x_sweep = [ (x_abs - 0.25*L) + r*L for r in rels ]

    # 2) Around each candidate X, try a small jitter to avoid slicing a vertex
    jitters = (0.0, +dx, -dx, +2*dx, -2*dx)

    best_area, best_face = 0.0, None

    for x0 in x_sweep:
        for j in jitters:
            x = x0 + j
            pln = gp_Pln(gp_Ax3(gp_Pnt(x, 0, 0), gp_Dir(1, 0, 0)))
            face_inf = BRepBuilderAPI_MakeFace(pln).Face()

            sec = BRepAlgoAPI_Section(shape_local, face_inf, True)
            sec.Approximation(True)
            sec.ComputePCurveOn1(True)
            sec.ComputePCurveOn2(True)
            sec.SetFuzzyValue(fuzzy)
            sec.Build()
            if not sec.IsDone():
                continue

            edges_comp = sec.Shape()

            # Closed wires first
            fb = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, False, False)
            closed_comp = fb.GetClosedWires()

            # If none, try to fix opens
            if _count_wires(closed_comp) == 0:
                fb_open = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, True, False)
                open_comp = fb_open.GetOpenWires()
                for w in _iter_wires(open_comp):
                    wf = ShapeFix_Wire(w)
                    wf.SetMaxTolerance(fuzzy)
                    wf.Perform()
                    fixed = wf.Wire()
                    face = BRepBuilderAPI_MakeFace(pln, fixed).Face()
                    gp_props = GProp_GProps()
                    brepgprop.SurfaceProperties(face, gp_props)
                    area = abs(gp_props.Mass())
                    if area > best_area:
                        best_area, best_face = area, face

            # Evaluate real closed wires (outer = largest area)
            for w in _iter_wires(closed_comp):
                face = BRepBuilderAPI_MakeFace(pln, w).Face()
                gp_props = GProp_GProps()
                brepgprop.SurfaceProperties(face, gp_props)
                area = abs(gp_props.Mass())
                if area > best_area:
                    best_area, best_face = area, face

            if best_face and best_area > 1e-6:
                # early out once we’ve got a good face
                g = GProp_GProps()
                brepgprop.SurfaceProperties(best_face, g)
                c = g.CentreOfMass()
                return float(c.Y()), float(c.Z())
            
    print(f"[sec] failed near X~{x_abs:.2f} (L={L:.1f}, H={H:.1f}, W={W:.1f}) with fuzzy={fuzzy:.3f}, dx={dx:.3f}")

    return None


# 2) centroid (y_c, z_c) of OUTER loop (gross section CG)
def _outer_cg_yz(shape_local, x_pos: float) -> Optional[Tuple[float,float]]:
    wire, plane = _outer_wire_at_x(shape_local, x_pos)
    if wire is None:
        # Fallback for mitred ends / open loops
        cg = _outer_cg_yz_slab(shape_local, x_pos, slab_thick=2.0)
        if cg is None:
            return None
        return cg
    face = BRepBuilderAPI_MakeFace(plane, wire).Face()
    g = GProp_GProps(); brepgprop.SurfaceProperties(face, g)
    c = g.CentreOfMass()
    return float(c.Y()), float(c.Z())



# 3) profile-specific 180° decision
def canonicalize_by_section_cg(shape_local, L, H, W, profile_type: str, x_frac: float = 0.25) -> Tuple[object, Tuple[float,float]]:
    """
    Decide a single 180° flip about X from the gross section CG at x=L*x_frac.
    Returns: (shape_local_fixed, (y_c, z_c)) for debugging.
    """
    x_pos = max(0.15*L, min(0.85*L, x_frac*L))  # avoid end features
    cg = _outer_cg_yz(shape_local, x_pos)
    if cg is None:
        # no section? keep as-is
        return shape_local, (None, None)

    y_c, z_c = cg

    if profile_type == "U":
        # Channel: toe must be at z≈0; if CG is nearer far flange, flip.
        if z_c > 0.5 * W:
            print(f"🔁 CG test (U): z_c={z_c:.2f} > W/2={0.5*W:.2f} → rotate 180° about X")
            shape_local = _rot_180_about_X(shape_local)

    elif profile_type == "L":
        # Angle: heel should be near (0,0); flip only if both are on far side.
        if (y_c > 0.5 * H) and (z_c > 0.5 * W):
            print(f"🔁 CG test (L): y_c,z_c in far halves → rotate 180° about X")
            shape_local = _rot_180_about_X(shape_local)

    # (You can print y_c,z_c for debug if useful)
    return shape_local, (y_c, z_c)

# --- robust, miter-aware selection of a “fully formed” section x ---
def _cg_from_best_section(shape_local, *, xmin, xmax, ymin, ymax, zmin, zmax, L, H, W):
    """
    Return (y_c, z_c) at an X where the loop is fully formed (miters avoided).
    Scans a few X positions in the safe core of the part and picks the largest-area loop.
    """
    # safe distance from mitred ends
    guard = max(0.10 * L, 1.5 * W)      # e.g. 130 mm for a 1300 mm part, or 1.5*75 = 112.5 mm
    x_lo  = xmin + guard
    x_hi  = xmax - guard
    if not (x_hi > x_lo):
        return None  # part too short for a safe slice

    # make 7 candidates across the “core” (can tweak count if you like)
    xs = [x_lo + i * (x_hi - x_lo) / 6.0 for i in range(7)]

    best_area, best_face = 0.0, None

    # sectioning tolerances (mm)
    fuzzy = max(0.10, 0.002 * max(H, W))
    dx    = max(0.10, 0.001 * L)
    jitters = (0.0, +dx, -dx)

    # finite face spanning Y/Z bbox (+margin) improves robustness vs infinite plane
    def _plane_face_x_span(x_abs, margin=5.0):
        pln = gp_Pln(gp_Ax3(gp_Pnt(x_abs, 0.0, 0.0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)))
        yu, Yv = float(ymin - margin), float(ymax + margin)
        zu, Zv = float(zmin - margin), float(zmax + margin)
        return BRepBuilderAPI_MakeFace(pln, yu, Yv, zu, Zv).Face(), pln

    for x0 in xs:
        for j in jitters:
            x = x0 + j
            face_rect, pln = _plane_face_x_span(x, margin=max(2.0, 0.02 * max(H, W)))
            sec = BRepAlgoAPI_Section(shape_local, face_rect, True)
            sec.Approximation(True)
            sec.ComputePCurveOn1(True)
            sec.ComputePCurveOn2(True)
            sec.SetFuzzyValue(fuzzy)
            sec.Build()
            if not sec.IsDone():
                continue

            edges_comp = sec.Shape()

            # closed wires first
            fb = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, False, False)
            closed_comp = fb.GetClosedWires()

            # iterate wires in a compound
            def _iter_wires(comp):
                exp = TopExp_Explorer(comp, TopAbs_WIRE)
                while exp.More():
                    yield topods.Wire(exp.Current())
                    exp.Next()

            # evaluate closed wires
            for w in _iter_wires(closed_comp):
                face = BRepBuilderAPI_MakeFace(pln, w).Face()
                gp_props = GProp_GProps(); brepgprop.SurfaceProperties(face, gp_props)
                area = abs(gp_props.Mass())
                if area > best_area:
                    best_area, best_face = area, face

            # light attempt to fix opens if still nothing great
            if best_area < 1e-6:
                fb_open = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, True, False)
                for w in _iter_wires(fb_open.GetOpenWires()):
                    wf = ShapeFix_Wire(w); wf.SetMaxTolerance(fuzzy); wf.Perform()
                    face = BRepBuilderAPI_MakeFace(pln, wf.Wire()).Face()
                    gp_props = GProp_GProps(); brepgprop.SurfaceProperties(face, gp_props)
                    area = abs(gp_props.Mass())
                    if area > best_area:
                        best_area, best_face = area, face

            if best_face and best_area > 1e-6:
                # early return once we’ve got a solid, large loop
                g = GProp_GProps(); brepgprop.SurfaceProperties(best_face, g)
                c = g.CentreOfMass()
                return float(c.Y()), float(c.Z())

    # diagnostics (optional)
    print(f"[sec] no closed loop in core region [{x_lo:.1f},{x_hi:.1f}] (miters?)")
    return None

def force_canonical_dstv_pose(shape_local, profile_type: str, x_frac: float = 0.25):
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape_local)
    L = float(xmax - xmin); H = float(ymax - ymin); W = float(zmax - zmin)

    print("-------------- Forcing Canoncical Pose")
    print(f"xmin={xmin:.2f}, xmax={xmax:.2f}, L={L:.2f}, H={H:.2f}, W={W:.2f}")

    cg = _cg_from_best_section(
        shape_local,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
        L=L, H=H, W=W
    )
    y_c, z_c = (None, None)
    if cg is not None:
        y_c, z_c = cg
        print(f"📍 CG (mitre-safe): y_c={y_c:.2f}, z_c={z_c:.2f} (H={H:.2f}, W={W:.2f})")
        if profile_type == "U" and z_c > 0.5 * W:
            print("🔁 CG test (U): far side → rotate 180° about X")
            shape_local = _rot_180_about_X(shape_local)
    elif True:
        print("❌ No CG provided (even after mitre-safe scan)")


    # Recompute AABB after potential flip
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape_local)
    L = float(xmax - xmin); H = float(ymax - ymin); W = float(zmax - zmin)
    return shape_local, (L, H, W, y_c, z_c)

def refine_orientation_by_flange_face(shape, obb_geom,
                                      face_normal_dir, position_dir,
                                      axis_label: str = "Y", profile_type: str = "U",
                                      shape_category: str = ""):
    """
    Find the largest face whose normal is within 10° of 'face_normal_dir' using a SIGNED test.
    Does NOT rotate here; prints diagnostics (area, side ±, offset). Returns (shape, obb_geom).
    """
    # --- local imports to keep this block drop-in safe ---
    import math
    import numpy as np
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomLProp import GeomLProp_SLProps

    # OBB center (world coords) as np
    aligned_center = np.array([
        obb_geom["aligned_center"].X(),
        obb_geom["aligned_center"].Y(),
        obb_geom["aligned_center"].Z()
    ], dtype=float)

    # Target normal (fn) and position axis (pd) as unit vectors
    fn = np.array([face_normal_dir.X(), face_normal_dir.Y(), face_normal_dir.Z()], dtype=float)
    nrm = np.linalg.norm(fn);  fn = fn / (nrm if nrm > 1e-12 else 1.0)

    pd = np.array([position_dir.X(), position_dir.Y(), position_dir.Z()], dtype=float)
    nrm = np.linalg.norm(pd);  pd = pd / (nrm if nrm > 1e-12 else 1.0)

    # signed alignment threshold
    cos_thr = math.cos(math.radians(10.0))

    best = None  # (area, side (+1/-1), centroid_np)

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        # explicit downcast to face
        face = topods.Face(exp.Current())

        # param mid-point on the surf for normal eval
        s = BRepAdaptor_Surface(face)
        umin, umax = s.FirstUParameter(), s.LastUParameter()
        vmin, vmax = s.FirstVParameter(), s.LastVParameter()
        u_mid = 0.5 * (umin + umax)
        v_mid = 0.5 * (vmin + vmax)

        # evaluate surface normal at (u_mid, v_mid)
        hsurf = s.Surface().Surface()
        props = GeomLProp_SLProps(hsurf, u_mid, v_mid, 1, 1e-6)
        if props.IsNormalDefined():
            n = np.array([props.Normal().X(), props.Normal().Y(), props.Normal().Z()], dtype=float)
            nrm = np.linalg.norm(n);  n = n / (nrm if nrm > 1e-12 else 1.0)

            cosang = float(np.dot(n, fn))  # SIGNED dot → keeps side info
            if abs(cosang) >= cos_thr:
                side = 1 if (cosang > 0.0) else -1

                # face area & centroid
                g = GProp_GProps()
                brepgprop.SurfaceProperties(face, g)
                area = float(abs(g.Mass()))
                c = g.CentreOfMass()
                centroid = np.array([c.X(), c.Y(), c.Z()], dtype=float)

                if (best is None) or (area > best[0]):
                    best = (area, side, centroid)

        exp.Next()

    if best is None:
        print(f"⚠️ Could not find flange/web face for {profile_type}")
        return shape, obb_geom

    area, side, centroid_np = best
    offset = float(np.dot(centroid_np - aligned_center, pd))
    print(
        f"🔎 {profile_type}: largest aligned face area={area:.1f}, "
        f"side={'+' if side > 0 else '-'}, {axis_label}-offset={offset:.2f}"
    )

    # No rotations here — do canonicalization later in DSTV-local.
    return shape, obb_geom


def refine_profile_orientation(shape, profile_match, obb_geom):
    """
    Hardened pipeline:
      1) Face-based refinement (signed normals) → diagnostics only.
      2) Move into DSTV-local (min-corner at origin; axes X,Y,Z as NC expects).
      3) Canonicalize: at most one 180° flip about X based on bbox near/far.
      4) Return the DSTV-local shape + fresh OBB.
    """
    st  = profile_match.get("Profile_type")
    cat = profile_match.get("Category")

    if st == "L":
        print("🔍 Refining angle orientation (signed normals)")
        shape, obb_geom = refine_orientation_by_flange_face(
            shape, obb_geom,
            face_normal_dir=obb_geom["aligned_dir_z"],  # leg face normal ≈ ±Z
            position_dir=obb_geom["aligned_dir_y"],     # report offset along Y
            axis_label="Y",
            profile_type="Angle",
            shape_category=cat
        )
    elif st == "U":
        print("🔍 Refining channel orientation (signed normals)")
        shape, obb_geom = refine_orientation_by_flange_face(
            shape, obb_geom,
            face_normal_dir=obb_geom["aligned_dir_z"],  # web faces normal ≈ ±Z
            position_dir=obb_geom["aligned_dir_y"],     # report offset along Y
            axis_label="Y",
            profile_type="Channel",
            shape_category=cat
        )
    else:
        print(f"ℹ️ No refinement needed for shape type '{st}'")
        return shape, obb_geom

    # 2) Go to DSTV-local (translate to min-corner, rotate to world DSTV axes)
    obb_geom = compute_obb_geometry(shape)
    shape_local, dstv_frame, (L, H, W) = _move_to_dstv_local(shape, obb_geom)

    # 3) Canonicalize once (flip about X only if bbox shows we’re nearer far-planes)
    if st == "U":
        shape_local = _canonicalize_channel(shape_local, H, W)
    elif st == "L":
        shape_local = _canonicalize_angle(shape_local,  H, W)

    # 4) Fresh OBB now that we’re in DSTV-local
    obb_final = compute_obb_geometry(shape_local)
    print(f"✅ Orientation canonicalized. Extents (L,H,W) = {obb_final['aligned_extents']}")
    return shape_local, obb_final


# --- OBB + 3D-CG based orientation for U and L (EA/UEA) ---

def _bbox_local(sh) -> Tuple[float,float,float,float,float,float]:
    box = Bnd_Box(); brepbndlib.Add(sh, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)

def _solid_centroid(shape) -> gp_Pnt:
    g = GProp_GProps()
    try:
        brepgprop.VolumeProperties(shape, g)
        if abs(g.Mass()) > 1e-12:
            return g.CentreOfMass()
    except Exception:
        pass
    try:
        g = GProp_GProps()
        brepgprop.SurfaceProperties(shape, g)
        if abs(g.Mass()) > 1e-12:
            return g.CentreOfMass()
    except Exception:
        pass
    xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(shape)
    return gp_Pnt(0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax))

def _rotX(shape, quarter_turns: int):
    """Rotate about local X axis through origin by k*90°."""
    k = quarter_turns % 4
    if k == 0:
        return shape
    angle = k * (math.pi / 2.0)
    ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    tr = gp_Trsf(); tr.SetRotation(ax, angle)
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

def _norm_yz(pt: gp_Pnt, ymin, ymax, zmin, zmax) -> Tuple[float, float]:
    H = max(ymax - ymin, 1e-12)
    W = max(zmax - zmin, 1e-12)
    return (float(pt.Y() - ymin) / H, float(pt.Z() - zmin) / W)

def orient_section_by_3d_cg(shape_local, profile_type: str, profile_dims: Optional[Dict] = None):
    """
    Assumes shape_local is already in OBB/DSTV-local axes:
      X=length, Y=height (leg), Z=width (other leg/web), right-handed, origin anywhere.
    Returns: (shape_out, diag) where diag has:
      - turns_about_X: int in {0,1,2,3}
      - y_norm, z_norm: centroid in [0,1] after orientation
      - L,H,W: section extents
      - method: 'U' or 'L'
      - confidence: 'ok'|'low'
    """
    xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(shape_local)
    L = xmax - xmin; H = ymax - ymin; W = zmax - zmin

    # ---- Stage 0: for Angles (EA/UEA) ensure longer leg maps to Y ----
    sel_turns_pre = 0
    if profile_type == "L" and profile_dims:
        # Try to read nominal leg sizes (common keys; adjust to your JSON as needed)
        legY_nom = float(profile_dims.get("leg_y", profile_dims.get("height", H)))
        legZ_nom = float(profile_dims.get("leg_z", profile_dims.get("width",  W)))
        want_Y, want_Z = (max(legY_nom, legZ_nom), min(legY_nom, legZ_nom))

        # match H->want_Y, W->want_Z; if swapped, do a 90° X-rotation to swap Y<->Z
        def _close(a,b): 
            # 2 mm or 1% tolerance (whichever larger)
            return abs(a-b) <= max(2.0, 0.01*max(abs(a),abs(b),1.0))

        if not (_close(H, want_Y) and _close(W, want_Z)):
            # check if a single 90° would make it match
            sh90 = _rotX(shape_local, 1)
            x2,x3,y2,y3,z2,z3 = _bbox_local(sh90)
            H2, W2 = (y3-y2), (z3-z2)
            if _close(H2, want_Y) and _close(W2, want_Z):
                shape_local = sh90
                sel_turns_pre = 1
                xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(shape_local)
                L = xmax - xmin; H = ymax - ymin; W = zmax - zmin
            else:
                # If still not matching, leave as-is; scoring step will handle heel placement.
                pass

    # ---- Stage 1: choose rotation by 3D-CG scoring ----
    best = None
    if profile_type == "U":
        # Only 0°/180° distinguish Z reflection
        candidates = (0, 2)
        for k in candidates:
            sh = _rotX(shape_local, k)
            xmin2,xmax2,ymin2,ymax2,zmin2,zmax2 = _bbox_local(sh)
            cg = _solid_centroid(sh)
            y_, z_ = _norm_yz(cg, ymin2, ymax2, zmin2, zmax2)
            # web on Z-max side; Y near middle
            score = abs(z_ - 0.75) + 0.5*abs(y_ - 0.5)
            pick = (score, k, y_, z_, sh)
            if (best is None) or (pick < best):
                best = pick
        method = "U"

    elif profile_type == "L":
        # Try all 4 quarters to put heel at origin
        candidates = (0, 1, 2, 3)
        for k in candidates:
            sh = _rotX(shape_local, k)
            xmin2,xmax2,ymin2,ymax2,zmin2,zmax2 = _bbox_local(sh)
            cg = _solid_centroid(sh)
            y_, z_ = _norm_yz(cg, ymin2, ymax2, zmin2, zmax2)
            score = (y_ + z_)  # heel near (0,0)
            pick = (score, k, y_, z_, sh)
            if (best is None) or (pick < best):
                best = pick
        method = "L"
    else:
        # Other profiles: no change
        cg = _solid_centroid(shape_local)
        y_, z_ = _norm_yz(cg, ymin, ymax, zmin, zmax)
        return shape_local, dict(
            turns_about_X=0, pre_turns=0, y_norm=y_, z_norm=z_, L=L, H=H, W=W,
            method=profile_type, confidence="ok"
        )

    score, k_sel, y_sel, z_sel, shape_out = best
    total_turns = (sel_turns_pre + k_sel) % 4

    # Confidence checks (cheap guardrails)
    conf = "ok"
    if method == "U" and (z_sel < 0.5 or abs(y_sel - 0.5) > 0.3):
        conf = "low"
    if method == "L" and (y_sel > 0.35 or z_sel > 0.35):
        conf = "low"

    # Final extents (unchanged by pure X-rotations)
    return shape_out, dict(
        turns_about_X=total_turns, pre_turns=sel_turns_pre,
        y_norm=y_sel, z_norm=z_sel, L=L, H=H, W=W,
        method=method, confidence=conf
    )


def compute_dstv_pose(primary_aligned_shape, profile_type: str, profile_dims: Dict|None = None,
                      channel_mode: str = "toe_at_z0"):
    # Stage 0: long leg → Y (UEA)
    shape0, pre_turns = _enforce_long_leg_on_Y_if_angle(primary_aligned_shape, profile_type, profile_dims or {})
    # Stage 1: arg-min over candidate X-rotations; translate each candidate to min-corner before scoring
    candidates = []
    rots = (0,2) if profile_type == "U" else (0,1,2,3)
    for k in rots:
        sh = _rotX(shape0, k)
        sh0 = _to_origin_min_corner(sh)  # this pose’s origin
        xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(sh0)
        H, W = (ymax-ymin, zmax-zmin)
        cg = _solid_centroid(sh0)
        y_, z_ = _norm_yz(cg, ymin, ymax, zmin, zmax)
        if profile_type == "U":
            if channel_mode == "toe_at_z0":
                # toes at Z=0, web at Z→1
                score = abs(z_ - 0.8) + 0.5*abs(y_ - 0.5); policy = "U/toe_at_z0"
            else:
                # web at Z=0, toes at Z→1  ⟵ likely your DSTV convention
                score = abs(z_ - 0.2) + 0.5*abs(y_ - 0.5); policy = "U/web_at_z0"
        elif profile_type == "L":
            score = (y_ + z_); policy = "L/heel_at_origin"
        else:
            score = 0.0; policy = "neutral"
        candidates.append((score, pre_turns+k, y_, z_, sh0, (H,W), policy))
    score, turns, y_sel, z_sel, refined_shape, (H,W), policy = min(candidates, key=lambda t: t[0])
    # Stage 2: build final DSTV frame & transform (world→dstv for this refined pose)
    # In this local pipeline we typically keep axes; origin is now (0,0,0)
    dstv_ax3 = gp_Ax3(gp_Pnt(0,0,0), gp_Dir(1,0,0).Crossed(gp_Dir(0,1,0)), gp_Dir(1,0,0))
    # Note: direction above ensures right-handed (X along length). If you keep fixed axes elsewhere, replace with your Ax3.
    # World→DSTV local (here: identity relative to the refined pose, but return for consistency)
    tr_world_to_dstv = gp_Trsf()  # identity if you are already in local
    # Confidence
    conf = "ok"
    if profile_type == "U" and policy.startswith("U/") and z_sel < 0.5: conf = "low"
    if profile_type == "L" and (y_sel > 0.35 or z_sel > 0.35): conf = "low"
    xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(refined_shape)
    L = xmax - xmin

    # --- POST: enforce origin at min-corner and non-negative coords ---
    refined_shape = _to_origin_min_corner(refined_shape)
    xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(refined_shape)
    # Make sure numerical noise didn't leave tiny negatives
    eps = 1e-6
    if xmin < -eps or ymin < -eps or zmin < -eps:
        refined_shape = _to_origin_min_corner(refined_shape)
        xmin,xmax,ymin,ymax,zmin,zmax = _bbox_local(refined_shape)

    # Recompute CG in FINAL pose for logs / downstream
    cg_pt = _solid_centroid(refined_shape)
    y_sel = (float(cg_pt.Y()) - ymin) / max(ymax - ymin, 1e-9)
    z_sel = (float(cg_pt.Z()) - zmin) / max(zmax - zmin, 1e-9)

    return refined_shape, dstv_ax3, tr_world_to_dstv, dict(
        turns_about_X=turns % 4, y_norm=y_sel, z_norm=z_sel, L=L, H=H, W=W,
        policy=policy, confidence=conf
    )