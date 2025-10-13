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


def refine_orientation_by_flange_face(shape: TopoDS_Shape, obb_geom: Dict[str, Any],
                                      face_normal_dir: gp_Dir, position_dir: gp_Dir,
                                      axis_label: str = "Y", profile_type: str = "U",
                                      shape_category: str = ""):
    """
    Detect orientation by the largest face whose normal aligns with face_normal_dir,
    and check its centroid position along position_dir. Rotate/translate if needed.
    """
    from OCC.Core.GeomLProp import GeomLProp_SLProps

    aligned_center = np.array([obb_geom["aligned_center"].X(),
                               obb_geom["aligned_center"].Y(),
                               obb_geom["aligned_center"].Z()])
    fn = _dir_to_np(face_normal_dir); fn /= max(np.linalg.norm(fn), 1e-12)
    pd = _dir_to_np(position_dir);    pd /= max(np.linalg.norm(pd), 1e-12)
    angle_threshold_rad = math.radians(10.0)

    largest_area = -1.0
    selected_centroid = None

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        surf = BRepAdaptor_Surface(face)
        umin, umax = surf.FirstUParameter(), surf.LastUParameter()
        vmin, vmax = surf.FirstVParameter(), surf.LastVParameter()
        u_mid = (umin + umax) / 2.0
        v_mid = (vmin + vmax) / 2.0

        handle = surf.Surface().Surface()
        props = GeomLProp_SLProps(handle, u_mid, v_mid, 1, 1e-6)
        if props.IsNormalDefined():
            n = np.array([props.Normal().X(), props.Normal().Y(), props.Normal().Z()], dtype=float)
            n /= max(np.linalg.norm(n), 1e-12)
            ang = math.acos(np.clip(abs(np.dot(n, fn)), 0.0, 1.0))
            if ang < angle_threshold_rad:
                gprops = GProp_GProps()
                brepgprop.SurfaceProperties(face, gprops)
                area = gprops.Mass()
                centroid = gprops.CentreOfMass()
                c = np.array([centroid.X(), centroid.Y(), centroid.Z()], dtype=float)
                if area > largest_area:
                    largest_area = area
                    selected_centroid = c
        explorer.Next()

    if selected_centroid is None:
        print(f"⚠️ Could not find flange face for {profile_type} orientation check")
        return shape, obb_geom

    vec_to_centroid = selected_centroid - aligned_center
    pos_offset = float(np.dot(vec_to_centroid, pd))
    print(f"🔍 {axis_label} position of largest flange face: {pos_offset:.2f} mm")

    if pos_offset < 0:
        if profile_type == "Channel":
            print(f"🔁 {profile_type} is reversed — rotating 180° around X axis")
            shape, _ = rotate_shape_around_axis(shape, obb_geom["aligned_center"], obb_geom["aligned_dir_x"], math.pi)
            obb_geom = compute_obb_geometry(shape)
            return shape, obb_geom

        if profile_type == "Angle" and shape_category != "EA":
            print(f"🔁 {profile_type} is reversed — translating to origin")
            obb_geom = compute_obb_geometry(shape)
            center_vec = np.array([obb_geom["aligned_center"].X(),
                                   obb_geom["aligned_center"].Y(),
                                   obb_geom["aligned_center"].Z()])
            dirx = _dir_to_np(obb_geom["aligned_dir_x"])
            diry = _dir_to_np(obb_geom["aligned_dir_y"])
            dirz = _dir_to_np(obb_geom["aligned_dir_z"])
            ext  = np.array(obb_geom["aligned_extents"])
            half = 0.5 * ext
            min_corner = center_vec - half[0]*dirx - half[1]*diry - half[2]*dirz
            translation = gp_Trsf()
            translation.SetTranslation(gp_Vec(gp_Pnt(min_corner[0], min_corner[1], min_corner[2]), gp_Pnt(0.0, 0.0, 0.0)))
            shape = BRepBuilderAPI_Transform(shape, translation, True).Shape()
            obb_geom = compute_obb_geometry(shape)
            return shape, obb_geom

        print(f"⚠️ No rotation logic defined for profile type '{profile_type}'")
        obb_geom = compute_obb_geometry(shape)
        return shape, obb_geom

    print(f"✅ {profile_type} is correctly oriented — no rotation needed.")
    return shape, obb_geom


def refine_profile_orientation(shape: TopoDS_Shape, profile_match: Dict[str, Any], obb_geom: Dict[str, Any]):
    st = profile_match.get("Profile_type")
    cat = profile_match.get("Category")
    if st == "L":
        print("🔍 Refining angle orientation")
        return refine_orientation_by_flange_face(
            shape, obb_geom,
            face_normal_dir=obb_geom["aligned_dir_z"],
            position_dir=obb_geom["aligned_dir_y"],
            axis_label="Y",
            profile_type="Angle",
            shape_category=cat
        )
    if st == "U":
        print("🔍 Refining channel orientation")
        return refine_orientation_by_flange_face(
            shape, obb_geom,
            face_normal_dir=obb_geom["aligned_dir_z"],
            position_dir=obb_geom["aligned_dir_y"],
            axis_label="Y",
            profile_type="Channel",
            shape_category=cat
        )
    print(f"ℹ️ No refinement needed for shape type '{st}'")
    return shape, obb_geom
