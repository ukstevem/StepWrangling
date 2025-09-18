from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_SOLID
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax3, gp_Trsf, gp_Pln
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds

from pathlib import Path
import numpy as np
import argparse
import hashlib
import os

# Optional diagnostics filled by compute_section_area() in robust mode.
# Callers can read this (e.g., for logging) but the function still just returns a float.
from typing import Dict, Any
CSA_DIAG: Dict[str, Any] = {}

# --- Helper: area of a single wire on a given plane (unsigned) ---
def _wire_area_on_plane(plane, wire):
    """Build a temporary planar face from 'wire' on 'plane' and return its unsigned area."""
    face = BRepBuilderAPI_MakeFace(plane, wire).Face()
    gp = GProp_GProps()
    brepgprop.SurfaceProperties(face, gp)
    return abs(gp.Mass())

# --- Helper : for multi-body parts ---
def count_solids_in_shape(shape) -> int:
    """Count standalone TopAbs_SOLID entities in the shape (detects multi-body STEP)."""
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n

def _section_islands_stats(solid, x0_abs, axis_dir=gp_Dir(1,0,0), tol=1e-6):
    """
    One slice diagnostic:
      - islands: number of disjoint OUTER loops (approx via bbox containment)
      - H, W: bbox of the union of all loops
      - area_sum: sum of unsigned areas of all outer loops (gross over islands)
    """
    bbox = Bnd_Box(); brepbndlib.Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    L = xmax - xmin
    eps = max(1e-3, 1e-4*L)
    x0 = max(xmin + eps, min(xmax - eps, x0_abs))

    plane = gp_Pln(gp_Ax3(gp_Pnt(x0, 0, 0), axis_dir))
    sec = BRepAlgoAPI_Section(solid, plane)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    # sec.SetFuzzyValue(tol)
    sec.Build()
    if not sec.IsDone():
        return {'ok': False}

    # get all closed wires
    fb = ShapeAnalysis_FreeBounds(sec.Shape(), tol, False, False)
    seq = fb.GetClosedWires()
    if seq is None or not hasattr(seq, "Length") or seq.Length() == 0:
        return {'ok': False}

    # collect wires with their bbox and area
    wires = []
    for i in range(1, seq.Length()+1):
        s = seq.Value(i)
        if s.ShapeType() == TopAbs_WIRE:
            w = topods.Wire(s)
            bb = Bnd_Box(); brepbndlib.Add(w, bb)
            _, y0, z0, _, y1, z1 = bb.Get()
            area = _wire_area_on_plane(plane, w)
            wires.append({'wire': w, 'bb': (y0,z0,y1,z1), 'area': float(area)})

    if not wires:
        return {'ok': False}

    # island inference by bbox containment (simple, fast):
    # a wire is an "island" if its bbox is not strictly inside any other wire's bbox.
    def bb_inside(bb_a, bb_b, margin=1e-3):
        ay0, az0, ay1, az1 = bb_a
        by0, bz0, by1, bz1 = bb_b
        return (ay0 >= by0 - margin and az0 >= bz0 - margin and
                ay1 <= by1 + margin and az1 <= bz1 + margin)

    outer_flags = []
    for i, wi in enumerate(wires):
        inside_any = False
        for j, wj in enumerate(wires):
            if i == j: continue
            if bb_inside(wi['bb'], wj['bb']):
                inside_any = True
                break
        outer_flags.append(not inside_any)

    # union bbox of all loops (good enough for H/W)
    ymins = [w['bb'][0] for w in wires]
    zmins = [w['bb'][1] for w in wires]
    ymaxs = [w['bb'][2] for w in wires]
    zmaxs = [w['bb'][3] for w in wires]
    H = float(max(ymaxs) - min(ymins))
    W = float(max(zmaxs) - min(zmins))

    islands = sum(1 for f in outer_flags if f)
    area_sum = float(sum(w['area'] for w, f in zip(wires, outer_flags) if f))

    return {'ok': True, 'islands': islands, 'H': H, 'W': W, 'area_sum': area_sum}

def plate_guard_heuristics(solid,
                           axis_dir=gp_Dir(1,0,0),
                           n_samples=9,
                           rel_end_margin=0.06,
                           abs_min_margin=15.0,
                           area_fill_min=0.85,
                           max_rel_spread=0.02,
                           islands_frac_max=0.2):
    """
    Decide if a part plausibly behaves like a PLATE, using only geometry.
    Returns (is_plausible_plate: bool, diag: dict).

    Heuristics:
      - If >1 SOLID -> not a plate (likely multi-body).
      - Cross-section fill = area_sum / (H*W) high across slices (median >= area_fill_min).
      - H and W vary very little along length (spread <= max_rel_spread).
      - Slices with multiple islands (disjoint loops) are rare (fraction <= islands_frac_max).
    """
    # Multi-body quick test
    n_solids = count_solids_in_shape(solid)
    if n_solids > 1:
        return False, {"reason": "multi_body_shape", "n_solids": n_solids}

    # Interior sampling window
    bb = Bnd_Box(); brepbndlib.Add(solid, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    L = xmax - xmin
    if L <= 1e-9:
        return False, {"reason": "degenerate_length"}

    margin = max(rel_end_margin*L, abs_min_margin)
    x_lo = xmin + margin
    x_hi = xmax - margin
    if x_hi <= x_lo:
        x_lo = xmin + 0.2*L
        x_hi = xmax - 0.2*L

    # sample
    xs, fills, Hs, Ws, islands_flags = [], [], [], [], []
    for i in range(max(5, n_samples)):  # force >=5 samples
        t = (i + 0.5) / max(5, n_samples)
        x0 = x_lo + t*(x_hi - x_lo)
        stats = _section_islands_stats(solid, x0, axis_dir=axis_dir)
        if not stats.get('ok'):
            continue
        H, W, area_sum = stats['H'], stats['W'], stats['area_sum']
        if H <= 0 or W <= 0:
            continue
        fill = float(area_sum / (H * W))
        xs.append(x0); fills.append(fill); Hs.append(H); Ws.append(W)
        islands_flags.append(stats['islands'] > 1)

    if not fills:
        return False, {"reason": "no_valid_sections"}

    import statistics as _st
    fill_med = float(_st.median(fills))
    H_spread = (max(Hs) - min(Hs)) / max(_st.median(Hs), 1e-9)
    W_spread = (max(Ws) - min(Ws)) / max(_st.median(Ws), 1e-9)
    islands_frac = sum(1 for f in islands_flags if f) / len(islands_flags)

    plausible = (
        fill_med >= area_fill_min and
        H_spread <= max_rel_spread and
        W_spread <= max_rel_spread and
        islands_frac <= islands_frac_max
    )

    diag = {
        "n_samples": len(fills),
        "fill_median": fill_med,
        "H_rel_spread": float(H_spread),
        "W_rel_spread": float(W_spread),
        "islands_fraction": float(islands_frac)
    }
    return plausible, diag


# --- Helper: identify OUTER wire (largest area) and any INNER wires (holes) ---
def _extract_outer_and_holes(edges_shape, plane, tol=1e-6):
    """
    From a section edge soup, return:
      outer_wire (TopoDS_Wire or None),
      inner_wires (list of TopoDS_Wire),
      area_outer (float, gross area from outer wire only).
    """
    fb = ShapeAnalysis_FreeBounds(edges_shape, tol, False, False)
    seq = fb.GetClosedWires()
    if seq is None or not hasattr(seq, "Length") or seq.Length() == 0:
        return None, [], 0.0

    wires = []
    for i in range(1, seq.Length()+1):
        s = seq.Value(i)
        if s.ShapeType() == TopAbs_WIRE:
            wires.append(topods.Wire(s))
    if not wires:
        return None, [], 0.0

    # OUTER = wire with maximum unsigned area on the plane
    areas = [_wire_area_on_plane(plane, w) for w in wires]
    i_outer = int(max(range(len(areas)), key=lambda k: areas[k]))
    outer = wires[i_outer]
    inners = [w for j, w in enumerate(wires) if j != i_outer]
    return outer, inners, float(areas[i_outer])


def robust_align_solid_from_geometry(solid, tol=1e-3):
    """
    Aligns a solid using the longest edge on the largest planar face.
    Returns:
    - aligned shape
    - transformation
    - coordinate system
    - largest face
    - local dir_x, dir_y, dir_z
    """
    # Step 1: Find largest planar face
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    largest_face = None
    max_area = 0.0

    while explorer.More():
        face = topods.Face(explorer.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        if area > max_area:
            max_area = area
            largest_face = face
        explorer.Next()

    if not largest_face:
        raise RuntimeError("No planar face found.")

    # Step 2: Get longest edge on that face
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

    # Step 3: Face normal → Y axis
    surf_adapt = BRepAdaptor_Surface(largest_face)
    if surf_adapt.GetType() != 0:
        raise RuntimeError("Largest face is not planar.")
    # dir_y = surf_adapt.Plane().Axis().Direction()
    dir_z = surf_adapt.Plane().Axis().Direction()

    # Step 4: Orthonormalize axes
    # dir_z = dir_x.Crossed(dir_y)
    # dir_y = dir_z.Crossed(dir_x)
    dir_y = dir_z.Crossed(dir_x)
    dir_x = dir_y.Crossed(dir_z)

    # Step 5: Build transformation
    origin = gp_Pnt(0, 0, 0)
    from_cs = gp_Ax3(origin, dir_x, dir_y)
    to_cs = gp_Ax3(origin, gp_Dir(1, 0, 0), gp_Dir(0, 1, 0))

    trsf = gp_Trsf()
    trsf.SetDisplacement(from_cs, to_cs)

    aligned = BRepBuilderAPI_Transform(solid, trsf, True).Shape()

    return aligned, trsf, to_cs, largest_face, dir_x, dir_y, dir_z


def ensure_right_handed(dir_x, dir_y, dir_z):
    """
    Ensures that the coordinate system is right-handed, particularly after transforms and rotations.
    This is needed to ensure DSTV coordinate compliance
    """
    x = np.array([dir_x.X(), dir_x.Y(), dir_x.Z()])
    y = np.array([dir_y.X(), dir_y.Y(), dir_y.Z()])
    z = np.array([dir_z.X(), dir_z.Y(), dir_z.Z()])

    if np.dot(np.cross(x, y), z) < 0:
        print("⚠️ Detected left-handed system. Reversing Z to enforce right-handed convention.")
        dir_z = gp_Dir(-dir_z.X(), -dir_z.Y(), -dir_z.Z())

    return dir_x, dir_y, dir_z


def compute_obb_geometry(aligned_shape):
    """
    Compute the OBB geometry of a shape that's already aligned.
    Returns a dictionary with:
    - aligned_extents: [length, height, width]
    - aligned_dir_x/y/z: gp_Dir
    - aligned_center: gp_Pnt
    """
    aabb = Bnd_Box()
    brepbndlib.Add(aligned_shape, aabb)
    
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = aabb.Get()
    except:
        raise RuntimeError("Failed to extract bounding box from shape.")

    length = xmax - xmin
    height = ymax - ymin
    width = zmax - zmin

    center = gp_Pnt(
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2
    )

    return {
        "aligned_extents": [length, height, width],
        "aligned_dir_x": gp_Dir(1, 0, 0),
        "aligned_dir_y": gp_Dir(0, 1, 0),
        "aligned_dir_z": gp_Dir(0, 0, 1),
        "aligned_center": center
    }

def get_volume_from_step(step_path: Path) -> float:
    """
    Reads a STEP file and returns its solid volume (in model units³).
    Raises on any read or transfer error.
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STEP file: {step_path!r}")

    # Transfer all roots into a single compound shape
    reader.TransferRoots()
    shape = reader.OneShape()  # TopoDS_Shape

    # Compute volume
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()

    return volume

def get_volume_from_shape(shape: TopoDS_Shape) -> float:
    """
    Computes and returns the enclosed volume of the given solid shape.
    The result is in the model’s cubic units (e.g. mm³ if your geometry is in mm).
    """
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()

# def compute_section_area(solid):
#     """
#     Computes the cross-sectional area by dividing the volume by the length.
#     Assumes solid is aligned with X as the length axis.
#     """
#     props = GProp_GProps()
#     brepgprop.VolumeProperties(solid, props)
#     volume = props.Mass()

#     bbox = Bnd_Box()
#     brepbndlib.Add(solid, bbox)
#     xmin, _, _, xmax, _, _ = bbox.Get()
#     length = xmax - xmin  # assumes X is aligned to length

#     area = volume / length if length > 0 else 0
#     return area

from OCC.Core.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


# def compute_section_area(solid, axis_dir=gp_Dir(1, 0, 0), sample_pos=None, tol=1e-6):
#     """
#     Computes a cross-sectional area by slicing the solid and building a face.
#     Falls back to volume/length if edge loops aren’t closed.
#     """
#     # 1. Bounding box to find slice location
#     bbox = Bnd_Box()
#     brepbndlib.Add(solid, bbox)
#     xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
#     length = xmax - xmin
#     if length <= tol:
#         return 0.0

#     # 2. Define slicing plane (default slice at midpoint)
#     x0 = xmin + (length / 2.0 if sample_pos is None else sample_pos)
#     plane = gp_Pln(gp_Ax3(gp_Pnt(x0, 0, 0), axis_dir))

#     # 3. Perform section
#     section = BRepAlgoAPI_Section(solid, plane)
#     section.ComputePCurveOn1(True)
#     section.Approximation(True)
#     section.Build()
#     if not section.IsDone():
#         raise RuntimeError("Section operation failed")
#     edges_shape = section.Shape()

#     # 4. Extract edges
#     exp = TopExp_Explorer(edges_shape, TopAbs_EDGE)
#     edges = []
#     while exp.More():
#         edges.append(exp.Current())
#         exp.Next()
#     if not edges:
#         return 0.0

#     # 5. Attempt robust wire and face construction
#     try:
#         wire_builder = BRepBuilderAPI_MakeWire()
#         for e in edges:
#             wire_builder.Add(e)
#         if hasattr(wire_builder, 'Close'):
#             wire_builder.Close()
#         if not wire_builder.IsDone():
#             raise RuntimeError("Wire not closed")
#         wire = wire_builder.Wire()

#         face_maker = BRepBuilderAPI_MakeFace(plane)
#         face_maker.Add(wire)
#         props = GProp_GProps()
#         brepgprop.SurfaceProperties(face_maker.Face(), props)
#         return abs(props.Mass())
#     except Exception:
#         # Fallback: approximate as volume/length
#         vol_props = GProp_GProps()
#         brepgprop.VolumeProperties(solid, vol_props)
#         volume = vol_props.Mass()
#         return volume / length



def compute_section_area(solid, axis_dir=gp_Dir(1, 0, 0), sample_pos=None, tol=1e-6):
    """
    Returns GROSS cross-sectional area (outer boundary only).

    Behaviour:
      - If sample_pos is given (offset from xmin): single slice.
      - Else (robust mode): sample along the length, cluster CSAs by similarity,
        take the median of the largest cluster (most commonly seen section).
    """
    # --- bounds ---
    bbox = Bnd_Box()
    brepbndlib.Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    L = xmax - xmin
    if L <= 1e-12:
        return 0.0

    # internal helper: gross CSA at absolute x0 (clamped to interior)
    def _gross_area_at_x(x0_abs: float) -> float | None:
        eps = max(1e-3, 1e-4 * L)  # avoid grazing ends
        x0 = max(xmin + eps, min(xmax - eps, x0_abs))
        plane = gp_Pln(gp_Ax3(gp_Pnt(x0, 0, 0), axis_dir))

        sec = BRepAlgoAPI_Section(solid, plane)
        sec.ComputePCurveOn1(True)
        sec.Approximation(True)
        # sec.SetFuzzyValue(tol)  # uncomment if loops are slightly gappy
        sec.Build()
        if not sec.IsDone():
            return None

        outer, inners, area_outer = _extract_outer_and_holes(sec.Shape(), plane, tol=tol)
        if outer is None:
            return None
        return float(area_outer)  # GROSS (outer only)

    # --- single-slice path (explicit sample_pos) ---
    if sample_pos is not None:
        x0_abs = xmin + float(sample_pos)
        a = _gross_area_at_x(x0_abs)
        if a is not None:
            return a
        # fallback
        gpv = GProp_GProps()
        brepgprop.VolumeProperties(solid, gpv)
        return gpv.Mass() / max(L, 1e-9)

    # --- robust multi-slice path ---
    # force a minimum of 5 samples; default target is 21 for stability
    n_samples_target = 21
    n_samples = max(5, n_samples_target)

    # interior sampling window (skip ends)
    rel_end_margin = 0.06
    abs_min_margin = 15.0
    margin = max(rel_end_margin * L, abs_min_margin)
    x_lo = xmin + margin
    x_hi = xmax - margin
    if x_hi <= x_lo:  # very short parts: shrink margin
        x_lo = xmin + 0.2 * L
        x_hi = xmax - 0.2 * L

    # collect CSA samples
    vals = []
    xs = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        x0_abs = x_lo + t * (x_hi - x_lo)
        a = _gross_area_at_x(x0_abs)
        if a is not None:
            vals.append(a)
            xs.append(x0_abs)

    if not vals:
        # ultimate fallback: volume/length
        gpv = GProp_GProps()
        brepgprop.VolumeProperties(solid, gpv)
        return gpv.Mass() / max(L, 1e-9)

    # --- cluster by relative tolerance, pick the largest cluster, then take median ---
    import statistics
    area_rel_tol = 0.03  # 3% similarity defines "same section" for clustering

    clusters: list[dict] = []  # each: {'ref': float, 'vals': list[float]}
    for a in sorted(vals):
        placed = False
        for c in clusters:
            ref = c['ref']
            if abs(a - ref) <= area_rel_tol * max(ref, 1e-9):
                c['vals'].append(a)
                # keep 'ref' near cluster center to avoid drift from outliers
                c['ref'] = statistics.median(c['vals'])
                placed = True
                break
        if not placed:
            clusters.append({'ref': a, 'vals': [a]})

    clusters.sort(key=lambda c: len(c['vals']), reverse=True)
    core = clusters[0]['vals'] if clusters else vals

    # robust representative of the "most common" cluster
    med_core = float(statistics.median(core))
    # If you literally want the "average of the most common", swap to:
    # med_core = float(sum(core) / len(core))

    # optional diagnostics (only if you've defined CSA_DIAG elsewhere)
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



def compute_average_section_area(solid, axis_dir=gp_Dir(1, 0, 0), n_samples=5, tol=1e-6):
    """
    Samples 'n_samples' cross-sections evenly along the length,
    returns the average area and warns on >1% variation.
    """
    bbox = Bnd_Box()
    brepbndlib.Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    length = xmax - xmin
    if length <= tol:
        return 0.0

    areas = []
    for i in range(n_samples):
        pos = (i + 0.5) * length / n_samples
        areas.append(compute_section_area(solid, axis_dir, sample_pos=pos, tol=tol))

    avg = sum(areas) / len(areas)
    diff = max(areas) - min(areas)
    if avg > 0 and diff > 0.01 * avg:
        print(f"Warning: section area varies by more than 1% (min={min(areas):.6f}, max={max(areas):.6f})")
    return avg


def swap_width_and_height_if_required(profile_match, shape, obb_geom):
    """
    Rotates shape 90° around X axis if profile classification flagged a mismatch
    in width vs height. Recomputes OBB after rotation.
    """
    requires_swap = profile_match.get("Requires_rotation", False)
    if not requires_swap:
        return shape, obb_geom

    print("🔁 Swapping height/width axes to match profile classification")

    try:
        trsf = gp_Trsf()
        trsf.SetRotation(gp_Ax1(obb_geom["aligned_center"], obb_geom["aligned_dir_x"]), np.pi / 2)
        shape_rotated = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
        obb_geom = compute_obb_geometry(shape_rotated)
        return shape_rotated, obb_geom
    except Exception as e:
        raise RuntimeError(f"Rotation failed during swap: {e}")



def compute_dstv_origin(center: gp_Pnt, extents: list, dir_x: gp_Dir, dir_y: gp_Dir, dir_z: gp_Dir):
    """
    Computes the DSTV origin (rear-bottom-left) for a shape aligned via OBB.

    Parameters:
    - center: gp_Pnt — the center of the OBB (post-alignment)
    - extents: [length, height, width] — full extents along dir_x, dir_y, dir_z
    - dir_x: gp_Dir — length/feed direction
    - dir_y: gp_Dir — height/web direction
    - dir_z: gp_Dir — width/flange direction

    Returns:
    - gp_Pnt — rear-bottom-left corner in the aligned local frame
    """

    # Compute offset from center to rear-bottom-left corner
    dx = -extents[0] / 2  # back along length
    dy = -extents[1] / 2  # down along height
    dz = -extents[2] / 2  # left along width

    # Offset vector from center to origin
    offset_vec = gp_Vec(dir_x).Scaled(dx) + gp_Vec(dir_y).Scaled(dy) + gp_Vec(dir_z).Scaled(dz)

    # Translate center to origin
    origin_local = center.Translated(offset_vec)
    return origin_local

from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

def align_obb_to_dstv_frame(shape, origin_local, dir_x, dir_y, dir_z):
    """
    Align a solid into the DSTV frame:
    1. Translates the origin to (0, 0, 0)
    2. Rotates the axes to X = length, Y = height, Z = width (DSTV frame)

    Parameters:
    - shape: the OCC shape to transform
    - origin_local: gp_Pnt, the local rear-bottom-left corner
    - dir_x, dir_y, dir_z: gp_Dir, aligned local axes (length, height, width)

    Returns:
    - Transformed shape aligned with DSTV frame
    - Final rotation transform
    """

    # STEP 1: Translate shape so origin_local becomes (0, 0, 0)
    translate_vec = gp_Vec(origin_local, gp_Pnt(0, 0, 0))
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(translate_vec)

    shape_translated = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

    # STEP 2: Rotate to align local frame with DSTV global frame
    # Local frame with current axes
    local_frame = gp_Ax3(gp_Pnt(0, 0, 0), dir_z, dir_x)  # Z = up, X = length
    # Target DSTV frame
    dstv_frame = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))  # Z up, X forward

    trsf_rotate = gp_Trsf()
    trsf_rotate.SetDisplacement(local_frame, dstv_frame)

    shape_rotated = BRepBuilderAPI_Transform(shape_translated, trsf_rotate, True).Shape()

    return shape_rotated, trsf_rotate

# from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax3, gp_Trsf
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

# Assumes ensure_right_handed is defined elsewhere and imported

# def align_obb_to_dstv_frame(shape,
#                             origin_local,
#                             dir_x,
#                             dir_y,
#                             dir_z,
#                             section_type=None,
#                             properties=None):
#     """
#     Align a solid into the DSTV frame:
#     1. Translate origin_local to (0, 0, 0).
#     2. Rotate axes to DSTV standard (X=length, Y=height, Z=thickness).
#     3. For angle sections, enforce right-handedness and shift the external corner to the origin.

#     Parameters:
#     - shape: OCC shape to transform
#     - origin_local: gp_Pnt, local rear-bottom-left corner
#     - dir_x, dir_y, dir_z: gp_Dir, principal axes
#     - section_type: str, one of 'beam', 'channel', or 'angle'
#     - properties: dict, must contain 'width' and 'height' for angles

#     Returns:
#     - Transformed shape
#     - Composite transformation gp_Trsf
#     """
#     # STEP 1: translate shape so origin_local → (0,0,0)
#     translate_vec = gp_Vec(origin_local, gp_Pnt(0, 0, 0))
#     trsf_translate = gp_Trsf()
#     trsf_translate.SetTranslation(translate_vec)
#     shape_translated = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

#     # STEP 2: rotate to align local frame with DSTV global frame
#     local_frame = gp_Ax3(gp_Pnt(0, 0, 0), dir_z, dir_x)
#     dstv_frame = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
#     trsf_rotate = gp_Trsf()
#     trsf_rotate.SetDisplacement(local_frame, dstv_frame)
#     shape_rotated = BRepBuilderAPI_Transform(shape_translated, trsf_rotate, True).Shape()

#                 # STEP 3: L-section handling by recomputing axis-aligned bounding box
#     if section_type == 'angle':
#         # Enforce right-handed coordinate system via existing helper
#         dir_x, dir_y, dir_z = ensure_right_handed(dir_x, dir_y, dir_z)

#         # --- Recompute axis-aligned bounding box on the rotated shape
#         from OCC.Core.Bnd import Bnd_Box
#         from OCC.Core.BRepBndLib import brepbndlib_Add

#         bb = Bnd_Box()
#         brepbndlib_Add(shape_rotated, bb)
#         xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()

#         # External convex corner: (xmax, ymax, zmin)
#         external_pt = gp_Pnt(xmax, ymax, zmin)

#         # Translate that corner to the origin
#         shift_vec = gp_Vec(external_pt, gp_Pnt(0, 0, 0))
#         trsf_extra = gp_Trsf()
#         trsf_extra.SetTranslation(shift_vec)
#         shape_rotated = BRepBuilderAPI_Transform(shape_rotated, trsf_extra, True).Shape()
#         trsf_rotate.Multiply(trsf_extra)

#                 # Note: for L-sections, after shifting the external corner, we intentionally allow negative interior coordinates,
# # so we do not further translate minima to zero. This preserves correct relative hole positions.

#         # Note: downstream export routines should use the current `shape_rotated` and `trsf_rotate`

#         # Note: downstream export routines should use the current `shape_rotated` and `trsf_rotate`

#     return shape_rotated, trsf_rotate



def rotate_shape_around_axis(shape, center, axis_dir, angle_rad):
    """
    Rotates a shape around an axis defined by a point (center) and a direction vector (axis_dir).
    
    Args:
        shape: TopoDS_Shape to rotate
        center: gp_Pnt — point on the rotation axis
        axis_dir: gp_Dir — direction of the rotation axis
        angle_rad: float — rotation angle in radians
    
    Returns:
        rotated_shape: transformed TopoDS_Shape
        trsf: gp_Trsf object used
    """
    axis = gp_Ax1(center, axis_dir)
    trsf = gp_Trsf()
    trsf.SetRotation(axis, angle_rad)
    transformer = BRepBuilderAPI_Transform(shape, trsf, True)
    rotated_shape = transformer.Shape()
    return rotated_shape, trsf



def refine_orientation_by_flange_face(shape, obb_geom, face_normal_dir, position_dir, axis_label="Y", profile_type="U", shape_category=""):
    """
    Generic version for both channels and angles. Detects orientation by evaluating the largest face
    facing `face_normal_dir` (e.g. Z for channel flanges), and checking its centroid position along `position_dir`.
    Returns (possibly rotated) shape and updated OBB.
    """
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    import numpy as np

    aligned_center = np.array([obb_geom["aligned_center"].X(),
                               obb_geom["aligned_center"].Y(),
                               obb_geom["aligned_center"].Z()])

    face_normal_dir = np.array([face_normal_dir.X(), face_normal_dir.Y(), face_normal_dir.Z()])
    face_normal_dir = face_normal_dir / np.linalg.norm(face_normal_dir)
    
    position_dir = np.array([position_dir.X(), position_dir.Y(), position_dir.Z()])
    position_dir = position_dir / np.linalg.norm(position_dir)

    angle_threshold_rad = np.deg2rad(10)

    largest_area = -1
    selected_centroid = None

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face = explorer.Current()
        surf = BRepAdaptor_Surface(face)
        umin, umax = surf.FirstUParameter(), surf.LastUParameter()
        vmin, vmax = surf.FirstVParameter(), surf.LastVParameter()
        u_mid = (umin + umax) / 2
        v_mid = (vmin + vmax) / 2

        handle = surf.Surface().Surface()
        props = GeomLProp_SLProps(handle, u_mid, v_mid, 1, 1e-6)

        if props.IsNormalDefined():
            normal = props.Normal()
            normal_vec = np.array([normal.X(), normal.Y(), normal.Z()])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            angle = np.arccos(np.clip(np.abs(np.dot(normal_vec, face_normal_dir)), 0, 1))

            if angle < angle_threshold_rad:
                gprops = GProp_GProps()
                brepgprop.SurfaceProperties(face, gprops)
                area = gprops.Mass()
                centroid = gprops.CentreOfMass()
                centroid_vec = np.array([centroid.X(), centroid.Y(), centroid.Z()])

                if area > largest_area:
                    largest_area = area
                    selected_centroid = centroid_vec

        explorer.Next()

    if selected_centroid is None:
        print(f"⚠️ Could not find flange face for {profile_type} orientation check")
        return shape, obb_geom

    vec_to_centroid = selected_centroid - aligned_center
    pos_offset = np.dot(vec_to_centroid, position_dir)

    print(f"🔍 {axis_label} position of largest flange face: {pos_offset:.2f} mm")

    if pos_offset < 0:
        if profile_type == "Channel":
            print(f"🔁 {profile_type} is reversed — rotating 180° around X axis")
            shape, trsf = rotate_shape_around_axis(shape, obb_geom["aligned_center"], obb_geom["aligned_dir_x"], np.pi)
        elif profile_type == "Angle" and shape_category != "EA":
            print(f"🔁 {profile_type} is reversed — translating to origin")
            # 5) recompute the OBB so we have up-to-date center, axes & extents
            obb_geom = compute_obb_geometry(shape)

            # extract as numpy arrays
            center_vec = np.array([
                obb_geom["aligned_center"].X(),
                obb_geom["aligned_center"].Y(),
                obb_geom["aligned_center"].Z()
            ])

            dir_x = np.array([
                obb_geom["aligned_dir_x"].X(),
                obb_geom["aligned_dir_x"].Y(),
                obb_geom["aligned_dir_x"].Z(),
            ])
            dir_y = np.array([
                obb_geom["aligned_dir_y"].X(),
                obb_geom["aligned_dir_y"].Y(),
                obb_geom["aligned_dir_y"].Z(),
            ])
            dir_z = np.array([
                obb_geom["aligned_dir_z"].X(),
                obb_geom["aligned_dir_z"].Y(),
                obb_geom["aligned_dir_z"].Z(),
            ])

            extents = np.array(obb_geom["aligned_extents"])  # [dx, dy, dz]

            # compute the minimum-corner of the OBB
            half = extents * 0.5
            min_corner = (
                center_vec
                - half[0] * dir_x
                - half[1] * dir_y
                - half[2] * dir_z
            )

            # 6) translate that min-corner to the global origin
            translation = gp_Trsf()
            translation.SetTranslation(
                gp_Vec(
                    gp_Pnt(min_corner[0], min_corner[1], min_corner[2]),
                    gp_Pnt(0.0, 0.0, 0.0)
                )
            )
            shape = BRepBuilderAPI_Transform(shape, translation, True).Shape()

            # 7) recompute OBB one last time so obb_geom is in the new frame
            obb_geom = compute_obb_geometry(shape)
            return shape, obb_geom

        else:
            print(f"⚠️ No rotation logic defined for profile type '{profile_type}'")
            obb_geom = compute_obb_geometry(shape)
        return shape, obb_geom
    else:
        print(f"✅ {profile_type} is correctly oriented — no rotation needed.")
        return shape, obb_geom


def refine_profile_orientation(shape, profile_match, obb_geom):
    shape_type = profile_match.get("Profile_type")
    shape_category = profile_match.get("Category")
    # print(profile_match)

    if shape_type == "L":
        print("🔍 Refining angle orientation")
        return refine_orientation_by_flange_face(
        shape, obb_geom,
        face_normal_dir=obb_geom["aligned_dir_z"],
        position_dir=obb_geom["aligned_dir_y"],
        axis_label="Y",
        profile_type="Angle",
        shape_category = shape_category
        )

    elif shape_type == "U":
        print("🔍 Refining channel orientation")
        return refine_orientation_by_flange_face(
            shape, obb_geom,
            face_normal_dir=obb_geom["aligned_dir_z"],
            position_dir=obb_geom["aligned_dir_y"],
            axis_label="Y",
            profile_type="Channel",
            shape_category = shape_category
        )

    else:
        print(f"ℹ️ No refinement needed for shape type '{shape_type}'")
        return shape, obb_geom

