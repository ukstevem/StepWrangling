from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax3, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import topods

import numpy as np

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
    dir_y = surf_adapt.Plane().Axis().Direction()

    # Step 4: Orthonormalize axes
    dir_z = dir_x.Crossed(dir_y)
    dir_y = dir_z.Crossed(dir_x)

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


# def compute_section_area(shape):
#     """
#     Compute the surface area of a face or shell.
#     If shape is a solid, warns user and returns 0.
#     """
#     if shape.ShapeType() != 4:  # 4 = TopAbs_FACE
#         print("⚠️ Warning: compute_section_area expected a face, not a solid.")
#         return 0

#     props = GProp_GProps()
#     face = topods.Face(shape)
#     brepgprop.SurfaceProperties(face, props)
#     return props.Mass()

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE

def compute_section_area_from_section(section_shape):
    """
    Compute the cross-sectional area from a section shape (wires).
    """
    total_area = 0.0

    # Extract wires from the section
    explorer = TopExp_Explorer(section_shape, TopAbs_WIRE)
    while explorer.More():
        wire = explorer.Current()
        face_maker = BRepBuilderAPI_MakeFace(wire)
        if face_maker.IsDone():
            face = face_maker.Face()
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()
            total_area += area
        explorer.Next()

    if total_area == 0.0:
        raise ValueError("❌ No valid faces created from wires — section area is zero.")

    return total_area


def compute_section_area(shape, slice_x=0.5, tol=1e-2, min_area=100, max_area=20000):
    """
    Computes the cross-sectional area of a solid by slicing at a given X,
    building a wire from edges, and computing the face area.

    Automatically rejects invalid areas.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln

    # Compute bounding box and slicing location
    aabb = Bnd_Box()
    brepbndlib.Add(shape, aabb)
    xmin, _, _, xmax, _, _ = aabb.Get()
    section_x = xmin + slice_x * (xmax - xmin)
    print(f"📏 Sectioning at X={section_x:.2f} with tolerance={tol}")

    # Cutting plane
    plane = gp_Pln(gp_Pnt(section_x, 0, 0), gp_Dir(1, 0, 0))
    section = BRepAlgoAPI_Section(shape, plane, False)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()

    section_shape = section.Shape()

    # Extract edges
    explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
    edges = []
    while explorer.More():
        edges.append(explorer.Current())
        explorer.Next()

    if not edges:
        raise RuntimeError("❌ No edges found in section shape.")

    # Build wire
    wire_maker = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire_maker.Add(edge)

    if not wire_maker.IsDone():
        raise RuntimeError("❌ Failed to create wire from section edges.")

    wire = wire_maker.Wire()
    if wire.IsNull():
        raise RuntimeError("❌ Wire is null.")

    # Build face and compute area
    face_maker = BRepBuilderAPI_MakeFace(wire)
    if not face_maker.IsDone():
        raise RuntimeError("❌ Failed to create face from wire.")

    face = face_maker.Face()
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass()

    print(f"✅ Section area: {area:.2f} mm²")

    # Sanity check
    if area < min_area or area > max_area:
        raise ValueError(f"❌ CSA {area:.2f} mm² out of expected range ({min_area}–{max_area} mm²)")

    return area




    print(f"✅ Section area: {area:.2f} mm²")
    return area



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



def refine_orientation_by_flange_face(shape, obb_geom, face_normal_dir, position_dir, axis_label="Y", profile_type="U"):
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
        elif profile_type == "Angle":
            print(f"🔁 {profile_type} is reversed — rotating 180° around Z axis")
            shape, trsf = rotate_shape_around_axis(shape, obb_geom["aligned_center"], obb_geom["aligned_dir_z"], np.pi)
        else:
            print(f"⚠️ No rotation logic defined for profile type '{profile_type}'")
            obb_geom = compute_obb_geometry(shape)
        return shape, obb_geom
    else:
        print(f"✅ {profile_type} is correctly oriented — no rotation needed.")
        return shape, obb_geom


def refine_profile_orientation(shape, profile_match, obb_geom):
    shape_type = profile_match.get("Profile_type")

    if shape_type == "L":
        print("🔍 Refining angle orientation")
        return refine_orientation_by_flange_face(
        shape, obb_geom,
        face_normal_dir=obb_geom["aligned_dir_z"],
        position_dir=obb_geom["aligned_dir_y"],
        axis_label="Y",
        profile_type="Angle"
        )

    elif shape_type == "U":
        print("🔍 Refining channel orientation")
        return refine_orientation_by_flange_face(
            shape, obb_geom,
            face_normal_dir=obb_geom["aligned_dir_z"],
            position_dir=obb_geom["aligned_dir_y"],
            axis_label="Y",
            profile_type="Channel"
        )

    else:
        print(f"ℹ️ No refinement needed for shape type '{shape_type}'")
        return shape, obb_geom  # ✅ This prevents the unpacking error
