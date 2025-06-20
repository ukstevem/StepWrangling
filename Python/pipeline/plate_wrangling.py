from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax3, gp_Trsf
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomLProp import GeomLProp_SLProps

from pipeline.geometry_utils import get_volume_from_step, get_volume_from_shape

THICKNESS_LIMIT_MM = 101.0
mat_density = .000000784

def compute_section_and_length_and_origin_from_obb(solid):
    obb = Bnd_OBB()
    brepbndlib.AddOBB(solid, obb, True, True)
    center = obb.Center()
    half_extents = [obb.XHSize(), obb.YHSize(), obb.ZHSize()]
    axes = [obb.XDirection(), obb.YDirection(), obb.ZDirection()]  # gp_Dir
    return center, half_extents, axes

def get_face_area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()

def is_planar_by_curvature(face, tol=1e-4):
    surface = BRep_Tool.Surface(face)
    try:
        umin, umax, vmin, vmax = surface.Bounds()
        u_mid = (umin + umax) / 2
        v_mid = (vmin + vmax) / 2
        props = GeomLProp_SLProps(surface, u_mid, v_mid, 2, tol)
        if not props.IsCurvatureDefined():
            return False
        return abs(props.MaxCurvature()) < tol and abs(props.MinCurvature()) < tol
    except:
        return False

def get_face_normal(face):
    surface = BRep_Tool.Surface(face)
    umin, umax, vmin, vmax = surface.Bounds()
    u_mid = (umin + umax) / 2
    v_mid = (vmin + vmax) / 2
    props = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    return props.Normal()  # gp_Dir

def align_plate_to_xy_plane(solid):
    """
    Detects a flat profile plate (thickness ≤ 101 mm), aligns so
      X = length, Y = width, Z = thickness (face normal),
    transforms into the world XY plane, and returns:
      (is_plate: bool,
       aligned_solid or None,
       dstv_ax3 or None,
       thickness_mm or None,
       reason: str)
    """
    try:
        # 1) Compute OBB
        center, half_extents, axes = compute_section_and_length_and_origin_from_obb(solid)
        # sort by descending half-extent
        idx_sorted = sorted(range(3), key=lambda i: -half_extents[i])
        length_idx, width_idx, thickness_idx = idx_sorted
        thickness_mm = 2.0 * half_extents[thickness_idx]
        length_mm = 2.0 * half_extents[length_idx]
        width_mm = 2.0 * half_extents[width_idx]
        step_mass = get_volume_from_shape(solid)*mat_density

        # 2) Collect nearly-planar faces
        faces = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            f = exp.Current()
            if is_planar_by_curvature(f):
                faces.append(f)
            exp.Next()
        if not faces:
            return False, solid, None, thickness_mm, length_mm, width_mm, step_mass, "No planar faces found"

        # 3) Largest face normal
        largest = max(faces, key=get_face_area)
        fn = get_face_normal(largest)
        if fn is None:
            return False, solid, None, thickness_mm, length_mm, width_mm, step_mass, "Could not determine face normal"

        # 4) Build DSTV axes (force correct types)
        cx = gp_Pnt(center.X(), center.Y(), center.Z())
        raw_x = axes[length_idx]
        zx = gp_Dir(fn.X(), fn.Y(), fn.Z()).Reversed()
        xx = gp_Dir(raw_x.X(), raw_x.Y(), raw_x.Z())
        yy = xx.Crossed(zx)
        dstv_ax3 = gp_Ax3(cx, zx, xx)

        # 5) Transform into world XY plane
        world_ax3 = gp_Ax3(gp_Pnt(0,0,0), gp_Dir(0,0,1), gp_Dir(1,0,0))
        tr = gp_Trsf()
        # use SetTransformation for OCC 7.9.0
        tr.SetTransformation(dstv_ax3, world_ax3)
        transformed = BRepBuilderAPI_Transform(solid, tr, True).Shape()

        if thickness_mm > THICKNESS_LIMIT_MM:
            print("Too thick for plate")
            return False, transformed, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass, "Too thick for plate"

        return True, transformed, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass, "Plate aligned and transformed to XY plane"

    except Exception as e:
        return False, None, None, None, None, None, None, f"Unhandled error: {e}"
