import math
import numpy as np
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Trsf
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomLProp import GeomLProp_SLProps

from pipeline.geometry_utils import plate_guard_heuristics

THICKNESS_LIMIT_MM = 101.0
mat_density = 0.000000784

# ---------- small helpers ----------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _proj_inplane_np(v_np, z_np):
    # project v onto plane orthogonal to z, and normalize
    return _unit(v_np - np.dot(v_np, z_np) * z_np)

def _acute_inplane_delta_deg(a_dir, b_dir, z_np):
    """
    Acute (0..90] in-plane angle between OCC directions a_dir, b_dir,
    measured in the plane orthogonal to z_np. Sign and 180° symmetry removed.
    Returns None if projection degenerates.
    """
    a = _proj_inplane_np(np.array([a_dir.X(), a_dir.Y(), a_dir.Z()], float), z_np)
    b = _proj_inplane_np(np.array([b_dir.X(), b_dir.Y(), b_dir.Z()], float), z_np)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return None
    signed = math.degrees(math.atan2(np.dot(np.cross(a, b), z_np), float(np.dot(a, b))))
    acute  = abs((signed + 180.0) % 360.0 - 180.0)
    return min(acute, 180.0 - acute)

def _angle_deg_between(a, b):
    a = _unit(a); b = _unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(c))

def _yaw_deg_from_xdir(xdir: gp_Dir) -> float | None:
    """Yaw of X relative to world +X in XY-plane (pre-transform)."""
    xv = np.array([xdir.X(), xdir.Y(), 0.0])
    n = np.linalg.norm(xv)
    if n < 1e-12:
        return None
    return math.degrees(math.atan2(xv[1], xv[0]))

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
    except Exception:
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

def _world_extents_mm(shape):
    obb = Bnd_OBB()
    brepbndlib.AddOBB(shape, obb, True, True)  # triangulation & use-shape-loc
    return 2.0 * obb.XHSize(), 2.0 * obb.YHSize(), 2.0 * obb.ZHSize()

# ---------- main ----------
def align_plate_to_xy_plane(
    solid,
    prefer_dir_x: gp_Dir | None = None,
    lock_x: bool = True,
    square_rel_tol: float = 1e-3,
    warn_obb_delta_deg: float | None = 1.0,  # set None to disable warning
    # NEW: optional absolute thickness gate (disabled by default)
    thickness_limit_mm: float | None = None,
):
    """
    Final plate alignment WITHOUT reintroducing random yaw:
      - Z = +largest planar face normal
      - X = locked to prefer_dir_x if provided, else:
          * if near-square/round: projected world +X (deterministic)
          * else: projected OBB long axis
      - Right-handed basis; map frame -> world

    Returns (9):
      (is_plate, aligned_solid, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass, message, sig)
      where sig = {
        'yaw_deg': float|None,
        'obb_vs_x_deg': float|None,
        'x_choice': 'prefer'|'worldX'|'obb_projected'|'fallback',
        'square_guard': bool,
        'dims_mm': {'L':..., 'W':..., 'T':...}
      }

    Notes:
      * thickness_mm is ALWAYS the smallest of the three aligned extents.
      * thickness_limit_mm is optional; if None, no absolute thickness gate is applied.
    """

    def align_plate_to_xy_plane(solid, *args, **kwargs):
        # --- Plate guardrail ---
        ok, diag = plate_guard_heuristics(solid, axis_dir=gp_Dir(1,0,0))
        if not ok:
            # Keep your return signature; veto plate early
            return (
                False,          # is_plate
                None,           # final_aligned_solid
                None,           # ax3
                0.0, 0.0, 0.0,  # thickness_mm, length_mm, width_mm
                0.0,            # step_mass
                f"Plate guard veto: {diag}",  # msg
                None            # sig / extra
            )


    try:
        # --- dimensions from OBB (not yaw) on the INPUT solid ---
        center, half_extents, axes = compute_section_and_length_and_origin_from_obb(solid)
        idx_sorted = sorted(range(3), key=lambda i: -half_extents[i])
        length_idx, width_idx, thickness_idx = idx_sorted
        length_mm    = 2.0 * half_extents[length_idx]
        width_mm     = 2.0 * half_extents[width_idx]
        thickness_mm = 2.0 * half_extents[thickness_idx]

        # mass from volume (assumes mat_density is defined elsewhere)
        props_v = GProp_GProps()
        brepgprop.VolumeProperties(solid, props_v)
        step_mass = props_v.Mass() * mat_density

        # --- largest planar face normal -> Z ---
        faces = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            f = exp.Current()
            if is_planar_by_curvature(f):
                faces.append(f)
            exp.Next()
        if not faces:
            sig = {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'n/a', 'square_guard': False,
                   'dims_mm': {'L': length_mm, 'W': width_mm, 'T': thickness_mm}}
            return False, solid, None, thickness_mm, length_mm, width_mm, step_mass, "No planar faces found", sig

        largest = max(faces, key=get_face_area)
        fn = get_face_normal(largest)
        if fn is None:
            sig = {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'n/a', 'square_guard': False,
                   'dims_mm': {'L': length_mm, 'W': width_mm, 'T': thickness_mm}}
            return False, solid, None, thickness_mm, length_mm, width_mm, step_mass, "Could not determine face normal", sig

        dir_z = gp_Dir(fn.X(), fn.Y(), fn.Z())
        if dir_z.Z() < 0:  # keep Z roughly "up"
            dir_z = gp_Dir(-dir_z.X(), -dir_z.Y(), -dir_z.Z())
        z_np = np.array([dir_z.X(), dir_z.Y(), dir_z.Z()], float)

        # --- choose X deterministically, avoid re-yaw ---
        def proj_to_plane(v_np):
            return _unit(v_np - np.dot(v_np, z_np) * z_np)

        x_choice = None
        x_np = None

        if lock_x and prefer_dir_x is not None:
            v = proj_to_plane(np.array([prefer_dir_x.X(), prefer_dir_x.Y(), prefer_dir_x.Z()], float))
            if np.linalg.norm(v) > 1e-12:
                x_np = v
                x_choice = 'prefer'

        is_square = False
        if x_np is None:
            # near-square / round check
            is_square = abs(length_mm - width_mm) <= max(square_rel_tol * max(length_mm, width_mm), 1e-6)
            if is_square:
                v = proj_to_plane(np.array([1.0, 0.0, 0.0], float))  # world +X
                if np.linalg.norm(v) < 1e-12:
                    v = proj_to_plane(np.array([0.0, 1.0, 0.0], float))  # fallback world +Y
                x_np = v
                x_choice = 'worldX'
            else:
                a = axes[length_idx]
                v = proj_to_plane(np.array([a.X(), a.Y(), a.Z()], float))
                if np.linalg.norm(v) < 1e-12:
                    # emergency: any X orthogonal to Z
                    vdir = gp_Ax3(gp_Pnt(0,0,0), dir_z).XDirection()
                    v = proj_to_plane(np.array([vdir.X(), vdir.Y(), vdir.Z()], float))
                    x_choice = 'fallback'
                else:
                    x_choice = 'obb_projected'
                x_np = v

        # stable sign toward +world X when possible
        if np.dot(x_np, np.array([1.0, 0.0, 0.0])) < 0:
            x_np = -x_np

        # right-handed: Y = Z × X ; then X = Y × Z (re-orthonormalize)
        y_np = _unit(np.cross(z_np, x_np))
        if np.dot(np.cross(x_np, y_np), z_np) < 0:
            y_np = -y_np
        x_np = _unit(np.cross(y_np, z_np))

        dir_x = gp_Dir(float(x_np[0]), float(x_np[1]), float(x_np[2]))

        # signature bits
        yaw_deg = _yaw_deg_from_xdir(dir_x)

        # compare OBB long axis to locked X as an ACUTE, IN-PLANE angle
        obb_vs_x_deg = None
        try:
            obb_long = axes[length_idx]  # gp_Dir from source OBB
            obb_vs_x_deg = _acute_inplane_delta_deg(dir_x, obb_long, z_np)
        except Exception:
            pass

        # --- build frame and transform ---
        cx = gp_Pnt(center.X(), center.Y(), center.Z())
        dstv_ax3 = gp_Ax3(cx, dir_z, dir_x)  # main=Z, XDir=X
        world_ax3 = gp_Ax3(gp_Pnt(0,0,0), gp_Dir(0,0,1), gp_Dir(1,0,0))

        tr = gp_Trsf()
        tr.SetTransformation(dstv_ax3, world_ax3)
        transformed = BRepBuilderAPI_Transform(solid, tr, True).Shape()

        # --- recompute dims from the final orientation and ENFORCE T=min(L,W,T) ---
        Lx, Ly, Lz = _world_extents_mm(transformed)  # extents along global X,Y,Z
        L, W, T = sorted((float(Lx), float(Ly), float(Lz)), reverse=True)  # L ≥ W ≥ T
        length_mm, width_mm, thickness_mm = L, W, T

        # update signature dims to FINAL numbers
        sig = {
            'yaw_deg': yaw_deg,
            'obb_vs_x_deg': obb_vs_x_deg,
            'x_choice': x_choice,
            'square_guard': bool(is_square),
            'dims_mm': {'L': length_mm, 'W': width_mm, 'T': thickness_mm},
        }

        # Optional absolute thickness gate (disabled by default)
        if thickness_limit_mm is not None and thickness_mm > thickness_limit_mm:
            return (False, transformed, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass,
                    "Too thick for plate", sig)

        return (True, transformed, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass,
                "Plate aligned to XY; thickness = min(X,Y,Z)", sig)

    except Exception as e:
        sig = {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'error', 'square_guard': False,
               'dims_mm': {'L': None, 'W': None, 'T': None}}
        return False, None, None, None, None, None, None, f"Unhandled error: {e}", sig
