# --- pythonocc-core imports ---
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Line, GeomAbs_Ellipse
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax3, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import breptools
import numpy as np, math

# ---------- helpers ----------
def _largest_planar_face(solid):
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    best, best_area = None, -1.0
    while exp.More():
        f = topods.Face(exp.Current())
        s = BRepAdaptor_Surface(f)
        if s.GetType() == GeomAbs_Plane:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(f, props)
            area = props.Mass()
            if area > best_area:
                best, best_area = f, area
        exp.Next()
    return best

def _outer_wire(face):
    try:
        w = breptools.OuterWire(face)
        if not w.IsNull():
            return w
    except Exception:
        pass
    # Fallback: pick the wire with largest perimeter estimated from straight edges
    choice, best_sum = None, -1.0
    wexp = TopExp_Explorer(face, TopAbs_WIRE)
    while wexp.More():
        w = topods.Wire(wexp.Current())
        total = 0.0
        eexp = TopExp_Explorer(w, TopAbs_EDGE)
        while eexp.More():
            e = topods.Edge(eexp.Current())
            c = BRepAdaptor_Curve(e)
            if c.GetType() == GeomAbs_Line:
                p1 = c.Value(c.FirstParameter()); p2 = c.Value(c.LastParameter())
                total += gp_Vec(p1, p2).Magnitude()
            eexp.Next()
        if total > best_sum:
            choice, best_sum = w, total
        wexp.Next()
    return choice

def _unit_np(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-16 else v

def _ensure_right_handed(dx: gp_Dir, dy: gp_Dir, dz: gp_Dir):
    x = np.array([dx.X(), dx.Y(), dx.Z()])
    y = np.array([dy.X(), dy.Y(), dy.Z()])
    z = np.array([dz.X(), dz.Y(), dz.Z()])
    if float(np.dot(np.cross(x, y), z)) < 0:
        # flip Y to enforce RH (could also flip X; choose one consistently)
        dy = gp_Dir(-dy.X(), -dy.Y(), -dy.Z())
    return dx, dy, dz

def _build_transform(dir_x: gp_Dir, dir_z: gp_Dir):
    # Orthonormalize via Y = Z × X, then X = Y × Z
    y = np.cross([dir_z.X(), dir_z.Y(), dir_z.Z()],
                 [dir_x.X(), dir_x.Y(), dir_x.Z()])
    y = _unit_np(y)
    dir_y = gp_Dir(float(y[0]), float(y[1]), float(y[2]))
    x2 = np.cross([dir_y.X(), dir_y.Y(), dir_y.Z()],
                  [dir_z.X(), dir_z.Y(), dir_z.Z()])
    x2 = _unit_np(x2)
    dir_x = gp_Dir(float(x2[0]), float(x2[1]), float(x2[2]))
    dir_x, dir_y, dir_z = _ensure_right_handed(dir_x, dir_y, dir_z)

    origin = gp_Pnt(0, 0, 0)
    # gp_Ax3(origin, main_dir(Z), XDir)
    from_cs = gp_Ax3(origin, dir_z, dir_x)
    world_cs = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))

    trsf = gp_Trsf()
    trsf.SetDisplacement(from_cs, world_cs)
    return trsf, from_cs, world_cs, dir_x, dir_y, dir_z

# ---------- main function ----------
def robust_align_length_to_X(solid, tol_len=1e-3, rel_tie=1e-6, debug=False):
    """
    Deterministic alignment (PCA/OBB version):
      * Choose largest planar face as reference (top/bottom).
      * Use its OUTER wire only (ignore holes).
      * Project outer wire edge points into the face plane and compute the
        principal in-plane direction (major PCA/OBB axis) as the 'length'.
      * If nearly isotropic in-plane spread, fall back to projecting world +X (or +Y) into the face.
      * Set Z = +face normal; X = chosen in-plane direction; Y = Z×X; enforce right-handedness.
      * Map (Z→+Z, X→+X) and return aligned shape & basis.

    Returns:
      aligned, trsf, world_cs, face, dir_x, dir_y, dir_z [, dbg]
    """
    dbg = {}

    # 1) Reference face (largest planar)
    face = _largest_planar_face(solid)
    if not face:
        raise RuntimeError("No planar face found.")
    srf = BRepAdaptor_Surface(face)
    if srf.GetType() != GeomAbs_Plane:
        raise RuntimeError("Reference face is not planar.")

    # 2) Face normal → prefer +Z
    n = srf.Plane().Axis().Direction()
    dir_z = gp_Dir(n.X(), n.Y(), n.Z())
    if dir_z.Z() < 0.0:
        dir_z = gp_Dir(-dir_z.X(), -dir_z.Y(), -dir_z.Z())
    nz = np.array([dir_z.X(), dir_z.Y(), dir_z.Z()], dtype=float)

    # 3) Outer wire
    wire = _outer_wire(face)
    if wire is None or wire.IsNull():
        raise RuntimeError("Failed to get outer wire of reference face.")

    # 4) Collect edge endpoints on the outer wire
    pts = []
    eexp = TopExp_Explorer(wire, TopAbs_EDGE)
    while eexp.More():
        e = topods.Edge(eexp.Current())
        c = BRepAdaptor_Curve(e)
        # Note: we don't filter by curve type; endpoints are fine for polygons/chamfers
        p1 = c.Value(c.FirstParameter()); p2 = c.Value(c.LastParameter())
        pts.append((p1.X(), p1.Y(), p1.Z()))
        pts.append((p2.X(), p2.Y(), p2.Z()))
        eexp.Next()

    if not pts:
        raise RuntimeError("No edge points found on outer wire.")

    # 5) Build a 2D basis (u, v) spanning the face plane
    #    seed chosen to avoid collinearity with nz
    seed = np.array([1.0, 0.0, 0.0], dtype=float) if abs(nz[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    u = seed - np.dot(seed, nz) * nz
    nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        # extreme edge-on fallback
        seed = np.array([0.0, 1.0, 0.0], dtype=float)
        u = seed - np.dot(seed, nz) * nz
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            # pathological geometry
            raise RuntimeError("Failed to build a stable in-plane basis.")
    u /= nu
    v = np.cross(nz, u); v /= float(np.linalg.norm(v))

    # 6) Project 3D points to 2D plane coords (u,v)
    P = np.empty((2, len(pts)), dtype=float)
    for i, (x, y, z) in enumerate(pts):
        p = np.array([x, y, z], dtype=float)
        P[0, i] = np.dot(p, u)
        P[1, i] = np.dot(p, v)

    # 7) PCA on 2D points (centered)
    Pc = P - P.mean(axis=1, keepdims=True)
    denom = max(P.shape[1] - 1, 1)
    C = (Pc @ Pc.T) / float(denom)  # 2x2 covariance
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    # largest eigenvector = major in-plane axis
    major2d = eigvecs[:, 1]
    lam_min, lam_max = float(eigvals[0]), float(eigvals[1])

    # 8) Degeneracy guard: nearly isotropic (circle-ish) → project world +X (else +Y)
    isotropic = (lam_max <= 1e-12) or ((lam_min / lam_max) > 0.95)
    if isotropic:
        wx = np.array([1.0, 0.0, 0.0], dtype=float)
        d = wx - np.dot(wx, nz) * nz
        nd = float(np.linalg.norm(d))
        if nd < 1e-12:
            wy = np.array([0.0, 1.0, 0.0], dtype=float)
            d = wy - np.dot(wy, nz) * nz
            nd = float(np.linalg.norm(d))
        d /= nd
    else:
        d = major2d[0] * u + major2d[1] * v
        d /= float(np.linalg.norm(d))

    # Prefer +world X to fix the sign deterministically
    if d[0] < 0.0:
        d = -d

    dir_x = gp_Dir(float(d[0]), float(d[1]), float(d[2]))

    # 9) Build transform and return
    trsf, from_cs, world_cs, dir_x, dir_y, dir_z = _build_transform(dir_x, dir_z)
    aligned = BRepBuilderAPI_Transform(solid, trsf, True).Shape()

    if debug:
        ang_xy = math.degrees(math.atan2(dir_x.Y(), dir_x.X()))
        dbg.update({
            "mode": "wire-PCA" if not isotropic else "round-fallback(+X projected)",
            "eigvals": (lam_min, lam_max),
            "dir_x": (dir_x.X(), dir_x.Y(), dir_x.Z()),
            "dir_z": (dir_z.X(), dir_z.Y(), dir_z.Z()),
            "angle_X_deg_after": ((ang_xy + 360.0) % 360.0)
        })

    return (aligned, trsf, world_cs, face, dir_x, dir_y, dir_z, dbg) if debug \
         else (aligned, trsf, world_cs, face, dir_x, dir_y, dir_z)

