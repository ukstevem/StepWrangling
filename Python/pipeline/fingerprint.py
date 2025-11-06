# pipeline/fingerprint.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple
import json, math, hashlib

# OCC imports kept local to functions that need them to avoid import cycles
# from OCC.Core.gp import gp_Trsf
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
# from OCC.Core.TopExp import TopExp_Explorer
# from OCC.Core.TopAbs import TopAbs_EDGE
# from OCC.Core.TopoDS import topods
# from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
# from OCC.Core.GeomAbs import GeomAbs_Circle

@dataclass(frozen=True)
class Fingerprint:
    route: str                     # 'section' | 'plate' | 'unknown'
    family_key: str                # coarse bucket (fast, route-aware)
    content_key: str               # strict match key (route artifact)
    hand: int = 0                  # -1, 0, +1
    fallback_key: str | None = None  # optional canonical mesh hash
    meta: Dict[str, Any] = None      # small extra bits (profile_type, dims…)

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        # serialize tuples/sets inside meta for safety
        return d

# ---------- generic helpers ----------

def _md5_of_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _quant(x: float, step: float) -> float:
    return round(round(x / step) * step, 6)  # stabilize float repr

def _dims_to_half_sorted(extents_xyz: Tuple[float, float, float], step=1.0) -> Tuple[float, float, float]:
    L, H, W = extents_xyz
    hx, hy, hz = (L/2.0, H/2.0, W/2.0)
    return tuple(sorted((_quant(hx, step), _quant(hy, step), _quant(hz, step)), reverse=True))

# ---------- UNKNOWN ROUTE ----------

def fingerprint_unknown(*, shape, extents_xyz: Tuple[float, float, float],
                        quant_step: float = 1.0) -> Fingerprint:
    """
    Coarse: tolerant OBB (sorted half-extents).
    Content: canonical mesh hash (tessellate→normalize→quantize).
    """
    family_key = json.dumps({
        "type": "obb_sorted_half_extents",
        "half": _dims_to_half_sorted(extents_xyz, step=quant_step)
    })
    family_key = hashlib.md5(family_key.encode()).hexdigest()

    fallback = _canonical_mesh_hash(shape)  # robust geometric tie-breaker
    # For unknowns, use the mesh hash also as content key.
    return Fingerprint(route="unknown",
                       family_key=family_key,
                       content_key=fallback,
                       hand=0,
                       fallback_key=None,
                       meta={})

def _canonical_mesh_hash(shape, deflection: float = 0.5, vert_quant: float = 0.1) -> str:
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepTools import breptools
    from OCC.Core.gp import gp_Pnt
    import hashlib, json

    try:
        BRepMesh_IncrementalMesh(shape, deflection)
        pts = []
        it = TopExp_Explorer(shape, TopAbs_FACE)
        while it.More():
            f = topods.Face(it.Current())
            loc = f.Location()
            triangulation = breptools.Triangulation(f, loc)
            if triangulation:
                nb = triangulation.NbNodes()
                tr = loc.Transformation()
                for i in range(1, nb + 1):
                    p = triangulation.Node(i)  # gp_Pnt
                    if not loc.IsIdentity():
                        q = gp_Pnt(p.X(), p.Y(), p.Z())
                        q.Transform(tr)
                        px, py, pz = q.X(), q.Y(), q.Z()
                    else:
                        px, py, pz = p.X(), p.Y(), p.Z()
                    pts.append((_quant(px, vert_quant), _quant(py, vert_quant), _quant(pz, vert_quant)))
            it.Next()
        if not pts:
            return hashlib.md5(b"mesh:empty").hexdigest()
        pts.sort()
        return hashlib.md5(json.dumps(pts).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.md5(b"mesh:error").hexdigest()

# ---------- SECTION ROUTE ----------

def fingerprint_section(*,
                        refined_shape,
                        extents_xyz: Tuple[float, float, float],  # (L, H, W) in DSTV pose
                        section_area: float,
                        profile_type: str | None,
                        raw_df_holes,     # pandas DF from classify_and_project_holes_dstv
                        endcuts: Dict[str, Any] | None = None,
                        dim_step: float = 1.0,
                        phi_step: float = 0.005,
                        nc1_hash: str | None = None) -> Fingerprint:
    """
    Coarse (family): route-aware — unsorted (L,H,W), profile tag, compactness phi.
    Content: normalized NC JSON (holes/slots/end cuts) + dims + hand flag.
    Fallback: canonical mesh hash in DSTV frame (optional to compute).
    """
    L, H, W = extents_xyz
    Lq, Hq, Wq = _quant(L, dim_step), _quant(H, dim_step), _quant(W, dim_step)
    phi = (section_area / (H * W)) if (H > 0 and W > 0) else 0.0
    phiq = _quant(phi, phi_step)
    tag = profile_type or "open"

    fam_obj = {"route": "section", "tag": tag, "L": Lq, "H": Hq, "W": Wq, "phi": phiq}
    family_key = hashlib.md5(json.dumps(fam_obj, sort_keys=True).encode()).hexdigest()

    hand = _hand_from_holes(raw_df_holes)

    nc_json = _normalize_nc_content(raw_df_holes, endcuts, profile_type, (Lq, Hq, Wq), hand)
    content_key = nc1_hash or _md5_of_json(nc_json)

    # Optional: tie-breaker if you ever see collisions
    # fallback = _canonical_mesh_hash(refined_shape)
    fallback = None

    return Fingerprint(
            route="section",
            family_key=family_key,
            content_key=content_key,
            hand=hand,
            fallback_key=fallback,
            meta={"profile_type": profile_type, "dims": (Lq, Hq, Wq)}
        )

def _hand_from_holes(df) -> int:
    if df is None or "Code" not in df.columns:
        return 0
    has_O = (df["Code"] == "O").any()
    has_U = (df["Code"] == "U").any()
    if has_O and not has_U: return +1
    if has_U and not has_O: return -1
    return 0

def _normalize_nc_content(df, endcuts, profile_type, dims_q, hand: int) -> Dict[str, Any]:
    """
    Build a compact, deterministic NC representation:
    - Round coords/diameters to 0.1 mm
    - Sort by (X,Y,face,type)
    - Include endcuts in a stable form if provided
    """
    round_c = lambda v: _quant(float(v), 0.1)
    holes = []
    if df is not None and len(df):
        # Expect columns: ["Code","Diameter (mm)","X (mm)","Y (mm)", ...]
        for row in df.itertuples(index=False):
            face = getattr(row, "Code", "")
            x = round_c(getattr(row, "X (mm)", 0.0))
            y = round_c(getattr(row, "Y (mm)", 0.0))
            d = _quant(getattr(row, "Diameter (mm)", 0.0), 0.01)
            # Extend later for slots/countersinks if your DF has them
            holes.append({"t": "hole", "f": face, "x": x, "y": y, "d": d})
    holes.sort(key=lambda h: (h["f"], h["x"], h["y"], h["d"]))

    ec_list = []
    if endcuts:
        # Accept either:
        #  A) simple mapping of {name: float_angle_deg}
        #  B) rich mapping of {name: {"type":..,"n":[...],"off":...}}
        # Build a small stable representation with rounding.
        def _round_endcut_value(val):
            # val can be float or dict
            if isinstance(val, (int, float)):
                # store as angle degrees, quantized to 0.1
                return {"angle_deg": _quant(float(val), 0.1)}
            if isinstance(val, dict):
                return {
                    "type": val.get("type", ""),
                    "n": [ _quant(float(v), 1e-3) for v in val.get("n", []) ],
                    "off": _quant(float(val.get("off", 0.0)), 0.1),
                }
            # unknown type → stringify to stay robust
            return {"value": str(val)}

        for k in sorted(endcuts.keys()):
            ec_list.append({k: _round_endcut_value(endcuts[k])})

    Lq,Hq,Wq = dims_q
    nc_json = {
        "v": 1,
        "profile": profile_type or "",
        "dims": {"L": Lq, "H": Hq, "W": Wq},
        "hand": hand,
        "features": holes,
        "endcuts": ec_list
    }
    return nc_json

# ---------- PLATE ROUTE ----------

def fingerprint_plate(*,
                      plate_shape, ax3,
                      L: float, W: float, T: float,
                      quant_LW: float = 1.0, quant_T: float = 0.01,
                      outline_hash: str | None = None,        # outer-only (optional)
                      outline_full_hash: str | None = None     # OUTER + INTERNALS (preferred)
                      ) -> Fingerprint:
    """
    Family (coarse): outer outline only (if available) → thickness-agnostic.
    Content (strict): FULL outline (outer + internal loops) + thickness.
    """
    Lq, Wq, Tq = _quant(L, quant_LW), _quant(W, quant_LW), _quant(T, quant_T)

    # Family key — prefer a real outer-outline hash; else fall back to L/W dims.
    if outline_hash:
        fam_key = _md5_of_json({"route":"plate","outer": outline_hash})
    else:
        fam_key = _md5_of_json({"route":"plate","L": Lq, "W": Wq})  # fallback

    # Content key — MUST include internals
    if outline_full_hash:
        content_key = _md5_of_json({"full": outline_full_hash, "T": Tq})
        hand = 0  # optional: add hand inference later if you want mirror-sensitivity
        return Fingerprint("plate", fam_key, content_key, hand, None,
                           meta={"dims": (Lq, Wq, Tq)})

    # Fallback path (older placeholder) — build loops in code (outer+holes)
    outline_canon, hand = _canonicalize_plate_loops_and_hand(plate_shape, ax3, L, W)
    fam_key_fallback = _md5_of_json({"route":"plate","outer": outline_canon["outer"]})
    content_key_fallback = _md5_of_json({"loops": outline_canon, "T": Tq})
    return Fingerprint("plate",
                       fam_key if outline_hash else fam_key_fallback,
                       content_key_fallback,
                       hand, None, meta={"dims": (Lq, Wq, Tq)})

def _canonicalize_plate_loops_and_hand(plate_shape, ax3, L, W):
    """
    Returns:
      - loops canonical form: {"outer":[[x,y],...], "holes":[[[x,y],...], ...]}
      - hand flag inferred by weighted Y-skew about midline (0 if symmetric)
    Assumes plate is already aligned by align_plate_to_xy_plane and ax3 is its local frame.
    """
    from OCC.Core.gp import gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepTools import breptools
    from OCC.Core.ShapeAnalysis import shapeanalysis
    from OCC.Core.BRepTools import breptools_UVBounds

    # world→local
    t = gp_Trsf(); t.SetTransformation(ax3); t.Invert()
    sh_loc = BRepBuilderAPI_Transform(plate_shape, t, True).Shape()

    # Extract 2D outer/inner loops from the biggest planar face (plate face)
    # For brevity we rely on triangulation UV bounds to get loops; you may already
    # have a robust 2D loop extractor—plug it here if so.
    outer = []
    holes = []
    features = []  # for hand inference (x,y,weight)

    f_it = TopExp_Explorer(sh_loc, TopAbs_FACE)
    biggest_area = -1.0
    biggest = None
    while f_it.More():
        f = topods.Face(f_it.Current())
        umin, umax, vmin, vmax = breptools.UVBounds(f)
        area_est = (umax-umin)*(vmax-vmin)
        if area_est > biggest_area:
            biggest_area = area_est; biggest = f
        f_it.Next()

    if biggest is not None:
        # This is a placeholder: replace with your actual 2D loop extraction (wires→polylines)
        # Here we just take the UV bounds rectangle as outer loop so the function compiles.
        outer = [[0.0, 0.0], [L, 0.0], [L, W], [0.0, W]]
        holes = []  # populate from inner wires if you have them

    # Hand inference from hole/cutout centroids (if you add them later)
    # For now, use simple symmetry test on outer only; returns 0 (=hand-neutral)
    hand = _infer_plate_hand_from_features(features, W)

    loops_canon = {"outer": _canon_poly(outer), "holes": [ _canon_poly(h) for h in holes ]}
    return loops_canon, hand

def _canon_poly(poly: Iterable[Iterable[float]], tol=1e-6) -> List[List[float]]:
    """Rotate-start-index to lexicographically minimal and return with closed=False."""
    pts = [[float(x), float(y)] for x,y in poly]
    if not pts: return pts
    # drop final duplicate if closed
    if len(pts) > 1 and abs(pts[0][0]-pts[-1][0])<tol and abs(pts[0][1]-pts[-1][1])<tol:
        pts = pts[:-1]
    # rotate to minimal start
    idx = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    pts = pts[idx:] + pts[:idx]
    return pts

def _infer_plate_hand_from_features(features: List[Tuple[float,float,float]], W: float, eps=1e-6) -> int:
    """features: list of (x,y,weight). Positive skew above midline = +1, below = -1."""
    s = sum(w * (y - W/2.0) for (_, y, w) in features)
    if abs(s) <= eps: return 0
    return 1 if s > 0 else -1

# at bottom of pipeline/fingerprint.py
def fp_to_kwargs(fp: Fingerprint | None) -> dict:
    if fp is None:
        return {
            "fp_route": "-",
            "fp_family_key": "-",
            "fp_content_key": "-",
            "fp_hand": 0,
            "fp_fallback_key": "",
            "fp_meta": "",
        }
    import json
    return {
        "fp_route": fp.route,
        "fp_family_key": fp.family_key,
        "fp_content_key": fp.content_key,
        "fp_hand": fp.hand,
        "fp_fallback_key": fp.fallback_key or "",
        "fp_meta": json.dumps(fp.meta or {}),  # optional, but useful
    }

