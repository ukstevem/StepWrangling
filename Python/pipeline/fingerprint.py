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
import pandas as pd

def _hand_from_holes(df: pd.DataFrame | None) -> int:
    """
    Infer 'hand' from hole face codes.
    +1  → only 'O' face present
    -1  → only 'U' face present
     0  → both/neither/unknown
    """
    if df is None or len(df) == 0:
        return 0

    # find a face/side column case-insensitively
    cols = {c.lower(): c for c in df.columns}
    face_col = None
    for name in ("code", "face", "side"):
        if name in cols:
            face_col = cols[name]
            break
    if not face_col:
        return 0

    ser = df[face_col].astype(str).str.upper().str.strip()
    has_o = ser.eq("O").any()
    has_u = ser.eq("U").any()

    if has_o and not has_u:
        return +1
    if has_u and not has_o:
        return -1
    return 0


# --- helpers used by normalization ---
def _swap_face_for_y_mirror(face: str) -> str:
    f = (face or "").upper()
    if f == "O": return "U"
    if f == "U": return "O"
    return f

def _sorted_features(features: list[dict]) -> list[dict]:
    return sorted(features, key=lambda h: (h.get("f",""), h.get("x",0.0), h.get("y",0.0), h.get("d",0.0), h.get("t","")))

def _endcuts_to_stable_list(endcuts: dict | None) -> list[dict]:
    """
    End-flip invariant representation:
    Ignore side labels; keep only normalized numeric content and sort.
    """
    out = []
    if not isinstance(endcuts, dict) or not endcuts:
        return out
    def _round_ec(ec):
        if not isinstance(ec, dict):
            return {}
        return {
            "type": str(ec.get("type","")),
            "n": [ _quant(float(v), 1e-3) for v in (ec.get("n") or []) ],
            "off": _quant(float(ec.get("off", 0.0)), 0.1),
        }
    for k in endcuts.keys():
        out.append(_round_ec(endcuts[k]))
    out.sort(key=lambda e: (e.get("type",""), tuple(e.get("n",[])), e.get("off",0.0)))
    return out


# --- SECTION ROUTE (drop-in replacement) -------------------------------------
def fingerprint_section(*,
                        refined_shape,
                        extents_xyz: Tuple[float, float, float],  # (L, H, W) in DSTV pose
                        section_area: float,
                        profile_type: str | None,
                        raw_df_holes,                     # pandas DF or None
                        endcuts: Dict[str, Any] | None = None,
                        web_thickness: float | None = None,
                        flange_thickness: float | None = None,
                        root_radius: float | None = None,
                        toe_radius: float | None = None,
                        dim_step: float = 1.0,
                        phi_step: float = 0.005,
                        xy_step: float = 0.1,             # coord quantization (mm)
                        d_step: float = 0.1,              # diameter quantization (mm)
                        ) -> Fingerprint:
    """
    Coarse (family): route='section', tag, H/W, tw/tf (if available), compactness φ.  (NO L)
    Content (strict): normalized NC (holes+endcuts) + dims (L,H,W) + 'hand'.
    Meta: section_leninv_key (holes normalized to nearest end → length-invariant).
    """
    L, H, W = extents_xyz
    Lq, Hq, Wq = _quant(L, dim_step), _quant(H, dim_step), _quant(W, dim_step)
    phi = (section_area / (H * W)) if (H > 0 and W > 0) else 0.0
    phiq = _quant(phi, phi_step)
    tag = (profile_type or "OPEN").upper()

    # Quantize thickness details if present (family only)
    twq = _quant(web_thickness, 0.1) if web_thickness is not None else None
    tfq = _quant(flange_thickness, 0.1) if flange_thickness is not None else None
    rrq = _quant(root_radius, 0.1) if root_radius is not None else None
    trq = _quant(toe_radius, 0.1) if toe_radius is not None else None

    # ---------- Family key (NO LENGTH) ----------
    fam_obj = {
        "route": "section", "tag": tag,
        "H": Hq, "W": Wq, "phi": phiq,
        "tw": twq, "tf": tfq, "rr": rrq, "tr": trq,
    }
    family_key = hashlib.md5(json.dumps(fam_obj, sort_keys=True).encode()).hexdigest()

    # ---------- Normalize holes / endcuts ----------
    df = raw_df_holes if isinstance(raw_df_holes, pd.DataFrame) and len(raw_df_holes) else None
    hand = _hand_from_holes(df)

    # strict (length-aware)
    nc_strict = _normalize_nc_content_section(
        df=df, endcuts=endcuts, tag=tag,
        dims=(Lq, Hq, Wq), hand=hand,
        xy_step=xy_step, d_step=d_step, length_invariant=False,
        mirror_x_origin_canonical=True,
    )
    content_key = _md5_of_json(nc_strict)

    # length-invariant (optional alt key for reports)
    nc_leninv = _normalize_nc_content_section(
        df=df, endcuts=endcuts, tag=tag,
        dims=(Lq, Hq, Wq), hand=hand,
        xy_step=xy_step, d_step=d_step, length_invariant=True,
        mirror_x_origin_canonical=True,
    )
    leninv_key = _md5_of_json(nc_leninv)

    return Fingerprint(route="section",
                       family_key=family_key,
                       content_key=content_key,
                       hand=hand,
                       fallback_key=None,
                       meta={
                           "profile_type": tag,
                           "dims": (Lq, Hq, Wq),
                           "section_leninv_key": leninv_key
                       })


def _normalize_nc_content_section(*,
                                  df: pd.DataFrame | None,
                                  endcuts: Dict[str, Any] | None,
                                  tag: str,
                                  dims: Tuple[float, float, float],   # (Lq,Hq,Wq)
                                  hand: int,
                                  xy_step: float,
                                  d_step: float,
                                  length_invariant: bool,
                                  mirror_x_origin_canonical: bool = True) -> Dict[str, Any]:
    """
    Build deterministic NC JSON and canonicalize over symmetries.
    - Quantize coords (xy_step) and diameters (d_step)
    - length_invariant: x_norm = min(X, L - X) (nearest end)
      else:             x_norm = X (but compare mirrored variants)
    - Canonicalization set:
        * X-flip (origin at other end) if mirror_x_origin_canonical
        * Y-mirror (swap O<->U) – helps when section is mirrored about web
      Pick lexicographically minimal serialized JSON.
    - Endcuts: collapse to a stable, side-agnostic sorted list so X-flip doesn’t change the key.
    - Always include dims, tag, hand.
    """
    Lq, Hq, Wq = dims

    # ----- pull hole rows
    features_raw: list[dict] = []
    if df is not None and len(df):
        cols = {c.lower(): c for c in df.columns}
        c_code = cols.get("code") or cols.get("face") or cols.get("side")
        c_x    = cols.get("x (mm)") or cols.get("x")
        c_y    = cols.get("y (mm)") or cols.get("y")
        c_d    = cols.get("diameter (mm)") or cols.get("diameter")

        for row in df.itertuples(index=False):
            features_raw.append({
                "t": "hole",
                "f": (getattr(row, c_code) if c_code else ""),
                "x": _quant(float(getattr(row, c_x)) if c_x else 0.0, xy_step),
                "y": _quant(float(getattr(row, c_y)) if c_y else 0.0, xy_step),
                "d": _quant(float(getattr(row, c_d)) if c_d else 0.0, d_step),
            })

    # Endcuts as flip-invariant stable list
    endcuts_norm = _endcuts_to_stable_list(endcuts)

    # ----- produce symmetry variants
    def _apply_ops(mirror_y: bool, flip_x: bool, nearest_end: bool) -> list[dict]:
        out = []
        for h in features_raw:
            x = float(h["x"]); y = float(h["y"]); d = float(h["d"]); f = str(h["f"])
            if flip_x and Lq > 0:
                x = max(Lq - x, 0.0)
            if mirror_y and Hq > 0:
                y = max(Hq - y, 0.0)
                # swap O<->U for open sections
                f = _swap_face_for_y_mirror(f)
            if nearest_end and Lq > 0:
                x = min(x, max(Lq - x, 0.0))
            out.append({"t":"hole","f":f,"x":_quant(x, xy_step),"y":_quant(y, xy_step),"d":_quant(d, d_step)})
        return _sorted_features(out)

    # Variant set:
    #  - length_invariant=True → only Y mirror vs not (nearest-end already handles X)
    #  - length_invariant=False → Cartesian of {no/yes Y} × {no/yes X flip}
    variants = []
    if length_invariant:
        for my in (False, True):
            feats = _apply_ops(mirror_y=my, flip_x=False, nearest_end=True)
            payload = {
                "v": 2, "route": "section", "profile": tag,
                "dims": {"L": Lq, "H": Hq, "W": Wq},
                "hand": hand, "features": feats, "endcuts": endcuts_norm
            }
            variants.append((json.dumps(payload, sort_keys=True, separators=(",",":")), payload))
    else:
        xs = (False, True) if mirror_x_origin_canonical else (False,)
        for my in (False, True):
            for fx in xs:
                feats = _apply_ops(mirror_y=my, flip_x=fx, nearest_end=False)
                payload = {
                    "v": 2, "route": "section", "profile": tag,
                    "dims": {"L": Lq, "H": Hq, "W": Wq},
                    "hand": hand, "features": feats, "endcuts": endcuts_norm
                }
                variants.append((json.dumps(payload, sort_keys=True, separators=(",",":")), payload))

    variants.sort(key=lambda t: t[0])
    return variants[0][1]


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

    fam_key = _md5_of_json({"route":"plate", "outer": outline_hash or "", "L": Lq, "W": Wq})

    # STRICT content: full geometry + size + thickness
    if outline_full_hash:
        print(f"[fingerprint_plate] FULL used  full={outline_full_hash[:12]}  LWT={Lq}×{Wq}×{Tq}")
        content_key = _md5_of_json({"full": outline_full_hash, "L": Lq, "W": Wq, "T": Tq})
        return Fingerprint("plate", fam_key, content_key, 0, None,
                           meta={"dims": (Lq, Wq, Tq), "fp_src": "full"})

    # Fallbacks (outer-only or computed loops)
    if outline_hash:
        print(f"[fingerprint_plate] OUTER used outer={outline_hash[:12]}  LWT={Lq}×{Wq}×{Tq}")
        content_key = _md5_of_json({"outer": outline_hash, "L": Lq, "W": Wq, "T": Tq})
        return Fingerprint("plate", fam_key, content_key, 0, None,
                           meta={"dims": (Lq, Wq, Tq), "fp_src": "outer"})

    print(f"[fingerprint_plate] FALLBACK used (no full/outer hash)  LWT={Lq}×{Wq}×{Tq}")
    outline_canon, hand = _canonicalize_plate_loops_and_hand(plate_shape, ax3, L, W)
    content_key = _md5_of_json({"loops": outline_canon, "L": Lq, "W": Wq, "T": Tq})
    return Fingerprint("plate", fam_key, content_key, hand, None,
                       meta={"dims": (Lq, Wq, Tq), "fp_src": "fallback"})

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

    return {
        "fp_route": str(getattr(fp, "route", "") or "").lower(),
        "fp_family_key": getattr(fp, "family_key", "") or "",
        "fp_content_key": getattr(fp, "content_key", "") or "",
        "fp_hand": int(getattr(fp, "hand", 0) or 0),
        "fp_fallback_key": getattr(fp, "fallback_key", "") or "",   # <- was missing
        "fp_meta": json.dumps(getattr(fp, "meta", {}) or {}, separators=(",", ":"), ensure_ascii=False),
    }

