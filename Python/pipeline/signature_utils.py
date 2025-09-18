## pipeline/signature_utils.py

import hashlib
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Vec


def compute_invariants(shape) -> dict[str, float]:
    """
    Compute basic geometric invariants for a solid shape.
    Returns a dict with volume, surface_area, bbox dims, principal inertia eigenvalues, centroid coords.
    """
    props_vol = GProp_GProps()
    brepgprop.VolumeProperties(shape, props_vol)
    volume = props_vol.Mass()

    props_area = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props_area)
    surface_area = props_area.Mass()

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    bbox_x, bbox_y, bbox_z = xmax - xmin, ymax - ymin, zmax - zmin

    centroid = props_vol.CentreOfMass()
    principal_props = props_vol.PrincipalProperties()
    try:
        inertia_ix, inertia_iy, inertia_iz = principal_props.Moments()
    except Exception:
        inertia_ix, inertia_iy, inertia_iz = props_vol.StaticMoments()

    return {
        "volume": volume,
        "surface_area": surface_area,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_z": bbox_z,
        "inertia_ix": inertia_ix,
        "inertia_iy": inertia_iy,
        "inertia_iz": inertia_iz,
        "centroid_x": centroid.X(),
        "centroid_y": centroid.Y(),
        "centroid_z": centroid.Z(),
    }


def make_signature(inv: dict[str, float], precision: int = 3, hash_type: str = "md5") -> str:
    """
    Round each invariant to `precision` decimals, concatenate with pipes, and hash.
    Returns hex digest string.
    """
    parts = [f"{round(val, precision):.{precision}f}" for val in inv.values()]
    raw = "|".join(parts).encode("utf-8")

    if hash_type.lower() == "md5":
        return hashlib.md5(raw).hexdigest()
    elif hash_type.lower() == "sha1":
        return hashlib.sha1(raw).hexdigest()
    return hashlib.md5(raw).hexdigest()


def compute_signature_info(
    aligned_shape,
    dir_x, dir_y, dir_z,
    precision: int = 3,
    hash_type: str = "md5"
) -> tuple[dict[str, float], str]:
    """
    High-level helper: takes an aligned shape and its original axes,
    computes invariants, chirality flag, and signature hash.
    Returns (invariants_dict_with_chirality, signature_hash).
    """
    # Capture original orientation vectors
    vec_x = gp_Vec(dir_x.XYZ())
    vec_y = gp_Vec(dir_y.XYZ())
    vec_z = gp_Vec(dir_z.XYZ())

    # Ensure right-handed for any geometry ops
    # (caller can enforce via ensure_right_handed)

    # Compute base invariants
    inv = compute_invariants(aligned_shape)

    # Detect chirality: +1 if right-handed, -1 if mirrored
    chirality_flag = 1 if vec_x.Crossed(vec_y).Dot(vec_z) > 0 else -1
    inv["chirality"] = chirality_flag

    # Generate a combined signature including chirality
    sig = make_signature(inv, precision, hash_type)
    return inv, sig


# === BEGIN tolerant, transform-invariant fingerprint additions ===
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict, Any, List, Optional

from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus


# ---- helpers: deterministic rounding and key building ----

def _q_mm(v, step):  # quantize to a step in mm
    return round(v / step) * step

def coarse_key_from_half_sorted(half_sorted, step=1.0):
    hx, hy, hz = half_sorted
    return tuple(_q_mm(x, step) for x in (hx, hy, hz))  # still sorted

def _roundf(v: float, nd: int) -> float:
    return float(round(v, nd))

def _nd_from_bin(binval: float) -> int:
    # number of decimals implied by a bin size (e.g., 0.1 -> 1 dp)
    # clamp at 0 to avoid negatives if someone passes bin >= 1
    import math
    if binval <= 0:
        return 3  # sensible fallback
    return max(0, int(abs(math.log10(binval))))

def _sorted_half_extents_from_aabb(shape) -> Tuple[float, float, float]:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    hx, hy, hz = (abs(xmax - xmin) / 2.0, abs(ymax - ymin) / 2.0, abs(zmax - zmin) / 2.0)
    # sort desc so orientation changes don’t change the tuple
    return tuple(sorted((hx, hy, hz), reverse=True))

def _principal_moments_sorted_safe(shape) -> Tuple[float, float, float]:
    # Your current code sometimes falls back to StaticMoments(), which are *first* moments (not inertia).
    # Here we try PrincipalProperties().Moments(); if it fails, return zeros (rare degenerate case).
    props_vol = GProp_GProps()
    brepgprop.VolumeProperties(shape, props_vol)
    try:
        I1, I2, I3 = props_vol.PrincipalProperties().Moments()
    except Exception:
        I1 = I2 = I3 = 0.0
    # guard tiny negative due to numeric noise
    I = [0.0 if (v < 0 and abs(v) < 1e-9) else float(v) for v in (I1, I2, I3)]
    return tuple(sorted(I))

def _face_type_counts(shape) -> Tuple[int, int, int, int, int]:
    n_plane = n_cyl = n_cone = n_torus = n_other = 0
    it = TopExp_Explorer(shape, TopAbs_FACE)
    while it.More():
        face = it.Current()
        stype = GeomAdaptor_Surface(BRep_Tool.Surface(face)).GetType()
        if   stype == GeomAbs_Plane:    n_plane += 1
        elif stype == GeomAbs_Cylinder: n_cyl   += 1
        elif stype == GeomAbs_Cone:     n_cone  += 1
        elif stype == GeomAbs_Torus:    n_torus += 1
        else:                           n_other += 1
        it.Next()
    return (n_plane, n_cyl, n_cone, n_torus, n_other)

@dataclass
class TolerantSignature:
    """Structured return for grouping and diagnostics."""
    sig_key: tuple
    sig_hash: str
    # diagnostics / for display:
    volume: float
    surface_area: float
    half_extents_sorted: Tuple[float, float, float]
    face_counts: Tuple[int, int, int, int, int]
    moments_sorted: Tuple[float, float, float]

def build_tolerant_signature(
    shape,
    *,
    half_extents_sorted=None,
    vol_bin: float | None = 0.1,    # None ⇒ exclude volume from key
    area_bin: float | None = 0.1,   # None ⇒ exclude area from key
    ext_bin: float = 0.1,
    mom_bin: float | None = 0.01,   # None ⇒ exclude principal moments
    include_chirality: bool = False,
    chirality_flag: int | None = None,
    hash_type: str = "sha256",
    use_faces: bool = True          # NEW: allow disabling face counts
) -> TolerantSignature:
    inv = compute_invariants(shape)
    vol  = float(inv["volume"])
    area = float(inv["surface_area"])

    # extents (prefer caller-supplied OBB half-extents; else AABB fallback)
    hx, hy, hz = half_extents_sorted or _sorted_half_extents_from_aabb(shape)

    # principal moments (sorted)
    moments = _principal_moments_sorted_safe(shape)

    # face-type counts or zeros if disabled
    faces = _face_type_counts(shape) if use_faces else (0, 0, 0, 0, 0)

    # tolerant rounding using your existing "bin⇒decimals" semantics
    vr  = None if vol_bin  is None else _roundf(vol,  _nd_from_bin(vol_bin))
    ar  = None if area_bin is None else _roundf(area, _nd_from_bin(area_bin))
    exr = tuple(_roundf(x, _nd_from_bin(ext_bin)) for x in (hx, hy, hz))
    mrr = None if mom_bin  is None else tuple(_roundf(x, _nd_from_bin(mom_bin)) for x in moments)

    # build key (deterministic, tolerant)
    key_parts = [exr]
    if vr  is not None: key_parts.append(vr)
    if ar  is not None: key_parts.append(ar)
    if use_faces:       key_parts.append(faces[:4])   # ignore "other"
    if mrr is not None: key_parts.append(mrr)
    if include_chirality:
        key_parts.append(int(chirality_flag if chirality_flag is not None else inv.get("chirality", 1)))

    sig_key = tuple(key_parts)

    # hash
    if hash_type.lower() in ("sha1", "sha-1"):
        sig_hash = hashlib.sha1(repr(sig_key).encode("utf-8")).hexdigest()
    elif hash_type.lower() == "md5":
        sig_hash = hashlib.md5(repr(sig_key).encode("utf-8")).hexdigest()
    else:
        sig_hash = hashlib.sha256(repr(sig_key).encode("utf-8")).hexdigest()

    return TolerantSignature(
        sig_key=sig_key,
        sig_hash=sig_hash,
        volume=vol,
        surface_area=area,
        half_extents_sorted=(hx, hy, hz),
        face_counts=faces,
        moments_sorted=moments,
    )


def bucket_by_signature(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    rows: iterable of {"id": str, "sig_hash": str}
    returns: {sig_hash: [ids...]} only for buckets with more than one member.
    """
    buckets: Dict[str, List[str]] = {}
    for r in rows:
        h = r["sig_hash"]
        buckets.setdefault(h, []).append(r["id"])
    return {h: ids for h, ids in buckets.items() if len(ids) > 1}

# === END tolerant, transform-invariant fingerprint additions ===



# --- Coarse, dimensions-only duplicate grouping ------------------------------

from typing import Iterable, Dict, List, Tuple

def _q_mm(v: float, step: float) -> float:
    """Quantize value v to a mm step."""
    return round(float(v) / float(step)) * float(step)

def coarse_extents_key(half_extents_sorted: Tuple[float, float, float], step: float = 1.0) -> Tuple[float, float, float]:
    """
    Build a coarse key from sorted half-extents (hx>=hy>=hz), quantized to 'step' mm.
    """
    hx, hy, hz = half_extents_sorted
    return (_q_mm(hx, step), _q_mm(hy, step), _q_mm(hz, step))

def bucket_by_coarse_extents(
    rows: Iterable[Dict[str, object]],
    *,
    step: float = 1.0,
    key_field: str = "half_extents_sorted",
    id_field: str = "id"
) -> Dict[Tuple[float, float, float], List[str]]:
    """
    Group items whose sorted half-extents are the same up to 'step' mm.
    Expects each row to have:
      - id_field (e.g. 'id')
      - key_field (tuple: (hx, hy, hz) sorted desc, in mm)
    Returns only buckets with more than one item.
    """
    buckets: Dict[Tuple[float, float, float], List[str]] = {}
    for r in rows:
        exr = r.get(key_field)
        if exr is None:
            continue
        k = coarse_extents_key(exr, step=step)
        buckets.setdefault(k, []).append(str(r[id_field]))
    return {k: ids for k, ids in buckets.items() if len(ids) > 1}
