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





