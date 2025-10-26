import pandas as pd
import numpy as np

# Columns we’ll try to coerce to float
_NUM_COLS = [
    "mass","obb_x","obb_y","obb_z","volume","surface_area",
    "bbox_x","bbox_y","bbox_z",
    "inertia_e1","inertia_e2","inertia_e3",
    "centroid_x","centroid_y","centroid_z","chirality"
]

# Path-ish columns -> strings
_PATH_COLS = [
    "step_path","native_step_path","stl_path","thumb_path","drilling_path",
    "dxf_path","nc1_path","brep_path","dxf_thumb_path"
]

# Everything you wanted in Pset_DSTV + identity
_KEEP_COLS = [
    "part_id","name",
    "section_shape","obj_type","issues","hash","assembly_hash","signature_hash",
    "obb_x","obb_y","obb_z","bbox_x","bbox_y","bbox_z",
    "mass","volume","surface_area",
    "centroid_x","centroid_y","centroid_z",
    "inertia_e1","inertia_e2","inertia_e3",
    "chirality",
] + _PATH_COLS

def build_parts_from_report(report_df: list[dict]) -> pd.DataFrame:
    """Collate report_df -> parts_df keyed by signature_hash, with numeric coercion and paths normalized."""
    df = pd.DataFrame(report_df).copy()

    # 1) Normalize paths to plain strings
    for c in _PATH_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"None": np.nan})

    # 2) Coerce numeric columns
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Choose key: use signature_hash as part_id (matches your instances.part_id)
    if "signature_hash" not in df.columns:
        raise ValueError("report_df is missing 'signature_hash' needed for part_id")
    df["part_id"] = df["signature_hash"]

    # 4) Robust section_shape:
    #    If empty, infer from obj_type when we can (e.g., 'Plate' -> 'PLATE')
    if "section_shape" in df.columns:
        df["section_shape"] = df["section_shape"].astype(str).str.strip()
        df.loc[df["section_shape"] == "", "section_shape"] = np.nan
    if "obj_type" in df.columns:
        mask = df["section_shape"].isna()
        df.loc[mask & (df["obj_type"].str.upper() == "PLATE"), "section_shape"] = "PLATE"
        # (extend here if you’d like: CHANNEL->"U", ANGLE->"L", etc.)

    # 5) Keep a compact set of useful columns (only those present)
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df_parts = df[keep].drop_duplicates(subset=["part_id"]).reset_index(drop=True)

    # Optional: sanity print
    cols_show = ["part_id","name","section_shape","mass","obb_x","obb_y","obb_z","bbox_x","bbox_y","bbox_z"]
    print("parts_df preview:\n", df_parts[[c for c in cols_show if c in df_parts.columns]].head())

    return df_parts
