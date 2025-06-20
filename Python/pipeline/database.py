import os
from datetime import datetime
import pandas as pd
import numpy as np
from supabase import create_client
# from config import SUPABASE_URL, SUPABASE_KEY
from pathlib import Path
from typing import List


SUPABASE_URL = "https://hguvsjpmiyeypfcuvvko.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhndXZzanBtaXlleXBmY3V2dmtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk2MjYyNzIsImV4cCI6MjA2NTIwMjI3Mn0.b-NxykPbsX9bwMZV_41AAMKi_dbAbA3D8DDnc0UolC4"

# supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def dump_df_to_supabase(df: pd.DataFrame, table_name="parts", projectnumber: str = None, chunk_size=500):
    # --- assume df has already been renamed and sanitized ---
    # 0) Inject projectnumber if you’re using that FK
    if projectnumber:
        df['projectnumber'] = projectnumber

    # 1) Drop imported_at so Postgres will fill it
    df = df.drop(columns=[c for c in ['imported_at'] if c in df.columns])

    # 2) Convert Paths, Timestamps, numpy types
    def sanitize(v):
        if isinstance(v, Path):        return str(v)
        if isinstance(v, pd.Timestamp): return v.isoformat()
        if isinstance(v, (np.generic,)): return v.item()
        return v
    df = df.applymap(sanitize)

    # 3) Insert in batches, relying on APIError on failure
    records = df.to_dict(orient="records")
    total = len(records)
    for start in range(0, total, chunk_size):
        batch = records[start : start + chunk_size]
        try:
            resp = supabase.table(table_name).insert(batch).execute()
        except APIError as e:
            # e.response is the HTTPX Response object
            raise RuntimeError(f"Insert failed on rows {start+1}-{start+len(batch)}: {e}") from e
        else:
            # Successful insert: resp.data contains the inserted rows
            print(f"Inserted rows {start+1}-{min(start+chunk_size, total)} (returned {len(resp.data)} rows)")

    print(f"✅ Successfully inserted {total} rows into `{table_name}`")


def normalize_report_df(df: pd.DataFrame, project_id,
                        keep_only: List[str] = None) -> pd.DataFrame:
    """
    Rename and clean up the report DataFrame so it matches the `parts` table schema.
    
    - Renames columns from:
        'Item ID', 'Item Type', 'STEP File', 'Image', 'Drilling Drawing',
        'DXF Thumb', 'Profile DXF', 'NC1 File', 'BREP', 'Mass (kg)',
        'X (mm)', 'Y (mm)', 'Z (mm)', 'Issues', 'Hash'
      to:
        item_id, item_type, step_file, image, drilling_drawing,
        dxf_thumb, profile_dxf, nc1_file, brep, mass_kg,
        x_mm, y_mm, z_mm, issues, hash
    
    - If `keep_only` is provided, drops all columns *not* in that list after renaming.
    
    Returns the cleaned DataFrame.
    """
    mapping = {
        "Item ID":          "item_id",
        "Item Type":        "item_type",
        "STEP File":        "step_file",
        "Image":            "image",
        "Drilling Drawing": "drilling_drawing",
        "DXF Thumb":        "dxf_thumb",
        "Profile DXF":      "profile_dxf",
        "NC1 File":         "nc1_file",
        "BREP":             "brep",
        "Mass (kg)":        "mass_kg",
        "X (mm)":           "x_mm",
        "Y (mm)":           "y_mm",
        "Z (mm)":           "z_mm",
        "Issues":           "issues",
        "Hash":             "hash",
    }
    
    # 1) Rename
    df = df.rename(columns=mapping)

    print(df.columns)
    
    # 2) Optionally drop any extra columns
    if keep_only is not None:
        # Always keep project_id (if present), plus imported_at if you had it
        protected = {"projectnumber", "imported_at"}
        allowed = set(keep_only) | protected
        df = df.loc[:, df.columns.intersection(allowed)]
    
    df['projectnumber'] = project_id

    return df