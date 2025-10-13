import os
from datetime import datetime
import pandas as pd
import numpy as np
from supabase import create_client
# from config import SUPABASE_URL, SUPABASE_KEY
from pathlib import Path
from typing import List
from postgrest.exceptions import APIError
from dotenv import load_dotenv

# load .env variables into process environment
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials in environment")


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
            pass
            # Successful insert: resp.data contains the inserted rows
            # print(f"Inserted rows {start+1}-{min(start+chunk_size, total)} (returned {len(resp.data)} rows)")

    print(f"✅ Successfully inserted {total} rows into `{table_name}`")


def normalize_report_df(df: pd.DataFrame, project_id, keep_only: list[str] | None = None) -> pd.DataFrame:
    """
    Map pipeline columns to the `parts` table columns.
    Uses the *current* report_df headers you showed.
    """
    mapping = {
        "name":             "item_id",
        "obj_type":         "item_type",
        "step_path":        "step_file",
        "stl_path":         "stl_file",
        "thumb_path":       "image",
        "drilling_path":    "drilling_drawing",
        "dxf_path":         "profile_dxf",
        "nc1_path":         "nc1_file",
        "brep_path":        "brep",
        "mass":             "mass_kg",
        "obb_x":            "x_mm",
        "obb_y":            "y_mm",
        "obb_z":            "z_mm",
        "issues":           "issues",
        "hash":             "hash",
        "dxf_thumb_path":   "dxf_thumb",
        "section_shape":    "section_shape",
        "assembly_hash":    "source_model_hash",
        "signature_hash":   "signature_hash",
        "volume":           "volume",
        "surface_area":     "surface_area",
        "bbox_x":           "bbox_x",
        "bbox_y":           "bbox_y",
        "bbox_z":           "bbox_z",
        "inertia_e1":       "inertia_e1",
        "inertia_e2":       "inertia_e2",
        "inertia_e3":       "inertia_e3",
        "centroid_x":       "centroid_x",
        "centroid_y":       "centroid_y",
        "centroid_z":       "centroid_z",
        "chirality":        "chirality",
    }

    out = df.rename(columns=mapping).copy()

    # Ensure projectnumber exists (your table uses this)
    out["projectnumber"] = project_id

    # Types/coercions (optional, but tends to avoid DB type errors)
    numeric_cols = [
        "mass_kg","x_mm","y_mm","z_mm","volume","surface_area",
        "bbox_x","bbox_y","bbox_z","inertia_e1","inertia_e2","inertia_e3",
        "centroid_x","centroid_y","centroid_z","chirality",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # File path columns -> strings (or empty)
    file_cols = ["step_file","stl_file","image","drilling_drawing","profile_dxf","nc1_file","brep","dxf_thumb"]
    for c in file_cols:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)

    # Optionally keep only a subset
    if keep_only is not None:
        protected = {"projectnumber"}  # always keep
        allowed = set(keep_only) | protected
        out = out.loc[:, [c for c in out.columns if c in allowed]]

    return out


import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from supabase import create_client
from postgrest.exceptions import APIError


def get_supabase_client(url: Optional[str] = None, key: Optional[str] = None):
    """
    Create and return a Supabase client using environment variables or provided credentials.
    """
    SUPABASE_URL = url or os.getenv("SUPABASE_URL")
    SUPABASE_KEY = key or os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing Supabase credentials in environment")

    return create_client(SUPABASE_URL, SUPABASE_KEY)


def sanitize_value(v: Any) -> Any:
    """
    Convert unsupported types (Path, Timestamp, numpy types) to native Python types for JSON.
    """
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def add_to_database(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    table_name: str,
    projectnumber: Optional[str] = None,
    allowed_columns: Optional[List[str]] = None,
    supabase_client: Optional[Any] = None,
    chunk_size: int = 500
) -> Dict[str, Any]:
    """
    Safely insert payload into the specified Supabase table in batches, with basic SQL-injection protection.

    Args:
        payload: A dict or list of dicts representing records to insert.
        table_name: Name of the target table in Supabase.
        projectnumber: Optional foreign key to inject into each record.
        allowed_columns: Optional whitelist of column names to include. Any keys not in this list will be dropped.
        supabase_client: Optional Supabase client instance. If not provided, a new one is created.
        chunk_size: Number of records per batch insert.

    Returns:
        A dict containing 'success': bool, and either 'data' with inserted rows or 'error' with exception details.
    """
    # Normalize payload to list
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Payload must be a dict or list of dicts")

    # Inject projectnumber if provided
    if projectnumber:
        for rec in records:
            rec['projectnumber'] = projectnumber

    sanitized_records: List[Dict[str, Any]] = []
    for rec in records:
        # Whitelist columns if provided
        rec_filtered = {}  # type: Dict[str, Any]
        for k, v in rec.items():
            if allowed_columns is not None and k not in allowed_columns and k != 'projectnumber':
                # drop unexpected column
                continue
            # basic key validation: no SQL meta-characters
            if any(c in k for c in [';', '--', '/*', '*/']):
                raise ValueError(f"Invalid column name detected: {k}")
            rec_filtered[k] = sanitize_value(v)
        sanitized_records.append(rec_filtered)

    # Get supabase client
    client = supabase_client or get_supabase_client()

    # Batch insert
    total = len(sanitized_records)
    inserted_data = []
    try:
        for start in range(0, total, chunk_size):
            batch = sanitized_records[start: start + chunk_size]
            # Using supabase REST client ensures parameterized inserts
            resp = client.table(table_name).insert(batch).execute()
            inserted_data.extend(resp.data or [])
    except APIError as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "data": inserted_data}
