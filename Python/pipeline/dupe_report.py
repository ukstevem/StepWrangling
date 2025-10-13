# dupe_report.py
# Drop-in replacement: builds duplicate-buckets and consolidated (unique+Quantity) reports.
# Outputs CSV + HTML for both.

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List
import pandas as pd
from html import escape
from urllib.parse import quote

__all__ = [
    # Duplicate buckets (existing surface kept)
    "build_dupe_buckets_df",
    "write_dupe_buckets_html",
    "write_dupe_buckets_csv",
    "generate_duplicate_reports",
    # New consolidated outputs
    "build_consolidated_items_df",
    "write_consolidated_html",
    "write_consolidated_csv",
    "generate_consolidated_report",
]

# -----------------------------------------------
# Utilities
# -----------------------------------------------

_PATH_COLS = [
    "step_path", "stl_path", "brep_path", "nc1_path", "drilling_path",
    "dxf_path", "dxf_thumb_path", "thumb_path"
]

def _exists(p) -> bool:
    try:
        return p is not None and str(p).strip() != "" and Path(p).is_file()
    except Exception:
        return False

def _as_uri_safe(p: str | Path) -> str | None:
    try:
        if not _exists(p):
            return None
        return Path(p).resolve().as_uri()
    except Exception:
        return None

def _first_non_null(series: pd.Series):
    for v in series:
        if pd.notna(v) and v != "":
            return v
    return pd.NA

def _pick_id_col(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Identifier column '{explicit}' not found in DataFrame.")
        return explicit
    for c in ("Item ID", "name", "Name", "part_number"):
        if c in df.columns:
            return c
    # Last resort: if there is an obvious single key-like column
    for c in df.columns:
        if c.lower() in {"id", "guid", "globalid"}:
            return c
    raise KeyError("No identifier column found. Tried: Item ID, name, Name, part_number, id/guid/globalid")

def _fmt_bbox_from_obb_row(row: pd.Series) -> str:
    try:
        x = float(row.get("obb_x", float("nan")))
        y = float(row.get("obb_y", float("nan")))
        z = float(row.get("obb_z", float("nan")))
        if pd.notna(x) and pd.notna(y) and pd.notna(z):
            return f"{x:.1f} × {y:.1f} × {z:.1f}"
    except Exception:
        pass
    return ""

def _fmt_bbox_from_xyz_row(row: pd.Series) -> str:
    candidates = [("X (mm)", "Y (mm)", "Z (mm)"), ("X","Y","Z")]
    for xs, ys, zs in candidates:
        try:
            if xs in row and ys in row and zs in row:
                x = float(row.get(xs, float("nan")))
                y = float(row.get(ys, float("nan")))
                z = float(row.get(zs, float("nan")))
                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    return f"{x:.1f} × {y:.1f} × {z:.1f}"
        except Exception:
            continue
    return ""

def _linkify(p: str | Path, label: str | None = None) -> str:
    """Simple file:// link (works locally in browsers)."""
    if not _exists(p):
        return "—"
    uri = _as_uri_safe(p)
    if not uri:
        return "—"
    lab = escape(label if label else Path(str(p)).name)
    return f'<a href="{uri}" target="_blank" rel="noopener">{lab}</a>'

def _imgify(p: str | Path, w: int = 120) -> str:
    uri = _as_uri_safe(p)
    if not uri:
        return ""
    return f'<img src="{uri}" style="max-width:{int(w)}px;height:auto;border-radius:4px;display:block;margin:0 auto;" />'

def _sortable_html(table_html: str, title: str) -> str:
    css = """
    <style>
      body { font-family: Arial, sans-serif; padding: 1rem; }
      h1 { font-size: 1.3rem; margin-bottom: .25rem; }
      p.meta { color:#555; margin-top:0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: .5rem; vertical-align: middle; }
      th { background: #f4f4f4; cursor: pointer; }
      tr:nth-child(even) { background: #fafafa; }
      a { color: #0066cc; text-decoration: none; }
      a:hover { text-decoration: underline; }
      td img { box-shadow: 0 1px 3px rgba(0,0,0,.1); }
    </style>
    """
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{escape(title)}</title>
    {css}
    <script src="https://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
  </head>
  <body>
    <h1>{escape(title)}</h1>
    <p class="meta">Click headers to sort.</p>
    {table_html}
  </body>
</html>
"""

# -----------------------------------------------
# Duplicate buckets (existing concept)
# -----------------------------------------------

def build_dupe_buckets_df(
    report_df: pd.DataFrame,
    dupe_index: Dict[Any, Iterable[Any]],
    *,
    col_item_id: str | None = None,
    mass_col: str = "Mass (kg)",
    include_members: bool = True,
    min_size: int = 2,  # <--- NEW
) -> pd.DataFrame:
    """
    Turn a duplicate index {signature: [ids...]} + full report into a tidy table
    with one row per duplicate group.

    Only groups with size >= min_size are included.
    """
    if report_df is None or report_df.empty or not dupe_index:
        return pd.DataFrame(columns=["Signature", "Quantity"])

    df = report_df.copy()
    id_col = _pick_id_col(df, col_item_id)
    df["_id_str_"] = df[id_col].astype(str)

    # Map: id -> row idx (first occurrence preferred)
    id_to_idx: Dict[str, int] = {}
    for i, v in enumerate(df["_id_str_"]):
        id_to_idx.setdefault(v, i)

    rows: List[dict] = []
    for sig, ids in dupe_index.items():
        members = [str(x) for x in (list(ids) if ids is not None else [])]
        members = [m for m in members if m in id_to_idx]
        if not members:
            continue

        # respect min_size
        if len(members) < int(min_size):
            continue

        # representative = first member's row
        rep_idx = id_to_idx[members[0]]
        rep = df.iloc[rep_idx].to_dict()

        row = {
            "Signature": sig,
            "Quantity": len(members),
            "Example ID": rep.get(id_col),
        }

        # BBox helper (best effort)
        bbox = _fmt_bbox_from_obb_row(rep)
        if not bbox:
            bbox = _fmt_bbox_from_xyz_row(rep)
        if bbox:
            row["BBox (mm)"] = bbox

        # Common helpful fields if present
        for c in ("obj_type", "Type", "Issues", mass_col):
            if c in df.columns:
                row[c] = rep.get(c)

        # File paths – keep representative path
        for c in _PATH_COLS:
            if c in df.columns:
                row[c] = rep.get(c)

        if include_members:
            row["Members"] = ", ".join(members)

        # Total mass if we know per-part mass
        if mass_col in df.columns:
            try:
                m = float(rep.get(mass_col))
                row["Total Mass (kg)"] = m * len(members)
            except Exception:
                pass

        rows.append(row)

    out = pd.DataFrame(rows)

    # Nice order
    desired = ["Signature", "Quantity", "obj_type", "Type", "Example ID", "BBox (mm)", "Members",
               mass_col, "Total Mass (kg)"] + _PATH_COLS
    cols = [c for c in desired if c in out.columns] + [c for c in out.columns if c not in desired]
    out = out[cols]
    return out



def write_dupe_buckets_html(df_dupes: pd.DataFrame, output_dir: str | Path, project_number: str) -> Path:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_number}_duplicates.html"

    if df_dupes is None or df_dupes.empty:
        html = _sortable_html("<p><em>No duplicate groups found.</em></p>", f"Duplicate Groups – {project_number}")
        out_path.write_text(html, encoding="utf-8")
        return out_path

    dfh = df_dupes.copy()

    # Linkify common path columns + add thumb image col
    if "thumb_path" in dfh.columns:
        dfh["Thumb"] = dfh["thumb_path"].apply(_imgify)

    link_map = {
        "step_path": "STEP",
        "nc1_path": "NC1",
        "brep_path": "BREP",
        "stl_path": "STL",
        "drilling_path": "Drilling",
        "dxf_path": "Profile DXF",
        "dxf_thumb_path": "DXF Thumb",
    }
    for raw_col, nice in link_map.items():
        if raw_col in dfh.columns:
            dfh[nice] = dfh[raw_col].apply(_linkify)

    show_cols = [c for c in ["Signature","Quantity","obj_type","Type","Example ID","BBox (mm)","Thumb",
                             "STEP","NC1","BREP","STL","Drilling","Profile DXF","DXF Thumb","Members",
                             "Mass (kg)","Total Mass (kg)"] if c in dfh.columns]
    table_html = dfh[show_cols].to_html(escape=False, index=False, classes="sortable")
    html = _sortable_html(table_html, f"Duplicate Groups – {project_number}")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_dupe_buckets_csv(df_dupes: pd.DataFrame, output_dir: str | Path, project_number: str) -> Path:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_number}_duplicates.csv"
    (df_dupes if df_dupes is not None else pd.DataFrame()).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def generate_duplicate_reports(
    report_df: pd.DataFrame,
    dupe_index: Dict[Any, Iterable[Any]],
    output_dir: str | Path,
    project_number: str,
    *,
    col_item_id: str | None = None,
    min_size: int = 2,  # <--- NEW
) -> Tuple[Path, Path]:
    """
    Build duplicate-bucket table and write HTML + CSV.
    Returns: (html_path, csv_path)

    Only groups with size >= min_size are included.
    """
    df_dupes = build_dupe_buckets_df(
        report_df,
        dupe_index,
        col_item_id=col_item_id,
        min_size=min_size,            # <--- pass it through
    )
    html_path = write_dupe_buckets_html(df_dupes, output_dir, project_number)
    csv_path  = write_dupe_buckets_csv(df_dupes, output_dir, project_number)
    return html_path, csv_path


# -----------------------------------------------
# Consolidated items (unique rows + Quantity)
# -----------------------------------------------

def build_consolidated_items_df(
    report_df: pd.DataFrame,
    dupe_index: Dict[Any, Iterable[Any]] | None = None,
    *,
    col_item_id: str | None = None,
    include_members: bool = True,
    mass_col: str = "Mass (kg)",
) -> pd.DataFrame:
    """
    Consolidate the full report to unique items with a Quantity column.

    Logic:
      • If an item's ID is in dupe_index, its group key is the duplicate signature.
      • Otherwise the group key is the item ID itself (singleton).
      • Keep the 'first non-null' value per column for a stable representative row.
      • Quantity = count per group. Also computes Total Mass if mass column present.
    """
    if report_df is None or report_df.empty:
        return pd.DataFrame(columns=["Quantity"])

    df = report_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    id_col = _pick_id_col(df, col_item_id)
    df["_id_str_"] = df[id_col].astype(str)

    # Build id -> signature (if duplicate) and signature -> members
    id_to_sig: Dict[str, Any] = {}
    sig_to_members: Dict[Any, list[str]] = {}

    if dupe_index:
        for sig, ids in dupe_index.items():
            ids_list = [str(x) for x in (list(ids) if ids is not None else [])]
            if not ids_list:
                continue
            sig_to_members.setdefault(sig, [])
            for i in ids_list:
                id_to_sig[i] = sig
                sig_to_members[sig].append(i)

    # Determine group key
    def _group_key(s: str):
        return id_to_sig.get(s, s)  # duplicate => signature; else own ID

    df["_group_key_"] = df["_id_str_"].map(_group_key)

    # Aggregation (compute size separately; don't include non-existent 'Quantity' in agg dict)
    grp = df.groupby("_group_key_", dropna=False)

    # First-non-null for all real columns
    agg = {col: _first_non_null for col in df.columns if col not in {"_id_str_", "_group_key_"}}

    grouped = grp.agg(agg)

    # Add Quantity as the group size
    quantity = grp.size().rename("Quantity")
    grouped = grouped.join(quantity)

    # Back to a clean frame
    grouped = grouped.reset_index().rename(columns={"_group_key_": "Group Key"})


    # Representative ID and BBox display
    if id_col in grouped.columns:
        grouped["Example ID"] = grouped[id_col]
    bbox = []
    for _, row in grouped.iterrows():
        b = _fmt_bbox_from_obb_row(row)
        if not b:
            b = _fmt_bbox_from_xyz_row(row)
        bbox.append(b)
    grouped["BBox (mm)"] = bbox

    # Members (IDs) for traceability
    if include_members:
        def _members_for(key):
            if key in sig_to_members:
                return ", ".join(sorted(set(sig_to_members[key])))
            # singleton (key is its own ID)
            if isinstance(key, str):
                return key
            return ""
        grouped["Members"] = grouped["Group Key"].map(_members_for)

    # Total Mass
    if mass_col in grouped.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            grouped["Total Mass (kg)"] = pd.to_numeric(grouped[mass_col], errors="coerce") * grouped["Quantity"]

    # Column order
    preferred = [
        "Quantity","obj_type","Type","Example ID","BBox (mm)","Members",
        "step_path","nc1_path","brep_path","stl_path","drilling_path","dxf_path","thumb_path",
        "Mass (kg)","Total Mass (kg)"
    ]
    cols = [c for c in preferred if c in grouped.columns] + [c for c in grouped.columns if c not in preferred + ["Group Key"]]
    grouped = grouped[cols + ([ "Group Key"] if "Group Key" in grouped.columns else [])]
    return grouped


def write_consolidated_html(
    df_consol: pd.DataFrame,
    output_dir: str | Path,
    project_number: str,
    *,
    title: str = "Consolidated Items",
) -> Path:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_number}_consolidated.html"

    if df_consol is None or df_consol.empty:
        html = _sortable_html("<p><em>No items.</em></p>", f"{title} – {project_number}")
        out_path.write_text(html, encoding="utf-8")
        return out_path

    dfh = df_consol.copy()

    # Render thumbnails and file links
    if "thumb_path" in dfh.columns:
        dfh["Thumb"] = dfh["thumb_path"].apply(_imgify)

    link_map = {
        "step_path": "STEP",
        "nc1_path": "NC1",
        "brep_path": "BREP",
        "stl_path": "STL",
        "drilling_path": "Drilling",
        "dxf_path": "Profile DXF",
    }
    for raw_col, nice in link_map.items():
        if raw_col in dfh.columns:
            dfh[nice] = dfh[raw_col].apply(_linkify)

    show_cols = [c for c in ["Quantity","obj_type","Type","Example ID","BBox (mm)","Thumb",
                             "STEP","NC1","BREP","STL","Drilling","Profile DXF","Members",
                             "Mass (kg)","Total Mass (kg)"] if c in dfh.columns]
    table_html = dfh[show_cols].to_html(escape=False, index=False, classes="sortable")
    html = _sortable_html(table_html, f"{title} – {project_number}")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_consolidated_csv(
    df_consol: pd.DataFrame,
    output_dir: str | Path,
    project_number: str,
) -> Path:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_number}_consolidated.csv"
    (df_consol if df_consol is not None else pd.DataFrame()).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def generate_consolidated_report(
    report_df: pd.DataFrame,
    dupe_index: Dict[Any, Iterable[Any]] | None,
    output_dir: str | Path,
    project_number: str,
    *,
    col_item_id: str | None = None,
) -> Tuple[Path, Path]:
    """
    Build consolidated items (unique rows with Quantity) and write HTML + CSV.
    Returns: (html_path, csv_path)
    """
    df_consol = build_consolidated_items_df(
        report_df=report_df,
        dupe_index=dupe_index,
        col_item_id=col_item_id,
        include_members=True,
    )
    html_path = write_consolidated_html(df_consol, output_dir, project_number)
    csv_path  = write_consolidated_csv(df_consol, output_dir, project_number)
    return html_path, csv_path
