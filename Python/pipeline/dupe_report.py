# pipeline/dupe_report.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Default column names in your report_df
COL_ITEM_ID   = "Item ID"
COL_ITEM_TYPE = "Item Type"
COL_STEP      = "STEP File"
COL_NC1       = "NC1 File"
COL_IMAGE     = "Image"

def _linkify_file(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        p = Path(path_str)
        # if the report stored Path objects, coerce to string gracefully
        if isinstance(path_str, Path):
            p = path_str
        u = p.resolve().as_uri()
        return f'<a href="{u}" download target="_blank">{p.name}</a>'
    except Exception:
        # Fallback to plain text
        return str(path_str)

def build_dupe_buckets_df(
    report_df: pd.DataFrame,
    dupe_index: Dict[str, List[str]],
    *,
    col_item_id: str = COL_ITEM_ID,
    col_item_type: str = COL_ITEM_TYPE,
    col_step: str = COL_STEP,
    col_nc1: str = COL_NC1,
    col_image: str = COL_IMAGE,    # ← add this
    min_size: int = 2
) -> pd.DataFrame:
    """
    Build a one-row-per-duplicate-group DataFrame from the main report and a dupe_index.
    dupe_index: {sig_hash: [Item IDs ...]}
    """
    rows = []
    if not dupe_index:
        return pd.DataFrame(columns=[
            "Group", "Group Short", "Count", "Type", "Example ID", "Thumb", "STEP", "NC1", "Members"
        ])

    for h, ids in dupe_index.items():
        if len(ids) < min_size:
            continue
        subset = report_df[report_df[col_item_id].isin(ids)].copy()
        if subset.empty:
            continue
        sample = subset.iloc[0]
        rows.append({
            "Group": h,
            "Group Short": h[:8],
            "Count": len(ids),
            "Type": str(sample.get(col_item_type, "")),
            "Example ID": str(sample.get(col_item_id, "")),
            "Thumb": str(sample.get(col_image, "")),         # ← store path (string)
            "STEP": str(sample.get(col_step, "")),
            "NC1": str(sample.get(col_nc1, "")),
            "Members": ", ".join(subset[col_item_id].astype(str).tolist()),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Group", "Group Short", "Count", "Type", "Example ID", "Thumb", "STEP", "NC1", "Members"
        ])

    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)

def write_dupe_buckets_html(
    df_buckets: pd.DataFrame,
    output_dir: str | Path,
    project_number: str,
    *,
    title: str = "Duplicate Buckets"
) -> Path:
    """
    Write a compact, sortable HTML page summarizing duplicate groups.
    Returns the path to the HTML file.
    """
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_number}_duplicates.html"

    # Make a copy for safe mutation
    df_html = df_buckets.copy()

    # Linkify file columns
    for col in ("STEP", "NC1"):
        if col in df_html.columns:
            df_html[col] = df_html[col].apply(_linkify_file)

    # Convert thumbnail path into an <img> tag with a file:// URI
    if "Thumb" in df_html.columns:
        def _imgify(path_str: str) -> str:
            if not path_str:
                return ""
            try:
                uri = Path(path_str).resolve().as_uri()
                return f'<img src="{uri}" style="max-width:120px; height:auto; border-radius:4px; display:block; margin:0 auto;"/>'
            except Exception:
                return ""
        df_html["Thumb"] = df_html["Thumb"].apply(_imgify)

    # Choose display columns (add Thumb between Example ID and STEP)
    display_cols = [c for c in [
        "Group Short", "Count", "Type", "Example ID", "Thumb", "STEP", "NC1", "Members"
    ] if c in df_html.columns]

    table_html = df_html[display_cols].to_html(escape=False, index=False, classes="sortable")

    css = """
    <style>
      body { font-family: Arial, sans-serif; padding: 1rem; }
      h1 { font-size: 1.3rem; margin-bottom: .5rem; }
      p.meta { color: #555; margin-top: 0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: .5rem; vertical-align: middle; }
      th { background: #f4f4f4; }
      tr:nth-child(even) { background: #fafafa; }
      a { color: #0066cc; text-decoration: none; }
      a:hover { text-decoration: underline; }
      small.badge { background:#eef; border:1px solid #ccd; border-radius:4px; padding:2px 6px; }
      td img { box-shadow: 0 1px 3px rgba(0,0,0,.1); }
    </style>
    """

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{title} – {project_number}</title>
    {css}
    <script src="https://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
  </head>
  <body>
    <h1>{title} <small class="badge">{project_number}</small></h1>
    <p class="meta">Groups are formed by the tolerant signature (transform-invariant). Click headers to sort.</p>
    {table_html}
  </body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Wrote duplicate bucket report: {out_path}")
    return out_path


def write_dupe_buckets_csv(
    df_buckets: pd.DataFrame,
    output_dir: str | Path,
    project_number: str
) -> Path:
    """
    Write the bucket summary as CSV. Returns the path to the CSV file.
    """
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project_number}_duplicates.csv"
    df_buckets.to_csv(out_path, index=False)
    print(f"✅ Wrote duplicate bucket CSV: {out_path}")
    return out_path

def generate_duplicate_reports(
    report_df: pd.DataFrame,
    dupe_index: Dict[str, List[str]],
    output_dir: str | Path,
    project_number: str,
    *,
    min_size: int = 2
) -> Tuple[Path, Path]:
    """
    Convenience wrapper: build bucket dataframe and write both HTML and CSV.
    Returns (html_path, csv_path).
    """
    df_buckets = build_dupe_buckets_df(report_df, dupe_index, min_size=min_size)
    html_path = write_dupe_buckets_html(df_buckets, output_dir, project_number)
    csv_path  = write_dupe_buckets_csv(df_buckets, output_dir, project_number)
    return html_path, csv_path
