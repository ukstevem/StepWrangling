# pipeline/fp_reports.py
from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
from html import escape

import pandas as pd

from pipeline.dupe_report import generate_consolidated_report


# ---------- Utilities for building group indexes ----------

def _valid_series(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str) != "-")

def build_indexes(
    report_df: pd.DataFrame,
    *,
    split_hand_in_family: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns (dupe_index_strict, dupe_index_coarse)

    - strict: exact manufactured content → (fp_route | fp_content_key)
    - coarse: family buckets → fp_family_key (optionally split by hand)
    """
    # STRICT
    mask_strict = _valid_series(report_df.get("fp_content_key", pd.Series([], dtype=object)))
    strict_df = report_df.loc[mask_strict, ["fp_route", "fp_content_key", "name"]].copy()
    strict_df["__key"] = strict_df["fp_route"].astype(str) + "|" + strict_df["fp_content_key"].astype(str)
    dupe_index_strict = strict_df.groupby("__key")["name"].apply(list).to_dict()

    # COARSE
    mask_coarse = _valid_series(report_df.get("fp_family_key", pd.Series([], dtype=object)))
    cols = ["fp_family_key", "name"] if not split_hand_in_family else ["fp_family_key", "fp_hand", "name"]
    coarse_df = report_df.loc[mask_coarse, cols].copy()

    if split_hand_in_family:
        coarse_df["__key"] = coarse_df["fp_family_key"].astype(str) + "|hand=" + coarse_df["fp_hand"].astype(str)
    else:
        coarse_df["__key"] = coarse_df["fp_family_key"].astype(str)

    dupe_index_coarse = coarse_df.groupby("__key")["name"].apply(list).to_dict()

    return dupe_index_strict, dupe_index_coarse


# ---------- Main report writer (reuses your consolidated report renderer) ----------

def generate_fp_reports(
    report_df: pd.DataFrame,
    output_dir,
    project_number: str,
    *,
    split_hand_in_family: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """
    Writes two consolidated reports:
      - <project>_coarse_consolidated.(html/csv)
      - <project>_strict_consolidated.(html/csv)
    Also writes a tiny index HTML that links to both.

    Returns: {"coarse": (html, csv), "strict": (html, csv), "index": index_html}
    """
    dupe_index_strict, dupe_index_coarse = build_indexes(
        report_df, split_hand_in_family=split_hand_in_family
    )

    coarse_html, coarse_csv = generate_consolidated_report(
        report_df=report_df,
        dupe_index=dupe_index_coarse,
        output_dir=output_dir,
        project_number=f"{project_number}_coarse",
    )

    strict_html, strict_csv = generate_consolidated_report(
        report_df=report_df,
        dupe_index=dupe_index_strict,
        output_dir=output_dir,
        project_number=f"{project_number}_strict",
    )

    # Index page linking both
    outdir = Path(output_dir)
    index_path = outdir / f"{project_number}_fingerprints.html"
    index_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(project_number)} – Fingerprint Reports</title>
<style>body{{font-family:system-ui,Segoe UI,Roboto,Arial; margin:20px}}</style></head>
<body>
  <h1>Fingerprint Reports – {escape(project_number)}</h1>
  <ul>
    <li><a href="{escape(Path(coarse_html).name)}">Coarse consolidated (family view)</a></li>
    <li><a href="{escape(Path(strict_html).name)}">Strict consolidated (content view)</a></li>
  </ul>
  <p>Tip: the strict view separates left/right and different hole sets.</p>
</body></html>""",
        encoding="utf-8"
    )

    return {
        "coarse": (coarse_html, coarse_csv),
        "strict": (strict_html, strict_csv),
        "index": str(index_path),
    }


# ---------- Extra: flat fingerprint table (sortable by browser) ----------

_FINGERPRINT_COLS = [
    "name", "obj_type", "section_shape",
    "obb_x", "obb_y", "obb_z",
    "fp_route", "fp_hand",
    "fp_family_key", "fp_content_key", "fp_fallback_key",
    "hash",  # your existing NC/DXF hash column
    "thumb_path", "drilling_path", "nc1_path", "dxf_path", "step_path",
]

def _short(h: str | None, n: int = 12) -> str:
    if not h or str(h).strip() in ("", "-"):
        return "-"
    s = str(h).strip()
    return s[:n]

def write_fingerprint_table_html(
    report_df: pd.DataFrame,
    output_dir,
    project_number: str,
) -> Tuple[str, str]:
    """
    Writes:
      - <project>_fingerprints_table.html (with thumbs + links)
      - <project>_fingerprints_table.csv  (flat data for Excel)
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = report_df.copy()

    # Ensure columns exist
    for c in _FINGERPRINT_COLS:
        if c not in df.columns:
            df[c] = "-"

    # Short display forms + keep full value in title tooltip
    df["fp_family_key_short"]  = df["fp_family_key"].apply(lambda x: _short(x, 12))
    df["fp_content_key_short"] = df["fp_content_key"].apply(lambda x: _short(x, 12))
    df["fp_fallback_short"]    = df["fp_fallback_key"].apply(lambda x: _short(x, 12))
    df["hash_short"]           = df["hash"].apply(lambda x: _short(x, 12))

    # Lightweight CSV
    csv_cols = [
        "name","obj_type","section_shape","obb_x","obb_y","obb_z",
        "fp_route","fp_hand","fp_family_key","fp_content_key","fp_fallback_key","hash",
        "thumb_path","drilling_path","nc1_path","dxf_path","step_path",
    ]
    csv_path = outdir / f"{project_number}_fingerprints_table.csv"
    df[csv_cols].to_csv(csv_path, index=False)

    # HTML (thumbs + file links)
    def _alink(label: str, path) -> str:
        if not path or str(path).strip() in ("", "-"):
            return "-"
        p = escape(str(path))
        return f'<a href="{p}">{escape(label)}</a>'

    rows_html: List[str] = []
    for r in df.to_dict(orient="records"):
        thumb = r.get("thumb_path") or ""
        thumb_cell = (
            f'<a href="{escape(str(r.get("drilling_path") or r.get("thumb_path") or ""))}">'
            f'<img src="{escape(str(thumb))}" alt="" '
            f'style="height:64px;max-width:160px;object-fit:contain;'
            f'border:1px solid #eee;border-radius:6px"/></a>'
            if thumb else "-"
        )
        links = " | ".join([
            _alink("Drill", r.get("drilling_path")),
            _alink("NC1",   r.get("nc1_path")),
            _alink("DXF",   r.get("dxf_path")),
            _alink("STEP",  r.get("step_path")),
        ])
        rows_html.append(f"""
<tr>
  <td>{escape(str(r["name"]))}</td>
  <td>{escape(str(r.get("obj_type","-")))}</td>
  <td>{escape(str(r.get("section_shape","-")))}</td>
  <td style="text-align:right">{escape(str(r.get("obb_x","-")))}</td>
  <td style="text-align:right">{escape(str(r.get("obb_y","-")))}</td>
  <td style="text-align:right">{escape(str(r.get("obb_z","-")))}</td>

  <td>{escape(str(r.get("fp_route","-")))}</td>
  <td style="text-align:center">{escape(str(r.get("fp_hand","-")))}</td>
  <td title="{escape(str(r.get('fp_family_key','-')))}">{escape(str(r.get('fp_family_key_short','-')))}</td>
  <td title="{escape(str(r.get('fp_content_key','-')))}">{escape(str(r.get('fp_content_key_short','-')))}</td>
  <td title="{escape(str(r.get('fp_fallback_key','-')))}">{escape(str(r.get('fp_fallback_short','-')))}</td>
  <td title="{escape(str(r.get('hash','-')))}">{escape(str(r.get('hash_short','-')))}</td>

  <td>{thumb_cell}</td>
  <td>{links}</td>
</tr>""")

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{escape(project_number)} – Fingerprints Table</title>
<style>
  body {{ font-family: system-ui, Segoe UI, Roboto, Arial; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e5e5e5; padding: 6px 8px; font-size: 13px; }}
  th {{ background: #fafafa; position: sticky; top: 0; z-index: 1; }}
  tbody tr:hover {{ background: #fffdf2; }}
  .hint {{ color:#666; font-size:12px; margin: 6px 0 16px; }}
</style>
</head>
<body>
  <h1>Fingerprint Table – {escape(project_number)}</h1>
  <p class="hint">Tip: open the CSV (<code>{escape(csv_path.name)}</code>) in Excel to sort by
  <code>fp_content_key</code> (strict) or <code>fp_family_key</code> (coarse). Hover hash cells to see full values.</p>

  <table>
   <thead>
    <tr>
      <th>Name</th>
      <th>Type</th>
      <th>Shape</th>
      <th>OBB&nbsp;X</th>
      <th>OBB&nbsp;Y</th>
      <th>OBB&nbsp;Z</th>

      <th>FP&nbsp;Route</th>
      <th>Hand</th>
      <th>Family&nbsp;Key</th>
      <th>Content&nbsp;Key</th>
      <th>Fallback</th>
      <th>Hash</th>

      <th>Thumb</th>
      <th>Links</th>
    </tr>
   </thead>
   <tbody>
     {''.join(rows_html)}
   </tbody>
  </table>
</body>
</html>"""

    html_path = outdir / f"{project_number}_fingerprints_table.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ Fingerprint table written: {html_path}")
    print(f"✅ CSV written:               {csv_path}")
    return str(html_path), str(csv_path)
