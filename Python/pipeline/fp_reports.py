# pipeline/fp_reports.py
from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
from pipeline.dupe_report import generate_consolidated_report
from pathlib import Path


def _valid_series(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str) != "-")

def _group_as_dict(df: pd.DataFrame, by: List[str]) -> Dict[str, List[str]]:
    """
    Group rows by `by` columns and return a dict: group_key -> [member names].
    Keys are stringified so they work with existing consolidated report.
    """
    # Build a string key even if `by` has multiple columns
    if len(by) == 1:
        keyser = df[by[0]].astype(str)
    else:
        keyser = df[by].astype(str).agg("|".join, axis=1)

    grouped = (
        pd.DataFrame({"__key": keyser, "name": df["name"]})
        .groupby("__key")["name"]
        .apply(list)
    )
    return grouped.to_dict()

def build_indexes(report_df: pd.DataFrame, *, split_hand_in_family: bool = False
                  ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns (dupe_index_strict, dupe_index_coarse)
    - strict: exact manufactured content (fp_content_key per route)
    - coarse: family buckets (fp_family_key), optionally split by hand
    """
    # Strict: (route, fp_content_key)
    mask_strict = _valid_series(report_df["fp_content_key"])
    strict_df = report_df.loc[mask_strict, ["fp_route", "fp_content_key", "name"]].copy()
    strict_df["__key"] = strict_df["fp_route"].astype(str) + "|" + strict_df["fp_content_key"].astype(str)
    dupe_index_strict = (
        strict_df.groupby("__key")["name"].apply(list).to_dict()
    )

    # Coarse: fp_family_key (optionally plus hand)
    mask_coarse = _valid_series(report_df["fp_family_key"])
    cols = ["fp_family_key", "name"] if not split_hand_in_family else ["fp_family_key", "fp_hand", "name"]
    coarse_df = report_df.loc[mask_coarse, cols].copy()

    if split_hand_in_family:
        # Key looks like: "<family_key>|hand=<-1/0/+1>"
        coarse_df["__key"] = coarse_df["fp_family_key"].astype(str) + "|hand=" + coarse_df["fp_hand"].astype(str)
    else:
        coarse_df["__key"] = coarse_df["fp_family_key"].astype(str)

    dupe_index_coarse = (
        coarse_df.groupby("__key")["name"].apply(list).to_dict()
    )

    return dupe_index_strict, dupe_index_coarse

def generate_fp_reports(report_df: pd.DataFrame, output_dir, project_number: str, *,
                        split_hand_in_family: bool = False) -> Dict[str, Tuple[str, str]]:
    """
    Writes two consolidated reports:
      - <project>_coarse_consolidated.(html/csv)
      - <project>_strict_consolidated.(html/csv)

    Returns a dict:
      {"coarse": (html_path, csv_path), "strict": (html_path, csv_path)}
    """
    dupe_index_strict, dupe_index_coarse = build_indexes(
        report_df, split_hand_in_family=split_hand_in_family
    )

    # Coarse families (route-agnostic; mirrors grouped unless split_hand_in_family=True)
    coarse_html, coarse_csv = generate_consolidated_report(
        report_df=report_df,
        dupe_index=dupe_index_coarse,
        output_dir=output_dir,
        project_number=f"{project_number}_coarse",
    )

    # Strict (manufacturing-identical content; splits L/R and hole-set diffs)
    strict_html, strict_csv = generate_consolidated_report(
        report_df=report_df,
        dupe_index=dupe_index_strict,
        output_dir=output_dir,
        project_number=f"{project_number}_strict",
    )

    return {"coarse": (coarse_html, coarse_csv), "strict": (strict_html, strict_csv)}


def _valid(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str) != "-")

def _build_indexes(df: pd.DataFrame, *, split_hand_in_family: bool = False
                   ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    # STRICT: exact manufactured content → (route | content_key)
    mask_strict = _valid(df["fp_content_key"])
    strict_df = df.loc[mask_strict, ["fp_route", "fp_content_key", "name"]].copy()
    strict_df["__key"] = strict_df["fp_route"].astype(str) + "|" + strict_df["fp_content_key"].astype(str)
    dupe_index_strict = strict_df.groupby("__key")["name"].apply(list).to_dict()

    # COARSE: family buckets → fp_family_key (optionally + hand)
    mask_coarse = _valid(df["fp_family_key"])
    cols = ["fp_family_key", "name"] if not split_hand_in_family else ["fp_family_key", "fp_hand", "name"]
    coarse_df = df.loc[mask_coarse, cols].copy()
    if split_hand_in_family:
        coarse_df["__key"] = coarse_df["fp_family_key"].astype(str) + "|hand=" + coarse_df["fp_hand"].astype(str)
    else:
        coarse_df["__key"] = coarse_df["fp_family_key"].astype(str)
    dupe_index_coarse = coarse_df.groupby("__key")["name"].apply(list).to_dict()

    return dupe_index_strict, dupe_index_coarse

def generate_fp_reports(report_df: pd.DataFrame, output_dir, project_number: str, *,
                        split_hand_in_family: bool = False) -> dict:
    """
    Writes two HTML reports using the SAME renderer you already use:
      - <project>_coarse_consolidated.html   (family view)
      - <project>_strict_consolidated.html   (content view, splits L/R)
    Also writes a tiny index HTML that links to both.
    Returns dict with file paths.
    """
    dupe_index_strict, dupe_index_coarse = _build_indexes(
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

    # simple index page
    outdir = Path(output_dir)
    index_path = outdir / f"{project_number}_fingerprints.html"
    index_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{project_number} – Fingerprint Reports</title>
<style>body{{font-family:system-ui,Segoe UI,Roboto,Arial; margin:20px}}</style></head>
<body>
  <h1>Fingerprint Reports – {project_number}</h1>
  <ul>
    <li><a href="{Path(coarse_html).name}">Coarse consolidated (family view)</a></li>
    <li><a href="{Path(strict_html).name}">Strict consolidated (content view)</a></li>
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
