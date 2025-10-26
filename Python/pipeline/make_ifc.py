"""
make_ifc.py — single script to run after DSTV conversion
Usage:
    python scripts/make_ifc.py --handoff "path/to/Handoff" --out "path/to/IFC/model.ifc"
If the handoff manifest is missing, it falls back to using the report CSV.
"""

import argparse
from pathlib import Path
import pandas as pd
from pipeline.ifc_exporter import write_ifc_from_handoff, write_ifc_from_report

def run_ifc_export(handoff_dir=None, report_csv=None, out_ifc=None, units: str ="MM"):
    """Run IFC export from either handoff manifest or consolidated report."""
    if handoff_dir:
        handoff_dir = Path(handoff_dir)
    if report_csv:
        report_csv = Path(report_csv)

    # Derive default IFC path
    if not out_ifc:
        if handoff_dir:
            out_ifc = handoff_dir.parent / "IFC" / "model.ifc"
        elif report_csv:
            out_ifc = Path(report_csv).parent.parent / "IFC" / "model.ifc"
        else:
            out_ifc = Path.cwd() / "model.ifc"
    out_ifc = Path(out_ifc)
    out_ifc.parent.mkdir(parents=True, exist_ok=True)

    # Preferred: use handoff manifest
    if handoff_dir and (handoff_dir / "manifest.json").exists():
        print(f"🔹 Exporting IFC from handoff: {handoff_dir}")
        path = write_ifc_from_handoff(handoff_dir, out_ifc, units=units,
                              lin_defl=None, ang_deg=20.0, verbose=True)
        print(f"✅ IFC created: {path}")
        return Path(path)

    # Fallback: report CSV
    if report_csv and Path(report_csv).exists():
        print(f"🔹 Exporting IFC from report CSV: {report_csv}")
        df = pd.read_csv(report_csv)
        path = write_ifc_from_report(df, out_ifc, geom_col='step_path', name_col='name', units=units)
        print(f"✅ IFC created: {path}")
        return Path(path)

    raise FileNotFoundError("Neither handoff manifest nor report CSV found; please provide --handoff or --report.")

def main():
    ap = argparse.ArgumentParser(description="Generate IFC after DSTV conversion.")
    ap.add_argument("--handoff", type=str, help="Path to Handoff folder (contains manifest.json)")
    ap.add_argument("--report", type=str, help="Path to consolidated report CSV (fallback)")
    ap.add_argument("--out", type=str, help="Output IFC path (default: IFC/model.ifc beside input)")
    ap.add_argument("--units", type=str, default="MILLIMETRE", choices=["MILLIMETRE","METRE"], help="IFC length unit")
    args = ap.parse_args()

    run_ifc_export(
        handoff_dir=args.handoff,
        report_csv=args.report,
        out_ifc=args.out,
        units=args.units
    )

if __name__ == "__main__":
    main()
