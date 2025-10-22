# rebuild_step.py
import argparse
from pipeline.rebuilder import rebuild_from_handoff

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True, help="Path to handoff_pack")
    ap.add_argument("--out", dest="out", required=True, help="Output STEP path")
    args = ap.parse_args()
    print(rebuild_from_handoff(args.inp, args.out))


# python rebuild_step.py --in path/to/handoff_pack --out rebuilt_assembly.step

# python rebuild_cli.py --in "C:\Dev\step-gemini\Extraction\Projects\AP6467A-0-709 - Batch 5 steelwork.step\cad\handoff_pack" --out rebuilt_assembly.step