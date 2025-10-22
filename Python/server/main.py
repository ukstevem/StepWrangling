# server/main.py
import os, json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# numpy for 4x4 transforms
try:
    import numpy as np
except Exception as e:
    raise SystemExit("This service requires numpy. pip install numpy") from e

# Import your pipeline
from pipeline.rebuilder import rebuild_from_handoff  # <- you already have this

# -------------------------
# Config
# -------------------------
HANDOFF_DIR = Path(os.environ.get("HANDOFF_DIR", "./handoff")).resolve()
OUTPUT_BASENAME = os.environ.get("OUTPUT_BASENAME", "rebuilt")  # prefix for rebuilt outputs

app = FastAPI(title="Rebuild API")

# Allow your local viewer to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Small IO helpers
# -------------------------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs

def _write_jsonl(path: Path, recs: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

def _ensure_m44(m):
    """Accept list-of-lists or flat list of 16; return (4,4) float numpy array (row-major)."""
    arr = np.array(m, dtype=float)
    if arr.size == 16:
        arr = arr.reshape(4,4)
    if arr.shape != (4,4):
        raise ValueError(f"Expected 4x4 matrix, got shape {arr.shape}")
    return arr

# -------------------------
# Subassembly operation on instances.jsonl (list-of-dicts in place)
# -------------------------
def _make_subassembly_records(
    records: List[Dict[str, Any]],
    selection_iids: List[str],
    subasm_name: str,
    parent_uid: str,
) -> Dict[str, Any]:
    """
    Mutates records:
      - Add a new assembly record (with is_assembly=True) named subasm_uid.
      - Reparent selected instance rows under that assembly.
      - Convert their transforms from world to local relative to the new assembly.
    Returns dict with subasm_uid and counts.
    """
    # 1) Find selected rows by instance_id or fallback to 'name'
    by_iid = {}
    for r in records:
        iid = r.get("instance_id") or r.get("name") or r.get("part_id")
        if iid:
            by_iid[iid] = r

    missing = [iid for iid in selection_iids if iid not in by_iid]
    if missing:
        raise HTTPException(status_code=400, detail=f"Instance IDs not found: {missing}")

    selected = [by_iid[iid] for iid in selection_iids]

    # 2) Choose subassembly pivot (first selected's current world transform)
    T_sub = _ensure_m44(selected[0]["T"])
    T_sub_inv = np.linalg.inv(T_sub)

    # 3) Create a deterministic-ish subassembly UID
    #    (you can switch to a hash if you prefer stable IDs across runs)
    subasm_uid = f"SUBASM_{subasm_name}"

    # 4) Add assembly record with its own transform (world) relative to parent
    asm_rec = {
        "is_assembly": True,
        "assembly_id": subasm_uid,      # new field used by rebuilder patch
        "name": subasm_name,
        "parent_asm": parent_uid,
        "T": T_sub.tolist(),            # assembly's world transform
    }
    records.append(asm_rec)

    # 5) Reparent each selected child under subassembly and convert T -> local
    for r in selected:
        T_world = _ensure_m44(r["T"])
        T_local = (T_sub_inv @ T_world)
        r["parent_asm"] = subasm_uid
        r["T"] = T_local.tolist()

    return {"subasm_uid": subasm_uid, "selected": len(selected)}

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "handoff": str(HANDOFF_DIR)}

@app.post("/api/make_subassembly")
def make_subassembly(payload: Dict[str, Any]):
    """
    Expected payload:
    {
      "selection": ["occ_001", "occ_002"],  # your instance_id strings (viewer sends node.name)
      "subasm_name": "Frame-01",
      "parent_uid": "__ROOT__"
    }
    """
    selection = payload.get("selection") or []
    subasm_name = (payload.get("subasm_name") or "Subassembly").strip()
    parent_uid = payload.get("parent_uid") or "__ROOT__"

    if len(selection) < 2:
        raise HTTPException(status_code=400, detail="Select at least two instances.")

    inst_path = HANDOFF_DIR / "instances.jsonl"
    if not inst_path.exists():
        raise HTTPException(status_code=404, detail=f"{inst_path} not found")

    records = _read_jsonl(inst_path)

    # Make subassembly in-place
    res = _make_subassembly_records(records, selection, subasm_name, parent_uid)

    # Persist
    _write_jsonl(inst_path, records)

    # Rebuild STEP+GLB+viewer_meta.json
    out_step = str((HANDOFF_DIR / f"{OUTPUT_BASENAME}.step").resolve())
    step_path = rebuild_from_handoff(str(HANDOFF_DIR), out_step)

    out = {
        "ok": True,
        "message": f"Subassembly '{subasm_name}' created",
        "subasm_uid": res["subasm_uid"],
        "step": step_path,
        "glb": str(Path(step_path).with_suffix(".glb")),
        "meta": str(Path(step_path).with_name("viewer_meta.json")),
    }
    return out
