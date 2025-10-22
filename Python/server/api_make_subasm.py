from fastapi import FastAPI, Request
from pathlib import Path
import json
from pipeline.rebuilder import rebuild_from_handoff
from server.main import make_subassembly  # your helper

app = FastAPI()

@app.post("/api/make_subassembly")
async def api_make_subassembly(req: Request):
    payload = await req.json()
    selection = payload.get("selection", [])
    subasm_name = payload.get("subasm_name", "NewSubassembly")
    parent_uid = payload.get("parent_uid", "__ROOT__")

    base = Path("/path/to/handoff")
    instances_path = base / "instances.jsonl"
    if not instances_path.exists():
        return {"ok": False, "message": "instances.jsonl not found"}

    # Load into memory as DataFrame or list of dicts
    records = [json.loads(line) for line in instances_path.read_text().splitlines() if line.strip()]

    # Apply your existing make_subassembly function
    df_instances = make_subassembly(records, selection, subasm_name, parent_uid)

    # Save updated instances
    with open(instances_path, "w", encoding="utf-8") as f:
        for rec in df_instances:
            f.write(json.dumps(rec) + "\n")

    # Rebuild the STEP + GLB + meta
    step_out = str(base / f"{subasm_name}_rebuilt.step")
    rebuild_from_handoff(str(base), step_out)

    return {"ok": True, "message": f"Subassembly {subasm_name} created", "step": step_out}
