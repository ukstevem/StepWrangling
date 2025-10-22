# -------------------------
# Manifest writer (viewer-friendly) with GLB/STEP switch
# -------------------------
from pathlib import Path
import json
import shutil

def _ensure_mat4(T):
    # Accept list of 16 or nested 4x4; return nested 4x4
    if isinstance(T, (list, tuple)) and len(T) == 16:
        return [list(T[0:4]), list(T[4:8]), list(T[8:12]), list(T[12:16])]
    if isinstance(T, (list, tuple)) and len(T) == 4 and all(len(r) == 4 for r in T):
        return [list(r) for r in T]
    raise ValueError(f"Bad transform, expected 16 elems or 4x4: {T}")

def _copy_if(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        if str(src.resolve()) != str(dst.resolve()):
            shutil.copyfile(str(src), str(dst))
        return True
    return False

def write_viewer_manifest(
    pack_root: str | Path,
    df_unique,
    df_instances,
    *,
    project_number: str | None = None,
    units: str = "MM",
    # how to find per-part GLBs or STEP files:
    unique_part_id_col="part_id",
    unique_name_col="name",
    unique_glb_col="glb_path",          # if not given, we’ll derive from step and rename
    unique_step_col="step_path",         # used as a fallback to deduce where the STEP should be
    parts_dir_name="parts",              # folder under pack_root for STEP files
    use_glb: bool = False,                # new switch; if False, write step_url instead of glb_url
    # instance mapping:
    inst_part_id_col="part_id",
    inst_name_col="name",
    inst_parent_col="parent_asm",        # optional
    inst_T_col="T",
    inst_id_col="instance_id",           # optional
    # assemblies (optional): pass a DataFrame or leave None
    df_assemblies=None,                   # expects columns: asm_id, name, parent_id, matrix
    # mesh inlining (optional):
    inline_mesh=False,                   # if True, embed vertices/triangles (and metadata) for each part
    inline_vertices_col="vertices",      # Nx3 floats
    inline_triangles_col="triangles",    # Mx3 ints
    inline_props_cols=None,              # list of property column names to carry into part dict
):
    """
    Writes: <pack_root>/manifest.json and copies per-part files into <pack_root>/defs/<part_id>.<ext>
    Controlled by use_glb flag:
        True  → store 'glb_url'
        False → store 'step_url' (using /parts/MEM-###.step)
    """

    print("Writing Viewer Manifest")

    pack_root = Path(pack_root)
    defs_dir = pack_root / "defs"
    parts_dir = pack_root / parts_dir_name
    defs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build parts entries
    parts_out = []
    partid_to_rel = {}

    for _, row in df_unique.iterrows():
        pid = str(row[unique_part_id_col])

        # Source file decision
        src_path = None
        rel_path = None

        if use_glb:
            # Use GLB logic (old default)
            if unique_glb_col in row and row[unique_glb_col]:
                src_path = Path(str(row[unique_glb_col]))
            elif unique_step_col in row and row[unique_step_col]:
                step_src = Path(str(row[unique_step_col]))
                guess = step_src.with_suffix(".glb")
                if guess.exists():
                    src_path = guess
            dst = defs_dir / f"{pid}.glb"
            if src_path and _copy_if(src_path, dst):
                rel_path = f"defs/{pid}.glb"
        else:
            # Use STEP logic
            step_name = f"{pid}.step"
            src_path = parts_dir / step_name
            dst = defs_dir / step_name
            if src_path.exists() and _copy_if(src_path, dst):
                rel_path = f"defs/{step_name}"

        part_rec = {"def_id": f"hash:{pid}"}

        # Inline or reference file
        if inline_mesh:
            verts = row.get(inline_vertices_col)
            tris  = row.get(inline_triangles_col)
            if verts is not None and tris is not None:
                V = [[float(x), float(y), float(z)] for (x, y, z) in verts]
                I = [[int(a), int(b), int(c)] for (a, b, c) in tris]
                part_rec["vertices"]  = V
                part_rec["triangles"] = I
            elif rel_path:
                key = "glb_url" if use_glb else "step_url"
                part_rec[key] = rel_path
        else:
            if rel_path:
                key = "glb_url" if use_glb else "step_url"
                part_rec[key] = rel_path

        # Extra properties
        if inline_props_cols:
            for c in inline_props_cols:
                if c in row and row[c] is not None:
                    part_rec[c] = row[c]

        parts_out.append(part_rec)
        partid_to_rel[pid] = rel_path

    # ---- Build instances
    instances_out = []
    for _, row in df_instances.iterrows():
        pid = str(row[inst_part_id_col])
        name = str(row[inst_name_col]) if inst_name_col and row.get(inst_name_col) is not None else pid
        occ_id = str(row[inst_id_col]) if inst_id_col and row.get(inst_id_col) is not None else f"occ:{name}"
        T = _ensure_mat4(row[inst_T_col])

        rec = {
            "occ_id": occ_id,
            "def_id": f"hash:{pid}",
            "name": name,
            "matrix": T,
        }
        if inst_parent_col and row.get(inst_parent_col) is not None:
            rec["parent_id"] = str(row[inst_parent_col])
        instances_out.append(rec)

    # ---- Assemblies (optional)
    assemblies_out = None
    if df_assemblies is not None:
        assemblies_out = []
        for _, row in df_assemblies.iterrows():
            a = {
                "asm_id":   str(row.get("asm_id") or row.get("id") or row.get("name")),
                "name":     str(row.get("name") or row.get("asm_id")),
                "parent_id": row.get("parent_id"),
                "matrix": _ensure_mat4(row.get("matrix") or [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
            }
            assemblies_out.append(a)

    # ---- Manifest root
    mani = {
        "version": "1.0",
        "units": units,
        "parts": parts_out,
        "instances": instances_out,
    }
    if project_number:
        mani["project_number"] = project_number
    if assemblies_out:
        mani["assemblies"] = assemblies_out

    (pack_root / "manifest.json").write_text(json.dumps(mani, indent=2), encoding="utf-8")
    return pack_root / "manifest.json"
