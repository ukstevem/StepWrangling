import os, json, hashlib, time
import pandas as pd
from pathlib import Path
from pipeline.manifest_writing import write_viewer_manifest


## DF columns to pass to Manifest

inline_props_cols = [
    # identity / traceability
    "name", "hash", "assembly_hash", "signature_hash", "obj_type", "issues",

    # file artefacts (handy to keep in IFC custom pset for provenance)
    "step_path", "native_step_path", "stl_path", "thumb_path", "drilling_path",
    "dxf_path", "nc1_path", "brep_path", "dxf_thumb_path",

    # physics / size
    "mass", "volume", "surface_area",
    "obb_x", "obb_y", "obb_z",
    "bbox_x", "bbox_y", "bbox_z",

    # section classification
    "section_shape",   # e.g. I / U / L / PLATE / ...
                       # (if you add designation/material later, include them too)

    # inertias & centroid (nice to have; goes in custom pset)
    "inertia_e1", "inertia_e2", "inertia_e3",
    "centroid_x", "centroid_y", "centroid_z",

    "chirality",
]

# OCC imports (OCP first, fallback to pythonocc)
try:
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    _OCC_FLAVOR = "OCP"
except Exception:
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    _OCC_FLAVOR = "pythonocc"

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_part_step(shape, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    status = w.Write(str(out_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP: {out_path}")

def _is_rigid_4x4(T, tol=1e-6):
    # T: iterable of 16 row-major numbers
    if len(T) != 16:
        return False
    # last row ~ [0,0,0,1]
    if abs(T[12])>tol or abs(T[13])>tol or abs(T[14])>tol or abs(T[15]-1)>tol:
        return False
    # optional: determinant ~ 1 (skip to keep it lightweight)
    return True

def _identity_T():
    # 4x4 row-major identity
    return [1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1]

def build_rebuilder_tables_from_report(
    report_df: pd.DataFrame,
    *,
    prefer_key: str = "auto",      # "auto" | "signature_hash" | "hash" | "tolerant"
    tolerant_sig_rows=None         # list[{"id": <name>, "sig_hash": <tolerant_hash>}] if available
):
    """
    From the existing report_df, produce df_unique and df_instances the rebuilder needs.

    - df_unique: one row per unique geometry definition
      columns: ["part_id", "name", "step_path"]   # step_path will be 'native_step_path' if available
    - df_instances: one row per occurrence
      columns: ["part_id", "T", "parent_asm", "instance_id", "name"]
    """
    if report_df is None or report_df.empty:
        raise ValueError("report_df is empty; nothing to export.")

    # Work on a copy
    rdf = report_df.copy()

    # --- Prefer native (unmodified/original) geometry if available ---
    # Define a single column we will use as the part definition geometry for the assembly.
    # If 'native_step_path' exists and is non-empty, use it; else fall back to 'step_path'.
    if "native_step_path" in rdf.columns:
        # normalize to str for length checks
        rdf["native_step_path"] = rdf["native_step_path"].astype(str)
        rdf["step_path"] = rdf["step_path"].astype(str)
        rdf["def_step_path"] = rdf["native_step_path"].where(
            rdf["native_step_path"].str.len() > 0, rdf["step_path"]
        )
    else:
        rdf["step_path"] = rdf["step_path"].astype(str)
        rdf["def_step_path"] = rdf["step_path"]

    # Keep only rows that actually point to a file
    rdf = rdf[rdf["def_step_path"].str.len() > 0]
    if rdf.empty:
        raise ValueError("No rows with a non-empty (native or normal) step_path in report_df.")

    # --- Choose a stable part key ---
    if prefer_key == "tolerant":
        if not tolerant_sig_rows:
            raise ValueError("prefer_key='tolerant' but tolerant_sig_rows not provided.")
        df_tol = pd.DataFrame(tolerant_sig_rows).rename(columns={"id": "name", "sig_hash": "tol_sig"})
        rdf = rdf.merge(df_tol[["name", "tol_sig"]], on="name", how="left")
        part_key_col = "tol_sig"
    elif prefer_key == "signature_hash":
        part_key_col = "signature_hash"
    elif prefer_key == "hash":
        part_key_col = "hash"
    elif prefer_key == "auto":
        part_key_col = "signature_hash" if ("signature_hash" in rdf.columns and rdf["signature_hash"].notna().any()) else "hash"
    else:
        raise ValueError(f"Unknown prefer_key: {prefer_key}")

    if part_key_col not in rdf.columns:
        raise KeyError(f"Chosen part key column '{part_key_col}' not found in report_df.")

    # --- df_unique: one row per part definition (use def_step_path) ---
    df_unique = (
        rdf.sort_values(["name", "def_step_path"])
           .groupby(part_key_col, as_index=False)
           .agg({"name": "first", "def_step_path": "first"})
           .rename(columns={part_key_col: "part_id", "def_step_path": "step_path"})
    )[["part_id", "name", "step_path"]]

    # --- df_instances: one per occurrence; identity T since we’re using native geometry ---
    def _identity_T():
        return [1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1]

    df_instances = (
        rdf.assign(
            part_id=rdf[part_key_col],
            parent_asm="A-ROOT",
            T=[_identity_T() for _ in range(len(rdf))],
            instance_id=rdf["name"]
        )
    )[["part_id", "T", "parent_asm", "instance_id", "name"]]

    return df_unique, df_instances


def export_handoff_from_existing(
    df_unique,
    df_instances,
    out_dir: str,
    *,
    # column mapping for unique parts
    part_id_col="part_id",
    name_col=None,                 # e.g. "name" or "mark"
    step_path_col=None,            # path to an existing per-part STEP (preferred)
    shape_col=None,                # TopoDS_Shape, only used if write_parts=True
    # column mapping for instances
    inst_part_id_col="part_id",
    inst_name_col=None,            # e.g. "instance_name"
    parent_asm_col=None,           # e.g. "parent_asm" or "assembly_id"
    transform_col="T",             # 16-number list (row-major)
    instance_id_col=None,          # optional unique instance id
    # behavior
    write_parts=False,             # set True only if you don’t already have per-part STEPs
    units="MM",
    schema="AP242"
):
    import shutil
    import re

    def _safe_name(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return "part"
        # keep readable, filesystem-safe
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        return s[:120]

    out = Path(out_dir)
    parts_dir = out / "parts_native"     # <- use readable native filenames here
    out.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # --- unique_parts.jsonl ---
    upath = out / "unique_parts.jsonl"
    with open(upath, "w", encoding="utf-8") as ufile:
        for _, row in df_unique.iterrows():
            pid = str(row[part_id_col])  # hash/signature (dedupe key)
            pname = str(row[name_col]) if name_col and row.get(name_col) is not None else pid

            if step_path_col and row.get(step_path_col):
                # Reuse existing per-part STEP file, but keep its native filename
                src = Path(str(row[step_path_col]))
                if not src.exists():
                    raise FileNotFoundError(f"Missing STEP for {pid}: {src}")

                # Keep original basename (e.g. MEM-0001_native.step)
                dst = parts_dir / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                ghash = _sha256_file(dst)
                # Keep relative path into the pack
                step_rel = str(dst.relative_to(out).as_posix())

            elif write_parts and shape_col:
                # Write per-part STEP from an in-memory TopoDS_Shape with readable name
                shape = row[shape_col]
                base = _safe_name(pname)
                dst = parts_dir / f"{base}_native.step"
                _write_part_step(shape, dst)
                ghash = _sha256_file(dst)
                step_rel = str(dst.relative_to(out).as_posix())

            else:
                raise ValueError(
                    "Provide either an existing step path (step_path_col) "
                    "or set write_parts=True with shape_col."
                )

            rec = {
                "part_id": pid,          # keep dedupe key (hash/signature)
                "name": pname,
                "step_path": step_rel,   # now points to parts_native/<original>.step
                "geom_hash": ghash
            }
            ufile.write(json.dumps(rec) + "\n")

    # --- instances.jsonl ---
    ipath = out / "instances.jsonl"
    with open(ipath, "w", encoding="utf-8") as ifile:
        for _, row in df_instances.iterrows():
            T = list(row[transform_col])
            if not _is_rigid_4x4(T):
                raise ValueError(f"Non-rigid or malformed transform for part {row[inst_part_id_col]}")

            rec = {
                "part_id": str(row[inst_part_id_col]),
                "T": T,
                "parent_asm": str(row[parent_asm_col]) if parent_asm_col and row.get(parent_asm_col) is not None else "A-ROOT",
            }
            if inst_name_col and row.get(inst_name_col) is not None:
                rec["name"] = str(row[inst_name_col])
            if instance_id_col and row.get(instance_id_col) is not None:
                rec["instance_id"] = str(row[instance_id_col])

            ifile.write(json.dumps(rec) + "\n")

    # --- viewer manifest (replaces the old manifest.json) ---
    # Build absolute glb_path column by assuming GLB next to the STEP file.
    dfu = df_unique.copy()
    dfu["glb_path"] = dfu["step_path"].apply(lambda rel:
        str((Path(out) / rel).with_suffix(".glb"))
    )
    viewer_manifest = write_viewer_manifest(
        pack_root=out,
        df_unique=dfu,
        df_instances=df_instances,
        units=units,
        project_number=None,          # set your project number here if you have it
        unique_part_id_col=part_id_col,
        unique_name_col=(name_col or "name"),
        unique_glb_col="glb_path",
        unique_step_col=step_path_col or "step_path",
        inst_part_id_col=inst_part_id_col,
        inst_name_col=inst_name_col or "name",
        inst_parent_col=parent_asm_col or "parent_asm",
        inst_T_col=transform_col,
        inst_id_col=instance_id_col or "instance_id",
        df_assemblies=None,            # or pass a dataframe if you have assemblies
        parts_dir_name="parts",        # where MEM-###.step lives
        inline_props_cols=inline_props_cols,
    )
    print(f"✅ Viewer manifest written: {viewer_manifest}")


    return str(out)


def export_handoff_from_report(
    report_df: pd.DataFrame,
    out_dir: str,
    *,
    prefer_key: str = "auto",            # "auto" | "signature_hash" | "hash" | "tolerant"
    tolerant_sig_rows=None,              # optional list of tolerant signatures (if using "tolerant")
    units="MM",
    schema="AP242",
    return_tables: bool = False
):
    """
    Build df_unique/df_instances from report_df and write the handoff pack.
    Returns the output dir; optionally also returns the two DataFrames.
    """
    df_unique, df_instances = build_rebuilder_tables_from_report(
        report_df,
        prefer_key=prefer_key,
        tolerant_sig_rows=tolerant_sig_rows
    )

    pack_dir = export_handoff_from_existing(
        df_unique=df_unique,
        df_instances=df_instances,
        out_dir=out_dir,
        part_id_col="part_id",
        name_col="name",
        step_path_col="step_path",
        inst_part_id_col="part_id",
        parent_asm_col="parent_asm",
        transform_col="T",
        write_parts=False,
        units=units,
        schema=schema,
    )

    if return_tables:
        return pack_dir, df_unique, df_instances
    return pack_dir
