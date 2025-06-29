import hashlib
import os
import argparse
import re
from pathlib import Path

# STEP export via OpenCascade (pythonOCC)
try:
    from OCC.Extend.DataExchange import write_step_file
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    can_write_step = True
except ImportError:
    can_write_step = False

# IFC export via IfcOpenShell
try:
    import ifcopenshell
    can_write_ifc = True
except ImportError:
    can_write_ifc = False


def hash_file(path: str, algorithm: str = 'sha256', chunk_size: int = 8192) -> str:
    """
    Compute the cryptographic hash of a file in chunks to handle large files.
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Available: {sorted(hashlib.algorithms_available)}")

    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_step_header(src: str, dest: str) -> None:
    """
    Normalize STEP file header by zeroing date/time, to ensure deterministic hashing.
    """
    with open(src, 'r') as f:
        data = f.read()
    # Zero out DATE_TIME field
    normalized = re.sub(
        r"(HEADER;.*DATE_TIME\([^)]*\))",
        "HEADER; DATE_TIME('0000-00-00T00:00:00');",
        data,
        flags=re.DOTALL
    )
    with open(dest, 'w') as f:
        f.write(normalized)


def _compose_solids(shapes) -> 'TopoDS_Compound':
    """
    Combine multiple OCC shapes into a single compound for STEP export.
    """
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for s in shapes:
        builder.Add(compound, s)
    return compound


def fingerprint_solids(solids, cad_file: str, algorithm: str = 'sha256') -> str:
    """
    Serialize solids to a CAD file (STEP/IFC), save it, and return a deterministic hash of geometry.

    :param solids: A single OCC shape, a list of shapes, or an ifcopenshell file object.
    :param cad_file: Target file path (.step/.stp or .ifc).
    :param algorithm: Hash algorithm for fingerprinting.
    :return: Hex digest of geometry data.
    """
    _, ext = os.path.splitext(cad_file)
    ext = ext.lower()
    os.makedirs(os.path.dirname(cad_file) or '.', exist_ok=True)

    # STEP export
    if ext in ('.step', '.stp'):
        if not can_write_step:
            raise RuntimeError("STEP export unavailable. Install pythonOCC.")
        step_shape = _compose_solids(solids) if isinstance(solids, (list, tuple)) else solids
        write_step_file(step_shape, cad_file)
        # Normalize header for deterministic hash
        norm_file = str(Path(cad_file).with_suffix('.norm.step'))
        normalize_step_header(cad_file, norm_file)
        hash_val = hash_file(norm_file, algorithm)
        os.remove(norm_file)

    # IFC export
    elif ext == '.ifc':
        if not can_write_ifc:
            raise RuntimeError("IFC export unavailable. Install IfcOpenShell.")
        if hasattr(solids, 'write'):
            solids.write(cad_file)
            hash_val = hash_file(cad_file, algorithm)
        else:
            raise RuntimeError("For IFC, 'solids' must be an ifcopenshell file object with write().")

    else:
        raise ValueError(f"Unsupported CAD extension '{ext}'. Use .step/.stp or .ifc.")

    return hash_val
