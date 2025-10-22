import os
import tempfile
from pathlib import Path

def resolve_output_path(
    rel_path: str,
    network_root: Path = Path(r"//pss-dc02/PSS_CAD/Customer Orders/Extraction"),
    fallback_root: Path = Path(r"c:/dev/step-gemini/pythin/Extraction")
) -> Path:
    """
    Given a relative path like "folder/subfolder/file.txt",
    decide whether to write it under network_root or under a local temp dir.
    Cleans any leading '..' segments from rel_path automatically.
    
    - If the network_root is reachable & writable, use it.
    - Otherwise, use fallback_root (by default, tempfile.gettempdir()).
    """

    # 1) Clean the relative path: drop any '..' or empty parts
    parts = [p for p in Path(rel_path).parts if p not in ("..", "", "\\", "/")]
    clean_rel = Path(*parts)

    # 2) Pick base
    if fallback_root is None:
        fallback_root = Path(tempfile.gettempdir())
    base = network_root
    try:
        # check it exists and is writable
        if not (network_root.exists() and os.access(network_root, os.W_OK)):
            raise OSError("Network share not writable")
    except Exception:
        base = fallback_root

    # 3) Build the final path
    full = base / clean_rel

    # 4) Make sure the directory exists
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


def is_network_available(net_root: Path) -> bool:
    """
    Returns True only if we can create & delete a zero-byte file in net_root.
    """
    probe = net_root / ".dstv_probe"
    try:
        # ensure the directory exists
        net_root.mkdir(parents=True, exist_ok=True)
        # try the probe
        with open(probe, "w") as f:
            pass
        probe.unlink()
        return True
    except Exception as e:
        print(f"[DEBUG] Network write test failed: {e!r}")
        return False



def get_root_dir(net_root, fb_root) -> Path:
    """
    Returns network_root if it exists and is writable;
    otherwise returns fallback_root (defaults to the OS temp dir).
    """
    net_root = Path(net_root)

    if fb_root is None:
        fb_root = Path(tempfile.gettempdir())
    else:
        fb_root = Path(fb_root)

    if is_network_available(net_root):
        print(f"[DEBUG] Using network root: {net_root}")
        return net_root
    else:
        print(f"[DEBUG] Falling back to local root: {fb_root}")
        return fb_root
    
    import os
import pandas as pd
import pathlib
from pipeline.path_management import get_root_dir

def create_project_directories(project_number, net_root, fb_root):

    # Get if network or local
    root = get_root_dir(net_root, fb_root)

    """
    Creates the output directory structure for a project if it doesn't exist.
    Returns the base path and subfolder paths.
    """
    base_path = os.path.join(root, "Projects", project_number)
    nc1_path = os.path.join(base_path, "Nc1")
    reports_path = os.path.join(base_path, "Reports")
    drilling_path = os.path.join(base_path, "Drilling")
    cad_path = os.path.join(base_path, "cad")
    step_path = os.path.join(cad_path, "step")
    brep_path = os.path.join(cad_path, "brep")
    dxf_path = os.path.join(cad_path, "dxf")
    dxf_thumb = os.path.join(dxf_path, "thumbs")
    thumb_path = os.path.join(base_path, "thumbs")
    stl_path = os.path.join(cad_path, "stl")
    handoff = os.path.join(cad_path, "handoff_pack")
    handoff_parts = os.path.join(handoff, "parts")
    IFC_path = os.path.join(cad_path, "IFC")


    for path in [nc1_path, reports_path, drilling_path, cad_path, thumb_path, step_path, stl_path, brep_path, dxf_path, dxf_thumb, handoff, handoff_parts]:
        os.makedirs(path, exist_ok=True)

    return base_path, nc1_path, reports_path, drilling_path, cad_path, thumb_path, step_path, stl_path, brep_path, dxf_path, dxf_thumb, handoff, handoff_parts, IFC_path