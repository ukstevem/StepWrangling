# pipeline/step_loader.py

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID


def load_step_file(filepath):
    """
    Loads a STEP file and returns a list of solids.
    Raises a RuntimeError with details if the file cannot be read.
    """
    reader = STEPControl_Reader()

    try:
        status = reader.ReadFile(filepath)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP file read failed with status code {status}: {filepath}")

        reader.TransferRoots()
        shape = reader.OneShape()

        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solid = explorer.Current()
            solids.append(solid)
            explorer.Next()

        print(f"Loaded {len(solids)} solids from {filepath}")
        return solids

    except Exception as e:
        raise RuntimeError(f"Exception occurred while loading STEP file '{filepath}': {e}") from e
