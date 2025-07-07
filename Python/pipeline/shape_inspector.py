from OCC.Core.STEPControl     import STEPControl_Reader
from OCC.Core.BRepBndLib      import brepbndlib
from OCC.Core.Bnd             import Bnd_Box
from OCC.Core.TopLoc          import TopLoc_Location
from OCC.Core.BRepBuilderAPI  import BRepBuilderAPI_Transform
from OCC.Core.gp              import gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt
import numpy as np

def load_step(path: str):
    """Read a STEP file and return its TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != 0:
        raise RuntimeError(f"Failed to read STEP: status {status}")
    reader.TransferRoots()
    return reader.OneShape()

def describe_shape(shape) -> dict:
    """
    Inspect a TopoDS_Shape in the *global* coordinate system.

    Returns a dict with:
      - centre:  (x,y,z) centroid of its axis‐aligned bounding box
      - extents: (L, W, H) = (Xsize, Ysize, Zsize)
      - axis_x, axis_y, axis_z: unit vectors for the shape’s local axes
    """
    # 1) Build the global bounding box
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    centre = ((xmin + xmax) / 2.0,
              (ymin + ymax) / 2.0,
              (zmin + zmax) / 2.0)
    extents = (xmax - xmin, ymax - ymin, zmax - zmin)

    # 2) Extract the shape’s location transform
    loc = shape.Location()  # TopLoc_Location
    trsf: gp_Trsf = loc.Transformation()

    # 3) From the gp_Trsf, pull out the rotation matrix
    mat = trsf.VectorialPart()  # gp_Mat
    # columns of mat are the local‐axes in global coords
    axis_x = (mat.Value(1,1), mat.Value(2,1), mat.Value(3,1))
    axis_y = (mat.Value(1,2), mat.Value(2,2), mat.Value(3,2))
    axis_z = (mat.Value(1,3), mat.Value(2,3), mat.Value(3,3))

    print("centre:",  np.array(centre))
    print("extents:", np.array(extents))
    print("X‐axis:",  np.array(axis_x))
    print("Y‐axis:", np.array(axis_y))
    print("Z‐axis:",  np.array(axis_z))

    return {
        'centre':  np.array(centre),
        'extents': np.array(extents),
        'axis_x':  np.array(axis_x),
        'axis_y':  np.array(axis_y),
        'axis_z':  np.array(axis_z),
    }

def apply_translation(shape, dx=0, dy=0, dz=0):
    """Return a new shape translated by (dx,dy,dz) in global coords."""
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

def apply_rotation(shape, axis_point, axis_dir, angle_deg):
    """
    Return a new shape rotated about a global axis.
      - axis_point: (x,y,z) a point on the rotation axis
      - axis_dir:   (ux,uy,uz) the unit direction of the axis
      - angle_deg:  rotation angle in degrees
    """
    p = gp_Pnt(*axis_point)
    u = gp_Vec(*axis_dir).Normalized()
    ax = gp_Ax1(p, u)
    tr = gp_Trsf()
    tr.SetRotation(ax, np.radians(angle_deg))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()
