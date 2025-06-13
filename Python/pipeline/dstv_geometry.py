from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Vec
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

import numpy as np
import pandas as pd

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
import numpy as np
import pandas as pd

def classify_and_project_holes_dstv(solid, dstv_frame, origin_dstv, width_mm, flange_span_mm):
    explorer = TopExp_Explorer(solid, TopAbs_EDGE)
    edges = []

    origin_np = np.array([origin_dstv.X(), origin_dstv.Y(), origin_dstv.Z()])
    L = np.array([dstv_frame.XDirection().X(), dstv_frame.XDirection().Y(), dstv_frame.XDirection().Z()])
    F = np.array([dstv_frame.YDirection().X(), dstv_frame.YDirection().Y(), dstv_frame.YDirection().Z()])
    W = np.array([dstv_frame.Direction().X(),   dstv_frame.Direction().Y(),   dstv_frame.Direction().Z()])

    width_tol = 0.5 * width_mm + 10  # for V face
    flange_tol = 15  # for U and O face distance from DSTV plane

    seen_keys = set()
    hole_rows = []

    while explorer.More():
        edge = topods.Edge(explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Circle:
            explorer.Next()
            continue

        circ = curve.Circle()
        center = circ.Location()
        axis = circ.Axis().Direction()

        center_np = np.array([center.X(), center.Y(), center.Z()])
        axis_np = np.array([axis.X(), axis.Y(), axis.Z()])
        axis_np /= np.linalg.norm(axis_np)
        radius = circ.Radius()

        v = center_np - origin_np
        d_web = abs(np.dot(v, W))
        d_flange = np.dot(v, F)

        dot_web = abs(np.dot(axis_np, W))
        dot_flange = np.dot(axis_np, F)

        # Classify by proximity and direction
        if d_web <= width_tol and dot_web > 0.7:
            code = 'V'
            y_axis = F
        elif abs(d_flange) <= flange_tol and dot_flange < -0.7:
            code = 'U'
            y_axis = W
        elif abs(d_flange - flange_span_mm) <= flange_tol and dot_flange > 0.7:
            code = 'O'
            y_axis = W
        else:
            explorer.Next()
            continue

        x = round(float(np.dot(v, L)), 2)
        y = round(float(np.dot(v, y_axis)), 2)

        key = (round(x, 1), round(y, 1), round(radius, 2), code)
        if key in seen_keys:
            explorer.Next()
            continue
        seen_keys.add(key)

        hole_rows.append({
            "Hole #":        len(hole_rows) + 1,
            "Code":          code,
            "Diameter (mm)": round(radius * 2, 2),
            "X (mm)":        x,
            "Y (mm)":        y
        })

        explorer.Next()

    return pd.DataFrame(hole_rows)

def check_duplicate_holes(df_holes, tolerance=0.1):
    """
    Checks for duplicate holes (same X, Y, and Code) within a given tolerance (in mm).
    returns a dataframe of duplicates, or False if no duplicates are found.
    """
    # Round to given tolerance (to handle small numeric noise)
    df_check = df_holes.copy()
    df_check["X_r"] = (df_check["X (mm)"] / tolerance).round().astype(int)
    df_check["Y_r"] = (df_check["Y (mm)"] / tolerance).round().astype(int)

    # Check for duplicates based on rounded X, Y, and face Code
    duplicates = df_check.duplicated(subset=["Code", "X_r", "Y_r"], keep=False)

    if duplicates.any():
        print(f"⚠️ Found {duplicates.sum()} potential duplicate holes.")
        dup_df = df_holes[duplicates]
        return True, dup_df
    else:
        return False , pd.DataFrame() 
