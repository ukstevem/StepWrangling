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

def classify_and_project_holes_dstv(solid, dstv_frame):
    origin_dstv = dstv_frame.Location()
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    cm = props.CentreOfMass()
    centroid = np.array([cm.X(), cm.Y(), cm.Z()])

    im = props.MatrixOfInertia()
    T = np.array([[im.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    ev, evec = np.linalg.eigh(T)
    order = np.argsort(ev)
    dirs = [gp_Vec(*evec[:, i]) for i in order]
    
    L = np.array([dstv_frame.XDirection().X(), dstv_frame.XDirection().Y(), dstv_frame.XDirection().Z()])
    F = np.array([dstv_frame.YDirection().X(), dstv_frame.YDirection().Y(), dstv_frame.YDirection().Z()])
    W = np.array([dstv_frame.Direction().X(),   dstv_frame.Direction().Y(),   dstv_frame.Direction().Z()])


    bbox = Bnd_Box()
    brepbndlib.Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    half_extents = 0.5 * np.sort(spans)[::-1]
    length, width, thickness = half_extents * 2

    origin_nc1 = np.array([origin_dstv.X(), origin_dstv.Y(), origin_dstv.Z()])


    tol_long = 0.7
    tol_web = 0.7
    hole_data = []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        if adaptor.GetType() == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            axis_dir = np.array([cyl.Axis().Direction().X(),
                                 cyl.Axis().Direction().Y(),
                                 cyl.Axis().Direction().Z()])
            norm = np.linalg.norm(axis_dir)
            if norm == 0:
                explorer.Next()
                continue
            axis_dir /= norm

            if abs(np.dot(axis_dir, L)) > tol_long:
                explorer.Next()
                continue

            pf = GProp_GProps()
            brepgprop.SurfaceProperties(face, pf)
            fc = np.array([pf.CentreOfMass().X(),
                           pf.CentreOfMass().Y(),
                           pf.CentreOfMass().Z()])

            if abs(np.dot(axis_dir, W)) > tol_web:
                code, rgb = "V", (1.0, 0.0, 0.0)
            else:
                side = np.dot(fc - centroid, F)
                code, rgb = ("O", (0.0, 1.0, 0.0)) if side > 0 else ("U", (0.0, 0.0, 1.0))

            hole_data.append((face, code, rgb, fc))
        explorer.Next()

    rows = []
    for idx, (face, code, rgb, fc) in enumerate(hole_data, start=1):
        adaptor = BRepAdaptor_Surface(face, True)
        diam = 2.0 * adaptor.Cylinder().Radius()
        v = fc - origin_nc1

        if code == "V":
            x = abs(float(np.dot(v, L)))
            y = abs(float(np.dot(v, F)))
        elif code == "U":
            x = abs(float(np.dot(v, L)))
            y = abs(float(np.dot(v, W)))
        elif code == "O":
            x = abs(float(np.dot(v, L)))
            y = abs(float(np.dot(v, W)))

        rows.append({
            "Hole #":        idx,
            "Code":          code,
            "Diameter (mm)": round(diam, 2),
            "X (mm)":        round(x, 2),
            "Y (mm)":        round(y, 2)
        })

    df_holes = pd.DataFrame(rows)
    return df_holes, hole_data, origin_nc1, L, F, W

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
