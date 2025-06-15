from OCC.Core.BRepTools import breptools
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.GeomAbs import GeomAbs_C1
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa

import os
import pyvista as pv
import ezdxf
from pathlib import Path

def export_solid_to_brep(solid, path, filename):
    success = breptools.Write(solid, f"{path}\{filename}.brep")
    if not success:
        raise RuntimeError(f"❌ Failed to write BREP to: {filename}")
    print(f"✅ BREP file written to {filename}")

def export_solid_to_step(solid, path, filename):
    step_path = f"{path}\{filename}.step"
    writer = STEPControl_Writer()
    writer.Transfer(solid, STEPControl_AsIs)
    status = writer.Write(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"❌ STEP export failed for {filename}")
    print(f"✅ STEP saved to {filename}")
    return step_path



def shape_to_thumbnail(solid, path, filename, deflection=0.5):
    # Mesh the shape
    BRepMesh_IncrementalMesh(solid, deflection)

    vertices = []
    faces = []

    topo = TopologyExplorer(solid)

    for face in topo.faces():
        triangulation = BRep_Tool.Triangulation(face, face.Location())
        if not triangulation:
            continue

        nb_nodes = triangulation.NbNodes()
        nb_triangles = triangulation.NbTriangles()

        offset = len(vertices)
        for i in range(1, nb_nodes + 1):
            pnt = triangulation.Node(i)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            faces.append([3, offset + n1 - 1, offset + n2 - 1, offset + n3 - 1])

    if not vertices or not faces:
        raise RuntimeError("❌ No triangulation data found in shape")

    pv_mesh = pv.PolyData(vertices, faces)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv_mesh, color="lightsteelblue", show_edges=True)
    plotter.set_background("white")
    plotter.camera_position = "iso"
    plotter.show(screenshot=f"{path}\{filename}.png")

    print(f"🖼️ Thumbnail saved to {filename}")
    
    return(f"{path}\{filename}.png")



import ezdxf
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa

def generate_plate_dxf(aligned_solid, filename, sampling_dist=1.0):
    """
    Extracts the largest planar face from `aligned_solid` (in the XY plane),
    samples each wire edge at ~`sampling_disbrepgprop.SurfacePropertiest` mm, and writes them as LWPolylines
    to DXF `filename`.
    """
    # 1) Find the largest face by area
    filename = Path(filename)
    best_face = None
    best_area = 0.0
    exp_f = TopExp_Explorer(aligned_solid, TopAbs_FACE)
    while exp_f.More():
        face = exp_f.Current()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        if area > best_area:
            best_area = area
            best_face = face
        exp_f.Next()

    if best_face is None:
        raise RuntimeError("No faces found on solid")

    # 2) Collect wires (outer contour + holes)
    wires_pts = []
    exp_w = TopExp_Explorer(best_face, TopAbs_WIRE)
    while exp_w.More():
        wire = exp_w.Current()
        pts = []
        exp_e = TopExp_Explorer(wire, TopAbs_EDGE)
        while exp_e.More():
            edge = exp_e.Current()
            # Instantiate curve adapter correctly
            curve = BRepAdaptor_Curve(edge)
            discretizer = GCPnts_UniformAbscissa(curve, sampling_dist)
            if discretizer.IsDone():
                for i in range(1, discretizer.NbPoints() + 1):
                    param = discretizer.Parameter(i)
                    p = curve.Value(param)
                    pts.append((p.X(), p.Y()))
            else:
                # fallback: start, mid, end
                p1 = curve.Value(curve.FirstParameter())
                p2 = curve.Value((curve.FirstParameter() + curve.LastParameter()) * 0.5)
                p3 = curve.Value(curve.LastParameter())
                pts.extend([(p1.X(), p1.Y()),
                            (p2.X(), p2.Y()),
                            (p3.X(), p3.Y())])
            exp_e.Next()
        # close the polyline
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        wires_pts.append(pts)
        exp_w.Next()

    # 3) Build and save DXF
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    for pts in wires_pts:
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, close=True)
    doc.saveas(filename)
    print(f"DXF written to {filename}")
    return filename

