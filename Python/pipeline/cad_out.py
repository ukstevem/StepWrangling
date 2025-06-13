from OCC.Core.BRepTools import breptools
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
import os
import pyvista as pv

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