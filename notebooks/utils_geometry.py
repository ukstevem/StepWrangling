from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Vec
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane

def make_section_face_at_start(solid, center, zaxis, half_length):
    """
    Cuts a planar section at the start end of the solid along the Z axis.
    Returns (section_face, section_origin)
    """
    # Create plane origin at -half length along Z
    origin = gp_Pnt(
        center.X() - half_length * zaxis.X(),
        center.Y() - half_length * zaxis.Y(),
        center.Z() - half_length * zaxis.Z()
    )
    plane = gp_Pln(origin, gp_Dir(zaxis))
    
    # Section operation
    section = BRepAlgoAPI_Section(solid, plane)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()

    # Build a face from section edges
    wire_builder = BRepBuilderAPI_MakeWire()
    exp = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while exp.More():
        wire_builder.Add(topods.Edge(exp.Current()))
        exp.Next()

    face = BRepBuilderAPI_MakeFace(wire_builder.Wire()).Face()
    return face, origin

def compute_obb_and_local_axes(solid):
    from OCC.Core.Bnd import Bnd_OBB
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.gp import gp_Vec

    # Compute OBB
    obb = Bnd_OBB()
    brepbndlib.AddOBB(solid, obb, True, True, False)

    # Get raw data
    center = obb.Center()
    axes = [obb.XDirection(), obb.YDirection(), obb.ZDirection()]
    half_extents = [obb.XHSize(), obb.YHSize(), obb.ZHSize()]

    # Sort half-extents to identify logical axes
    sorted_indices = np.argsort(half_extents)
    web_idx, flange_idx, length_idx = sorted_indices

    # Assign named axes + extents
    xaxis = gp_Vec(axes[web_idx])
    yaxis = gp_Vec(axes[flange_idx])
    zaxis = gp_Vec(axes[length_idx])

    he_X = half_extents[web_idx]
    he_Y = half_extents[flange_idx]
    he_Z = half_extents[length_idx]

    return {
        'center': center,
        'xaxis': xaxis,
        'yaxis': yaxis,
        'zaxis': zaxis,
        'he_X': he_X,
        'he_Y': he_Y,
        'he_Z': he_Z,
        'axes': (xaxis, yaxis, zaxis),
        'half_extents': (he_X, he_Y, he_Z)
    }

def cut_section_area(solid, origin, normal):
    """
    Cut a planar section of a solid at the specified origin and normal, return face and area.

    Parameters:
        solid  - TopoDS_Shape (solid)
        origin - gp_Pnt
        normal - gp_Dir

    Returns:
        (face, area)
    """
    # Build cutting plane
    plane = gp_Pln(origin, normal)
    section = BRepAlgoAPI_Section(solid, plane)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()

    # Extract edges and build a face
    wire_builder = BRepBuilderAPI_MakeWire()
    explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while explorer.More():
        wire_builder.Add(topods.Edge(explorer.Current()))
        explorer.Next()

    if wire_builder.IsDone():
        face = BRepBuilderAPI_MakeFace(wire_builder.Wire()).Face()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        return face, props.Mass()
    else:
        return None, 0.0

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods

def cut_section_at_start(solid, center, zaxis, half_length):
    """
    Cut a section at the 'start' end of the solid along the Z axis.

    Parameters:
        solid       - TopoDS_Shape
        center      - gp_Pnt (center of bounding box)
        zaxis       - gp_Dir (local Z axis of the part)
        half_length - float (half-length of the solid along Z)

    Returns:
        (face, origin) - section face and the plane origin as gp_Pnt
    """
    # Compute plane origin at the start end
    ox = center.X() - half_length * zaxis.X()
    oy = center.Y() - half_length * zaxis.Y()
    oz = center.Z() - half_length * zaxis.Z()
    origin = gp_Pnt(ox, oy, oz)

    # Construct plane and section
    plane = gp_Pln(origin, gp_Dir(zaxis))
    section = BRepAlgoAPI_Section(solid, plane)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()

    # Extract wire from section edges
    wire_builder = BRepBuilderAPI_MakeWire()
    explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while explorer.More():
        wire_builder.Add(topods.Edge(explorer.Current()))
        explorer.Next()

    if wire_builder.IsDone():
        face = BRepBuilderAPI_MakeFace(wire_builder.Wire()).Face()
        return face, origin
    else:
        return None, origin

import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

def extract_four_corners_section(face, xaxis, yaxis, origin):
    """
    Extract 4 ordered corner points from a section face.

    Parameters:
        face   - BRep face assumed to be a planar 4-corner section.
        xaxis  - Local x-axis (gp_Dir or gp_Vec).
        yaxis  - Local y-axis (gp_Dir or gp_Vec).
        origin - gp_Pnt used as projection origin for sorting.

    Returns:
        ordered_corners - List of gp_Pnt in consistent winding order.
    """
    # Convert origin and axes to NumPy
    origin_np = np.array([origin.X(), origin.Y(), origin.Z()])
    x_np = np.array([xaxis.X(), xaxis.Y(), xaxis.Z()])
    y_np = np.array([yaxis.X(), yaxis.Y(), yaxis.Z()])

    explorer = TopExp_Explorer(face, TopAbs_VERTEX)
    pts_3d = []

    while explorer.More():
        vertex = topods.Vertex(explorer.Current())
        pt = BRep_Tool.Pnt(vertex)
        pts_3d.append(pt)
        explorer.Next()

    if len(pts_3d) != 4:
        raise ValueError(f"Expected 4 corners, got {len(pts_3d)}")

    # Project to local 2D coordinates
    pts_2d = []
    for pt in pts_3d:
        vec = np.array([pt.X(), pt.Y(), pt.Z()]) - origin_np
        px = np.dot(vec, x_np)
        py = np.dot(vec, y_np)
        pts_2d.append((px, py))

    # Sort bottom-left to top-right
    indexed = sorted(zip(pts_2d, pts_3d), key=lambda p: (p[0][1], p[0][0]))
    ordered_corners = [p[1] for p in indexed]
    return ordered_corners

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Dir

def extract_section_faces(solid, section_origin, zaxis, tolerance=1e-3):
    """
    Extract planar section faces near the given section_origin plane normal to zaxis.
    Returns a list of (face, area, centroid, normal).
    """
    section_faces = []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() != GeomAbs_Plane:
            explorer.Next()
            continue

        plane = surf.Plane()
        face_normal = plane.Axis().Direction()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        centroid = props.CentreOfMass()
        area = props.Mass()

        # Check if face lies in section plane (dot with zaxis ≈ 0 and position close)
        dot = abs(face_normal.Dot(gp_Dir(zaxis)))
        if dot < tolerance:
            dist = plane.Distance(section_origin)
            if dist < tolerance:
                section_faces.append((face, area, centroid, face_normal))

        explorer.Next()

    return section_faces



def get_face_area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()



def should_flip_zaxis_for_uea(corners_3d, xaxis, yaxis, origin, matched_profile):
    """
    Determines if Z axis needs to be flipped for a UEA profile based on leg orientation.

    Parameters:
    - corners_3d: list of gp_Pnt defining the section at current end
    - xaxis, yaxis: gp_Vec (local axes from OBB)
    - origin: gp_Pnt (section origin)
    - matched_profile: dict from classifier including width, height, type

    Returns:
    - True if zaxis should be reversed
    """
    # Project each 3D point into local XY frame
    local_y = np.array([yaxis.X(), yaxis.Y(), yaxis.Z()])
    local_x = np.array([xaxis.X(), xaxis.Y(), xaxis.Z()])
    origin_np = np.array([origin.X(), origin.Y(), origin.Z()])

    pts_2d = []
    for pt in corners_3d:
        vec = np.array([pt.X(), pt.Y(), pt.Z()]) - origin_np
        px = np.dot(vec, local_x)
        py = np.dot(vec, local_y)
        pts_2d.append((px, py))

    # Measure Y extents to detect leg positions
    ys = [py for _, py in pts_2d]
    min_y = min(ys)
    max_y = max(ys)
    span_y = max_y - min_y

    # If this is a UEA, the long leg should be in -Y direction
    is_uea = matched_profile.get("type", "").upper() == "UEA"

    if not is_uea:
        return False  # No flip needed for EA, Beam, etc.

    # Determine whether the longer leg is on bottom (i.e., larger extent downward)
    longer_leg_down = abs(min_y) > abs(max_y)

    return not longer_leg_down  # If longer leg is not down, we should flip Z


def extract_ordered_section_corners(face, xaxis, yaxis, origin, expected=4):
    """
    Extracts and orders corners from a planar section face based on local XY axes.
    Returns ordered list of gp_Pnt.

    Parameters:
        face (TopoDS_Face): Section face to analyze
        xaxis, yaxis (gp_Vec): local coordinate frame
        origin (gp_Pnt): local frame origin
        expected (int): number of corners expected (default=4)
    """
    origin_np = np.array([origin.X(), origin.Y(), origin.Z()])
    local_x = np.array([xaxis.X(), xaxis.Y(), xaxis.Z()])
    local_y = np.array([yaxis.X(), yaxis.Y(), yaxis.Z()])

    explorer = TopExp_Explorer(face, TopAbs_VERTEX)
    pts_3d = []

    while explorer.More():
        vert = topods.Vertex(explorer.Current())
        pt = BRep_Tool.Pnt(vert)
        pts_3d.append(pt)
        explorer.Next()

    if len(pts_3d) < expected:
        raise ValueError(f"Expected at least {expected} corners, got {len(pts_3d)}")

    # Project into 2D (local XY)
    pts_2d = []
    for pt in pts_3d:
        vec = np.array([pt.X(), pt.Y(), pt.Z()]) - origin_np
        px = np.dot(vec, local_x)
        py = np.dot(vec, local_y)
        pts_2d.append((px, py))

    # Sort by Y descending (bottom to top), then X ascending (left to right)
    indexed = sorted(zip(pts_2d, pts_3d), key=lambda v: (-v[0][1], v[0][0]))

    ordered_pts_3d = [pt3d for _, pt3d in indexed[:expected]]

    return ordered_pts_3d
