from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Vec
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane

# Helper: draw an axis arrow
def draw_axis_arrow(display, origin, direction, scale=100, color='BLACK'):
    end = gp_Pnt(
        origin.X() + direction.X() * scale,
        origin.Y() + direction.Y() * scale,
        origin.Z() + direction.Z() * scale,
    )
    edge = BRepBuilderAPI_MakeEdge(origin, end).Edge()
    display.DisplayShape(edge, color=color, update=False)


def draw_plane(display, origin, normal, size=100, color="MAGENTA"):
    """
    Draw a planar square face centered at `origin` with normal `normal`.

    Parameters:
        display - The OCC display context.
        origin  - gp_Pnt
        normal  - gp_Dir or gp_Vec
        size    - float, size of the plane (length of one edge)
        color   - str, display color
    """
    # Convert normal to vector
    n = gp_Vec(normal) if hasattr(normal, 'XYZ') else gp_Vec(normal.X(), normal.Y(), normal.Z())

    # Create U, V vectors orthogonal to the normal
    u = n.Crossed(gp_Vec(0, 0, 1))
    if u.Magnitude() < 1e-6:
        u = gp_Vec(1, 0, 0)
    v = n.Crossed(u)
    u.Normalize()
    v.Normalize()

    u *= size / 2
    v *= size / 2

    o = gp_Vec(origin.X(), origin.Y(), origin.Z())
    corners = [
        gp_Pnt(*(o + u + v)),
        gp_Pnt(*(o - u + v)),
        gp_Pnt(*(o - u - v)),
        gp_Pnt(*(o + u - v))
    ]

    # Create wire from corners
    edges = [BRepBuilderAPI_MakeEdge(corners[i], corners[(i + 1) % 4]).Edge() for i in range(4)]
    wire_builder = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire_builder.Add(edge)
    
    face = BRepBuilderAPI_MakeFace(wire_builder.Wire()).Face()
    display.DisplayShape(face, color=color, update=False)

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
import numpy as np

def draw_obb(display, center, xaxis, yaxis, zaxis, he_X, he_Y, he_Z, color="WHITE"):
    import numpy as np
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt

    c = np.array([center.X(), center.Y(), center.Z()])
    axes = [xaxis, yaxis, zaxis]
    half_extents = [he_X, he_Y, he_Z]
    axis_vecs = [np.array([a.X(), a.Y(), a.Z()]) for a in axes]

    corners = []
    for dx in (-1, 1):
        for dy in (-1, 1):
            for dz in (-1, 1):
                offset = (
                    dx * half_extents[0] * axis_vecs[0] +
                    dy * half_extents[1] * axis_vecs[1] +
                    dz * half_extents[2] * axis_vecs[2]
                )
                pt = gp_Pnt(*(c + offset))
                corners.append(pt)

    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    for i, j in edges:
        edge = BRepBuilderAPI_MakeEdge(corners[i], corners[j]).Edge()
        display.DisplayShape(edge, color=color, update=False)



from OCC.Core.AIS import AIS_TextLabel
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TCollection import TCollection_ExtendedString

def label_2d_text(display, label_id, position, text, height=24, color="BLACK"):
    ais_label = AIS_TextLabel()
    ais_label.SetText(TCollection_ExtendedString(text))
    ais_label.SetPosition(position)

    col_map = {
        "BLACK": (0.0, 0.0, 0.0),
        "WHITE": (1.0, 1.0, 1.0),
        "RED": (1.0, 0.0, 0.0),
        "GREEN": (0.0, 1.0, 0.0),
        "BLUE": (0.0, 0.0, 1.0),
        "CYAN": (0.0, 1.0, 1.0),
        "YELLOW": (1.0, 1.0, 0.0),
        "MAGENTA": (1.0, 0.0, 1.0),
        "GRAY": (0.5, 0.5, 0.5)
    }
    r, g, b = col_map.get(color.upper(), (0.0, 0.0, 0.0))
    text_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)

    ais_label.SetColor(text_color)
    display.Context.Display(ais_label, True)
    display.Context.UpdateCurrentViewer()
