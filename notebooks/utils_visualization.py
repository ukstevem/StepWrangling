from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Vec, gp_Trsf, gp_Ax3
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.AIS import AIS_TextLabel
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TCollection import TCollection_ExtendedString

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


def draw_obb(display, center, xaxis, yaxis, zaxis, he_X, he_Y, he_Z, color="WHITE"):
    import numpy as np
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt

    c = np.array([center.X(), center.Y(), center.Z()])
    axis_vecs = [np.array([a.X(), a.Y(), a.Z()]) for a in [xaxis, yaxis, zaxis]]
    half_extents = [he_X, he_Y, he_Z]

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

def translate_face(face, direction_vec, distance):
    # Clone the vector so we can safely scale it
    vec = gp_Vec(direction_vec.X(), direction_vec.Y(), direction_vec.Z())
    vec.Multiply(distance)

    trsf = gp_Trsf()
    trsf.SetTranslation(vec)

    transformed = BRepBuilderAPI_Transform(face, trsf, True).Shape()
    return transformed

def draw_legend(display, origin=(0, 0, 0), spacing=50):
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.gp import gp_Pln, gp_Vec

    labels = [
        ("START", "RED"),
        ("U FACE", "GREEN"),
        ("V FACE", "BLUE")
    ]

    for i, (label, color) in enumerate(labels):
        offset_vec = gp_Vec(spacing * i, 0, 0)
        legend_origin = gp_Pnt(*origin)
        legend_origin.Translate(offset_vec)

        plane = gp_Pln(legend_origin, gp_Dir(0, 0, 1))  # faces up
        face = BRepBuilderAPI_MakeFace(plane, -5, 5, -5, 5).Shape()

        display.DisplayShape(face, color=color, transparency=0.2, update=False)

        # Optionally: add on-screen text label if needed
        display.DisplayMessage(legend_origin, label)

def make_face(origin, axis1, axis2, he1, he2, normal, offset=10.0):
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    try:
        if he1 <= 0 or he2 <= 0:
            raise ValueError(f"Invalid half-extents: he1={he1}, he2={he2}")

        origin_pnt = gp_Pnt(*origin)

        # Offset the origin along the normal
        normal_vec = normal  # already a gp_Vec
        if normal_vec.Magnitude() == 0:
            raise ValueError("Normal vector has zero magnitude")

        normal_vec.Normalize()
        normal_vec.Multiply(offset)
        origin_pnt.Translate(normal_vec)

        normal_dir = gp_Dir(normal)

        # Attempt to use axis1 as x_dir
        try:
            x_dir = gp_Dir(axis1)
            # Ensure axis1 is not parallel to normal
            _ = gp_Ax3(origin_pnt, normal_dir, x_dir)
        except:
            # Fallback: use orthogonal direction from axis2
            fallback = axis2.Crossed(normal_vec)
            if fallback.Magnitude() == 0:
                raise ValueError("Fallback x direction is degenerate")
            x_dir = gp_Dir(fallback)

        ax3 = gp_Ax3(origin_pnt, normal_dir, x_dir)
        plane = gp_Pln(ax3)

        face = BRepBuilderAPI_MakeFace(plane, -he1, he1, -he2, he2).Shape()
        if face.IsNull():
            raise RuntimeError("BRepBuilderAPI_MakeFace returned a null shape")

        return face

    except Exception as e:
        print(f"❌ make_face failed: {e}")
        return None




def draw_local_axes(display, center, xaxis, yaxis, zaxis, length=100):
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

    # Fix: ensure center is gp_Pnt
    if not isinstance(center, gp_Pnt):
        center = gp_Pnt(center.X(), center.Y(), center.Z())

    def draw_axis(vec, color):
        end_point = gp_Pnt(
            center.X() + vec.X() * length,
            center.Y() + vec.Y() * length,
            center.Z() + vec.Z() * length
        )
        edge = BRepBuilderAPI_MakeEdge(center, end_point).Edge()
        display.DisplayShape(edge, color=color, update=False)

    draw_axis(xaxis, "RED")
    draw_axis(yaxis, "GREEN")
    draw_axis(zaxis, "BLUE")

    display.FitAll()




def visualize_dstv_faces(display, obb_data, axes, offset=10.0):
    """
    Visualizes START, O, U, V, and H faces per DSTV definition.

    Axes must be aligned:
    - z = beam length direction (START normal)
    - y = height (U face normal)
    - x = width (V face normal)
    """
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.gp import gp_Dir, gp_Ax3, gp_Pln

    def make_plane_face(origin, normal_vec, x_vec, he1, he2):
        origin_pnt = gp_Pnt(*origin)
        ax3 = gp_Ax3(origin_pnt, gp_Dir(normal_vec), gp_Dir(x_vec))
        plane = gp_Pln(ax3)
        return BRepBuilderAPI_MakeFace(plane, -he1, he1, -he2, he2).Shape()

    def translate_point(base, vec, dist):
        return [
            base.X() + vec.X() * dist,
            base.Y() + vec.Y() * dist,
            base.Z() + vec.Z() * dist
        ]

    center = obb_data["center"]
    if not isinstance(center, gp_Pnt):
        center = gp_Pnt(center.X(), center.Y(), center.Z())

    he_X = obb_data["he_X"]
    he_Y = obb_data["he_Y"]
    he_Z = obb_data["he_Z"]

    # START face (cutting face at front)
    origin_start = translate_point(center, axes["z"], -(he_Z + offset))
    face_start = make_plane_face(origin_start, axes["z"], axes["x"], he_X, he_Y)

    # U face (height, up)
    origin_u = translate_point(center, axes["y"], +(he_Y + offset))
    face_u = make_plane_face(origin_u, axes["y"], axes["x"], he_X, he_Z)

    # O face (opposite U)
    origin_o = translate_point(center, axes["y"], -(he_Y + offset))
    face_o = make_plane_face(origin_o, -axes["y"], axes["x"], he_X, he_Z)

    # V face (width)
    origin_v = translate_point(center, axes["x"], +(he_X + offset))
    face_v = make_plane_face(origin_v, axes["x"], axes["y"], he_Y, he_Z)

    # H face (opposite V)
    origin_h = translate_point(center, axes["x"], -(he_X + offset))
    face_h = make_plane_face(origin_h, -axes["x"], axes["y"], he_Y, he_Z)

    # Display faces
    if face_start: display.DisplayShape(face_start, color="RED", transparency=0.7, update=False)    # START
    if face_u:     display.DisplayShape(face_u,     color="GREEN", transparency=0.7, update=False)  # U (top)
    if face_o:     display.DisplayShape(face_o,     color="ORANGE", transparency=0.7, update=False) # O
    if face_v:     display.DisplayShape(face_v,     color="BLUE", transparency=0.7, update=False)   # V (side)
    if face_h:     display.DisplayShape(face_h,     color="CYAN", transparency=0.7, update=True)    # H
