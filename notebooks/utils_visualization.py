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

import numpy as np

# Helper: draw an axis arrow
def draw_axis_arrow(display, origin, direction, scale=100, color='BLACK'):
    end = gp_Pnt(
        origin.X() + direction.X() * scale,
        origin.Y() + direction.Y() * scale,
        origin.Z() + direction.Z() * scale,
    )
    edge = BRepBuilderAPI_MakeEdge(origin, end).Edge()
    display.DisplayShape(edge, color=color, update=False)


# def draw_plane(display, origin, normal, size=100, color="MAGENTA"):
#     """
#     Draw a planar square face centered at `origin` with normal `normal`.

#     Parameters:
#         display - The OCC display context.
#         origin  - gp_Pnt
#         normal  - gp_Dir or gp_Vec
#         size    - float, size of the plane (length of one edge)
#         color   - str, display color
#     """
#     # Convert normal to vector
#     n = gp_Vec(normal) if hasattr(normal, 'XYZ') else gp_Vec(normal.X(), normal.Y(), normal.Z())

#     # Create U, V vectors orthogonal to the normal
#     u = n.Crossed(gp_Vec(0, 0, 1))
#     if u.Magnitude() < 1e-6:
#         u = gp_Vec(1, 0, 0)
#     v = n.Crossed(u)
#     u.Normalize()
#     v.Normalize()

#     u *= size / 2
#     v *= size / 2

#     o = gp_Vec(origin.X(), origin.Y(), origin.Z())
#     corners = [
#         gp_Pnt(*(o + u + v)),
#         gp_Pnt(*(o - u + v)),
#         gp_Pnt(*(o - u - v)),
#         gp_Pnt(*(o + u - v))
#     ]

#     # Create wire from corners
#     edges = [BRepBuilderAPI_MakeEdge(corners[i], corners[(i + 1) % 4]).Edge() for i in range(4)]
#     wire_builder = BRepBuilderAPI_MakeWire()
#     for edge in edges:
#         wire_builder.Add(edge)
    
#     face = BRepBuilderAPI_MakeFace(wire_builder.Wire()).Face()
#     display.DisplayShape(face, color=color, update=False)


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


# def visualize_dstv_faces(display, obb_data, axes, offset=10.0):
#     """
#     Visualizes START, U, O, V, H faces per DSTV definition.
#     Returns face_data dict mapping:
#       pid -> {
#         'shape': TopoDS_Face,
#         'origin': (x,y,z),
#         'normal': (nx,ny,nz)
#       }
#     """
#     from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln
#     from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

#     def make_plane_face(origin, normal_vec, x_vec, he1, he2):
#         pnt   = gp_Pnt(*origin)
#         ax3   = gp_Ax3(pnt, gp_Dir(*normal_vec), gp_Dir(*x_vec))
#         plane = gp_Pln(ax3)
#         return BRepBuilderAPI_MakeFace(plane, -he1, he1, -he2, he2).Shape()

#     def translate_point(center, vec, dist):
#         return (
#             center.X() + vec.Coord(1) * dist,
#             center.Y() + vec.Coord(2) * dist,
#             center.Z() + vec.Coord(3) * dist
#         )

#     # unpack OBB
#     center = obb_data["center"]
#     he_X, he_Y, he_Z = obb_data["he_X"], obb_data["he_Y"], obb_data["he_Z"]

#     # shorthand to extract gp_Dir coords
#     def dir_to_tuple(d: gp_Dir):
#         return (d.Coord(1), d.Coord(2), d.Coord(3))

#     face_data = {}

#     # --- START face (we'll call its pid 'start') ---
#     origin_start = translate_point(center, axes["z"], -(he_Z + offset))
#     normal_start = dir_to_tuple(axes["z"])
#     # choose your local X for the face’s U-direction:
#     x_start = dir_to_tuple(axes["x"])
#     face_start = make_plane_face(origin_start, normal_start, x_start, he_X, he_Y)
#     display.DisplayShape(face_start, color="RED", transparency=0.7, update=False)
#     face_data['start'] = {
#         'shape':  face_start,
#         'origin': origin_start,
#         'normal': normal_start
#     }

#     # --- U face ---
#     origin_u = translate_point(center, axes["y"], +(he_Y + offset))
#     normal_u = dir_to_tuple(axes["y"])
#     x_u      = dir_to_tuple(axes["x"])
#     face_u   = make_plane_face(origin_u, normal_u, x_u, he_X, he_Z)
#     display.DisplayShape(face_u, color="GREEN", transparency=0.7, update=False)
#     face_data['U'] = {
#         'shape':  face_u,
#         'origin': origin_u,
#         'normal': normal_u
#     }

#     # --- O face (bottom) ---
#     origin_o = translate_point(center, axes["y"], -(he_Y + offset))
#     # bottom face flips the Y-axis
#     normal_o = tuple(-c for c in dir_to_tuple(axes["y"]))
#     x_o      = dir_to_tuple(axes["x"])
#     face_o   = make_plane_face(origin_o, normal_o, x_o, he_X, he_Z)
#     display.DisplayShape(face_o, color="ORANGE", transparency=0.7, update=False)
#     face_data['O'] = {
#         'shape':  face_o,
#         'origin': origin_o,
#         'normal': normal_o
#     }

#     # --- V face (side) ---
#     origin_v = translate_point(center, axes["x"], +(he_X + offset))
#     normal_v = dir_to_tuple(axes["x"])
#     x_v      = dir_to_tuple(axes["y"])
#     face_v   = make_plane_face(origin_v, normal_v, x_v, he_Y, he_Z)
#     display.DisplayShape(face_v, color="BLUE", transparency=0.7, update=False)
#     face_data['V'] = {
#         'shape':  face_v,
#         'origin': origin_v,
#         'normal': normal_v
#     }

#     # --- H face (opposite side) ---
#     origin_h = translate_point(center, axes["x"], -(he_X + offset))
#     normal_h = tuple(-c for c in dir_to_tuple(axes["x"]))
#     x_h      = dir_to_tuple(axes["y"])
#     face_h   = make_plane_face(origin_h, normal_h, x_h, he_Y, he_Z)
#     display.DisplayShape(face_h, color="CYAN", transparency=0.7, update=True)
#     face_data['H'] = {
#         'shape':  face_h,
#         'origin': origin_h,
#         'normal': normal_h
#     }

#     return face_data

def visualize_dstv_faces(display, obb_data, axes, local_cs, offset=10.0):
    """
    Draws and returns DSTV faces: start, O, U, V, H.
    Faces are displayed in world-space, and their origins/normals returned in local_cs.

    Returns:
      face_data: dict[label] = {'shape':TopoDS_Face, 'origin':(x,y,z), 'normal':(nx,ny,nz)}
    """
    from OCC.Core.gp                  import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Trsf, gp_Vec
    from OCC.Core.BRepBuilderAPI      import BRepBuilderAPI_MakeFace

    # world -> local transformer
    world_to_local = gp_Trsf()
    world_to_local.SetTransformation(gp_Ax3(), local_cs)

    center = obb_data["center"]
    he_X, he_Y, he_Z = obb_data["he_X"], obb_data["he_Y"], obb_data["he_Z"]

    def translate_point(c, vec, dist):
        return (
            c.X() + vec.Coord(1)*dist,
            c.Y() + vec.Coord(2)*dist,
            c.Z() + vec.Coord(3)*dist
        )

    def make_plane(origin, normal_vec, x_vec, h1, h2):
        pnt = gp_Pnt(*origin)
        ax3 = gp_Ax3(pnt, gp_Dir(*normal_vec), gp_Dir(*x_vec))
        pln = gp_Pln(ax3)
        return BRepBuilderAPI_MakeFace(pln, -h1, h1, -h2, h2).Shape()

    def dir_tuple(d):
        return (d.Coord(1), d.Coord(2), d.Coord(3))

    # Define faces: start (front), O (bottom), U (top), V (side), H (opposite)
    definitions = [
        ('start', -(he_Z + offset), axes['z'], axes['x'], he_X, he_Y, 'RED'),
        ('O',     -(he_Y + offset), axes['y'], axes['x'], he_X, he_Z, 'ORANGE'),
        ('U',     +(he_Y + offset), axes['y'], axes['x'], he_X, he_Z, 'GREEN'),
        ('V',     +(he_X + offset), axes['x'], axes['y'], he_Y, he_Z, 'BLUE'),
        ('H',     -(he_X + offset), axes['x'], axes['y'], he_Y, he_Z, 'CYAN')
    ]

    face_data = {}
    for label, dist_sign, norm_axis, x_axis, h1, h2, color in definitions:
        # world origin
        origin_world = translate_point(center, norm_axis, dist_sign)
        normal_world = dir_tuple(norm_axis)
        xvec_world   = dir_tuple(x_axis)
        face_shape   = make_plane(origin_world, normal_world, xvec_world, h1, h2)
        display.DisplayShape(face_shape, color=color, transparency=0.7,
                             update=(label=='H'))

        # local origin
        wp = gp_Pnt(*origin_world)
        lp = wp.Transformed(world_to_local)
        origin_local = (lp.Coord(1), lp.Coord(2), lp.Coord(3))

        # local normal
        wv = gp_Vec(*normal_world)
        wv.Transform(world_to_local)
        wv.Normalize()
        normal_local = (wv.Coord(1), wv.Coord(2), wv.Coord(3))

        face_data[label] = {
            'shape':  face_shape,
            'origin': origin_local,
            'normal': normal_local
        }

    return face_data


from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

import numpy as np
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

def visualize_projected_holes(display, df_holes, obb_data,
                              sphere_radius=5.0, axis_length=100.0):
    """
    Overlays on `display`:
      • A green sphere at each hole intersection point.
      • A white line indicating the hole’s axis (normal) direction.

    Parameters:
      display         – your active OCC display.
      df_holes        – DataFrame with X,Y,Z (local) & Axis X/Y/Z (local).
      obb_data        – dict with center, xaxis,yaxis,zaxis for OBB local→global.
      sphere_radius   – radius of the debug spheres (mm).
      axis_length     – length of the normal axis line (mm).
    """
    # Build local→global transform
    origin = np.array([obb_data['center'].X(),
                       obb_data['center'].Y(),
                       obb_data['center'].Z()])
    x = np.array([obb_data['xaxis'].X(), obb_data['xaxis'].Y(), obb_data['xaxis'].Z()])
    y = np.array([obb_data['yaxis'].X(), obb_data['yaxis'].Y(), obb_data['yaxis'].Z()])
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    R = np.vstack([x, y, z]).T  # columns = local axes

    def local_to_global(pt_local):
        return R.dot(pt_local) + origin

    for _, row in df_holes.iterrows():
        # Local coords
        pt_loc = np.array([row["X"], row["Y"], row["Z"]])
        n_loc  = np.array([row["Axis X"], row["Axis Y"], row["Axis Z"]])
        n_loc /= np.linalg.norm(n_loc)

        # Convert to global
        pt_glob  = local_to_global(pt_loc)
        end_glob = local_to_global(pt_loc + axis_length * n_loc)

        # Draw green sphere at the hole intersection
        sphere = BRepPrimAPI_MakeSphere(gp_Pnt(*pt_glob), sphere_radius).Shape()
        display.DisplayShape(sphere, color="GREEN", update=False)

        # Draw white line for the axis/normal
        edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*pt_glob),
                                       gp_Pnt(*end_glob)).Edge()
        display.DisplayShape(edge, color="WHITE", update=True)

    display.Repaint()

# Example usage:
# visualize_projected_holes(display, df_projected_holes, obb_data)


import numpy as np
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

def visualize_assigned_holes(display, df_assigned, obb_data, sphere_radius=5.0):
    """
    Overlays spheres at each assigned hole point on the active `display`.
    
    Parameters:
      display       – your active OCC display
      df_assigned   – DataFrame with columns ['Face','X','Y','Z', ...] in local coords
      obb_data      – dict with 'center','xaxis','yaxis','zaxis' for local→global transform
      sphere_radius – radius of debug spheres (mm)
    """
    # Build local→global transform
    origin = np.array([obb_data['center'].X(),
                       obb_data['center'].Y(),
                       obb_data['center'].Z()])
    x = np.array([obb_data['xaxis'].X(), obb_data['xaxis'].Y(), obb_data['xaxis'].Z()]); x /= np.linalg.norm(x)
    y = np.array([obb_data['yaxis'].X(), obb_data['yaxis'].Y(), obb_data['yaxis'].Z()]); y /= np.linalg.norm(y)
    z = np.cross(x, y); z /= np.linalg.norm(z)
    R = np.vstack([x, y, z]).T

    def local_to_global(pt_local):
        return R.dot(pt_local) + origin

    # Draw a sphere at each hole point
    for _, row in df_assigned.iterrows():
        pt_local = np.array([row["X"], row["Y"], row["Z"]])
        pt_global = local_to_global(pt_local)
        p = gp_Pnt(*pt_global)
        sph = BRepPrimAPI_MakeSphere(p, sphere_radius).Shape()
        display.DisplayShape(sph, color="MAGENTA", update=False)

    display.Repaint()

# Example:
# visualize_assigned_holes(display, df_assigned, obb_data)


import numpy as np
import pandas as pd
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Vec, gp_Dir

def section_holes_on_dstv_faces(solid, obb_data, face_data, section_type):
    """
    Section the solid with each DSTV face plane at the actual part surface,
    extract trimmed circular hole edges, and record both local and global coords.

    Returns DataFrame: ['Face','X','Y','Diameter (mm)','Xg','Yg','Zg'] 
    where X,Y are local face-plane coords and Xg,Yg,Zg are global intersection points.
    """
    # Build local→global rotation
    ctr = obb_data['center']
    origin = np.array([ctr.X(), ctr.Y(), ctr.Z()])
    x = np.array([obb_data['xaxis'].X(), obb_data['xaxis'].Y(), obb_data['xaxis'].Z()]); x/=np.linalg.norm(x)
    y = np.array([obb_data['yaxis'].X(), obb_data['yaxis'].Y(), obb_data['yaxis'].Z()]); y/=np.linalg.norm(y)
    z = np.cross(x, y); z/=np.linalg.norm(z)
    R = np.vstack([x, y, z]).T
    def to_local(pt):
        return R.T.dot(pt - origin)

    DSTV_FACE_MAP = {'I':['O','U','V'],'U':['H','U','O'],'L':['H','U']}
    face_codes = DSTV_FACE_MAP[section_type.upper()]

    rows = []
    for code in face_codes:
        s = code.lower()
        # half‐extent per face
        he = obb_data['he_Y'] if code in ['U','O'] else obb_data['he_X']
        # global normal and compute plane origin on part surface
        ng = face_data[f'normal_{s}']
        normal = gp_Dir(ng.X(), ng.Y(), ng.Z())
        sign = 1 if code in ['U','V'] else -1
        offset_vec = gp_Vec(normal) * (sign * he)
        plane_origin = gp_Pnt(ctr.X() + offset_vec.X(),
                              ctr.Y() + offset_vec.Y(),
                              ctr.Z() + offset_vec.Z())
        plane = gp_Pln(plane_origin, normal)

        sec = BRepAlgoAPI_Section(solid, plane)
        sec.ComputePCurveOn1(True)
        sec.Approximation(True)
        sec.Build()
        section_shape = sec.Shape()

        explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
        # global axes for projection
        axg = np.array([face_data[f'axis_{s}_x'].X(), 
                        face_data[f'axis_{s}_x'].Y(), 
                        face_data[f'axis_{s}_x'].Z()])
        ayg = np.array([face_data[f'axis_{s}_y'].X(), 
                        face_data[f'axis_{s}_y'].Y(), 
                        face_data[f'axis_{s}_y'].Z()])
        axl = R.T.dot(axg)
        ayl = R.T.dot(ayg)
        plane_origin_local = to_local(np.array([plane_origin.X(),
                                                plane_origin.Y(),
                                                plane_origin.Z()]))

        while explorer.More():
            edge = explorer.Current()
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Circle:
                circ = curve.Circle()
                cpt = circ.Location()
                cg = np.array([cpt.X(), cpt.Y(), cpt.Z()])
                center_local = to_local(cg)
                rel = center_local - plane_origin_local
                X = float(np.dot(rel, axl))
                Y = float(np.dot(rel, ayl))
                diam = 2 * circ.Radius()
                rows.append({
                    'Face': code,
                    'X': X, 'Y': Y,
                    'Diameter (mm)': float(diam),
                    'Xg': float(cg[0]),
                    'Yg': float(cg[1]),
                    'Zg': float(cg[2]),
                })
            explorer.Next()

    return pd.DataFrame(rows, columns=['Face','X','Y','Diameter (mm)','Xg','Yg','Zg'])


def visualize_holes_global(display, df, sphere_radius=5.0):
    """
    Overlays spheres at the global intersection points (Xg,Yg,Zg) on the existing display.
    """
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

    for _, row in df.iterrows():
        p = gp_Pnt(row['Xg'], row['Yg'], row['Zg'])
        sph = BRepPrimAPI_MakeSphere(p, sphere_radius).Shape()
        display.DisplayShape(sph, color="MAGENTA", update=False)

    display.Repaint()

# Example usage:
# df_holes_sectioned = section_holes_on_dstv_faces(solid, obb_data, face_data, profile_type)
# print(df_holes_sectioned)
# visualize_holes_global(display, df_holes_sectioned)

from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir

def make_hole_marker(center: gp_Pnt,
                     normal: gp_Dir,
                     radius: float,
                     axis_scale: float = 4.0) -> TopoDS_Compound:
    """
    Build a compound marker for one hole:
      - a sphere of given radius at `center`
      - a short axis line along `normal`
    """
    # 1) Compute half the desired axis length
    total_len = axis_scale * radius
    half_len  = total_len / 2.0

    # 2) Build a pure vector from the direction and scale it
    v = gp_Vec(normal.X(), normal.Y(), normal.Z())  # now vector of magnitude 1
    v.Scale(half_len)                               # in-place scaling → magnitude = half_len

    # 3) Generate the two end‐points as new translated Pnts
    p1 = center.Translated(v.Reversed())  # back half_len along normal
    p2 = center.Translated(v)             # forward half_len along normal

    # 4) Create the sphere
    sphere = BRepPrimAPI_MakeSphere(center, radius).Shape()

    # 5) Create the edge between p1 and p2
    edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()

    # 6) Pack both into a compound
    builder = BRep_Builder()
    comp    = TopoDS_Compound()
    builder.MakeCompound(comp)
    builder.Add(comp, sphere)
    builder.Add(comp, edge)

    return comp


from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

def show_hole_marker(display,
                     center,
                     normal,
                     radius,
                     axis_scale: float = 2.0,
                     color=(1.0, 0.0, 0.0),
                     width: float = 1.0,
                     transparency: float = 0.0):
    """
    Create a sphere+axis marker at (center, normal, radius), wrap it in AIS,
    display it in the given OCC-Qt `display`, style it, and return the AIS object.

    Parameters:
    -----------
    display         : your OCC.Core.AIS.AIS_InteractiveContext or the helper
                      that exposes .Context.Display and .Context.Redisplay
    center          : gp_Pnt
    normal          : gp_Dir
    radius          : float
    axis_scale      : float, multiple of radius for the axis length
    color           : tuple(r,g,b) in [0..1]
    width           : float, line width for the axis
    transparency    : float in [0..1], 0 = opaque, 1 = fully transparent

    Returns:
    --------
    ais_marker      : AIS_Shape
    """
    # 1) build the raw marker compound
    marker = make_hole_marker(center, normal, radius, axis_scale)

    # 2) wrap as AIS_Shape
    ais_marker = AIS_Shape(marker)

    # 3) display it
    display.Context.Display(ais_marker, True)

    # 4) style
    r, g, b = color
    ais_marker.SetColor(Quantity_Color(r, g, b, Quantity_TOC_RGB))
    ais_marker.SetWidth(width)
    if transparency > 0.0:
        ais_marker.SetTransparency(transparency)

    # 5) refresh with both presentation & display update
    display.Context.Redisplay(ais_marker, True, True)

    return ais_marker
