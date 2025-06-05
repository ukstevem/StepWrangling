from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Circ
from utils_visualization import make_face
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_Circle
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

import numpy as np
import pandas as pd

def compute_dstv_origins_and_axes_from_section(ordered_corners, xaxis, yaxis, zaxis):
    """
    Compute DSTV origins at standard corner locations from a section face.
    
    ordered_corners: list of 4 gp_Pnt, ordered in [origin_U, origin_V, origin_O, opposite]
    xaxis, yaxis, zaxis: gp_Dir defining local axes
    """
    if len(ordered_corners) != 4:
        raise ValueError("Expected 4 ordered corners (U, V, O, opposite)")

    origin_u, origin_v, origin_o, _ = ordered_corners

    vec_x = gp_Vec(xaxis)
    vec_y = gp_Vec(yaxis)
    vec_z = gp_Vec(zaxis)

    face_axes = {
        'U': (gp_Vec(-zaxis.X(), -zaxis.Y(), -zaxis.Z()), vec_x.Reversed()),  # Length and horizontal
        'V': (gp_Vec(-zaxis.X(), -zaxis.Y(), -zaxis.Z()), vec_y),             # Length and vertical
        'O': (gp_Vec(-zaxis.X(), -zaxis.Y(), -zaxis.Z()), vec_x)              # Length and outside
    }

    origins = {
        'U': origin_u,
        'V': origin_v,
        'O': origin_o
    }

    return origins, face_axes

def color_solid_by_profile(display, solid, profile_type, update=True):
    """
    Colors a solid based on its DSTV profile type.

    Parameters:
        display       - pythonOCC display object
        solid         - TopoDS_Solid to color
        profile_type  - str, one of: "I", "U", "L", "Unknown"
        update        - whether to update the view immediately
    """
    color_map = {
        "I": "YELLOW",
        "U": "ORANGE",
        "L": "CYAN",
        "Unknown": "RED"
    }
    color = color_map.get(profile_type, "WHITE")
    display.DisplayShape(solid, color=color, update=update)

from OCC.Core.gp import gp_Vec, gp_Pnt

def compute_dstv_origins_and_axes_from_section(corners, xaxis, yaxis, zaxis):
    """
    Compute DSTV origins and face axes based on 4 ordered section corners.

    Parameters:
        corners - List of 4 gp_Pnt, ordered bottom-left to top-right.
        xaxis   - gp_Dir or gp_Vec for local web direction (U axis).
        yaxis   - gp_Dir or gp_Vec for local flange direction (V axis).
        zaxis   - gp_Dir or gp_Vec for length direction (O axis).

    Returns:
        origins    - Dict with 'V', 'U', 'O' corner points (gp_Pnt).
        face_axes  - Dict of 2D axis frames per face ('V', 'U', 'O').
    """
    if len(corners) != 4:
        raise ValueError("Expected exactly 4 ordered corners")

    # Unpack corners assuming bottom-left winding
    bl, br, tl, tr = corners

    # Assign DSTV origins (conventionally placed at the far end of each face)
    origin_v = br  # Web face (usually vertical web)
    origin_o = tl  # Outside face (usually top flange)
    origin_u = bl  # Underside face (bottom flange)

    # Axes per face: (X axis direction, Y axis direction)
    face_axes = {
        'V': (gp_Vec(-zaxis.X(), -zaxis.Y(), -zaxis.Z()), gp_Vec(yaxis)),
        'O': (gp_Vec(-zaxis.X(), -zaxis.Y(), -zaxis.Z()), gp_Vec(xaxis)),
        'U': (gp_Vec(-zaxis.X(), -zaxis.Y(), -zaxis.Z()), gp_Vec(-xaxis.X(), -xaxis.Y(), -xaxis.Z()))
    }

    return {
        'V': origin_v,
        'O': origin_o,
        'U': origin_u
    }, face_axes

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

def visualize_dstv_origins(display, origins, face_axes, axis_length=40):
    """
    Visualize DSTV origin points and associated axis frames.

    Parameters:
        display     - OCC display context
        origins     - Dict of face -> gp_Pnt ('V', 'O', 'U')
        face_axes   - Dict of face -> (gp_Vec, gp_Vec) representing (X, Y) axes
        axis_length - Length of axis arrows
    """
    color_map = {
        'V': 'YELLOW',
        'O': 'CYAN',
        'U': 'MAGENTA'
    }

    for face, origin in origins.items():
        color = color_map.get(face, 'WHITE')
        display.DisplayShape(origin, color=color, update=False)

        x_axis, y_axis = face_axes[face]
        pnt_origin = origin

        # Draw X axis
        end_x = gp_Pnt(
            pnt_origin.X() + x_axis.X() * axis_length,
            pnt_origin.Y() + x_axis.Y() * axis_length,
            pnt_origin.Z() + x_axis.Z() * axis_length
        )
        edge_x = BRepBuilderAPI_MakeEdge(pnt_origin, end_x).Edge()
        display.DisplayShape(edge_x, color='RED', update=False)

        # Draw Y axis
        end_y = gp_Pnt(
            pnt_origin.X() + y_axis.X() * axis_length,
            pnt_origin.Y() + y_axis.Y() * axis_length,
            pnt_origin.Z() + y_axis.Z() * axis_length
        )
        edge_y = BRepBuilderAPI_MakeEdge(pnt_origin, end_y).Edge()
        display.DisplayShape(edge_y, color='GREEN', update=False)

    display.FitAll()

def get_dstv_face_axes(profile_type, xaxis, yaxis, zaxis):
    """
    Maps OBB axes to DSTV face axes based on profile type.

    Returns a dictionary of vectors: {'U': ..., 'V': ..., 'O': ...}
    """
    # Default mapping for most profiles
    axis_u = -xaxis  # U face (side or bottom)
    axis_v = yaxis  # V face (top)
    axis_o = -zaxis  # O face (start/back)

    # Special case for L profiles (angle steel)
    if profile_type == "L":
        axis_u = xaxis  # L leg pointing +X = U face
        axis_v = yaxis  # vertical leg = V face

    return {"U": axis_u, "V": axis_v, "O": axis_o}

# def draw_dstv_faces(display,
#     center, he_X, he_Y, he_Z,
#     xaxis, yaxis, zaxis,
#     profile_type,
#     offset=10.0):
#     """
#     Shades the DSTV O, U, and V faces and also returns their origin/normal dict.

#     Returns
#     -------
#     plane_defs : dict[str, dict]
#         e.g. {
#           'O': {'origin': (x,y,z), 'normal': (nx,ny,nz)},
#           'U': {...}, 'V': {...}
#         }
#     """
#     from OCC.Core.gp import gp_Pnt

#     # 1) figure out which axes are which for this profile
#     face_axes = get_dstv_face_axes(profile_type, xaxis, yaxis, zaxis)

#     # 2) compute O-face origin at center − Z * he_Z
#     origin_o = [
#         center.X() + face_axes["O"].Coord(i+1) * -he_Z
#         for i in range(3)
#     ]
#     # then U and V, offset from O
#     origin_u = [origin_o[i] + face_axes["U"].Coord(i+1) * he_X
#                 for i in range(3)]
#     origin_v = [origin_o[i] + face_axes["V"].Coord(i+1) * he_Y
#                 for i in range(3)]

#     # 3) build your plane_defs dict
#     plane_defs = {}
#     for pid, origin, axis in [
#         ('O', origin_o, face_axes['O']),
#         ('U', origin_u, face_axes['U']),
#         ('V', origin_v, face_axes['V']),
#     ]:
#         plane_defs[pid] = {
#             'origin': tuple(origin),
#             'normal': (
#                 axis.Coord(1),
#                 axis.Coord(2),
#                 axis.Coord(3)
#             )
#         }

#     # 4) actually create & draw the faces
#     face_start = make_face(origin_o, xaxis, yaxis, he_X, he_Y, face_axes["O"], offset)
#     face_u     = make_face(origin_u, yaxis, zaxis, he_Y, he_Z, face_axes["U"], offset)
#     face_v     = make_face(origin_v, xaxis, zaxis, he_X, he_Z, face_axes["V"], offset)

#     display.DisplayShape(face_start, color="RED",   transparency=0.7, update=False)
#     display.DisplayShape(face_u,     color="GREEN", transparency=0.7, update=False)
#     display.DisplayShape(face_v,     color="BLUE",  transparency=0.7, update=True)

#     # return the dict so you can feed it straight into project_holes_to_planes()
#     return plane_defs


def get_circular_edges(solid: TopoDS_Shape):
    explorer = TopExp_Explorer(solid, TopAbs_EDGE)
    circular_edges = []
    while explorer.More():
        edge = explorer.Current()
        curve_adaptor = BRepAdaptor_Curve(edge)
        if curve_adaptor.GetType() == Geom_Circle:
            circle = curve_adaptor.Circle()
            circular_edges.append((edge, circle))
        explorer.Next()
    return circular_edges



def project_point_to_plane_axes(point, origin, axis_x, axis_y):
    """
    Projects a 3D point onto a plane defined by origin and orthonormal (axis_x, axis_y).
    Returns local (x, y) coordinates.
    """
    vec = np.array([point.X(), point.Y(), point.Z()]) - np.array([origin.X(), origin.Y(), origin.Z()])
    x_local = np.dot(vec, np.array([axis_x.X(), axis_x.Y(), axis_x.Z()]))
    y_local = np.dot(vec, np.array([axis_y.X(), axis_y.Y(), axis_y.Z()]))
    return x_local, y_local

import numpy as np
import pandas as pd
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder

def project_cylinders_onto_faces(solid, obb_data, face_data, section_type, threshold=0.98, max_distance=0.8):
    def transform_point_to_local(point, obb_data):
        origin = np.array([obb_data['center'].X(), obb_data['center'].Y(), obb_data['center'].Z()])
        x = np.array([obb_data['xaxis'].X(), obb_data['xaxis'].Y(), obb_data['xaxis'].Z()])
        y = np.array([obb_data['yaxis'].X(), obb_data['yaxis'].Y(), obb_data['yaxis'].Z()])
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        R = np.vstack([x, y, z]).T
        return np.dot(R.T, point - origin)

    def transform_vector_to_local(vec, obb_data):
        x = np.array([obb_data['xaxis'].X(), obb_data['xaxis'].Y(), obb_data['xaxis'].Z()])
        y = np.array([obb_data['yaxis'].X(), obb_data['yaxis'].Y(), obb_data['yaxis'].Z()])
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        R = np.vstack([x, y, z]).T
        return np.dot(R.T, vec)

    def line_plane_intersection(p0, v, p_face, n):
        denom = np.dot(v, n)
        if np.abs(denom) < 1e-6:
            return None
        t = np.dot(p_face - p0, n) / denom
        return p0 + t * v

    DSTV_FACE_MAP = {
        'I': ['O', 'U', 'V'],
        'U': ['H', 'U', 'O'],
        'L': ['H', 'U'],
    }

    face_codes = DSTV_FACE_MAP.get(section_type.upper(), [])
    if not face_codes:
        raise ValueError(f"Unsupported section type: {section_type}")

    rows = []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)

    while explorer.More():
        face = explorer.Current()
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() == GeomAbs_Cylinder:
            cyl = surf.Cylinder()

            center_global = np.array([
                cyl.Axis().Location().X(),
                cyl.Axis().Location().Y(),
                cyl.Axis().Location().Z()
            ])
            dir_global = np.array([
                cyl.Axis().Direction().X(),
                cyl.Axis().Direction().Y(),
                cyl.Axis().Direction().Z()
            ])

            center_local = transform_point_to_local(center_global, obb_data)
            dir_local = transform_vector_to_local(dir_global, obb_data)
            diameter = 2 * cyl.Radius()

            for face_code in face_codes:
                suffix = face_code.lower()
                try:
                    origin = face_data[f'origin_{suffix}']
                    normal = face_data[f'normal_{suffix}']
                except KeyError:
                    continue

                face_origin = np.array([origin.X(), origin.Y(), origin.Z()])
                face_normal = np.array([normal.X(), normal.Y(), normal.Z()])
                face_origin_local = transform_point_to_local(face_origin, obb_data)
                face_normal_local = transform_vector_to_local(face_normal, obb_data)

                # alignment = np.abs(np.dot(dir_local, face_normal_local))
                # if alignment < threshold:
                #     continue

                alignment = np.dot(dir_local, face_normal_local)
                # only holes whose axis points “outward” from the face
                if alignment < threshold:
                    continue

                intersection = line_plane_intersection(center_local, dir_local, face_origin_local, face_normal_local)
                if intersection is None:
                    continue

                distance = np.abs(np.dot((intersection - face_origin_local), face_normal_local))
                if distance > max_distance * max(obb_data['he_X'], obb_data['he_Y'], obb_data['he_Z']):
                    continue

                rows.append({
                    "Face": face_code,
                    "X": intersection[0],
                    "Y": intersection[1],
                    "Z": intersection[2],
                    "Diameter (mm)": diameter,
                    "Axis X": dir_local[0],
                    "Axis Y": dir_local[1],
                    "Axis Z": dir_local[2],
                })
                
        explorer.Next()

    return pd.DataFrame(rows)


import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder

def assign_holes_to_faces_from_solid(solid, obb_data, face_data, section_type, tol=0.8):
    """
    Extracts cylinders from 'solid', maps them into OBB-local space,
    and assigns each to all DSTV faces it lies on (by signed plane distance).
    Returns a DataFrame with columns:
      ['Hole #', 'Face', 'X', 'Y', 'Z', 'Diameter (mm)', 'Axis X', 'Axis Y', 'Axis Z']
    """
    # Build local transform from obb_data
    origin = np.array([obb_data['center'].X(), obb_data['center'].Y(), obb_data['center'].Z()])
    x = np.array([obb_data['xaxis'].X(), obb_data['xaxis'].Y(), obb_data['xaxis'].Z()]); x /= np.linalg.norm(x)
    y = np.array([obb_data['yaxis'].X(), obb_data['yaxis'].Y(), obb_data['yaxis'].Z()]); y /= np.linalg.norm(y)
    z = np.cross(x, y); z /= np.linalg.norm(z)
    R = np.vstack([x, y, z]).T

    def to_local_point(pt):
        return R.T.dot(pt - origin)
    def to_local_vec(v):
        return R.T.dot(v)

    # DSTV face map
    DSTV_FACE_MAP = {'I':['O','U','V'], 'U':['H','U','O'], 'L':['H','U']}
    face_codes = DSTV_FACE_MAP.get(section_type.upper(), [])

    # Precompute local face planes
    local_faces = {}
    for code in face_codes:
        s = code.lower()
        og = face_data[f'origin_{s}']
        ng = face_data[f'normal_{s}']
        axg= face_data[f'axis_{s}_x']; ayg = face_data[f'axis_{s}_y']
        o_loc = to_local_point(np.array([og.X(), og.Y(), og.Z()]))
        n_loc = to_local_vec(np.array([ng.X(), ng.Y(), ng.Z()])); n_loc /= np.linalg.norm(n_loc)
        x_loc = to_local_vec(np.array([axg.X(), axg.Y(), axg.Z()])); x_loc /= np.linalg.norm(x_loc)
        y_loc = to_local_vec(np.array([ayg.X(), ayg.Y(), ayg.Z()])); y_loc /= np.linalg.norm(y_loc)
        local_faces[code] = {'origin':o_loc, 'normal':n_loc, 'axis_x':x_loc, 'axis_y':y_loc}

    rows = []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    hole_id = 1

    while explorer.More():
        face = explorer.Current()
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            cg = np.array([cyl.Axis().Location().X(),
                           cyl.Axis().Location().Y(),
                           cyl.Axis().Location().Z()])
            dg = np.array([cyl.Axis().Direction().X(),
                           cyl.Axis().Direction().Y(),
                           cyl.Axis().Direction().Z()])
            center = to_local_point(cg)
            axis   = to_local_vec(dg); axis /= np.linalg.norm(axis)
            diam   = 2 * cyl.Radius()

            for code, f in local_faces.items():
                o = f['origin']; n = f['normal']
                # signed distance from face plane
                dist = np.dot(center - o, n)
                he = obb_data['he_Y'] if code in ['U','O'] else obb_data['he_X']
                if abs(dist) > tol * he:
                    continue
                # project into local face X/Y
                dx = np.dot(center - o, f['axis_x'])
                dy = np.dot(center - o, f['axis_y'])
                rows.append({
                    'Hole #': hole_id,
                    'Face': code,
                    'X': dx, 'Y': dy, 'Z': dist,
                    'Diameter (mm)': diam,
                    'Axis X': axis[0],
                    'Axis Y': axis[1],
                    'Axis Z': axis[2],
                })
            hole_id += 1
        explorer.Next()

    return pd.DataFrame(rows)

