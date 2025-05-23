from OCC.Core.gp import gp_Pnt, gp_Vec
from utils_visualization import make_face

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

def draw_dstv_faces(display, center, he_X, he_Y, he_Z, xaxis, yaxis, zaxis, profile_type, offset=1.0):
    """
    Shades the DSTV U, V, and START (O) faces based on classified profile and aligned OBB axes.
    """
    from OCC.Core.gp import gp_Pnt

    offset = 10

    # Get DSTV face direction vectors based on profile
    face_axes = get_dstv_face_axes(profile_type, xaxis, yaxis, zaxis)

    # Compute origin of START face (center - Z)
    origin_o = [
        center.X() + face_axes["O"].Coord(1) * -he_Z,
        center.Y() + face_axes["O"].Coord(2) * -he_Z,
        center.Z() + face_axes["O"].Coord(3) * -he_Z
    ]

    # Compute U and V face origins offset from O face origin
    origin_u = [
        origin_o[i] + face_axes["U"].Coord(i + 1) * he_X for i in range(3)
    ]
    origin_v = [
        origin_o[i] + face_axes["V"].Coord(i + 1) * he_Y for i in range(3)
    ]

    # Create and display the faces
    face_start = make_face(origin_o, xaxis, yaxis, he_X, he_Y, face_axes["O"], offset)
    face_u = make_face(origin_u, yaxis, zaxis, he_Y, he_Z, face_axes["U"], offset)
    face_v = make_face(origin_v, xaxis, zaxis, he_X, he_Z, face_axes["V"], offset)

    display.DisplayShape(face_start, color="RED", transparency=0.7, update=False)   # START face
    display.DisplayShape(face_u, color="GREEN", transparency=0.7, update=False)     # U face
    display.DisplayShape(face_v, color="BLUE", transparency=0.7, update=True)       # V face
