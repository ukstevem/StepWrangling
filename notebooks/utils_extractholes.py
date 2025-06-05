from OCC.Core.gp import gp_Ax3, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

def mark_holes_in_local_cs(solid, local_cs: gp_Ax3):
    """
    Detects holes in `solid` (as before), builds a compound of spheres+axes in global coords,
    then transforms that entire compound into the user-supplied local coordinate system `local_cs`.
    """
    # 1) First get your global markers
    global_markers = mark_holes(solid)

    # 2) Build a transformation from the *global* default CS (gp_Ax3()) into your `local_cs`
    global_cs = gp_Ax3()  
    trsf = gp_Trsf()
    trsf.SetTransformation(global_cs, local_cs)

    # 3) Apply that transform to the whole compound in one go
    transformer = BRepBuilderAPI_Transform(global_markers, trsf, True)
    local_markers = transformer.Shape()

    return local_markers

# --- Usage example: ---
# Suppose you've defined your local origin & axes as:
#   local_origin = gp_Pnt( 10, 5, 0 )
#   local_x_dir  = gp_Dir( 0, 1, 0 )
#   local_y_dir  = gp_Dir( 1, 0, 0 )
#   local_cs     = gp_Ax3( local_origin, local_z_dir, local_x_dir )
#
# Then:
#   my_solid       = <your loaded TopoDS_Shape>
#   hole_markers   = mark_holes_in_local_cs(my_solid, local_cs)
#   viewer.DisplayShape(hole_markers)
