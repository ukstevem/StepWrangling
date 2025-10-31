# pipeline/vis_utils.py
from pathlib import Path
import math
import numpy as np

from OCC.Core.gp import gp_Ax3, gp_Ax2, gp_Pln, gp_Pnt, gp_Dir, gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve, GeomAbs_BezierCurve

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir, gp_Trsf

# --- OCC CG marker ------------------------------------------------------------
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

def add_cg_marker_occ(display, y_c: float, z_c: float, radius: float = 3.0):
    """
    Draw a small red sphere at the DSTV-local centroid (0, y_c, z_c) in the OCC viewer.
    Call this BEFORE saving the debug PNG.
    """
    if y_c is None or z_c is None:
        return  # nothing to draw
    p = gp_Pnt(0.0, float(y_c), float(z_c))  # X=0 slice; DSTV-local coords
    sph = BRepPrimAPI_MakeSphere(p, float(radius)).Shape()
    # Display in red; 'update=False' to batch draw calls; update right before saving
    display.DisplayShape(sph, update=False, color="RED")

def _trsf_world_to_local(ax3: gp_Ax3) -> gp_Trsf:
    """
    Build a gp_Trsf that takes world coordinates -> local coords of ax3.
    ax3 is the gp_Ax3 you use for DSTV frame (origin, Z, X).
    """
    world = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    t = gp_Trsf()
    t.SetDisplacement(world, ax3)  # both are gp_Ax3 now ✅
    return t



def _edge_polyline_local(edge, trsf: gp_Trsf, max_seg=60):
    """Sample an EDGE into points in local coords (after trsf). Returns Nx3 array."""
    crv = BRepAdaptor_Curve(edge)
    f, l = crv.FirstParameter(), crv.LastParameter()
    typ = crv.GetType()

    # choose samples based on curve type
    if typ in (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse):
        n = 40
    elif typ in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
        n = 60
    else:
        n = 50
    n = min(max_seg, n)

    ts = np.linspace(f, l, n)
    pts = []
    for t in ts:
        p = crv.Value(t)                  # world
        p_l = p.Transformed(trsf)         # local
        pts.append([p_l.X(), p_l.Y(), p_l.Z()])
    return np.asarray(pts, float)


def save_yz_section_debug_png(shape, dstv_frame: gp_Ax3, out_path: Path, *,
                              H: float = None, W: float = None,
                              title: str = None, figsize=(7,7), dpi=150,
                              cg_yz: tuple[float,float] | None = None):
    """
    Save a Y–Z plot (DSTV local) of the shape edges with axis arrows.
    - Y vertical (up), Z horizontal (right), origin at (0,0) in local frame.
    - If H/W are provided they’re drawn as guide ticks.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # transform shape to local
    t_loc = _trsf_world_to_local(dstv_frame)
    sh_loc = BRepBuilderAPI_Transform(shape, t_loc, True).Shape()

    # compute local bbox for autoscale
    box = Bnd_Box(); brepbndlib.Add(sh_loc, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    # we care about Y–Z
    y0, y1 = float(ymin), float(ymax)
    z0, z1 = float(zmin), float(zmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_title(title or "DSTV Local Y–Z")

    # plot edges
    exp = TopExp_Explorer(sh_loc, TopAbs_EDGE)
    for _ in iter(lambda: exp.More(), False):
        e = exp.Current()
        exp.Next()
        poly = _edge_polyline_local(e, gp_Trsf(), max_seg=60)  # already local; identity trsf
        if poly.size == 0:
            continue
        Y = poly[:,1]; Z = poly[:,2]
        ax.plot(Z, Y, linewidth=0.8, alpha=0.9, color="black")  # Z on x-axis, Y on y-axis

    # axis arrows (trihedron in Y–Z)
    # origin at (0,0)
    ax.arrow(0, 0, (W or (z1 - z0) * 0.25), 0, head_width=(H or (y1 - y0))*0.02, length_includes_head=True, color="blue")
    ax.text((W or (z1 - z0) * 0.27), 0, "Z", color="blue", va="center")
    ax.arrow(0, 0, 0, (H or (y1 - y0) * 0.25), head_width=(W or (z1 - z0))*0.02, length_includes_head=True, color="green")
    ax.text(0, (H or (y1 - y0) * 0.27), "Y", color="green", ha="center")

    # guides for H/W if provided
    if H is not None:
        ax.axhline(0, color="#88bb88", lw=0.6)
        ax.axhline(H, color="#88bb88", lw=0.6, ls="--")
        ax.text((z0+z1)/2, H, f" H={H:.1f}", color="#558855", ha="center", va="bottom", fontsize=8)
    if W is not None:
        ax.axvline(0, color="#8888bb", lw=0.6)
        ax.axvline(W, color="#8888bb", lw=0.6, ls="--")
        ax.text(W, (y0+y1)/2, f"W={W:.1f}", color="#555588", va="center", ha="left", fontsize=8)

     # CG marker (section centroid in local Y–Z)
    if cg_yz is not None:
        y_c, z_c = cg_yz
        ax.plot(z_c, y_c, marker='o', markersize=6, color='red')
        ax.text(z_c, y_c, "  CG", color='red', va='center')

    # styling
    padY = 0.08 * max(1.0, abs(y1 - y0))
    padZ = 0.08 * max(1.0, abs(z1 - z0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(0, z0) - padZ, max(W if W is not None else z1, z1) + padZ)
    ax.set_ylim(min(0, y0) - padY, max(H if H is not None else y1, y1) + padY)
    ax.set_xlabel("Z (mm) — flange direction")
    ax.set_ylabel("Y (mm) — height")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

from OCC.Core.gp import gp_Pnt
from OCC.Display.OCCViewer import rgb_color
