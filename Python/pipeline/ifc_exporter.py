# IFC exporter — robust, styled, with adaptive+welded meshing
# - No ifcopenshell.api helpers (works on minimal builds)
# - No ifcopenshell.util.placement usage
# - Resolves MEM-####[_native].step/.stp from handoff/parts (and nearby dirs)
# - Adaptive, welded triangulation to keep counts sane
# - Direct per-instance tessellation on Body subcontext + normals + simple style

import json, math
from pathlib import Path
from typing import List, Tuple
import ifcopenshell
from ifcopenshell import guid as _ifc_guid

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.TopologyUtils import TopologyExplorer

# ---------------------
# Logging
# ---------------------

def _log(msg: str):
    print(f"[ifc_exporter] {msg}")

# ---------------------
# Geometry loading
# ---------------------

def _load_shape_from_path(path: Path):
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".step", ".stp"):
        rdr = STEPControl_Reader()
        status = rdr.ReadFile(str(path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP read failed: {path}")
        rdr.TransferRoots()
        return rdr.OneShape()
    if suf == ".brep":
        from OCC.Core.TopoDS import TopoDS_Shape
        sh = TopoDS_Shape()
        ok = breptools_Read(sh, str(path), None)
        if not ok:
            raise RuntimeError(f"BREP read failed: {path}")
        return sh
    raise ValueError(f"Unsupported geometry file: {path}")

# ---------------------
# Meshing (adaptive + welded)
# ---------------------

def _tri_arrays_from_shape(shape,
                           lin_defl: float | None = None,
                           ang_deg: float = 20.0,
                           rel: bool = True,
                           in_parallel: bool = True,
                           weld_tol_mm: float = 0.10
                           ) -> Tuple[List[Tuple[float,float,float]], List[Tuple[int,int,int]]]:
    """
    Adaptive triangulation with vertex welding across faces.
    FIXES:
      - Applies each face's TopLoc_Location transform to node coords
      - Flips triangle winding for TopAbs_REVERSED faces
    Returns (V, I) with 0-based indices.
    """
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.BRep import BRep_Tool
    from OCC.Extend.TopologyUtils import TopologyExplorer

    bb = Bnd_Box(); brepbndlib.Add(shape, bb, True)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    diag = max(((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2) ** 0.5, 1e-6)

    if lin_defl is None:
        # Coarse but clean default; good for mm models with holes
        lin_defl = max(0.5, min(2.5, diag / 250.0))

    BRepMesh_IncrementalMesh(shape, float(lin_defl), bool(rel), math.radians(float(ang_deg)), bool(in_parallel))

    raw_V: List[Tuple[float,float,float]] = []
    raw_I: List[Tuple[int,int,int]] = []
    base = 0

    topo = TopologyExplorer(shape)
    for face in topo.faces():
        # Get triangulation and its location (transform)
        loc = face.Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if not tri:
            continue

        # Apply location transform to nodes
        # NOTE: tri.Node(i) is a gp_Pnt in 'local' coords; transform to 3D
        n_nodes = tri.NbNodes()
        trsf = loc.Transformation()
        for i in range(1, n_nodes+1):
            p = tri.Node(i).Transformed(trsf)
            raw_V.append((float(p.X()), float(p.Y()), float(p.Z())))

        # Triangles (flip winding for reversed faces)
        is_reversed = (face.Orientation() == TopAbs_REVERSED)
        for i in range(1, tri.NbTriangles()+1):
            a, b, c = tri.Triangle(i).Get()
            if is_reversed:
                # swap to flip winding
                raw_I.append((base + a - 1, base + c - 1, base + b - 1))
            else:
                raw_I.append((base + a - 1, base + b - 1, base + c - 1))

        base += n_nodes

    if not raw_V or not raw_I:
        raise RuntimeError("No triangles generated; try increasing lin_defl or ang_deg.")

    # ---- Weld with simple spatial quantization (reduces seams/dupes)
    q = float(max(weld_tol_mm, 1e-6))
    def qkey(pt): return (int(round(pt[0]/q)), int(round(pt[1]/q)), int(round(pt[2]/q)))

    remap, welded_V, key_to_new = {}, [], {}
    for idx, v in enumerate(raw_V):
        key = qkey(v)
        widx = key_to_new.get(key)
        if widx is None:
            widx = len(welded_V)
            welded_V.append(v)
            key_to_new[key] = widx
        remap[idx] = widx

    welded_I = [(remap[i], remap[j], remap[k]) for (i, j, k) in raw_I]
    clean_I = [(i, j, k) for (i, j, k) in welded_I if (i != j and j != k and i != k)]
    if not clean_I:
        raise RuntimeError("All triangles collapsed after welding; decrease weld_tol_mm.")

    return welded_V, clean_I

# ---------------------
# IFC boilerplate & placement utils
# ---------------------

def _new_ifc(schema: str = "IFC4", units: str = "MILLIMETRE"):
    from ifcopenshell.guid import new as new_guid
    m = ifcopenshell.file(schema=schema)

    project = m.create_entity("IfcProject", GlobalId=new_guid(), Name="Project")
    m.add(project)

    # Units
    if str(units).upper().startswith("MM"):
        length_unit = m.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Prefix="MILLI", Name="METRE")
    else:
        length_unit = m.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    angle_unit  = m.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
    solid_angle = m.create_entity("IfcSIUnit", UnitType="SOLIDANGLEUNIT", Name="STERADIAN")
    time_unit   = m.create_entity("IfcSIUnit", UnitType="TIMEUNIT", Name="SECOND")
    project.UnitsInContext = m.create_entity("IfcUnitAssignment", Units=[length_unit, angle_unit, solid_angle, time_unit])

    # Context + subcontext
    origin = m.create_entity("IfcCartesianPoint", Coordinates=(0.0,0.0,0.0))
    z_dir  = m.create_entity("IfcDirection", DirectionRatios=(0.0,0.0,1.0))
    x_dir  = m.create_entity("IfcDirection", DirectionRatios=(1.0,0.0,0.0))
    wcs    = m.create_entity("IfcAxis2Placement3D", Location=origin, Axis=z_dir, RefDirection=x_dir)

    context = m.create_entity("IfcGeometricRepresentationContext",
                              ContextIdentifier="Model", ContextType="Model",
                              CoordinateSpaceDimension=3, Precision=1e-5,
                              WorldCoordinateSystem=wcs)
    project.RepresentationContexts = [context]

    subctx = m.create_entity("IfcGeometricRepresentationSubContext",
                             ContextIdentifier="Body", ContextType="Model",
                             TargetView="MODEL_VIEW", ParentContext=context)

    # Spatial tree
    def _lp(rel_to=None, x=0,y=0,z=0):
        p = m.create_entity("IfcCartesianPoint", Coordinates=(float(x),float(y),float(z)))
        a = m.create_entity("IfcAxis2Placement3D", Location=p)
        return m.create_entity("IfcLocalPlacement", PlacementRelTo=rel_to, RelativePlacement=a)

    from ifcopenshell.guid import new as guid
    site_pl = _lp(None); bldg_pl = _lp(site_pl); storey_pl = _lp(bldg_pl)
    site   = m.create_entity("IfcSite", GlobalId=guid(), Name="Site", ObjectPlacement=site_pl); m.add(site)
    bldg   = m.create_entity("IfcBuilding", GlobalId=guid(), Name="Building", ObjectPlacement=bldg_pl); m.add(bldg)
    storey = m.create_entity("IfcBuildingStorey", GlobalId=guid(), Name="Level 0", Elevation=0.0, ObjectPlacement=storey_pl); m.add(storey)

    m.add(m.create_entity("IfcRelAggregates", GlobalId=guid(), RelatingObject=project, RelatedObjects=[site]))
    m.add(m.create_entity("IfcRelAggregates", GlobalId=guid(), RelatingObject=site,    RelatedObjects=[bldg]))
    m.add(m.create_entity("IfcRelAggregates", GlobalId=guid(), RelatingObject=bldg,    RelatedObjects=[storey]))

    return m, subctx, storey


def _axis2placement3d(model, origin, xdir, zdir):
    p = model.create_entity("IfcCartesianPoint", Coordinates=tuple(origin))
    x = model.create_entity("IfcDirection", DirectionRatios=tuple(xdir))
    z = model.create_entity("IfcDirection", DirectionRatios=tuple(zdir))
    return model.create_entity("IfcAxis2Placement3D", Location=p, Axis=z, RefDirection=x)


def _matrix4_to_axes(M):
    x = [float(M[0][0]), float(M[1][0]), float(M[2][0])]
    y = [float(M[0][1]), float(M[1][1]), float(M[2][1])]
    z = [float(M[0][2]), float(M[1][2]), float(M[2][2])]
    t = [float(M[0][3]), float(M[1][3]), float(M[2][3])]
    def nrm(v):
        L = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
        return [v[0]/L, v[1]/L, v[2]/L]
    return t, nrm(x), nrm(z)


def _local_placement_from_mat(model, parent_placement, M4x4):
    origin, xdir, zdir = _matrix4_to_axes(M4x4)
    a2p = _axis2placement3d(model, origin, xdir, zdir)
    return model.create_entity("IfcLocalPlacement", PlacementRelTo=parent_placement, RelativePlacement=a2p)

# ---------------------
# Styling + TFS builder
# ---------------------

def _ensure_style(model):
    if getattr(model, "_simple_style", None):
        return model._simple_style
    colour = model.create_entity("IfcColourRgb", Red=0.9, Green=0.9, Blue=0.9)
    shading = model.create_entity("IfcSurfaceStyleShading", SurfaceColour=colour, Transparency=0.0)
    surf_style = model.create_entity("IfcSurfaceStyle", Name="Default", Side="BOTH", Styles=[shading])
    pres_assign = model.create_entity("IfcPresentationStyleAssignment", Styles=[surf_style])
    def _apply(rep_item):
        model.create_entity("IfcStyledItem", Item=rep_item, Styles=[pres_assign], Name=None)
    model._simple_style = _apply
    return _apply


def _compute_vertex_normals(V, I):
    n = [[0.0, 0.0, 0.0] for _ in V]
    for (i, j, k) in I:
        x1,y1,z1 = V[i]; x2,y2,z2 = V[j]; x3,y3,z3 = V[k]
        ux,uy,uz = x2-x1, y2-y1, z2-z1
        vx,vy,vz = x3-x1, y3-y1, z3-z1
        nx,ny,nz = (uy*vz - uz*vy), (uz*vx - ux*vz), (ux*vy - uy*vx)
        n[i][0]+=nx; n[i][1]+=ny; n[i][2]+=nz
        n[j][0]+=nx; n[j][1]+=ny; n[j][2]+=nz
        n[k][0]+=nx; n[k][1]+=ny; n[k][2]+=nz
    for m in n:
        L = (m[0]*m[0] + m[1]*m[1] + m[2]*m[2]) ** 0.5 or 1.0
        m[0]/=L; m[1]/=L; m[2]/=L
    return [tuple(m) for m in n]


def _make_tfs(model, subctx, V, I, apply_style=None):
    pts = model.create_entity("IfcCartesianPointList3D", CoordList=[(float(x),float(y),float(z)) for (x,y,z) in V])
    idx = [[i+1,j+1,k+1] for (i,j,k) in I]
    tfs = model.create_entity("IfcTriangulatedFaceSet", Coordinates=pts, CoordIndex=idx)
    # normals
    normals = _compute_vertex_normals(V, I)
    tfs.Normals = normals
    try:
        tfs.NormalIndex = idx
    except Exception:
        try:
            tfs.PnIndex = idx
        except Exception:
            pass
    try:
        tfs.Closed = True
    except Exception:
        pass
    if apply_style:
        apply_style(tfs)
    body = model.create_entity("IfcShapeRepresentation",
                               ContextOfItems=subctx,
                               RepresentationIdentifier="Body",
                               RepresentationType="Tessellation",
                               Items=[tfs])
    return body

# ---------------------
# Export entrypoint
# ---------------------

def write_ifc_from_handoff(handoff_dir, out_ifc_path, *, units="MILLIMETRE",
                           default_class="IfcMember",
                           parts_dir_name="parts",
                           lin_defl=None, ang_deg=20.0, verbose=True):
    handoff_dir = Path(handoff_dir)
    mani_path = handoff_dir / "manifest.json"
    if not mani_path.exists():
        raise FileNotFoundError(f"No manifest.json at {mani_path}")

    mani = json.loads(mani_path.read_text(encoding="utf-8"))
    parts = mani.get("parts", [])
    insts = mani.get("instances", [])
    _log(f"Manifest parts: {len(parts)}, instances: {len(insts)}")

    # def_id -> MEM name
    defid_to_mem = {}
    for inst in insts:
        did = str(inst.get("def_id")) if inst.get("def_id") is not None else None
        nm  = inst.get("name") or inst.get("occ_id") or ""
        if did and isinstance(nm, str) and nm.upper().startswith("MEM-"):
            defid_to_mem.setdefault(did, nm)

    # IFC model
    model, subctx, storey = _new_ifc(schema="IFC4", units=units)
    apply_style = _ensure_style(model)

    # Candidate parts dirs
    candidate_parts_dirs = [
        handoff_dir / parts_dir_name,
        handoff_dir.parent / parts_dir_name,
        handoff_dir.parent.parent / parts_dir_name,
    ]
    seen, parts_dirs = set(), []
    for d in candidate_parts_dirs:
        try:
            rp = d.resolve()
        except Exception:
            rp = d
        if d.exists() and str(rp) not in seen:
            parts_dirs.append(d); seen.add(str(rp))
    if verbose:
        _log(f"Searching parts in: {[str(d) for d in parts_dirs]}")

    def _find_mem_step(mem_name: str) -> Path | None:
        pats = [f"{mem_name}.step", f"{mem_name}.stp", f"{mem_name}_native.step", f"{mem_name}_native.stp"]
        for root in parts_dirs if parts_dirs else [handoff_dir]:
            for p in pats:
                cand = root / p
                if cand.exists():
                    return cand
        return None

    # Build meshes per def_id
    mesh_cache = {}
    missing_geom = 0
    for p in parts:
        def_id = p.get("def_id") or p.get("id") or p.get("hash") or ""
        def_id = str(def_id)
        if not def_id:
            _log("Part without def_id skipped")
            continue

        gpath = None
        # explicit path in manifest
        for k in ("step_url","step_path","path","file"):
            v = p.get(k)
            if v:
                cand = handoff_dir / v if not Path(v).is_absolute() else Path(v)
                if cand.exists():
                    gpath = cand; break
        # fallback via MEM
        if not gpath:
            mem_name = defid_to_mem.get(def_id)
            if mem_name:
                cand = _find_mem_step(mem_name)
                if cand:
                    gpath = cand
                    if verbose:
                        _log(f"Resolved via MEM: {def_id} → {cand}")

        if not gpath:
            missing_geom += 1
            if verbose:
                _log(f"Missing geometry for {def_id}")
            continue

        try:
            shape = _load_shape_from_path(gpath)
            V, I = _tri_arrays_from_shape(shape, lin_defl=lin_defl, ang_deg=ang_deg)
            mesh_cache[def_id] = (V, I)
            if verbose:
                _log(f"Meshed {gpath.name}: V={len(V)} I={len(I)} for {def_id}")
        except Exception as e:
            if verbose:
                _log(f"Mesh fail {gpath}: {e}")

    # Place instances (robust, with dup detection)
    created = 0
    skipped = 0
    dup_skipped = 0

    def _hashable_matrix(M):
        try:
            return tuple(round(float(x), 6) for row in M for x in row)
        except Exception:
            return None

    seen_keys = set()
    _log(f"Instance rows (reported): {len(insts)}")

    for inst in insts:
        def_id = str(inst.get("def_id")) if inst.get("def_id") is not None else ""
        if not def_id:
            skipped += 1
            _log("Instance with empty def_id skipped")
            continue

        name   = inst.get("name") or def_id
        occ_id = inst.get("occ_id")
        M      = inst.get("matrix") or [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        key    = ("occ", str(occ_id)) if occ_id else ("dnm", def_id, name, _hashable_matrix(M))
        if key in seen_keys:
            dup_skipped += 1
            _log(f"Duplicate instance skipped: occ_id={occ_id} def_id={def_id} name={name}")
            continue
        seen_keys.add(key)

        mesh = mesh_cache.get(def_id)
        if not mesh:
            skipped += 1
            _log(f"No mesh for instance def_id={def_id}; skipped")
            continue

        plc = _local_placement_from_mat(model, storey.ObjectPlacement, M)
        V, I = mesh
        body_shape = _make_tfs(model, subctx, V, I, apply_style=apply_style)

        prod = model.create_entity(default_class, Name=name, ObjectPlacement=plc)
        pds = model.create_entity("IfcProductDefinitionShape", Representations=[body_shape])
        prod.Representation = pds

        rel = model.create_entity("IfcRelContainedInSpatialStructure",
                                  GlobalId=_ifc_guid.new(),
                                  RelatingStructure=storey,
                                  RelatedElements=[prod])
        model.add(rel)
        created += 1

    if verbose:
        _log(f"Meshes: {len(mesh_cache)} (missing geom: {missing_geom})")
        _log(f"Instances created: {created}, skipped (no mesh/def_id): {skipped}, duplicates skipped: {dup_skipped}")

    out_ifc_path = Path(out_ifc_path)
    out_ifc_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_ifc_path))
    if verbose:
        _log(f"Wrote IFC: {out_ifc_path}")
    return str(out_ifc_path)

# ---------------------
# Optional fallback: CSV/DF report → IFC (flat placement)
# ---------------------

def write_ifc_from_report(report_df_or_path, out_ifc_path, *,
                          units="MILLIMETRE",
                          geom_col="step_path",
                          name_col="name",
                          parts_dir_name="parts",
                          lin_defl=None, ang_deg=20.0, verbose=True):
    """
    Build a simple IFC directly from a CSV/DataFrame report.
    Each row is a part placed at the storey origin (no transform).
    Geometry path is taken from `geom_col` if present; otherwise we try to
    resolve MEM-####[_native].step/.stp under likely parts directories
    next to the output IFC.
    """
    # Load DataFrame
    try:
        import pandas as pd
        if hasattr(report_df_or_path, "__dataframe__") or hasattr(report_df_or_path, "columns"):
            df = report_df_or_path
        else:
            df = pd.read_csv(report_df_or_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read report: {e}")

    out_ifc_path = Path(out_ifc_path)
    model, subctx, storey = _new_ifc(schema="IFC4", units=units)
    apply_style = _ensure_style(model)

    # Candidate parts dirs around the IFC folder
    cad_dir = out_ifc_path.parent.parent if out_ifc_path.parent.name.upper() == "IFC" else out_ifc_path.parent
    candidate_parts_dirs = [
        cad_dir / parts_dir_name,
        cad_dir.parent / parts_dir_name,
    ]
    seen, parts_dirs = set(), []
    for d in candidate_parts_dirs:
        try:
            rp = d.resolve()
        except Exception:
            rp = d
        if d.exists() and str(rp) not in seen:
            parts_dirs.append(d); seen.add(str(rp))
    if verbose:
        _log(f"[report] Searching parts in: {[str(d) for d in parts_dirs]}")

    def _find_by_name(base_name: str) -> Path | None:
        pats = [f"{base_name}.step", f"{base_name}.stp", f"{base_name}_native.step", f"{base_name}_native.stp"]
        for root in parts_dirs:
            for p in pats:
                cand = root / p
                if cand.exists():
                    return cand
        return None

    created = 0
    for _, row in df.iterrows():
        name = str(row.get(name_col) or "Part")
        # 1) explicit path
        gpath = None
        pv = row.get(geom_col)
        if isinstance(pv, str) and pv:
            cand = Path(pv)
            gpath = cand if cand.exists() else None
        # 2) MEM-based by name
        if gpath is None and name.upper().startswith("MEM-"):
            gpath = _find_by_name(name)
        if gpath is None:
            if verbose:
                _log(f"[report] Missing geometry for row name={name}")
            continue

        try:
            shape = _load_shape_from_path(gpath)
            V, I = _tri_arrays_from_shape(shape, lin_defl=lin_defl, ang_deg=ang_deg)
            body_shape = _make_tfs(model, subctx, V, I, apply_style=apply_style)
            prod = model.create_entity("IfcMember", Name=name, ObjectPlacement=None)
            # place at storey origin
            prod.ObjectPlacement = _local_placement_from_mat(model, storey.ObjectPlacement,
                                                              [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            pds = model.create_entity("IfcProductDefinitionShape", Representations=[body_shape])
            prod.Representation = pds
            rel = model.create_entity("IfcRelContainedInSpatialStructure",
                                      GlobalId=_ifc_guid.new(),
                                      RelatingStructure=storey,
                                      RelatedElements=[prod])
            model.add(rel)
            created += 1
        except Exception as e:
            if verbose:
                _log(f"[report] Mesh fail {gpath}: {e}")

    out_ifc_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_ifc_path))
    if verbose:
        _log(f"[report] Wrote IFC with {created} items → {out_ifc_path}")
    return str(out_ifc_path)
