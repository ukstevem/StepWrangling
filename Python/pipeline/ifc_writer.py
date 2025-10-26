from pathlib import Path
import math
import pandas as pd
import ifcopenshell
import ifcopenshell.api as api


# ---- helpers ----------------------------------------------------------

def _to_float(x):
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except Exception:
        return None

def _flatten_matrix_row_major_4x4(m):
    if m is None:
        return None
    if isinstance(m, (list, tuple)) and len(m) == 16:
        vals = [_to_float(v) for v in m]
        return vals if all(v is not None for v in vals) else None
    try:
        flat = []
        for r in m:
            flat.extend(r if isinstance(r, (list, tuple)) else [r])
        vals = [_to_float(v) for v in flat]
        if len(vals) == 16 and all(v is not None for v in vals):
            return vals
    except Exception:
        pass
    return None

def _matrix_4x4_nested(m):
    """Return a 4x4 row-major nested list of floats, or None if invalid."""
    def _flt(x):
        try:
            v = float(x)
            if math.isnan(v):
                return None
            return v
        except Exception:
            return None

    if isinstance(m, (list, tuple)) and len(m) == 4 and all(isinstance(r, (list, tuple)) and len(r) == 4 for r in m):
        rows = [[_flt(v) for v in r] for r in m]
        if all(v is not None for r in rows for v in r):
            return rows
        return None

    try:
        flat = list(m)
        if len(flat) == 16:
            flat = [_flt(v) for v in flat]
            if any(v is None for v in flat):
                return None
            return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    except Exception:
        pass

    try:
        rows, acc = [], []
        for v in m:
            acc.append(_flt(v))
            if len(acc) == 4:
                rows.append(acc); acc = []
        if len(rows) == 4 and all(v is not None for r in rows for v in r):
            return rows
    except Exception:
        pass
    return None

def _identity_4x4_nested():
    return [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]

def _identity_4x4():
    return [1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1]

def _ifc_class_from_section_shape(val):
    if val is None:
        return "IfcBuildingElementProxy"
    s = str(val).strip().upper()
    if s in {"I","U","H","W","CHANNEL","BEAM","T"}: return "IfcBeam"
    if s in {"L","ANGLE","BAR","ROD","MEMBER"}:     return "IfcMember"
    if s in {"PLATE","FLAT","SHEET"}:               return "IfcPlate"
    return "IfcBuildingElementProxy"

def _length_mm_from_part_row(p: dict):
    cand = []
    for k in ("obb_x","obb_y","obb_z","bbox_x","bbox_y","bbox_z"):
        v = _to_float(p.get(k))
        if v is not None:
            cand.append(v)
    return max(cand) if cand else None

# columns we’ll copy into the custom property set when present
_PSET_DSTV_COLS = [
    "section_shape","obj_type","issues","hash","assembly_hash","signature_hash",
    "obb_x","obb_y","obb_z","bbox_x","bbox_y","bbox_z",
    "mass","volume","surface_area",
    "centroid_x","centroid_y","centroid_z",
    "inertia_e1","inertia_e2","inertia_e3",
    "chirality",
    "step_path","native_step_path","stl_path","thumb_path","drilling_path",
    "dxf_path","nc1_path","brep_path","dxf_thumb_path",
]

# ---------- Manual Pset / Quantities helpers ----------
def _as_ifc_value(m, v):
    """Return a proper IFC measure entity (IfcReal/IfcInteger/IfcBoolean/IfcLabel) or None."""
    if v is None:
        return None
    # filter bad numerics
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        # choose Integer for int-like values, else Real
        if abs(fv - round(fv)) < 1e-9:
            return m.create_entity("IfcInteger", int(round(fv)))
        return m.create_entity("IfcReal", fv)
    except Exception:
        # booleans
        if isinstance(v, bool):
            return m.create_entity("IfcBoolean", bool(v))
        # strings & everything else
        s = str(v)
        if s == "" or s.lower() == "nan" or s.lower() == "none":
            return None
        return m.create_entity("IfcLabel", s)


def _ensure_qto(m, product, name="BaseQuantities"):
    qto = m.create_entity("IfcElementQuantity", Name=name, Quantities=[])
    m.create_entity("IfcRelDefinesByProperties", RelatedObjects=[product], RelatingPropertyDefinition=qto)
    return qto

# quantities
def _qto_add_length(m, qto, name, value_mm: float):
    q = m.create_entity("IfcQuantityLength", Name=name, LengthValue=float(value_mm))
    qto.Quantities = tuple(list(qto.Quantities or []) + [q])

def _qto_add_weight(m, qto, name, value_kg: float):
    q = m.create_entity("IfcQuantityWeight", Name=name, WeightValue=float(value_kg))
    qto.Quantities = tuple(list(qto.Quantities or []) + [q])

def _ensure_pset(m, product, name="Pset_DSTV"):
    pset = m.create_entity("IfcPropertySet", Name=name, HasProperties=[])
    m.create_entity("IfcRelDefinesByProperties", RelatedObjects=[product], RelatingPropertyDefinition=pset)
    return pset

# properties
def _pset_add_props(m, pset, props: dict):
    if not props:
        return
    new_props = list(pset.HasProperties or [])
    for k, v in props.items():
        nominal = _as_ifc_value(m, v)   # <— uses the new typed-measure helper
        if nominal is None:
            continue
        new_props.append(
            m.create_entity("IfcPropertySingleValue", Name=str(k), NominalValue=nominal)
        )
    pset.HasProperties = tuple(new_props)
# ------------------------------------------------------


# ---- main -------------------------------------------------------------

def write_ifc(parts_df: pd.DataFrame,
              instances_df: pd.DataFrame,
              out_path,
              ifc_version: str = "IFC4X3",
              storey_name: str = "Level 0") -> str:

    out_path = Path(out_path)

    print("[IFC_WRITER] Module:", __file__)
    print("[IFC_WRITER] parts_df cols:", list(parts_df.columns))
    print("[IFC_WRITER] instances_df cols:", list(instances_df.columns))


    # index parts by part_id
    parts_by_id = {str(r["part_id"]): r.to_dict()
                   for _, r in parts_df.iterrows()
                   if pd.notna(r.get("part_id"))}

    # IFC file
    m = api.run("project.create_file", version=ifc_version)

    # --- Project -------------------------------------------------------
    try:
        proj = api.run("project.create_project", m, name="DSTV Export")
    except Exception:
        proj = m.create_entity("IfcProject", Name="DSTV Export")

    # --- Units: mm, kg (explicit SI units; no helpers needed) ----------
    u_length = m.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Prefix="MILLI", Name="METRE")
    u_mass   = m.create_entity("IfcSIUnit", UnitType="MASSUNIT",   Prefix="KILO",  Name="GRAM")
    u_area   = m.create_entity("IfcSIUnit", UnitType="AREAUNIT",   Name="SQUARE_METRE")
    u_vol    = m.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
    u_angle  = m.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
    units = [u_length, u_mass, u_area, u_vol, u_angle]
    try:
        api.run("unit.assign_unit", m, units=units)
    except Exception:
        ua = m.create_entity("IfcUnitAssignment", Units=units)
        try:
            proj.UnitsInContext = ua
        except Exception:
            pass

    # --- Geometric representation context ------------------------------
    try:
        model_ctx = api.run("context.add_context", m, context_type="Model", context_identifier="Model")
        api.run("context.add_subcontext", m, parent=model_ctx, context_type="Model",
                context_identifier="Body", target_view="MODEL_VIEW", predefined_type="MODEL_VIEW")
    except Exception:
        wcs = m.create_entity("IfcAxis2Placement3D",
                              Location=m.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)))
        m.create_entity("IfcGeometricRepresentationContext",
                        ContextIdentifier="Model",
                        ContextType="Model",
                        CoordinateSpaceDimension=3,
                        Precision=1e-6,
                        WorldCoordinateSystem=wcs)

    # --- Spatial structure ---------------------------------------------
    site   = api.run("root.create_entity", m, ifc_class="IfcSite", name="Default Site")
    bldg   = api.run("root.create_entity", m, ifc_class="IfcBuilding", name="Default Building")
    storey = api.run("root.create_entity", m, ifc_class="IfcBuildingStorey", name=str(storey_name))
    api.run("aggregate.assign_object", m, relating_object=site,  products=[bldg])
    api.run("aggregate.assign_object", m, relating_object=bldg, products=[storey])

    # --- Elements -------------------------------------------------------
    for _, inst in instances_df.iterrows():
        iid   = inst.get("instance_id")
        pid   = inst.get("part_id")
        name  = inst.get("name")
        T     = inst.get("T")

        if pd.isna(pid):
            continue

        p = parts_by_id.get(str(pid), {})
        ifc_class = _ifc_class_from_section_shape(p.get("section_shape"))
        elem_name = str(iid or name or p.get("name") or pid)
        elem = api.run("root.create_entity", m, ifc_class=ifc_class, name=elem_name)

        mat = _matrix_4x4_nested(T) or _identity_4x4_nested()
        api.run("geometry.edit_object_placement", m, product=elem, matrix=mat)

        # --- BaseQuantities (manual) ---
        qto = _ensure_qto(m, elem, name="BaseQuantities")
        L = _length_mm_from_part_row(p)
        if L is not None:
            _qto_add_length(m, qto, name="Length", value_mm=float(L))

        print("[IFC_WRITER] QTO for", elem_name, "len:", len(qto.Quantities or ()))

        mass = _to_float(p.get("mass"))
        if mass is None:
            # optional: fallback via volume (mm³) → mass (kg)
            vol_mm3 = _to_float(p.get("volume"))
            if vol_mm3 is not None:
                density_kg_m3 = 7850.0
                mass = density_kg_m3 * (vol_mm3 * 1e-9)
        if mass is not None:
            _qto_add_weight(m, qto, name="GrossWeight", value_kg=float(mass))

        # --- Common pset (manual) ---
        if ifc_class in ("IfcBeam", "IfcMember"):
            common_name = "Pset_BeamCommon" if ifc_class == "IfcBeam" else "Pset_MemberCommon"
            pset_common = _ensure_pset(m, elem, name=common_name)
            ref_val = p.get("name") or p.get("section_shape")
            if ref_val:
                _pset_add_props(m, pset_common, {"Reference": ref_val})

        # --- Custom DSTV pset (manual) ---
        pset_dstv = _ensure_pset(m, elem, name="Pset_DSTV")

        # build clean props dict, skipping None/NaN
        props = {}

        print("[IFC_WRITER] element:", elem_name, "part_id:", pid)
        print("[IFC_WRITER] raw part dict keys:", list(p.keys()))
        print("[IFC_WRITER] sample fields:", {k: p.get(k) for k in ["mass","obb_x","bbox_x","section_shape","step_path"] if k in p})

        for c in _PSET_DSTV_COLS:
            if c not in p:
                continue
            v = p[c]
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
            except Exception:
                # keep strings/bools
                pass
            props[c] = v

        if not props:
            print("[IFC_WRITER] No props for", elem_name)
        else:
            shown = {k: props[k] for k in ["mass","obb_x","obb_y","obb_z","bbox_x","bbox_y","bbox_z","section_shape","step_path"] if k in props}
            print("[IFC_WRITER] Props for", elem_name, shown)

        props["debug_marker"] = "pset_dstv_written"

        _pset_add_props(m, pset_dstv, props)

        api.run("spatial.assign_container", m, relating_structure=storey, products=[elem])

    # --- write file -----------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.write(str(out_path))
    return str(out_path)
