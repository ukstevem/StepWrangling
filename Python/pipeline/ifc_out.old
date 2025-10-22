import ifcopenshell
import ifcopenshell.api
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.GProp import GProp_GProps

def ensure_ok(result, msg=""):
    if isinstance(result, str):
        raise RuntimeError(f"{msg} → {result}")
    return result

def export_solid_to_ifc(solid, ifc_path, name="Solid", material="S355JR", profile_type="UNSPECIFIED"):

    ifc = ifcopenshell.api.run("project.create_file")

    project = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="STEP Export Project"),
        "Creating IfcProject"
    )

    context = ensure_ok(
        ifcopenshell.api.run("context.add_context", ifc, context_type="Model"),
        "Adding Model context"
    )

    ensure_ok(
        ifcopenshell.api.run("unit.assign_unit", ifc, length="millimetre"),
        "Assigning unit"
    )

    site = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Default Site"),
        "Creating IfcSite"
    )
    building = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Default Building"),
        "Creating IfcBuilding"
    )
    storey = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name="Default Storey"),
        "Creating IfcBuildingStorey"
    )

    ensure_ok(
        ifcopenshell.api.run("aggregate.assign_object", ifc, product=project, relating_object=site),
        "Assigning project→site"
    )
    ensure_ok(
        ifcopenshell.api.run("aggregate.assign_object", ifc, product=site, relating_object=building),
        "Assigning site→building"
    )
    ensure_ok(
        ifcopenshell.api.run("aggregate.assign_object", ifc, product=building, relating_object=storey),
        "Assigning building→storey"
    )

    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepTools import BRepTools

    # Log volume (should be > 0)
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    print(f"📦 Solid volume: {props.Mass()}")

    # Optional: dump to .brep for inspection
    BRepTools.Write(solid, "debug_solid.brep")

    shape = ensure_ok(
        ifcopenshell.api.run("geometry.add_occ_shape", ifc, shape=solid, context=context),
        "Adding OCC shape"
    )

    beam = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBeam", name=name),
        "Creating IfcBeam"
    )

    ensure_ok(
        ifcopenshell.api.run("geometry.assign_representation", ifc, product=beam, representation=shape),
        "Assigning geometry representation"
    )
    ensure_ok(
        ifcopenshell.api.run("spatial.assign_container", ifc, product=beam, relating_structure=storey),
        "Assigning beam→storey"
    )
    ensure_ok(
        ifcopenshell.api.run("material.assign_material", ifc, product=beam, type="IfcMaterial", name=material),
        "Assigning material"
    )
    import os
    print("Saving IFC to:", os.path.abspath(ifc_path))
    ifc.write(ifc_path)
    print(f"✅ IFC saved to {ifc_path}")
