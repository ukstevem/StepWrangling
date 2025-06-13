import ifcopenshell
import ifcopenshell.api
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

def ensure_ok(result, msg=""):
    if isinstance(result, str):
        raise RuntimeError(f"{msg} → {result}")
    return result


import ifcopenshell
import ifcopenshell.api
from OCC.Core.TopoDS import TopoDS_Shape

def ensure_ok(result, msg="Unknown IFC error"):
    if isinstance(result, str):
        raise RuntimeError(f"{msg}: {result}")
    return result

def export_solid_to_ifc(solid: TopoDS_Shape, ifc_path: str, name="Solid", material="S355JR", profile_type="UNSPECIFIED"):
    # Create a new IFC file
    ifc = ifcopenshell.api.run("project.create_file")

    # Create core project structure
    project = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="STEP Export Project"),
        "Failed to create IfcProject"
    )

    context = ensure_ok(
        ifcopenshell.api.run("context.add_context", ifc, context_type="Model"),
        "Failed to add context"
    )

    ensure_ok(
        ifcopenshell.api.run("unit.assign_unit", ifc, length="millimetre"),
        "Failed to assign unit"
    )

    # Create site > building > storey hierarchy
    site = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Default Site"),
        "Failed to create IfcSite"
    )
    building = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Default Building"),
        "Failed to create IfcBuilding"
    )
    storey = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name="Default Storey"),
        "Failed to create IfcBuildingStorey"
    )

    ensure_ok(
        ifcopenshell.api.run("aggregate.assign_object", ifc, product=project, relating_object=site),
        "Failed to assign site to project"
    )
    ensure_ok(
        ifcopenshell.api.run("aggregate.assign_object", ifc, product=site, relating_object=building),
        "Failed to assign building to site"
    )
    ensure_ok(
        ifcopenshell.api.run("aggregate.assign_object", ifc, product=building, relating_object=storey),
        "Failed to assign storey to building"
    )

    # Convert OCC solid to IfcShapeRepresentation
    shape = ensure_ok(
        ifcopenshell.api.run("geometry.add_occ_shape", ifc, shape=solid, context=context),
        "Failed to add shape from OCC solid"
    )

    # Create IfcBeam (or IfcBuildingElementProxy, etc.)
    beam = ensure_ok(
        ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBeam", name=name),
        "Failed to create IfcBeam"
    )

    ensure_ok(
        ifcopenshell.api.run("geometry.assign_representation", ifc, product=beam, representation=shape),
        "Failed to assign geometry representation"
    )

    ensure_ok(
        ifcopenshell.api.run("spatial.assign_container", ifc, product=beam, relating_structure=storey),
        "Failed to assign beam to storey"
    )

    ensure_ok(
        ifcopenshell.api.run("material.assign_material", ifc, product=beam, type="IfcMaterial", name=material),
        "Failed to assign material"
    )

    # Save IFC file
    ifc.write(ifc_path)
    print(f"✅ IFC file saved to: {ifc_path}")
