import ifcopenshell
import ifcopenshell.api
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

def export_solid_to_ifc(solid, ifc_path, name, material):
    # Create a new IFC file (version can be changed)
    ifc = ifcopenshell.api.run("project.create_file")

    # Add a project
    project = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="STEP Export Project")
    context = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    ifcopenshell.api.run("unit.assign_unit", ifc, length="millimetre")

    # Create a site, building, storey, and product placement
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Default Site")
    building = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Default Building")
    storey = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name="Default Storey")
    ifcopenshell.api.run("aggregate.assign_object", ifc, product=project, relating_object=site)
    ifcopenshell.api.run("aggregate.assign_object", ifc, product=site, relating_object=building)
    ifcopenshell.api.run("aggregate.assign_object", ifc, product=building, relating_object=storey)

    # Convert solid into IfcShapeRepresentation
    shape = ifcopenshell.api.run("geometry.add_occ_shape", ifc, shape=solid, context=context)

    # Create a beam (or generic IfcBuildingElementProxy)
    beam = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBeam", name=name)
    ifcopenshell.api.run("geometry.assign_representation", ifc, product=beam, representation=shape)
    ifcopenshell.api.run("spatial.assign_container", ifc, product=beam, relating_structure=storey)

    # Assign material if you want
    ifcopenshell.api.run("material.assign_material", ifc, product=beam, type="IfcMaterial", name=material)

    # Save to disk
    ifc.write(ifc_path)
    print(f"✅ IFC file saved to: {ifc_path}")
