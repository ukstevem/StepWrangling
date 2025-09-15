# main.py - Entry point for DSTV pipeline

from datetime import datetime
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pipeline.step_loader import load_step_file
from pipeline.signature_utils import compute_signature_info
from pipeline.geometry_utils import (robust_align_solid_from_geometry, 
                                     ensure_right_handed, 
                                     compute_obb_geometry, 
                                     compute_section_area,
                                     swap_width_and_height_if_required,
                                     compute_dstv_origin,
                                     align_obb_to_dstv_frame,
                                     refine_profile_orientation)
from pipeline.shape_inspector import (describe_shape)
from pipeline.assembly_management import (
                                     fingerprint_solids
                                     )
from pipeline.classification import classify_profile
from pipeline.summary_writer import section_result_table
from pipeline.dstv_geometry import (classify_and_project_holes_dstv,
                                    check_duplicate_holes,
                                    analyze_end_faces_web_and_flange
                                     )
from pipeline.dstv_writer import (generate_nc1_file, 
                                  assemble_dstv_header_data)
from pipeline.drilling import (generate_hole_projection_html)
from pipeline.cad_out import (export_solid_to_brep, 
                              shape_to_thumbnail, 
                              export_solid_to_step,
                              generate_plate_dxf,
                              render_dxf_drawing,
                              export_solid_to_brep_and_fingerprint,
                              export_profile_dxf_with_pca
                              )
from pipeline.plate_wrangling import (align_plate_to_xy_plane)
from pipeline.report_builder import (record_solid,
                                     df_to_html_with_images,
                                     df_to_excel_with_images)
from pipeline.path_management import (resolve_output_path,  
                                  create_project_directories)
from pipeline.database import (dump_df_to_supabase, normalize_report_df,
                                add_to_database
                                )
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
from pathlib import Path



def dstv_pipeline(step_file, root_model, project_number, matl_grade):

    # Log start time
    start_time = datetime.now()
    print(f"Start Time : {start_time}")
    # Generate folder structure and get paths

    # Static
    BASE = Path(__file__).parent
    media_path = BASE / "media"
    json_path = BASE / "data/Shape_classifier_info.json"
    network_root = BASE / "G:/Customer Orders/Extraction"
    fallback_root = BASE / "C:/dev/step-gemini/Extraction"

    # Project Driven
    base_path, nc_path, report_path, drilling_path, cad_path, thumb_path, step_path, brep_path, dxf_path, dxf_thumb = create_project_directories(step_file, network_root, fallback_root)

    if not os.path.isfile(root_model):
        raise FileNotFoundError(f"Could not find a STEP file at: {root_model!r}")
    solids = load_step_file(root_model)

    # setup for fingerprint and saving out of the root model
    
    # setting paths
    root_name = Path(root_model).name            # e.g. "part123.step"
    cad_file  = Path(cad_path) / root_name       # e.g. ".../Extraction/Proj1/Models/part123.step"
    # compute an MD5 of the newly written STEP
    hash_type = 'md5'
    source_model_hash = fingerprint_solids(solids, str(cad_file), hash_type) # To be able to confirm model used at later date.

    #write data to source_model table
    # now insert a row into `source_model`
    payload = {
        "filename": Path(root_model).name,
        "hash":     source_model_hash,
        "project":  project_number
    }
    response = add_to_database(
        payload,
        table_name="source_model",
        # if you want to enforce column whitelisting:
        allowed_columns=["filename", "hash", "project"]
    )

    if not response.get("success"):
        # log or re-raise depending on how you want to handle failures
        raise RuntimeError(f"DB insert failed: {response['error']}")
    print(f"Inserted source_model row: {response['data']!r}")


    # Now get going on extraction
    report_rows = []    #report array complied per part in loop

    for i, shape_orig in enumerate(solids):
        # set variable defaults
        member_id = f"MEM-{i+1:04d}"
        step_file = ""
        thumbnail_file = ""
        html_file = ""
        dxf_file = ""
        nc1_file = ""
        brep_file = ""
        step_mass = 0
        obb_x = 0
        obb_y = 0
        obb_z = 0
        object_type = "-"
        issues = "-"
        hash = ""
        dxf_thumb_file = ""
        section_shape = ""
        assembly_hash = ""
        signature_hash = ""
        volume = ""
        surface_area = ""
        bbox_x = ""
        bbox_y = ""
        bbox_z = ""
        inertia_e1 = ""
        inertia_e2 = ""
        inertia_e3 = ""
        centroid_x = ""
        centroid_y = ""
        centroid_z = ""
        chirality = ""
        print(f"\n🟦 Processing {member_id}... {datetime.now().strftime('%H:%M:%S')}")

        try:

            # print("=== Original Shape ===")
            # metrics0 = describe_shape(shape_orig)

            # STEP 1: Align to major geometry
            primary_aligned_shape, trsf, cs, largest_face, dir_x, dir_y, dir_z = robust_align_solid_from_geometry(shape_orig)
            dir_x, dir_y, dir_z = ensure_right_handed(dir_x, dir_y, dir_z)

            # Get signature details of initial shape
            inv, signature_hash = compute_signature_info(
                primary_aligned_shape,
                dir_x, dir_y, dir_z,
                precision=3,
                hash_type="md5"
            )

            # print("=== Primary Aligned Shape ===")
            # metrics1 = describe_shape(primary_aligned_shape)

            # STEP 2: Compute OBB geometry
            obb_geom = compute_obb_geometry(primary_aligned_shape)

            # STEP 3: Section area
            section_area = compute_section_area(primary_aligned_shape)

            print(f"Section area: {section_area:.2f}")

            # STEP 4: Classification signature
            cs_data = {
                "span_web": obb_geom["aligned_extents"][1],
                "span_flange": obb_geom["aligned_extents"][2],
                "area": section_area,
                "length": obb_geom["aligned_extents"][0]
            }

            print(cs_data)

            # STEP 5: Match profile
            profile_match = classify_profile(cs_data, json_path, tol_dim=1.0, tol_area=.05)
            
            print(f"Profile Match complete : {profile_match}")

            # STEP 5.5: no match? check for plate
            if profile_match is None:
                is_plate, final_aligned_solid, ax3, thickness_mm, length_mm, width_mm, step_mass, msg = align_plate_to_xy_plane(primary_aligned_shape)
                # print(f"Basic X/Y alignment complete, is plate? {is_plate}")
                # print(f"Is Plate: {is_plate}, aligned_solid: {final_aligned_solid}, Mass: {step_mass}, Message: {msg}")
                obb_x = f'{obb_geom["aligned_extents"][0]:.2f}'
                obb_y = f'{obb_geom["aligned_extents"][1]:.2f}'
                obb_z = f'{obb_geom["aligned_extents"][2]:.2f}'
                step_mass = f'{step_mass:.0f}'
                if is_plate:
                    # Branch to take care of plate
                    object_type = "Plate"
                    # print("Creating profile DXF.")
                    hash, dxf_file, dxf_thumb_file = export_profile_dxf_with_pca(final_aligned_solid, f"{dxf_path}\{member_id}.dxf", f"{dxf_thumb}\{member_id}" )
                else:
                    # print(f"Solid: {final_aligned_solid}, Thumbnail Path {thumb_path}, Member ID: {member_id}, Mseeage: {msg}")
                    print("ℹ️  Object not identified", msg)
                    object_type = "Other"
                    hash =""

                step_file = export_solid_to_step(final_aligned_solid, step_path, member_id)
                brep_fingerprint, brep_file = export_solid_to_brep(final_aligned_solid, brep_path, member_id)
                thumbnail_file = shape_to_thumbnail(final_aligned_solid, thumb_path, member_id)

            else:
                print("✅ Section match found")
                # print(f"Profile Match Data: {profile_match}, {profile_match['JSON']}\n {profile_match['STEP']}")

                # STEP 6: Adjust orientation if needed
                primary_aligned_shape, obb_geom = swap_width_and_height_if_required(
                    profile_match, primary_aligned_shape, obb_geom
                )

                print(f"🔄 Post-swap extents: {obb_geom['aligned_extents']}")

                # print("=== after Width/Height Swap ===")
                # metrics2 = describe_shape(primary_aligned_shape)

                step_vals = profile_match["STEP"]
                
                # generic DSTV frame
                origin_local = compute_dstv_origin(
                    obb_geom["aligned_center"],
                    obb_geom["aligned_extents"],
                    obb_geom["aligned_dir_x"],
                    obb_geom["aligned_dir_y"],
                    obb_geom["aligned_dir_z"],
                )
                primary_aligned_shape, _ = align_obb_to_dstv_frame(
                    primary_aligned_shape,
                    origin_local,
                    obb_geom["aligned_dir_x"],
                    obb_geom["aligned_dir_y"],
                    obb_geom["aligned_dir_z"],
                )

                # print("=== after align OBB to DSTV ===")
                # metrics3 = describe_shape(primary_aligned_shape)

                dstv_frame = gp_Ax3(
                    gp_Pnt(0, 0, 0),
                    gp_Dir(obb_geom["aligned_dir_z"].XYZ()),  # thickness
                    gp_Dir(obb_geom["aligned_dir_x"].XYZ()),  # length
                )

                # STEP 8: Final orientation refinement
                refined_shape, obb_geom = refine_profile_orientation(
                    primary_aligned_shape, profile_match, compute_obb_geometry(primary_aligned_shape)
                )

                obb_geom = compute_obb_geometry(refined_shape)

                print(f"✅ Orientation refinement complete. New extents: {obb_geom['aligned_extents']}")

                # print("=== after final refinement ===")
                # metrics2 = describe_shape(primary_aligned_shape)

                # cad output
                # export_solid_to_brep(shape_orig, brep_path, member_id)
                object_type = f"{profile_match['Category']} {profile_match['Designation']}"
                section_shape = f"{profile_match['Profile_type']}"
                step_file = export_solid_to_step(refined_shape, step_path, member_id)
                brep_fingerprint, brep_file = export_solid_to_brep(refined_shape, brep_path, member_id)

                # # after refine_profile_orientation and compute_obb_geometry
                # print(">>> OBB extents (L, H, W):", obb_geom["aligned_extents"])
                # for axis_name in ["x", "y", "z"]:
                #     vec = obb_geom[f"aligned_dir_{axis_name}"]
                #     xyz = vec.XYZ()
                #     print(f">>> aligned_dir_{axis_name}: "
                #         f"({xyz.X():.6f}, {xyz.Y():.6f}, {xyz.Z():.6f})")

                ## define the DSTV frame and axis direction from geometry
                dstv_frame = gp_Ax3(
                    gp_Pnt(0, 0, 0),
                    gp_Dir(obb_geom["aligned_dir_z"].XYZ()),
                    gp_Dir(obb_geom["aligned_dir_x"].XYZ())
                )

                # print(section_result_table(profile_match))
                web_cuts = analyze_end_faces_web_and_flange(refined_shape, dstv_frame)
                # print(web_cuts)

                # STEP 9: Classify and check holes
                step_vals = profile_match["STEP"]
                raw_df_holes = classify_and_project_holes_dstv(refined_shape, 
                                                                dstv_frame,
                                                                dstv_frame.Location(),
                                                                step_vals["width"],
                                                                step_vals["height"])
                # For report
                obb_x = f'{step_vals["length"]:.2f}'
                obb_y = f'{step_vals["height"]:.2f}'
                obb_z = f'{step_vals["width"]:.2f}'
                step_mass = f'{step_vals["mass"]:.0f}'
                # print(raw_df_holes)

                check_duplicate_holes(raw_df_holes, tolerance=0.5)

                # STEP 9.5: Collate DSTV header
                dstv_header_data = assemble_dstv_header_data(project_number, 
                                                            nc_path, 
                                                            matl_grade, 
                                                            member_id, 
                                                            profile_match)

                # STEP 10: Output
                # NC1               
                nc1_file, hash = generate_nc1_file(raw_df_holes, dstv_header_data, nc_path, web_cuts)
                # Drawing
                html_file = generate_hole_projection_html(raw_df_holes, dstv_header_data, media_path, drilling_path, web_cuts)
                # Thumbnail
                thumbnail_file = shape_to_thumbnail(primary_aligned_shape, thumb_path, member_id)

        except Exception as e:
            print(f"❌ Error in {member_id}: {e}")
            record_solid(report_rows, 
                     name = member_id,
                     step_path = step_file,
                     thumb_path = thumbnail_file,
                     drilling_path = html_file,
                     dxf_path = dxf_file,
                     nc1_path = nc1_file,
                     brep_path = brep_file,
                     mass = step_mass,
                     obb_x = obb_x,
                     obb_y = obb_y,
                     obb_z = obb_z,
                     obj_type = object_type,
                     issues = str(e),
                     hash = hash,
                     dxf_thumb_path = dxf_thumb_file,
                     section_shape = section_shape,
                     assembly_hash = source_model_hash,
                     signature_hash = signature_hash,
                     volume = inv["volume"],
                     surface_area = inv["surface_area"],
                     bbox_x = inv["bbox_x"],
                     bbox_y = inv["bbox_y"],
                     bbox_z = inv["bbox_z"],
                     inertia_e1 = inv["inertia_ix"],
                     inertia_e2 = inv["inertia_iy"],
                     inertia_e3 = inv["inertia_iz"],
                     centroid_x = inv["centroid_x"],
                     centroid_y = inv["centroid_y"],
                     centroid_z = inv["centroid_z"],
                     chirality = inv["chirality"]
             )

        # # Export summary
        record_solid(report_rows, 
                     name = member_id,
                     step_path = step_file,
                     thumb_path = thumbnail_file,
                     drilling_path = html_file,
                     dxf_path = dxf_file,
                     nc1_path = nc1_file,
                     brep_path = brep_file,
                     mass = step_mass,
                     obb_x = obb_x,
                     obb_y = obb_y,
                     obb_z = obb_z,
                     obj_type = object_type,
                     issues = issues,
                     hash = hash,
                     dxf_thumb_path = dxf_thumb_file,
                     section_shape = section_shape,
                     assembly_hash = source_model_hash,
                     signature_hash = signature_hash,
                     volume = inv["volume"],
                     surface_area = inv["surface_area"],
                     bbox_x = inv["bbox_x"],
                     bbox_y = inv["bbox_y"],
                     bbox_z = inv["bbox_z"],
                     inertia_e1 = inv["inertia_ix"],
                     inertia_e2 = inv["inertia_iy"],
                     inertia_e3 = inv["inertia_iz"],
                     centroid_x = inv["centroid_x"],
                     centroid_y = inv["centroid_y"],
                     centroid_z = inv["centroid_z"],
                     chirality = inv["chirality"]
                    )

    report_df = pd.DataFrame(report_rows)

    # print(report_df)

    # Create HTML report
    df_to_html_with_images(report_df, report_path, project_number)

    # Log to Supabase
    report_for_db = normalize_report_df(report_df, project_number)
    dump_df_to_supabase(report_for_db)

    print("\n✅ Pipeline completed. Summary saved.")
    finish_time = datetime.now()
    print(f"Finish Time : {finish_time}")
    print(f"Processing Duration : {finish_time - start_time}")



if __name__ == "__main__":

    home_path = r"C:/Dev/step-gemini/Python/data"
    step1 = "C25001-1-0101-MAIN.step"
    step2 = "MEM-026.step"     #Corner Leg Beam
    step3 = "MEM-210.step"     #Heavy Beam
    step4 = "MEM-003.step"     #Plate
    step5 = "MEM-739.step"     #Formed Plate
    step6 = "MEM-569.step"     #Plain Beam
    step7 = "MEM-418.step"     #Beam flange coped
    step8 = "MEM-546.step"     #Beam web coped
    step9 = "MEM-2122.step"    #Washer
    step10 = "MEM-2124.step"    #Bolt Head
    step11 = "0444-1 ANGLED.step"
    step12 = "ncTest.step"
    step13 = "TestEA.step"
    step14 = "TestEAMirror.step"
    step15 = "TestUEA.step"
    step16 = "TestUEAMirror.step"
    step17 = "TestPFC.step"
    step18 = "MEM-0027.step"     #Misaligned plate
    step19 = "MEM-0057.step"     #Misaligned Plate 2
    step20 = "02086 - Steelwork.stp"
    step21 = "MEM-0001.step"    #02086 - Imported Channel Ident issue"
    step22 = "MEM-1181.step"
    step23 = "MEM-1180.step"
    step24 = "MEM-1207.step"
    step25 = "MEM-0602.step"
    step26 = "4v1800 Panel Off.step"

    # For Angle NC1 file 
    step507 = "MEM-0507.step"
    step515 = "MEM-0515.step"
    step523 = "MEM-0523.step"
    step530 = "MEM-0530.step"
    step538 = "MEM-0538.step"    
    step602 = "MEM-0602.step"
    step610 = "MEM-0610.step"
    step164 = "MEM-1164.step"
    step166 = "MEM-1166.step"     

    # For cranked Beams NC1
    step219 = "MEM-0219.step"
    step234 = "MEM-0234.step"
    step254 = "MEM-0254.step"
    step276 = "MEM-0276.step"
    step300 = "MEM-0300.step"    
    step161 = "MEM-0161.step"
    step185 = "MEM-0185.step"
    step198 = "MEM-0198.step"
    step163 = "MEM-0163.step"    
    step221 = "MEM-0221.step"
    step1074 = "MEM-1074.step"
    step1051 = "MEM-1051.step"

    # cranked beam assembly 1028 rev B
    step1028 = "C25001-1-1028.step"

    # Kirby
    step23601 = r"kirby\275 kV 5.37m High Level Post Insulator x6.stp"
    step23602 = r"kirby\275 kV PQ Capacitor Voltage Transformer x3.stp"
    step23603 = r"kirby\275 kV Surge Arrestor Strc Transformer SGT2 x3.stp"
    step23604 = r"kirby\275 kW Current Transformer Structure x3.stp"
    step23605 = r"kirby\275 kW Rotary Disconnector Struc.stp"
    step23606 = r"kirby\275kW 2.7m Post Insulator Structure.stp"

    # BWB
    step669 = "P669K-6v3000-p2b2-EF2end.step"

    # 02138 - ERG Structures
    step02138 = "AP6467A-0-709 - Batch 5 steelwork.step"

    # 02140 - ERG Castle Environmental
    step02140 = "AP5630-0-872 STEEL LADDER STP.step"

    step_file = step02140
    step_path = str(Path(home_path).joinpath(step_file))

    project = "02140"
    # project = "test"
    # project = "02086"
    grade = "Mixed"

    dstv_pipeline(step_file, step_path, project, grade)