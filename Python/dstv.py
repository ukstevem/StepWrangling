# main.py - Entry point for DSTV pipeline

import os
import pandas as pd
from datetime import datetime
from pipeline.step_loader import load_step_file
from pipeline.geometry_utils import (robust_align_solid_from_geometry, 
                                     ensure_right_handed, 
                                     compute_obb_geometry, 
                                     compute_section_area,
                                     swap_width_and_height_if_required,
                                     compute_dstv_origin,
                                     align_obb_to_dstv_frame,
                                     refine_profile_orientation
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
from pipeline.database import (dump_df_to_supabase, normalize_report_df)
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
from pathlib import Path



def dstv_pipeline(root_model, project_number, matl_grade):


    # Generate folder structure and get paths

    # Static
    BASE = Path(__file__).parent
    media_path = BASE / "media"
    json_path = BASE / "data/Shape_classifier_info.json"
    network_root = BASE / "G:/Customer Orders/Extraction"
    fallback_root = BASE / "C:/dev/step-gemini/Extraction"

    # Project Driven
    base_path, nc_path, report_path, drilling_path, cad_path, thumb_path, step_path, brep_path, dxf_path, dxf_thumb = create_project_directories(project_number, network_root, fallback_root)

    if not os.path.isfile(root_model):
        raise FileNotFoundError(f"Couldn’t find STEP file at: {root_model!r}")
    solids = load_step_file(root_model)
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
        step_mass = ""
        obb_x = "0"
        obb_y = "0"
        obb_z = "0"
        object_type = "-"
        issues = "-"
        hash = ""
        dxf_thumb_file = ""
        print(f"\n🟦 Processing {member_id}... {datetime.now().strftime('%H:%M:%S')}")

        try:

            # Thumbnail
            thumbnail_file = shape_to_thumbnail(shape_orig, thumb_path, member_id)

            # STEP 1: Align to major geometry
            primary_aligned_shape, trsf, cs, largest_face, dir_x, dir_y, dir_z = robust_align_solid_from_geometry(shape_orig)
            dir_x, dir_y, dir_z = ensure_right_handed(dir_x, dir_y, dir_z)

            # STEP 2: Compute OBB geometry
            obb_geom = compute_obb_geometry(primary_aligned_shape)
            
            # STEP 3: Section area
            # section_area = compute_section_area(primary_aligned_shape)
            section_area = compute_section_area(primary_aligned_shape)

            # print(f"Section area: {section_area:.2f}")

            # STEP 4: Classification signature
            cs_data = {
                "span_web": obb_geom["aligned_extents"][1],
                "span_flange": obb_geom["aligned_extents"][2],
                "area": section_area,
                "length": obb_geom["aligned_extents"][0]
            }

            # print(cs_data)

            # STEP 5: Match profile
            profile_match = classify_profile(cs_data, json_path, tol_dim=1.0, tol_area=.05)
            
            # print(f"Profile Match Data: {profile_match['JSON']}\n {profile_match['STEP']}")

            # STEP 5.5: no match? check for plate
            if profile_match is None:
                is_plate, final_aligned_solid, ax3, thickness_mm, length_mm, width_mm, step_mass, msg = align_plate_to_xy_plane(primary_aligned_shape)
                                
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
                    print("ℹ️  Object not identified", msg)
                    object_type = "Other"
                    hash =""

                step_file = export_solid_to_step(final_aligned_solid, step_path, member_id)
                brep_fingerprint, brep_file = export_solid_to_brep(final_aligned_solid, brep_path, member_id)

            else:
                print("✅ Section match found")

                # STEP 6: Adjust orientation if needed
                primary_aligned_shape, obb_geom = swap_width_and_height_if_required(
                    profile_match, primary_aligned_shape, obb_geom
                )

                print(f"🔄 Post-swap extents: {obb_geom['aligned_extents']}")

                # STEP 7: Compute DSTV origin and align
                origin_local = compute_dstv_origin(
                    obb_geom["aligned_center"],
                    obb_geom["aligned_extents"],
                    obb_geom["aligned_dir_x"],
                    obb_geom["aligned_dir_y"],
                    obb_geom["aligned_dir_z"]
                )

                primary_aligned_shape, trsf = align_obb_to_dstv_frame(
                    primary_aligned_shape,
                    origin_local,
                    obb_geom["aligned_dir_x"],
                    obb_geom["aligned_dir_y"],
                    obb_geom["aligned_dir_z"]
                )

                # STEP 8: Final orientation refinement
                refined_shape, obb_geom = refine_profile_orientation(
                    primary_aligned_shape, profile_match, compute_obb_geometry(primary_aligned_shape)
                )

                print(f"✅ Orientation refinement complete. New extents: {obb_geom['aligned_extents']}")

                # cad output
                # export_solid_to_brep(shape_orig, brep_path, member_id)
                object_type = f"{profile_match['Profile_type']} {profile_match['Designation']}"
                step_file = export_solid_to_step(refined_shape, step_path, member_id)
                brep_fingerprint, brep_file = export_solid_to_brep(refined_shape, brep_path, member_id)


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
                                                            root_model, 
                                                            matl_grade, 
                                                            member_id, 
                                                            profile_match)

                # STEP 10: Output
                # NC1               
                nc1_file, hash = generate_nc1_file(raw_df_holes, dstv_header_data, nc_path, web_cuts)
                # Drawing
                html_file = generate_hole_projection_html(raw_df_holes, dstv_header_data, media_path, drilling_path, web_cuts)


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
                     dxf_thumb_path = dxf_thumb_file
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
                     dxf_thumb_path = dxf_thumb_file
                    )

    report_df = pd.DataFrame(report_rows)
    # save_summary_table(summary_df, os.path.join(OUTPUT_DIR, "summary.html"))
    print("\n✅ Pipeline completed. Summary saved.")
    # report_df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

    # Create HTML report
    df_to_html_with_images(report_df, report_path, project_number)

    # Log to Supabase
    report_for_db = normalize_report_df(report_df, project_number)
    dump_df_to_supabase(report_for_db)



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

    step_file = step1   
    step_path = str(Path(home_path).joinpath(step_file))

    # project = "10206"
    project = "10206"
    grade = "S355"

    dstv_pipeline(step_path, project, grade)