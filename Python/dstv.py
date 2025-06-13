# main.py - Entry point for DSTV pipeline

import os
import pandas as pd
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
                                    check_duplicate_holes
                                     )
from pipeline.dstv_writer import (generate_nc1_file, 
                                  create_project_directories, 
                                  assemble_dstv_header_data)
from pipeline.drawing import generate_hole_projection_html
from pipeline.cad_out import (export_solid_to_brep, 
                              shape_to_thumbnail, 
                              export_solid_to_step)

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3

def dstv_pipeline(root_model, project_number, matl_grade):

    # Generate folder structure and get paths

    # Project Driven
    base_path, nc_path, report_path, drawing_path, cad_path, thumb_path, step_path, brep_path = create_project_directories(project_number)

    # Static
    media_path = "../../data/media/"
    json_path = "../../data/Shape_classifier_info.json"

    solids = load_step_file(root_model)
    results = []

    for i, shape_orig in enumerate(solids):
        member_id = f"MEM-{i+1:03d}"
        print(f"\n🟦 Processing {member_id}...")

        try:

            # Thumbnail
            shape_to_thumbnail(shape_orig, thumb_path, member_id)
            # cad output
            # export_solid_to_brep(shape_orig, brep_path, member_id)
            export_solid_to_step(shape_orig, step_path, member_id)

            # STEP 1: Align to major geometry
            primary_aligned_shape, trsf, cs, largest_face, dir_x, dir_y, dir_z = robust_align_solid_from_geometry(shape_orig)
            dir_x, dir_y, dir_z = ensure_right_handed(dir_x, dir_y, dir_z)

            # STEP 2: Compute OBB geometry
            obb_geom = compute_obb_geometry(primary_aligned_shape)

            # print(f"📏 Extents (L x H x W): {obb_geom['aligned_extents']}")
            # print(f"📍 Center: {obb_geom['aligned_center'].X():.2f}, {obb_geom['aligned_center'].Y():.2f}, {obb_geom['aligned_center'].Z():.2f}")

            # STEP 3: Section area
            # section_area = compute_section_area(primary_aligned_shape)
            section_area = compute_section_area(primary_aligned_shape)

            print(f"Section area: {section_area:.2f}")

            # STEP 4: Classification signature
            cs_data = {
                "span_web": obb_geom["aligned_extents"][1],
                "span_flange": obb_geom["aligned_extents"][2],
                "area": section_area,
                "length": obb_geom["aligned_extents"][0]
            }

            # STEP 5: Match profile
            profile_match = classify_profile(cs_data, json_path, tol_dim=1.0, tol_area=5)
            if profile_match is None:
                raise ValueError("❌ No matching profile found")
            # else:
                # print("✅ Match found:")
                # print(f"  → Designation: {profile_match['Designation']}")
                # print(f"  → Category: {profile_match['Category']}")
                # print(f"  → Type: {profile_match['Profile_type']}")
                # print(f"  → Requires rotation: {profile_match['Requires_rotation']}")

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

            ## define the DSTV frame and axis direction from geometry
            dstv_frame = gp_Ax3(
                gp_Pnt(0, 0, 0),
                gp_Dir(obb_geom["aligned_dir_z"].XYZ()),
                gp_Dir(obb_geom["aligned_dir_x"].XYZ())
            )

            # print(section_result_table(profile_match))

            # STEP 9: Classify and check holes
            # raw_df_holes, hole_data, origin_nc1, L, F, W = classify_and_project_holes_dstv(
            #     refined_shape, dstv_frame
            # )
            step_vals = profile_match["STEP"]
            raw_df_holes = classify_and_project_holes_dstv(refined_shape, 
                                                            dstv_frame,
                                                            dstv_frame.Location(),
                                                            step_vals["width"],
                                                            step_vals["height"])

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
            generate_nc1_file(raw_df_holes, dstv_header_data, nc_path)
            # Drawing
            generate_hole_projection_html(raw_df_holes, dstv_header_data, media_path, drawing_path)
            
            # results.append({
            #     "ID": member_id,
            #     "Designation": profile_match.get("designation"),
            #     "Weight": round(L * F * W * 7.85e-6, 2),
            #     "Comments": "✓"
            # })

        except Exception as e:
            print(f"❌ Error in {member_id}: {e}")
            results.append({
                "ID": member_id,
                "Designation": "-",
                "Weight": 0,
                "Comments": str(e)
            })


    # # Export summary
    # summary_df = pd.DataFrame(results)
    # save_summary_table(summary_df, os.path.join(OUTPUT_DIR, "summary.html"))
    # print("\n✅ Pipeline completed. Summary saved.")

if __name__ == "__main__":

    # step_path = "../../data/C25001-1-0101.step"
    step_path = "../../data/MEM-026.step"
    # step_path = "../../data/0444-1 ANGLED.step"
    # step_path = "../../data/ncTest.step"
    # step_path = "../../data/TestEA.step"
    # step_path = "../../data/TestEAMirror.step"
    # step_path = "../../data/TestUEA.step"
    # step_path = "../../data/TestUEAMirror.step"
    # step_path = "../../data/TestPFC.step"

    dstv_pipeline(step_path, "10206", "S355")