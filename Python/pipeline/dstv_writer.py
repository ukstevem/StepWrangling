import os
import pandas as pd
import pathlib


def create_project_directories(project_number):
    """
    Creates the output directory structure for a project if it doesn't exist.
    Returns the base path and subfolder paths.
    """
    base_path = os.path.join("../../data", "Projects", project_number)
    nc1_path = os.path.join(base_path, "Nc1")
    reports_path = os.path.join(base_path, "Reports")
    drawings_path = os.path.join(base_path, "Drawings")

    for path in [nc1_path, reports_path, drawings_path]:
        os.makedirs(path, exist_ok=True)

    return base_path, nc1_path, reports_path, drawings_path

def assemble_dstv_header_data(project_number, step_path, matl_grade, member_id, profile_match):
    """
    Assembles header data dictionary for DSTV NC1 file.
    """
    filename_stem = pathlib.Path(step_path).stem
    out_filename = f"{filename_stem}-{member_id}"
    model_filename = f"{filename_stem}.step"

    step_vals = profile_match["STEP"]

    header_dict = {
        "project_number": project_number,
        "model_filename": model_filename,
        "material_grade": matl_grade,
        "quantity": 1,
        "Designation": profile_match["Designation"],
        "Mass": step_vals["mass"],
        "Height": step_vals["height"],
        "Width": step_vals["width"],
        "CSA": step_vals["area"],
        "web_thickness": step_vals["web_thickness"],
        "flange_thickness": step_vals["flange_thickness"],
        "root_radius": step_vals["root_radius"],
        "toe_radius": step_vals["toe_radius"],
        "code_profile": profile_match["Profile_type"],
        "Length": step_vals["length"],
        "out_filename": out_filename
    }

    return header_dict

def generate_nc1_file(df_holes, header_data, nc_path):

    """
    Write DSTV NC1 ST-block header to .nc1 file with specified order
    """
    faces = ['V', 'U', 'O']  #Face priority order

    filename = f"{nc_path}\{header_data['out_filename']}.nc1"
    with open(filename, 'w') as f:
        f.write('ST\n')
        f.write(f"  {header_data['project_number']}\n")
        f.write(f"  {header_data['model_filename']}\n")
        f.write(f"  Drill-Cut\n")
        f.write(f"  {header_data['out_filename']}\n")
        f.write(f"  {header_data['material_grade']}\n")
        f.write(f"  {header_data['quantity']}\n")
        f.write(f"  {header_data['Designation']}\n")
        f.write(f"  {header_data['code_profile']}\n")
        f.write(f"    {header_data['Length']:8.2f}\n")
        f.write(f"    {header_data['Height']:8.2f}\n")
        f.write(f"    {header_data['Width']:8.2f}\n")
        f.write(f"    {header_data['flange_thickness']:8.2f}\n")
        f.write(f"    {header_data['web_thickness']:8.2f}\n")
        f.write(f"    {header_data['root_radius']:8.2f}\n")
        f.write(f"    {header_data['Mass']:8.2f}\n")
        f.write('        0.00\n') #surface area
        # Following the spec, three zeros
        f.write('        0.00\n')
        f.write('        0.00\n')
        f.write('        0.00\n')
        f.write('        0.00\n')
        f.write('  -\n')
        f.write('  -\n')
        f.write('  -\n')
        f.write('  -\n')
        
        # BO blocks by face
        for face in faces:
            df_face = df_holes[df_holes['Code'] == face]
            if df_face.empty:
                continue
            f.write('BO\n')
            for _, row in df_face.iterrows():
                x = row['X (mm)']
                y = row['Y (mm)']
                d = row['Diameter (mm)']
                # Right-align numeric columns: x, y (8-wide), diameter (6-wide)
                f.write(f"  {face.lower()}  {x:8.2f} {y:8.2f} {d:6.2f}\n")
                
        f.write('EN\n')
    print(f"DSTV header written to {filename}")