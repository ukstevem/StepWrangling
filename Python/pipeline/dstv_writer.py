import os
import tempfile
from pathlib import Path

def assemble_dstv_header_data(project_number, step_path, matl_grade, member_id, profile_match):
    """
    Assembles header data dictionary for DSTV NC1 file.
    """
    filename_stem = Path(step_path).stem
    out_filename = f"{filename_stem}-{member_id}"
    model_filename = f"{filename_stem}.step"

    step_vals = profile_match["STEP"]
    # print(profile_match)

    header_dict = {
        "project_number": project_number,
        "model_filename": model_filename,
        "material_grade": matl_grade,
        "quantity": 1,
        "Designation": f'{profile_match["Category"]}{profile_match["Designation"]}',
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

import hashlib
from pathlib import Path

def nc1_group_key(nc1_path: Path | str, skip_first: int = 5) -> str:
    """
    Return the MD5 of the NC1 file *from* line skip_first+1 onward.
    Lines are split on '\n' (after normalizing CRLF → LF), and then
    re-joined with '\n' before hashing.
    """
    p = Path(nc1_path)
    # read & normalize line endings
    text = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")

    # drop the first `skip_first` lines
    relevant = lines[skip_first:]
    # if you want to preserve a trailing newline, you can do:
    # data = "\n".join(relevant) + "\n"
    data = "\n".join(relevant)

    return hashlib.md5(data.encode("utf-8")).hexdigest()


def generate_nc1_file(df_holes, header_data, nc_dir, web_cut) -> tuple[Path, str]:
    """
    Write a DSTV NC1 to nc_dir/<out_filename>.nc1, then return (path, hash).
    """
    nc_dir = Path(nc_dir)
    nc_dir.mkdir(parents=True, exist_ok=True)

    out_file = nc_dir / f"{header_data['out_filename']}.nc1"

    # Write with explicit '\n' line endings:
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ST\n")
        f.write(f"  {header_data['project_number']}\n")
        f.write(f"  {header_data['model_filename']}\n")
        f.write("  Drill-Cut\n")
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
        f.write("        0.00\n")  # surface area
        f.write(f"   {web_cut['start_web']:8.1f}\n")
        f.write(f"   {web_cut['end_web']:8.1f}\n")
        f.write(f"   {web_cut['start_flange']:8.1f}\n")
        f.write(f"   {web_cut['end_flange']:8.1f}\n")
        f.write("  -\n" * 4)
        if df_holes.empty:
            pass
        else:
        # BO blocks by face
            for face in ['V', 'U', 'O']:
                df_face = df_holes[df_holes['Code'] == face]
                if df_face.empty:
                    continue
                f.write("BO\n")
                for _, row in df_face.iterrows():
                    x = row['X (mm)']
                    y = row['Y (mm)']
                    d = row['Diameter (mm)']
                    f.write(f"  {face.lower()}  {x:8.2f} {y:8.2f} {d:6.2f}\n")

        f.write("EN\n")

    # Compute the hash of the freshly written file:
    nc1_hash = nc1_group_key(out_file)

    print(f"✅ DSTV written to {out_file}, hash={nc1_hash}")
    return out_file, nc1_hash
