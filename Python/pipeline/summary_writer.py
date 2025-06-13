import pandas as pd

def section_result_table(result, precision=2):
    """
    Pretty print classification result as a one-row DataFrame with rounded floats.
    """
    if not result:
        print("No classification result.")
        return

    step_vals = result["STEP"]
    json_vals = result["JSON"]
    
    data = {
        "Field": [
            "Designation", "Category", "Profile Type", "Match Score", "Rotation Required",
            "Height (mm)", "Width (mm)", "CSA (mm²)", "Length (mm)", "Mass (Kg)",
            "Web Thickness (mm)", "Flange Thickness (mm)", "Root Radius (mm)", "Toe Radius (mm)"
        ],
        "STEP File": [
            result["Designation"],
            result["Category"],
            result["Profile_type"],
            f"{result['Match_score']:.2f}",
            result["Requires_rotation"],
            f"{step_vals['height']:.2f}",
            f"{step_vals['width']:.2f}",
            f"{step_vals['area']:.2f}",
            f"{step_vals['length']:.2f}",
            f"{step_vals['mass']:.2f}",
            "","","",""
        ],
        "JSON Spec": [
            "", "", "", "", "",
            f"{json_vals['height']:.2f}",
            f"{json_vals['width']:.2f}",
            f"{json_vals['csa']:.2f}",
            f"{json_vals['length']:.2f}",
            "",
            f"{json_vals['web_thickness']:.2f}",
            f"{json_vals['flange_thickness']:.2f}",
            f"{json_vals['root_radius']:.2f}",
            f"{json_vals['toe_radius']:.2f}"
        ]
    }

    dataframe = pd.DataFrame(data)

    return dataframe




