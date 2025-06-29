import pandas as pd
from pathlib import Path
import os


def record_solid(
    report_rows,
    name: str,
    step_path: str,  # path to the file this solid created
    thumb_path: str,  # path to a thumbnail image for that output
    drilling_path: str,
    dxf_path: str,
    nc1_path: str,
    brep_path: str,
    mass: str,
    obb_x: int,
    obb_y: int,
    obb_z: int,
    obj_type: str,
    issues: str,
    hash: str,
    dxf_thumb_path : str,
    section_shape : str,
    assembly_hash : str,
    ):
    row = {
        "Item ID": name,
        "Item Type": obj_type,
        "STEP File": Path(rf"{step_path}"),
        "Image": Path(rf"{thumb_path}"),
        "Drilling Drawing": Path(rf"{drilling_path}"),
        "DXF Thumb": Path(rf"{dxf_thumb_path}"),
        "Profile DXF": Path(rf"{dxf_path}"),
        "Section Shape": section_shape,
        "NC1 File": Path(rf"{nc1_path}"),
        "BREP": Path(rf"{brep_path}"),
        "Mass (kg)": mass,
        "X (mm)": obb_x,
        "Y (mm)": obb_y,
        "Z (mm)": obb_z,
        "Issues": issues,
        "Hash": hash,
        "Assembly Hash" : assembly_hash
    }
    return report_rows.append(row)

def df_to_html_with_images(df, output_dir, project_number):
    """
    Exports a DataFrame to a styled, sortable HTML report with:
      - Clickable file:/// links
      - Thumbnail images
      - Custom CSS
      - Column‐header sorting via sorttable.js
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{project_number}.html"

    # 1) Prepare a copy for HTML
    df_html = df.copy()

    def make_file_link(path):
        p = Path(path).resolve()
        return f'<a href="{p.as_uri()}" download target="_blank">{p.name}</a>'

    for col in ['STEP File', 'Drilling Drawing', 'Profile DXF', 'NC1 File', 'BREP']:
        if col in df_html:
            df_html[col] = df_html[col].apply(make_file_link)

    for col in ['Image', 'DXF Thumb']:
        if col in df_html:
            df_html[col] = df_html[col].apply(
            lambda p: f'<img src="{p}" style="max-width:120px; height:auto; border-radius:4px;"/>'
        )

    # 2) Get the table HTML with a `sortable` class
    table_html = df_html.to_html(
        escape=False,
        index=False,
        classes="sortable"   # ← pandas will emit <table class="sortable" …>
    )

    # 3) Embed CSS + sorttable.js
    css = """
    <style>
      body { font-family: Arial, sans-serif; padding: 1em; }
      h1 { font-size: 1.5em; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 0.5em; vertical-align: middle; cursor: pointer; }
      th { background-color: #f4f4f4; }
      tr:nth-child(even) { background-color: #fafafa; }
      tr:hover { background-color: #f1f1f1; }
      a { color: #0077cc; text-decoration: none; }
      a:hover { text-decoration: underline; }
      img { display: block; margin: 0 auto; }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Project {project_number} Report</title>
    {css}
    <!-- sorttable.js makes any table with class="sortable" clickable -->
    <script src="https://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
  </head>
  <body>
    <h1>Project {project_number} Pipeline Report</h1>
    {table_html}
  </body>
</html>
"""

    print(f"✅ Writing report to HTML at {html_path}")
    html_path.write_text(html, encoding="utf-8")



# 3b) Export to Excel with images inserted into cells (using XlsxWriter):
import os
from pathlib import Path
import pandas as pd

def df_to_excel_with_images(
    df: pd.DataFrame,
    excel_dir: str,
    project_number: str,
    file_cols=None,
    image_col="Image"
):
    """
    Exports df to <excel_dir>/<project_number>.xlsx, inserting:
      • Clickable external: links for any column in file_cols
      • Embedded thumbnails for image_col
    """
    # 1) Defaults for your schema
    if file_cols is None:
        file_cols = ["STEP File", "Drilling Drawing", "Profile DXF", "NC1 File"]

    excel_dir = Path(excel_dir)
    excel_dir.mkdir(parents=True, exist_ok=True)
    out_path = excel_dir / f"{project_number}.xlsx"

    # 2) Get column indices
    file_col_inds = [(c, df.columns.get_loc(c)) for c in file_cols if c in df.columns]
    img_col_ind  = df.columns.get_loc(image_col) if image_col in df.columns else None

    # 3) Write DataFrame and then overwrite links/images
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        sheet = "Report"
        df.to_excel(writer, sheet_name=sheet, index=False, startrow=1)
        wb  = writer.book
        ws  = writer.sheets[sheet]

        # 4) Header row styling
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#F0F0F0"})
        for col_idx, hdr in enumerate(df.columns):
            ws.write(0, col_idx, hdr, hdr_fmt)

        # 5) Loop over each row record
        for row_idx, record in enumerate(df.to_dict(orient="records"), start=1):
            # a) hyperlinks for each file column
            for col_name, col_idx in file_col_inds:
                fp = record[col_name]
                if fp:
                    ws.write_url(
                        row_idx, col_idx,
                        f"external:{fp}",
                        string=Path(fp).name
                    )
            # b) insert image if column exists
            if img_col_ind is not None:
                thumb = record[image_col]
                if thumb and Path(thumb).exists():
                    ws.set_row(row_idx, 80)  # enough height
                    ws.insert_image(
                        row_idx, img_col_ind,
                        thumb,
                        {"x_scale": 0.5, "y_scale": 0.5, "x_offset": 2, "y_offset": 2}
                    )

    print(f"Wrote Excel report to {out_path}")
