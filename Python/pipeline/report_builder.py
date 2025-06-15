import pandas as pd
from pathlib import Path
import os


def record_solid(
    report_rows,
    name: str,
    step_path: str,  # path to the file this solid created
    thumb_path: str,  # path to a thumbnail image for that output
    drawing_path: str,
    dxf_path: str,
    nc1_path: str,
    obb_x: int,
    obb_y: int,
    obb_z: int,
    obj_type: str,
    issues: str
    ):
    row = {
        "Item ID": name,
        "Item Type": obj_type,
        "STEP File": Path(rf"../../{step_path}"),
        "Image": Path(rf"../../{thumb_path}"),
        "Drilling Drawing": Path(rf"../../{drawing_path}"),
        "Profile DXF": Path(rf"../../{dxf_path}"),
        "NC1 File": Path(rf"../../{nc1_path}"),
        "X": obb_x,
        "Y": obb_y,
        "Z": obb_z,
        "Issues": issues
    }
    return report_rows.append(row)


def df_to_html_with_images(df, output_dir, project_number):
    """
    Exports a DataFrame to a styled HTML report with:
      - Clickable file:/// links that launch native apps
      - Thumbnail images
      - Custom CSS for a cleaner look
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{project_number}.html"

    # 1) Prepare a copy for HTML
    df_html = df.copy()

    # Helper to make file:// links
    def make_file_link(path):
        p = Path(path).resolve()
        uri = p.as_uri()  # yields file:///C:/...
        name = p.name
        return f'<a href="{uri}" target="_blank">{name}</a>'

    # Columns with files
    for col in ['STEP File', 'Drilling Drawing', 'Profile DXF', 'NC1 File']:
        if col in df_html:
            df_html[col] = df_html[col].apply(make_file_link)

    # Image thumbnails
    if 'Image' in df_html:
        df_html['Image'] = df_html['Image'].apply(
            lambda p: f'<img src="{p}" style="max-width:120px; height:auto; border-radius:4px;"/>'
        )

    # 2) Build the full HTML with embedded CSS
    css = """
    <style>
      body { font-family: Arial, sans-serif; padding: 1em; }
      h1 { font-size: 1.5em; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 0.5em; vertical-align: middle; }
      th { background-color: #f4f4f4; text-align: left; }
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
  </head>
  <body>
    <h1>Project {project_number} Pipeline Report</h1>
    {df_html.to_html(escape=False, index=False)}
  </body>
</html>
"""

    # 3) Write out
    print(f"Writing report to HTML at {html_path}")
    html_path.write_text(html, encoding="utf-8")




# # 3b) Export to Excel with images inserted into cells (using XlsxWriter):
# def df_to_excel_with_images(df, excel_path):
#     with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
#         df.to_excel(writer, sheet_name="report", index=False, startrow=1)
#         workbook  = writer.book
#         worksheet = writer.sheets["report"]

#         # Write header formatting (optional)
#         header_format = workbook.add_format({"bold": True, "bg_color": "#F0F0F0"})
#         for col_num, value in enumerate(df.columns.values):
#             worksheet.write(0, col_num, value, header_format)

#         # Insert hyperlinks and thumbnails
#         for row_idx, row in enumerate(report_rows, start=1):
#             # hyperlink in “output_file” column
#             file_col = df.columns.get_loc("output_file")
#             worksheet.write_url(
#                 row_idx, file_col,
#                 f"external:{row['output_file']}",
#                 string=os.path.basename(row['output_file'])
#             )
#             # image in “thumbnail” column
#             thumb_col = df.columns.get_loc("thumbnail")
#             worksheet.set_row(row_idx, 80)  # adjust row height
#             worksheet.insert_image(
#                 row_idx, thumb_col,
#                 row["thumbnail"],
#                 {"x_scale": 0.5, "y_scale": 0.5, "x_offset": 2, "y_offset": 2}
#             )

# df_to_excel_with_images(df, "pipeline_report.xlsx")
