import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import BytesIO
from base64 import b64encode
import os
import pandas as pd

def generate_hole_projection_html(
    df_holes,
    header_data,
    logo_path,
    drawing_path
):

    print("Generating Drawing")

    logo = logo_path+"\PSS_Standard_RGB.png"
    df = df_holes.copy()
    df_sorted = df.sort_values(by=["Code", "Diameter (mm)", "X (mm)", "Y (mm)"]).copy()
    df_sorted["Hole ID"] = (
        df_sorted.groupby("Code").cumcount() + 1
    ).astype(str).str.zfill(2)
    df_sorted["ID"] = df_sorted["Code"] + "-" + df_sorted["Hole ID"]

    length = header_data["Length"]
    y_min, y_max = df_sorted["Y (mm)"].min(), df_sorted["Y (mm)"].max()

    # Set panel layout based on code_profile
    profile_type = header_data.get("code_profile", "").upper()
    if profile_type == "L":
        face_codes = ["V", "U"]
    else:
        face_codes = ["O", "V", "U"]
    n_faces = len(face_codes)
    
    def plot_face(face_code, ax, title):
        from matplotlib.ticker import MaxNLocator
    
        face_df = df_sorted[df_sorted["Code"] == face_code]
        ax.set_title(f"{title} View ({face_code} Face)", fontsize=10)
        ax.set_aspect('equal')
    
        # Actual axis max values
        x_max = float(header_data.get("Length", 2000))
        y_max = (
            float(header_data.get("Height", 300)) if face_code == "V"
            else float(header_data.get("Width", 100))
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
    
        # Function to generate ticks <= max, and append max if missing
        def make_ticks(max_val, locator, threshold=0.02):
            ticks = [tick for tick in locator.tick_values(0, max_val) if tick <= max_val]
            if ticks:
                last_tick = max(ticks)
                if abs(max_val - last_tick) / max_val > threshold:
                    ticks.append(max_val)
            else:
                ticks.append(max_val)
            return sorted(set(round(t, 2) for t in ticks))
    
        x_locator = MaxNLocator(nbins=6, integer=True)
        y_locator = MaxNLocator(nbins=5, integer=True)
    
        ax.set_xticks(make_ticks(x_max, x_locator))
        ax.set_yticks(make_ticks(y_max, y_locator))
    
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle="--", linewidth=0.5)
    
        # Plot hole positions and labels
        for i, (_, row) in enumerate(face_df.iterrows()):
            x, y = row["X (mm)"], row["Y (mm)"]
            ax.plot(x, y, marker='+', color='black', markersize=8, mew=1.5)
            dy = -30 if i % 2 == 0 else 20
            ax.text(x, y + dy, row["ID"], fontsize=6, ha='center')



    # Create figure with fixed panels
    fig, axs = plt.subplots(n_faces, 1, figsize=(12, 3 * n_faces))
    if n_faces == 1:
        axs = [axs]
    face_labels = {"O": "Top", "V": "Middle", "U": "Bottom", "H": "Side"}
    for ax, face in zip(axs, face_codes):
        label = face_labels.get(face, face)
        plot_face(face, ax, label)
    plt.tight_layout()

    # Convert figure to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Convert logo to base64
    with open(logo, "rb") as f:
        logo_base64 = b64encode(f.read()).decode("utf-8")
    logo_data_url = f"data:image/png;base64,{logo_base64}"

    # Title block HTML
    title_block_html = f"""
    <table style="width:100%; border: 1px solid black; border-collapse: collapse; font-size: 14px; margin: 0;">
      <tr>
        <td rowspan="4" style="border: 1px solid black; text-align:center; width: 160px;">
          <img src="{logo_data_url}" style="max-height:100px; max-width:140px;" alt="Logo" />
        </td>
        <td style="border: 1px solid black;"><strong>Project</strong></td>
        <td style="border: 1px solid black;">{header_data['project_number']}</td>
        <td style="border: 1px solid black;"><strong>NC File</strong></td>
        <td style="border: 1px solid black;">{header_data['out_filename']}</td>
      </tr>
      <tr>
        <td style="border: 1px solid black;"><strong>Designation</strong></td>
        <td style="border: 1px solid black;">{header_data['Designation']}</td>
        <td style="border: 1px solid black;"><strong>Length</strong></td>
        <td style="border: 1px solid black;">{header_data['Length']:.0f}mm</td>
      </tr>
      <tr>
        <td style="border: 1px solid black;"><strong>Mass</strong></td>
        <td style="border: 1px solid black;">{header_data['Mass']:.2f}kg</td>
        <td style="border: 1px solid black;"><strong>Material</strong></td>
        <td style="border: 1px solid black;">{header_data['material_grade']}</td>
      </tr>
      <tr>
        <td style="border: 1px solid black;"><strong>Quantity</strong></td>
        <td style="border: 1px solid black;">{header_data['quantity']}</td>
        <td style="border: 1px solid black;"><strong>Doc Ref</strong></td>
        <td style="border: 1px solid black;">{header_data['out_filename']}</td>
      </tr>
    </table>
    """

    # HTML document
    html = f"""
    <html>
    <head>
    <style>
      body {{
        font-family: monospace;
        margin: 0;
        padding: 0;
        border: 4px solid black;
      }}
      img {{
        display: block;
        margin: 0 auto;
        max-width: 100%;
        width: 100%;
      }}
      h2 {{
        text-align: center;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
      }}
      th, td {{
        border: 1px solid black;
        padding: 4px 8px;
        text-align: center;
      }}
      @media print {{
        img {{
          page-break-after: never;
        }}
      }}
    </style>
    </head>
    <body>
        <h2>{header_data['out_filename']} Drilling Schedule</h2>
        <img src="data:image/png;base64,{img_base64}" />
    """

    # Add hole tables per face
    for face in face_codes:
        face_df = df_sorted[df_sorted["Code"] == face][["ID", "X (mm)", "Y (mm)", "Diameter (mm)"]].reset_index(drop=True)
        html += f"<h2>Hole Table - {face} Face</h2><table><tr><th>ID</th><th>X (mm)</th><th>Y (mm)</th><th>Diameter (mm)</th></tr>"
        for _, row in face_df.iterrows():
            html += f"<tr><td>{row['ID']}</td><td>{row['X (mm)']}</td><td>{row['Y (mm)']}</td><td>{row['Diameter (mm)']}</td></tr>"
        html += "</table>"

    html += '<div style="height: 40px;"></div>' + title_block_html
    html += "</body></html>"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(drawing_path), exist_ok=True)
    
    # Write to file
    html_filename = f"{drawing_path}\{header_data['out_filename']}.html"
    with open(f"{html_filename}", "w") as f:
        f.write(html)
    
    print(f"✅ Full hole drawing saved to: {header_data['out_filename']}.html")
    return html_filename