import os
from pathlib import Path
from html import escape
from urllib.parse import quote
import pandas as pd

_PATH_COLS = [
    "step_path", "stl_path", "brep_path", "nc1_path", "drilling_path",
    "dxf_path", "dxf_thumb_path", "thumb_path"
]

# ---------- path helpers ----------

def _safe_rel_for_web(target: str | Path, start_dir: str | Path) -> tuple[str | None, str | None]:
    """
    Return a forward-slash relative path from start_dir -> target.
    If it can't be made (e.g., different drives on Windows), return (None, reason).
    """
    try:
        t = Path(target).resolve(strict=False)
        s = Path(start_dir).resolve(strict=False)
        rel = os.path.relpath(str(t), start=str(s))
        return rel.replace("\\", "/"), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _href_from_report_to(fp: str, report_dir: Path) -> tuple[str | None, str | None]:
    """
    Create a forward-slash relative URL from the report folder to the target file.
    Returns (rel_url, err).
    """
    try:
        rel = os.path.relpath(str(Path(fp).resolve()), start=str(report_dir.resolve()))
        return rel.replace("\\", "/"), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ---------- viewer writer ----------

def ensure_stl_viewer(out_dir) -> str:
    """
    Creates an HTML STL viewer in out_dir if it doesn't already exist.
    Returns the absolute path to the viewer file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viewer = out_dir / "stl_viewer.html"
    if viewer.exists():
        return str(viewer)

    viewer.write_text("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>STL Viewer (file:// friendly + ?file= over HTTP)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    html,body { margin:0; height:100%; background:#0f1115; color:#e5e7eb; font:14px system-ui, sans-serif; }
    #app { position:fixed; inset:0; }
    #msg {
      position: fixed; top:10px; left:50%; transform:translateX(-50%);
      background: rgba(0,0,0,.6); padding:8px 12px; border-radius:8px; z-index: 10;
    }
    #bar {
      position:fixed; top:10px; right:10px; z-index:11; display:flex; gap:8px; align-items:center;
    }
    #bar input[type=file] { display:none; }
    #bar label, #bar button {
      background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
      padding:6px 10px; border-radius:6px; cursor:pointer;
    }
    .drop { outline:2px dashed rgba(255,255,255,.3); outline-offset:-10px; }
    a { color:#8ab4ff; }
  </style>
</head>
<body>
  <div id="app"></div>
  <div id="msg">Loading…</div>
  <div id="bar">
    <input id="fileInput" type="file" accept=".stl" />
    <label for="fileInput">Choose .stl</label>
    <button id="helpBtn" title="Tips">?</button>
  </div>

  <!-- Three.js r146 global (UMD) build so OrbitControls/STLLoader exist on THREE.* -->
  <script src="https://unpkg.com/three@0.146.0/build/three.min.js"></script>
  <script src="https://unpkg.com/three@0.146.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://unpkg.com/three@0.146.0/examples/js/loaders/STLLoader.js"></script>

  <script>
  (function(){
    const app = document.getElementById('app');
    const msg = document.getElementById('msg');
    function setStatus(t, err=false){
      msg.textContent = t;
      msg.style.background = err ? 'rgba(180,0,0,.6)' : 'rgba(0,0,0,.6)';
      msg.style.display = t ? 'block' : 'none';
    }
    window.setStatus = setStatus;

    // --- Three.js scene ---
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('app').appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1115);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.001, 1e9);
    camera.position.set(4,2,4);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8); dir.position.set(3,4,5); scene.add(dir);

    const grid = new THREE.GridHelper(10, 10);
    grid.material.transparent = true; grid.material.opacity = 0.2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(1));

    function fit(obj){
      const box = new THREE.Box3().setFromObject(obj);
      const size = new THREE.Vector3(); box.getSize(size);
      const center = new THREE.Vector3(); box.getCenter(center);
      const maxDim = Math.max(size.x, size.y, size.z);
      const dist = (maxDim/2) / Math.tan(THREE.MathUtils.degToRad(camera.fov/2)) * 1.35;
      camera.position.copy(center).add(new THREE.Vector3(dist, dist*0.6, dist));
      camera.near = Math.max(dist/1000, 0.001); camera.far = dist*1000; camera.updateProjectionMatrix();
      controls.target.copy(center); controls.update();
    }

    function clearMeshes(){
      [...scene.children].forEach(o=>{
        if(o.isMesh){
          o.geometry.dispose();
          (Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());
          scene.remove(o);
        }
      });
    }

    function addMeshFromGeometry(geo){
      geo.computeVertexNormals(); geo.center();
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ metalness:0.15, roughness:0.85 }));
      clearMeshes(); scene.add(mesh); fit(mesh);
      return mesh;
    }

    function loadFromBuffer(arrayBuffer, name='(buffer)'){
      try{
        const loader = new THREE.STLLoader();
        const geo = loader.parse(arrayBuffer);
        addMeshFromGeometry(geo);
        setStatus(`Loaded ${name}`);
      }catch(e){
        console.error(e); setStatus('Failed to parse STL buffer', true);
      }
    }

    function loadFromURL(url){
      setStatus('Loading STL…');
      const loader = new THREE.STLLoader();
      console.log('[viewer] requesting STL:', url);
      loader.load(
        url,
        (geo)=>{
          try{
            addMeshFromGeometry(geo);
            setStatus('Loaded ' + url);
          }catch(e){
            console.error('[viewer] post-load processing failed:', e);
            setStatus('Failed to process STL after load', true);
          }
        },
        (xhr)=>{
          if (xhr && xhr.total) {
            const pct = Math.round((xhr.loaded / xhr.total) * 100);
            setStatus('Loading STL… ' + pct + '%');
          }
        },
        (err)=>{
          console.error('[viewer] load error:', err);
          setStatus('Failed to load STL (check relative path & that server root exposes /cad/).', true);
        }
      );
    }

    // Drag & Drop + File Input (works under file://)
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', e=>{
      const f = e.target.files && e.target.files[0]; if(!f) return;
      const r = new FileReader(); r.onload=()=>loadFromBuffer(r.result, f.name);
      r.onerror=()=>setStatus('Failed to read file', true);
      r.readAsArrayBuffer(f);
    });

    ['dragenter','dragover','dragleave','drop'].forEach(ev=>{
      window.addEventListener(ev, e=>{
        e.preventDefault(); e.stopPropagation();
        if(ev==='dragenter'||ev==='dragover') app.classList.add('drop');
        else if(ev==='dragleave'||ev==='drop') app.classList.remove('drop');
      });
    });

    window.addEventListener('drop', e=>{
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if(!f){ setStatus('No file dropped', true); return; }
      if(!/\\.stl$/i.test(f.name)){ setStatus('Please drop an .stl file', true); return; }
      const r = new FileReader(); r.onload=()=>loadFromBuffer(r.result, f.name);
      r.onerror=()=>setStatus('Failed to read dropped file', true);
      r.readAsArrayBuffer(f);
    });

    // ?file= support (normalize backslashes)
    const params = new URLSearchParams(location.search);
    const raw = params.get('file');
    const q = raw ? raw.replace(/\\\\/g, '/') : null;

    if (location.protocol === 'file:') {
      if (q) setStatus('This page is file:// — URL loads are blocked by the browser. Use “Choose .stl” or drag & drop.', true);
    } else if (q) {
      loadFromURL(q);
    } else {
      setStatus('Add ?file=relative/path.stl, or use “Choose .stl” or drag & drop.');
    }

    // Resize / Render
    window.addEventListener('resize', ()=>{
      camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    (function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();

    // Help
    document.getElementById('helpBtn').addEventListener('click', ()=>{
      alert([
        'Tips:',
        '• If this page is opened via file://, use “Choose .stl” or drag & drop.',
        '• For clickable links (?file=...), serve the folder:  python -m http.server 8000',
        '  and open  http://localhost:8000/Reports/0000.html',
        '• Start the server at the project root so /Reports and /cad are both visible.',
        '• Always use forward slashes in URLs: ../cad/stl/MEM-0001.stl'
      ].join('\\n'));
    });
  })();
  </script>
</body>
</html>
""", encoding="utf-8")

    return str(viewer)

# ---------- data capture ----------

def record_solid(report_rows, **kwargs):
    """
    Append one row of raw data. Underscore* keys are raw file paths; the HTML
    builder will turn them into links/thumbs later.
    """
    row = dict(kwargs)
    for k in _PATH_COLS:
        v = row.get(k, None)
        if isinstance(v, Path):
            v = str(v)
        if v is not None and isinstance(v, str):
            v = v.strip()
            if v == "":
                v = None
        row[k] = v
    report_rows.append(row)

    # ---------- CSV report ----------

def df_to_csv_exports(df: pd.DataFrame, output_dir, project_number: str) -> tuple[str, str]:
    """
    Exports two CSVs to <output_dir>/:
      • <project_number>.csv        -> display-friendly subset (like HTML table), file cols as relative paths/URIs
      • <project_number>_raw.csv    -> full raw df as provided

    Returns (display_csv_path, raw_csv_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Where the HTML report lives (we reuse this so relative links match your HTML)
    html_path = output_dir / f"{project_number}.html"
    report_dir = html_path.parent

    dfh = df.copy()

    # Ensure raw path columns exist
    for c in _PATH_COLS:
        if c not in dfh.columns:
            dfh[c] = None

    def _exists(p) -> bool:
        try:
            return p is not None and Path(p).is_file()
        except Exception:
            return False

    def _uri(p: str) -> str | None:
        try:
            return Path(p).resolve().as_uri()
        except Exception:
            return None

    def _rel_or_uri(p) -> str:
        """
        Prefer a web-relative path (so it matches the HTML report’s links).
        Fall back to file:// URI. Return empty string if missing.
        """
        if not _exists(p):
            return ""
        rel_web, err = _href_from_report_to(p, report_dir)
        if rel_web is not None:
            return rel_web.replace("\\", "/")
        u = _uri(p)
        return u or ""

    # --- STL viewer link support (mirrors HTML) ---
    viewer_path = Path(ensure_stl_viewer(output_dir))
    viewer_dir  = viewer_path.parent
    viewer_name = viewer_path.name

    def _stl_view_href(p: str) -> str:
        """
        Build `stl_viewer.html?file=<relative>` (empty if STL missing or path can't be built).
        Uses robust relative path from viewer_dir -> STL file.
        """
        if not _exists(p):
            return ""
        rel_web, err = _safe_rel_for_web(p, viewer_dir)
        if rel_web is None:
            # Can't compute a relative path from the viewer location (e.g., different drives)
            return ""
        return f"{viewer_name}?file={quote(rel_web, safe='/')}"

    # Build display columns (mirrors HTML's base columns)
    base_candidates = [
        ("name", "Name"),
        ("obj_type", "Type"),
        ("issues", "Issues"),
        ("mass", "Mass (kg)"),
        ("obb_x", "X (mm)"),
        ("obb_y", "Y (mm)"),
        ("obb_z", "Z (mm)"),
        # legacy fallbacks
        ("Item ID", "Name"),
        ("Item Type", "Type"),
        ("Issues", "Issues"),
        ("Mass (kg)", "Mass (kg)"),
        ("X (mm)", "X (mm)"),
        ("Y (mm)", "Y (mm)"),
        ("Z (mm)", "Z (mm)"),
    ]

    display_df = pd.DataFrame(index=dfh.index)
    for src, dst in base_candidates:
        if src in dfh.columns and dst not in display_df.columns:
            display_df[dst] = dfh[src]

    # File/path columns (store paths, not HTML/img) + STL View
    display_df["STEP File"]   = dfh["step_path"].apply(_rel_or_uri)
    display_df["STL"]         = dfh["stl_path"].apply(_rel_or_uri)
    display_df["STL View"]    = dfh["stl_path"].apply(_stl_view_href)  # <-- new column
    display_df["BREP"]        = dfh["brep_path"].apply(_rel_or_uri)
    display_df["NC1 File"]    = dfh["nc1_path"].apply(_rel_or_uri)
    display_df["Drilling Drawing"] = dfh["drilling_path"].apply(_rel_or_uri)
    display_df["Profile DXF"] = dfh["dxf_path"].apply(_rel_or_uri)
    display_df["DXF Thumb"]   = dfh["dxf_thumb_path"].apply(_rel_or_uri)
    display_df["Image"]       = dfh["thumb_path"].apply(_rel_or_uri)

    # Keep only columns that actually exist (nice order, with STL View after STL)
    file_cols = ["STEP File", "STL", "STL View", "BREP", "NC1 File", "Drilling Drawing", "Profile DXF", "DXF Thumb", "Image"]
    ordered_cols = [c for c in ["Name", "Type", "Issues", "Mass (kg)", "X (mm)", "Y (mm)", "Z (mm)"] if c in display_df.columns]
    ordered_cols += [c for c in file_cols if c in display_df.columns]
    display_df = display_df[ordered_cols]

    # Write CSVs
    display_csv = output_dir / f"{project_number}.csv"
    raw_csv     = output_dir / f"{project_number}_raw.csv"

    display_df.to_csv(display_csv, index=False, encoding="utf-8")
    dfh.to_csv(raw_csv, index=False, encoding="utf-8")

    print(f"✅ Wrote display CSV to {display_csv}")
    print(f"✅ Wrote raw CSV to {raw_csv}")
    return str(display_csv), str(raw_csv)



# ---------- HTML report ----------

def df_to_html_with_images(df: pd.DataFrame, output_dir, project_number: str) -> str:
    """
    Styled, sortable HTML report.
      • Emits web-relative links when possible (HTTP-friendly)
      • STL viewer link (opens stl_viewer.html) uses robust relative paths
      • Graceful fallbacks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{project_number}.html"
    report_dir = html_path.parent  # e.g., .../Reports

    dfh = df.copy()

    # Ensure raw columns exist so .apply() never crashes
    for c in _PATH_COLS:
        if c not in dfh.columns:
            dfh[c] = None

    def _exists(p) -> bool:
        try:
            return p is not None and Path(p).is_file()
        except Exception:
            return False

    def _uri(p: str) -> str | None:
        try:
            return Path(p).resolve().as_uri()
        except Exception:
            return None

    def _link_if_file(p, label=None):
        if not _exists(p):
            return "—"
        lab = escape(label if label else Path(p).name)

        # Prefer a web-relative link (works under http://localhost)
        rel_web, err = _href_from_report_to(p, report_dir)
        if rel_web is not None:
            return f'<a href="{quote(rel_web, safe="/")}" target="_blank" rel="noopener">{lab}</a>'

        # Fallback to file:// if we can't compute a relative web path (e.g., different drive)
        u = _uri(p)
        if u:
            return f'<a href="{u}" target="_blank" rel="noopener">{lab}</a>'
        return "—"

    def _img_if_file(p, w=120):
        if _exists(p):
            rel_web, err = _href_from_report_to(p, report_dir)
            if rel_web is not None:
                return f'<img src="{quote(rel_web, safe="/")}" style="max-width:{int(w)}px;height:auto;border-radius:4px;" />'
            # fallback to file:// if needed
            u = _uri(p)
            if u:
                return f'<img src="{u}" style="max-width:{int(w)}px;height:auto;border-radius:4px;" />'
        return ""

    # STL viewer link
    viewer_path = Path(ensure_stl_viewer(output_dir))
    viewer_dir  = viewer_path.parent
    viewer_name = viewer_path.name

    def _stl_view_link(p):
        if not _exists(p):
            return "—"
        rel_web, err = _safe_rel_for_web(p, viewer_dir)
        if rel_web is None:
            fname = Path(p).name
            print(f"[STL VIEW WARN] Could not build relative path from {viewer_dir} -> {p} :: {err}")
            return f'<span title="Viewer link unavailable: {escape(err)}">⚠ {escape(fname)}</span>'
        href = f'{viewer_name}?file={quote(rel_web, safe="/")}'
        return f'<a href="{href}" target="_blank" rel="noopener">View 3D</a>'

    # Display columns
    dfh["STEP File"]         = dfh["step_path"].apply(lambda p: _link_if_file(p, "STEP"))
    dfh["STL"]               = dfh["stl_path"].apply(lambda p: _link_if_file(p, "STL"))
    dfh["STL View"]          = dfh["stl_path"].apply(_stl_view_link)
    dfh["BREP"]              = dfh["brep_path"].apply(lambda p: _link_if_file(p, "BREP"))
    dfh["NC1 File"]          = dfh["nc1_path"].apply(_link_if_file)
    dfh["Drilling Drawing"]  = dfh["drilling_path"].apply(_link_if_file)
    dfh["Profile DXF"]       = dfh["dxf_path"].apply(_link_if_file)
    dfh["DXF Thumb"]         = dfh["dxf_thumb_path"].apply(_img_if_file)
    dfh["Image"]             = dfh["thumb_path"].apply(_img_if_file)

    # Base columns: support both newer keys and legacy "Item *" keys
    base_candidates = [
        ("name", "Name"),
        ("obj_type", "Type"),
        ("issues", "Issues"),
        ("mass", "Mass (kg)"),
        ("obb_x", "X (mm)"),
        ("obb_y", "Y (mm)"),
        ("obb_z", "Z (mm)"),
        # legacy fallbacks
        ("Item ID", "Name"),
        ("Item Type", "Type"),
        ("Issues", "Issues"),
        ("Mass (kg)", "Mass (kg)"),
        ("X (mm)", "X (mm)"),
        ("Y (mm)", "Y (mm)"),
        ("Z (mm)", "Z (mm)"),
    ]

    base_cols = []
    for src, dst in base_candidates:
        if src in dfh.columns and dst not in base_cols:
            dfh[dst] = dfh[src]
            base_cols.append(dst)

    file_cols = ["STEP File", "STL", "STL View", "BREP", "NC1 File", "Drilling Drawing", "Profile DXF", "DXF Thumb", "Image"]
    show_cols = base_cols + [c for c in file_cols if c in dfh.columns]

    if not show_cols:
        table_html = dfh.to_html(escape=False, index=False, classes="sortable")
    else:
        table_html = dfh[show_cols].to_html(escape=False, index=False, classes="sortable")

    css = """
    <style>
      body { font-family: Arial, sans-serif; padding: 1em; }
      h1 { font-size: 1.5em; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 0.5em; vertical-align: middle; }
      th { background-color: #f4f4f4; cursor: pointer; }
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
    <title>Project {escape(str(project_number))} Report</title>
    {css}
    <script src="https://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
  </head>
  <body>
    <h1>Project {escape(str(project_number))} Pipeline Report</h1>
    {table_html}
  </body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ Writing report to HTML at {html_path}")
    return str(html_path)

# ---------- Excel export (unchanged) ----------

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
    if file_cols is None:
        file_cols = ["STEP File", "Drilling Drawing", "Profile DXF", "NC1 File"]

    excel_dir = Path(excel_dir)
    excel_dir.mkdir(parents=True, exist_ok=True)
    out_path = excel_dir / f"{project_number}.xlsx"

    file_col_inds = [(c, df.columns.get_loc(c)) for c in file_cols if c in df.columns]
    img_col_ind  = df.columns.get_loc(image_col) if image_col in df.columns else None

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        sheet = "Report"
        df.to_excel(writer, sheet_name=sheet, index=False, startrow=1)
        wb  = writer.book
        ws  = writer.sheets[sheet]

        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#F0F0F0"})
        for col_idx, hdr in enumerate(df.columns):
            ws.write(0, col_idx, hdr, hdr_fmt)

        for row_idx, record in enumerate(df.to_dict(orient="records"), start=1):
            for col_name, col_idx in file_col_inds:
                fp = record[col_name]
                if fp:
                    ws.write_url(
                        row_idx, col_idx,
                        f"external:{fp}",
                        string=Path(fp).name
                    )
            if img_col_ind is not None:
                thumb = record[image_col]
                if thumb and Path(thumb).exists():
                    ws.set_row(row_idx, 80)
                    ws.insert_image(
                        row_idx, img_col_ind,
                        thumb,
                        {"x_scale": 0.5, "y_scale": 0.5, "x_offset": 2, "y_offset": 2}
                    )

    print(f"Wrote Excel report to {out_path}")
