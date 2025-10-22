# pipeline/rebuilder.py
import json
from pathlib import Path
from typing import Optional

# OCC imports (OCP first, fallback to pythonocc)
try:
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_Label
    from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCP.TDataStd import TDataStd_Name
    from OCP.gp import gp_Trsf
    from OCP.STEPControl import STEPControl_Reader, STEPControl_AsIs
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.Interface import Interface_Static
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopoDS import TopoDS_Shape
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.RWGltf import RWGltf_CafWriter
    from OCP.TColStd import TColStd_IndexedDataMapOfStringString, TColStd_MapOfAsciiString
    from OCP.Message import Message_ProgressRange
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    _HAS_RWGltf = True
    _OCC_FLAVOR = "OCP"
except Exception:
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
    from OCC.Core.TDF import TDF_Label
    from OCC.Core.TCollection import TCollection_ExtendedString,TCollection_AsciiString
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.gp import gp_Trsf
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TColStd import TColStd_IndexedDataMapOfStringString, TColStd_MapOfAsciiString
    from OCC.Core.Message import Message_ProgressRange
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import BRepBndLib
    try:
        from OCC.Core.RWGltf import RWGltf_CafWriter  # pythonocc name
        _HAS_RWGltf = True
    except Exception:
        _HAS_RWGltf = False
    _OCC_FLAVOR = "pythonocc"

# --- robust app + document creation for XCAF across OCP/pythonocc builds ---

def _get_xcaf_app():
    # OCP usually: GetApplication_s(); some builds expose GetApplication()
    if hasattr(XCAFApp_Application, "GetApplication"):
        return XCAFApp_Application.GetApplication()
    if hasattr(XCAFApp_Application, "GetApplication_s"):
        return XCAFApp_Application.GetApplication_s()
    # Last resort: ctor (rare)
    return XCAFApp_Application()

def _construct_doc(fmt):
    """
    Try both doc constructors:
      - TDocStd_Document(str)
      - TDocStd_Document(TCollection_ExtendedString)
    Return a doc or None.
    """
    # 1) plain str
    try:
        return TDocStd_Document(fmt)
    except Exception:
        pass
    # 2) extended string
    try:
        return TDocStd_Document(TCollection_ExtendedString(fmt))
    except Exception:
        pass
    return None

def _new_xcaf_document(app):
    """
    Tries multiple formats & calling conventions.
    Accepts a directly-constructed doc if doc.Main() is accessible,
    even if NewDocument reports False/None on this binding.
    """
    # Formats seen across OCC bindings
    formats = ("MDTV-CAF", "MDTV-XCAF", "XmlXCAF", "BinXCAF", "XmlOcaf", "BinOcaf")

    # Signature variants to try for NewDocument:
    #   app.NewDocument(str, doc)
    #   app.NewDocument(ExtendedString, doc)
    def _try_newdoc(app, fmt, doc):
        # (a) str, doc
        try:
            ok = app.NewDocument(fmt, doc)
            if ok is None:
                ok = True
            # If Main() works, accept as success even if ok==False on this build.
            _ = doc.Main()
            return True
        except TypeError:
            pass
        except Exception:
            # Keep trying other signatures
            pass
        # (b) ExtendedString, doc
        try:
            ok = app.NewDocument(TCollection_ExtendedString(fmt), doc)
            if ok is None:
                ok = True
            _ = doc.Main()
            return True
        except Exception:
            return False

    last_errors = []
    for fmt in formats:
        # Try direct construction first
        doc = _construct_doc(fmt)
        if doc is None:
            last_errors.append(f"ctor failed for fmt='{fmt}'")
            continue

        # If doc.Main() already works, some builds don't even require NewDocument
        try:
            _ = doc.Main()
            # Still try NewDocument to register services; accept success if either works.
            if _try_newdoc(app, fmt, doc):
                return doc
            # If NewDocument complains but doc.Main() is fine, accept the doc.
            return doc
        except Exception as e:
            # Need NewDocument to wire services; attempt all signatures
            if _try_newdoc(app, fmt, doc):
                return doc
            last_errors.append(f"NewDocument failed for fmt='{fmt}'")

    # If we’re here, everything failed. Provide actionable diagnostics.
    app_attrs = [a for a in dir(app) if "NewDocument" in a or "GetApplication" in a]
    raise RuntimeError(
        "Failed to create XCAF document.\n"
        f"Tried formats: {formats}\n"
        f"Attempts: {', '.join(last_errors) or 'no constructor/No Main() access'}\n"
        f"App methods found: {app_attrs}"
    )

# --- XCAF helpers with _s compatibility ---

def _update_assemblies(shape_tool):
    """Force-regen assembly graph before export (compat for UpdateAssemblies vs UpdateAssemblies_s)."""
    try:
        shape_tool.UpdateAssemblies()
    except AttributeError:
        getattr(shape_tool, "UpdateAssemblies_s")()


def _iface_set_cval(key: str, val: str):
    """Interface_Static.SetCVal compatibility (SetCVal vs SetCVal_s)."""
    try:
        Interface_Static.SetCVal(key, val)
    except AttributeError:
        setter = getattr(Interface_Static, "SetCVal_s", None)
        if not callable(setter):
            raise
        setter(key, val)

def _get_shape_tool(doc):
    """Return the XCAF shape tool, regardless of '_s' static naming."""
    try:
        # OCP/pythonocc variant without suffix
        return XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    except AttributeError:
        # Static variant with '_s'
        return XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def _load_shape_from_step(path: Path) -> TopoDS_Shape:
    rdr = STEPControl_Reader()
    if rdr.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed STEP read: {path}")
    rdr.TransferRoots()
    return rdr.OneShape()

def _trsf_from_4x4(M):
    # row-major: a11 a12 a13 tx  a21 a22 a23 ty  a31 a32 a33 tz  0 0 0 1
    a11,a12,a13,tx, a21,a22,a23,ty, a31,a32,a33,tz, *_ = M
    T = gp_Trsf()
    T.SetValues(a11,a12,a13,tx,  a21,a22,a23,ty,  a31,a32,a33,tz)
    return T

def _loc_from_4x4(M):
    return TopLoc_Location(_trsf_from_4x4(M))

def _set_label_name(label: TDF_Label, name: Optional[str]):
    if not name:
        return
    ext = TCollection_ExtendedString(name)
    # Try common static forms first
    try:
        # pythonocc usually supports this static call
        TDataStd_Name.Set(label, ext)
        return
    except TypeError:
        pass
    except AttributeError:
        pass
    # Some OCP builds expose a static `_s` variant
    try:
        setter = getattr(TDataStd_Name, "Set_s", None)
        if callable(setter):
            setter(label, ext)
            return
    except Exception:
        pass
    # Fallback: attach an instance attribute and set via instance method
    # 1) Create attribute
    try:
        name_attr = TDataStd_Name()
        # Some bindings require AddAttribute; others allow Set directly after creation
        try:
            label.AddAttribute(name_attr)
        except Exception:
            # If AddAttribute isn't required/supported, continue
            pass
        # 2) Call instance method: name_attr.Set(ExtendedString)
        name_attr.Set(ext)
        return
    except Exception as e:
        # Last resort: ignore naming rather than crash
        # (you can `print` or `log` e here if you want diagnostics)
        return


def _make_asm_label(shape_tool, name="A-ROOT"):
    """
    Create a ROOT label as a registered free shape (empty COMPOUND).
    XCAF will treat it as an assembly once components are added.
    """
    try:
        from OCP.TopoDS import TopoDS_Compound
        from OCP.BRep import BRep_Builder
    except Exception:
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    # Register root as a free shape
    try:
        lbl = shape_tool.AddShape(comp)
    except TypeError:
        lbl = shape_tool.NewShape()
        shape_tool.SetShape(lbl, comp)

    _set_label_name(lbl, name)
    return lbl

def _bbox_diag(shape) -> float:
    """Return bounding-box diagonal in model units (mm if your doc is mm)."""
    box = Bnd_Box()
    # useNonDestructive=True keeps shape intact; dont use deflection here
    BRepBndLib.Add(shape, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dx, dy, dz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    return (dx*dx + dy*dy + dz*dz) ** 0.5

def _triangulate_shape(shape, units: str):
    """
    Ensure triangulation exists for a shape (recursively).
    Picks a sensible absolute linear deflection based on bbox, with clamps.
    """
    diag = max(_bbox_diag(shape), 1e-3)
    # Choose absolute deflection: ~1/200 of diag, clamped
    # Tune clamps by units (millimeters vs meters)
    if str(units).upper().startswith("M") and len(str(units)) == 1:  # 'M' for meters
        min_defl, max_defl = 0.0005, 0.01    # meters → 0.5–10 mm
    else:  # MM or default
        min_defl, max_defl = 0.5, 10.0       # millimeters
    lin_defl = max(min_defl, min(max_defl, diag / 200.0))
    ang_defl = 0.1  # radians (~5.7°) – decent quality

    # relative=False means lin_defl is absolute in model units
    # parallel=True to speed up on multicore
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)

def _triangulate_doc(doc, shape_tool, units: str):
    """
    Triangulate all free shapes in the document so RWGltf_CafWriter
    finds Poly_Triangulation on faces.
    """
    # Collect top-level/free shapes
    try:
        from OCP.TDF import TDF_LabelSequence as _LabelSeq
    except Exception:
        from OCC.Core.TDF import TDF_LabelSequence as _LabelSeq

    roots = _LabelSeq()
    shape_tool.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        lbl = roots.Value(i)
        shp = shape_tool.GetShape(lbl)
        if not shp or shp.IsNull():
            continue
        _triangulate_shape(shp, units)

def _triangulate_part_definitions(shape_tool, part_labels: dict, units: str):
    """
    Triangulate the actual part definition shapes (not the assembly containers).
    RWGltf needs Poly_Triangulation present on the definition shapes that
    components reference.
    """
    for pid, lbl in part_labels.items():
        try:
            shp = shape_tool.GetShape(lbl)
            if not shp or shp.IsNull():
                continue
            _triangulate_shape(shp, units)
        except Exception as e:
            print(f"[rebuilder] warn: triangulation failed for part '{pid}': {e}")

# --- GLB Export Helpers ---

def _export_glb_from_doc(doc, out_glb: Path) -> bool:
    """
    Export a GLB using OCCT's RWGltf_CafWriter if available.
    Tries both common Perform() overloads.
    """
    if not _HAS_RWGltf:
        print("[rebuilder] RWGltf not available; skipping GLB export.")
        return False

    try:
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        fname = TCollection_AsciiString(str(out_glb).replace("\\", "/"))
        writer = RWGltf_CafWriter(fname, True)  # True = .glb (binary)

        # Prepare optional args required by different overloads
        fileInfo = TColStd_IndexedDataMapOfStringString()  # leave empty
        progress = Message_ProgressRange()

        # --- Try overload #1: Perform(doc, fileInfo, progress)
        try:
            ok = bool(writer.Perform(doc, fileInfo, progress))
        except TypeError:
            ok = False

        # --- Fallback to overload #2:
        #     Perform(doc, rootLabels, labelFilter, fileInfo, progress)
        if not ok:
            try:
                shape_tool = _get_shape_tool(doc)
                from OCP.TDF import TDF_LabelSequence as _LabelSeq  # try OCP first
            except Exception:
                from OCC.Core.TDF import TDF_LabelSequence as _LabelSeq

            roots = _LabelSeq()
            try:
                # OCP/pythonocc both usually provide GetFreeShapes(seq)
                shape_tool.GetFreeShapes(roots)
            except Exception:
                # If that fails, just proceed with empty roots (writer may export whole doc)
                pass

            labelFilter = TColStd_MapOfAsciiString()  # empty filter → export all

            ok = bool(writer.Perform(doc, roots, labelFilter, fileInfo, progress))

        if not ok:
            print("[rebuilder] RWGltf_CafWriter.Perform returned False")
            return False

        # Some builds expose Write(); safe to call if present
        if hasattr(writer, "Write"):
            try:
                writer.Write()
            except Exception:
                pass

        print(f"[rebuilder] wrote GLB → {out_glb.name}")
        return True

    except Exception as e:
        print(f"[rebuilder] GLB export failed: {e}")
        return False
    
# --- Rebuilder ---

def rebuild_from_handoff(handoff_dir: str, out_step: str) -> str:
    base = Path(handoff_dir)
    man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    units  = man.get("units", "MM")
    schema = man.get("schema", "AP242")

    # XCAF app + document (robust across bindings)
    app = _get_xcaf_app()
    doc = _new_xcaf_document(app)
    shape_tool = _get_shape_tool(doc)

    # Root & label caches
    root = _make_asm_label(shape_tool, "A-ROOT")
    asm_labels = {"A-ROOT": root}
    part_labels = {}

    # Load unique part definitions and register once
    for line in (base / "unique_parts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec["part_id"]
        shp = _load_shape_from_step(base / rec["step_path"])
        lbl = shape_tool.AddShape(shp)     # definition once
        _set_label_name(lbl, rec.get("name") or pid)
        part_labels[pid] = lbl

    inst_lines = (base / "instances.jsonl").read_text(encoding="utf-8").splitlines()
    inst_recs = [json.loads(l) for l in inst_lines if l.strip()]

    # PASS 1: create/mount assemblies (container nodes)
    for rec in inst_recs:
        if rec.get("is_assembly"):
            asm_id = rec.get("assembly_id") or rec.get("name")
            if not asm_id:
                continue
            # make label for this assembly id if not present
            if asm_id not in asm_labels:
                asm_labels[asm_id] = _make_asm_label(shape_tool, asm_id)
            parent = rec.get("parent_asm", "A-ROOT")
            if parent not in asm_labels:
                asm_labels[parent] = _make_asm_label(shape_tool, parent)
            # mount this assembly under its parent with its transform
            loc = _loc_from_4x4(rec["T"])
            shape_tool.AddComponent(asm_labels[parent], asm_labels[asm_id], loc)
            _set_label_name(asm_labels[asm_id], rec.get("name") or asm_id)

    # PASS 2: mount part occurrences
    seen_occurrence_names = set()
    for rec in inst_recs:
        if rec.get("is_assembly"):
            continue  # already handled
        parent = rec.get("parent_asm", "A-ROOT")
        if parent not in asm_labels:
            asm_labels[parent] = _make_asm_label(shape_tool, parent)
        loc = _loc_from_4x4(rec["T"])  # NOTE: now treated as LOCAL if parent is a subasm
        comp_lbl = shape_tool.AddComponent(asm_labels[parent], part_labels[rec["part_id"]], loc)

        # Name occurrence: prefer instance_id → name → part_id; keep unique
        base_name = rec.get("instance_id") or rec.get("name") or rec["part_id"]
        name = base_name
        i = 1
        while name in seen_occurrence_names:
            i += 1
            name = f"{base_name}__{i}"
        seen_occurrence_names.add(name)
        _set_label_name(comp_lbl, name)



    _update_assemblies(shape_tool)
    print(f"[rebuilder] unique parts: {len(part_labels)}; parents: {len(asm_labels)}")

    # Write AP242 with names & units
    _iface_set_cval("write.step.unit", units)
    _iface_set_cval("write.step.schema", "AP242" if schema.upper()=="AP242" else "AP214IS")


    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    if not w.Transfer(doc, STEPControl_AsIs):
        raise RuntimeError("STEPCAF transfer failed")
    if w.Write(out_step) != IFSelect_RetDone:
        raise RuntimeError(f"Write failed: {out_step}")
    
    # ------------------------------------------------------------
    # Ensure shapes are triangulated for GLB (avoids 'skipped … without triangulation')
    # ------------------------------------------------------------
    try:
        _triangulate_part_definitions(shape_tool, part_labels, units)
    except Exception as e:
         print(f"[rebuilder] warning: part triangulation failed ({e}) – GLB export may skip nodes")


    # ------------------------------------------------------------
    # Export GLB for the web viewer (if RWGltf is available)
    # ------------------------------------------------------------
    # out_step_path = Path(out_step)
    # out_glb_path  = out_step_path.with_suffix(".glb")
    # _export_glb_from_doc(doc, out_glb_path)
    
    # ------------------------------------------------------------
    # Generate viewer_meta.json for the 3D web viewer
    # ------------------------------------------------------------

    try:
        meta = {"model_name": Path(out_step).stem,
                "by_instance_id": {},
                "parents": ["__ROOT__"]}

        inst_path = base / "instances.jsonl"
        if inst_path.exists():
            with open(inst_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    iid = rec.get("instance_id") or rec.get("name") or rec["part_id"]
                    parent = rec.get("parent_asm", "__ROOT__")
                    meta["by_instance_id"][iid] = {
                        "name": rec.get("name", iid),
                        "part_uid": rec["part_id"],
                        "parent_uid": parent,
                    }
                    meta["parents"].append(parent)

        # Deduplicate parent list
        meta["parents"] = sorted(set(meta["parents"]))

        # Save next to the STEP file
        viewer_meta_path = Path(out_step).with_name("viewer_meta.json")
        viewer_meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[rebuilder] wrote viewer metadata → {viewer_meta_path.name}")
    except Exception as e:
        print(f"[rebuilder] warning: failed to write viewer_meta.json ({e})")


    return out_step