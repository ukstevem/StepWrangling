import json

def prepare_classification_input(obb_data, profile_type, section_face):
    """
    Prepares measured profile data for classification.
    Expects:
        - obb_data: dict with keys 'he_X', 'he_Y', 'he_Z'
        - profile_type: "I", "U", "L", etc.
        - section_face: TopoDS_Face from which CSA is computed
    Returns:
        dict suitable for classify_profile()
    """
    from utils_geometry import get_face_area  # if not already at top of file

    he_X = obb_data['he_X']
    he_Y = obb_data['he_Y']
    he_Z = obb_data['he_Z']

    # Assume: he_X = flange span, he_Y = web height, he_Z = half-length
    area = get_face_area(section_face)

    return {
        'span_flange': 2 * he_X,
        'span_web':    2 * he_Y,
        'length':      2 * he_Z,
        'area':        area,
        'profile_type': profile_type
    }

def classify_profile(cs, json_path, tol_dim=1.0, tol_area=0.05):
    """
    Attempts to classify a structural profile by matching its dimensions and area.
    Tries both the original and swapped width/height to handle OBB axis flips.
    """
    with open(json_path) as f:
        lib = json.load(f)

    def try_match(height, width):
        best, best_score = None, float('inf')
        for cat, ents in lib.items():
            for name, info in ents.items():
                dh = abs(height - info["height"])
                dw = abs(width  - info["width"])
                if dh > tol_dim or dw > tol_dim:
                    continue
                ea = abs(cs["area"] - info["csa"]) / info["csa"]
                if ea > tol_area:
                    continue
                el = abs(cs["length"] - info.get("length", cs["length"])) / info.get("length", cs["length"])
                score = dh + dw + ea * 100 + el * 100
                if score < best_score:
                    best_score = score
                    best = {
                        **info,
                        "Designation": name,
                        "Category": cat,
                        "Measured_height": height,
                        "Measured_width": width,
                        "Measured_area": cs["area"],
                        "Measured_length": cs["length"],
                        "Match_score": score,
                        "Profile_type": cs.get("profile_type", "Unknown")
                    }
        return best, best_score

    # Try both normal and swapped width/height
    original, score1 = try_match(cs["span_web"], cs["span_flange"])
    swapped, score2  = try_match(cs["span_flange"], cs["span_web"])

    if original and (not swapped or score1 <= score2):
        return original
    elif swapped:
        swapped["Swapped_dimensions"] = True
        return swapped
    else:
        return None

# Only checks with x and y one way
# def classify_profile(cs, json_path, tol_dim=1.0, tol_area=0.05):
#     # cs already contains cs['span_flange'], cs['span_web'], cs['area'], cs['length']
#     with open(json_path) as f:
#         lib = json.load(f)
#     best, best_score = None, float('inf')
#     for cat, ents in lib.items():
#         for name, info in ents.items():
#             dh = abs(cs['span_web'] - info['height'])
#             dw = abs(cs['span_flange']    - info['width'])
#             if dh > tol_dim or dw > tol_dim: 
#                 continue
#             ea = abs(cs['area'] - info['csa'])/info['csa']
#             if ea > tol_area: 
#                 continue
#             el = abs(cs['length'] - info.get('length', cs['length']))/info.get('length', cs['length'])
#             score = dh + dw + ea*100 + el*100
#             if score < best_score:
#                 best_score = score
#                 best = {
#                   **info,
#                   "Designation":     name,
#                   "Category":        cat,
#                   "Measured_height": cs['span_web'],
#                   "Measured_width":  cs['span_flange'],
#                   "Measured_area":   cs['area'],
#                   "Measured_length": cs['length'],
#                   "Match_score":     score,
#                   "Profile_type":    cs.get("profile_type", "Unknown")
#                 }    
#     return best

# # cs = compute_section_and_length_and_origin(solid)
# # profile = classify_profile(cs, "Shape_classifier_info.json")
