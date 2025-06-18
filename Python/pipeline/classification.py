import json

def classify_profile(cs, json_path, tol_dim=1.0, tol_area=0.05):
    """
    Attempts to classify a structural profile by matching its dimensions and area.
    Tries both the original and swapped width/height to detect if a 90° rotation is needed.
    """
    with open(json_path) as f:
        lib = json.load(f)

    def try_match(height, width):
        best, best_score = None, float('inf')
        for cat, ents in lib.items():
            for name, info in ents.items():
                dh = abs(height - info["height"])
                dw = abs(width  - info["width"])
                ea = abs(cs["area"] - info["csa"]) / info["csa"]
                # print(f"[CSA] checking {name}: cs_area={cs['area']:.1f}, lib_csa={info['csa']:.1f}, rel_err={ea:.1%}, tol_area={tol_area:.1%}")
                if ea > tol_area:
                    # print(f"  → SKIPPING {name} (CSA error {ea:.1%} > tol_area)\n")
                    continue
                if dh > tol_dim or dw > tol_dim:
                    continue
                el = abs(cs["length"] - info.get("length", cs["length"])) / info.get("length", cs["length"])
                score = dh + dw + ea * 100 + el * 100
                if score < best_score:
                    best_score = score
                    best = {
                        "Designation": name,
                        "Category": cat,
                        "Profile_type": info["code_profile"],
                        "Match_score": score,
                        "Requires_rotation": False,  # to be set later
                    
                        "JSON": {
                            "height": info["height"],
                            "width": info["width"],
                            "csa": info["csa"],
                            "length": 0,
                            "Mass": info.get("mass", 0.0)*cs["length"]/1000,
                            "web_thickness": info.get("web_thickness", 0.0),
                            "flange_thickness": info.get("flange_thickness", 0.0),
                            "root_radius": info.get("root_radius", 0.0),
                            "toe_radius":  info.get("toe_radius", 0.0)
                        },
                        "STEP": {
                            "height": height,
                            "width": width,
                            "area": float(cs["area"]),
                            "length": info.get("length", cs["length"]),
                            "mass": info.get("mass", 0.0)*cs["length"]/1000,
                            "web_thickness": info.get("web_thickness", 0.0),
                            "flange_thickness": info.get("flange_thickness", 0.0),
                            "root_radius": info.get("root_radius", 0.0),
                            "toe_radius":  info.get("toe_radius", 0.0)
                        }
                    }
        return best, best_score

    # Try both normal and swapped width/height
    original, score1 = try_match(cs["span_web"], cs["span_flange"])
    swapped, score2  = try_match(cs["span_flange"], cs["span_web"])
   
    best_match = None
    if original and (not swapped or score1 <= score2):
        original["Requires_rotation"] = False
        best_match = original
    elif swapped:
        swapped["Requires_rotation"] = True
        best_match = swapped

    # Patch angle logic if applicable
    if best_match and best_match.get("Profile_type") == "L":
        wt = best_match["JSON"]["web_thickness"]
        best_match["JSON"]["flange_thickness"] = wt
        best_match["STEP"]["flange_thickness"] = wt

    return best_match

