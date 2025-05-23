def report_summary(obb_data, matched_profile, profile_type, section_info, dstv_info):
    """
    Print a compact diagnostic report for structural element processing.
    
    Parameters:
    - obb_data: dict with keys xaxis, yaxis, zaxis, he_X, he_Y, he_Z
    - matched_profile: dict from classification output
    - profile_type: string, e.g., 'I', 'U', 'L', 'Unknown'
    - section_info: dict with keys:
        - 'z_cut': 'start' or 'end'
        - 'corners': list of section gp_Pnt
        - 'z_flipped': bool
        - 'leg_orientation_ok': bool or None
    - dstv_info: dict with keys:
        - 'origins': dict of origin points
        - 'face_axes': dict of face axis vectors
    """
    print("\n===== 🔎 Summary Report =====")

    print(f"\n🔹 OBB extents:")
    print(f"   X: {2 * obb_data['he_X']:.1f} mm")
    print(f"   Y: {2 * obb_data['he_Y']:.1f} mm")
    print(f"   Z: {2 * obb_data['he_Z']:.1f} mm")

    print("\n🔹 Local axes assignment:")
    print("   X → Web direction")
    print("   Y → Flange direction")
    print("   Z → Length direction")

    print("\n🔍 Profile classification:")
    print(f"   Type: {profile_type}")
    if matched_profile:
        print(f"   Match: {matched_profile['Designation']} [{matched_profile['Category']}]  Score: {matched_profile['Match_score']:.2f}")
        if matched_profile.get("Swapped_dimensions"):
            print("   ⚠️  Used swapped width/height for matching")

    print("\n📐 Section plane analysis:")
    print(f"   Cut at: Z-{section_info['z_cut']}")
    print(f"   Corners detected: {len(section_info['corners'])}")
    print(f"   Z axis flipped: {'✅ Yes' if section_info['z_flipped'] else '❌ No'}")
    if section_info.get("leg_orientation_ok") is not None:
        print(f"   Leg orientation valid: {'✅ Yes' if section_info['leg_orientation_ok'] else '❌ No'}")

    print("\n📎 DSTV Origins:")
    for k, pt in dstv_info['origins'].items():
        print(f"   {k}: ({pt.X():.1f}, {pt.Y():.1f}, {pt.Z():.1f})")

    print("\n🧭 DSTV Face Axes:")
    for face, (xv, yv) in dstv_info['face_axes'].items():
        print(f"   {face}: X=({xv.X():.2f},{xv.Y():.2f},{xv.Z():.2f}), Y=({yv.X():.2f},{yv.Y():.2f},{yv.Z():.2f})")

    print("\n============================\n")
