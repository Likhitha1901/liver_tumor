import numpy as np
import pandas as pd
from scipy.ndimage import label

COUINAUD_INFO = {
    1: {"name": "Segment I (Caudate Lobe)", "lobe": "Caudate", "artery": "Caudate branches (LHA/RHA)", "drainage": "IVC direct"},
    2: {"name": "Segment II (Left Lateral Superior)", "lobe": "Left Lateral", "artery": "Left Hepatic Artery (LHA)", "drainage": "Left Hepatic Vein"},
    3: {"name": "Segment III (Left Lateral Inferior)", "lobe": "Left Lateral", "artery": "Left Hepatic Artery (LHA)", "drainage": "Left Hepatic Vein"},
    4: {"name": "Segment IVa/b (Left Medial / Quadrate)", "lobe": "Left Medial", "artery": "Middle/Left Hepatic Artery", "drainage": "Middle Hepatic Vein"},
    5: {"name": "Segment V (Right Anterior Inferior)", "lobe": "Right Anterior", "artery": "Right Hepatic Artery (RHA)", "drainage": "Middle/Right Hepatic Vein"},
    6: {"name": "Segment VI (Right Posterior Inferior)", "lobe": "Right Posterior", "artery": "Right Hepatic Artery (RHA)", "drainage": "Right Hepatic Vein"},
    7: {"name": "Segment VII (Right Posterior Superior)", "lobe": "Right Posterior", "artery": "Right Hepatic Artery (RHA)", "drainage": "Right Hepatic Vein"},
    8: {"name": "Segment VIII (Right Anterior Superior)", "lobe": "Right Anterior", "artery": "Right Hepatic Artery (RHA)", "drainage": "Middle/Right Hepatic Vein"},
}

def analyze_couinaud_segments(couinaud_mask, liver_mask, tumor_mask, voxel_vol_cm3, spacing_mm):
    """
    Computes rigorous segmentational metrics for Couinaud segments I to VIII.
    Returns:
        df: pandas DataFrame with volumetric breakdown in cm³
        summary: dictionary with high-level clinical findings
    """
    rows = []
    total_liver_vox = int(np.sum(liver_mask > 0))
    total_tumor_vox = int(np.sum(tumor_mask > 0))
    
    total_liver_cm3 = round(total_liver_vox * voxel_vol_cm3, 2)
    total_tumor_cm3 = round(total_tumor_vox * voxel_vol_cm3, 2)
    total_healthy_cm3 = max(0.0, round(total_liver_cm3 - total_tumor_cm3, 2))
    overall_burden = round((total_tumor_cm3 / total_liver_cm3 * 100.0) if total_liver_cm3 > 0 else 0.0, 2)
    
    involved_segments = []
    clear_segments = []
    
    for seg_id in range(1, 9):
        info = COUINAUD_INFO[seg_id]
        seg_mask = (couinaud_mask == seg_id) & (liver_mask > 0)
        seg_tum_mask = (couinaud_mask == seg_id) & (tumor_mask > 0)
        
        seg_liv_vox = int(np.sum(seg_mask))
        seg_tum_vox = int(np.sum(seg_tum_mask))
        
        seg_liv_cm3 = round(seg_liv_vox * voxel_vol_cm3, 2)
        seg_tum_cm3 = round(seg_tum_vox * voxel_vol_cm3, 2)
        seg_hea_cm3 = max(0.0, round(seg_liv_cm3 - seg_tum_cm3, 2))
        burden = round((seg_tum_cm3 / seg_liv_cm3 * 100.0) if seg_liv_cm3 > 0 else 0.0, 2)
        
        is_involved = seg_tum_cm3 >= 0.5
        status = "Involved (Tumor Present)" if is_involved else "Clear (Healthy Parenchyma)"
        if is_involved:
            involved_segments.append(f"Segment {seg_id}")
        else:
            clear_segments.append(f"Segment {seg_id}")
            
        rows.append({
            "Segment ID": seg_id,
            "Anatomical Name": info["name"],
            "Lobe / Sector": info["lobe"],
            "Total Volume (cm3)": seg_liv_cm3,
            "Healthy Parenchyma (cm3)": seg_hea_cm3,
            "Tumor Volume (cm3)": seg_tum_cm3,
            "Tumor Burden (%)": burden,
            "Feeding Artery": info["artery"],
            "Clinical Status": status
        })
        
    df = pd.DataFrame(rows)
    
    # Analyze distinct tumor lesions (connected components)
    labeled_tumors, num_lesions = label(tumor_mask)
    lesion_sizes_cm3 = []
    for l_id in range(1, num_lesions + 1):
        l_vox = int(np.sum(labeled_tumors == l_id))
        l_cm3 = round(l_vox * voxel_vol_cm3, 2)
        if l_cm3 >= 0.1: # filter noise
            lesion_sizes_cm3.append(l_cm3)
            
    lesion_sizes_cm3.sort(reverse=True)
    if not lesion_sizes_cm3 and total_tumor_cm3 > 0:
        lesion_sizes_cm3 = [total_tumor_cm3]
        
    # Estimate max lesion diameter (in cm)
    if total_tumor_vox > 0:
        # Approximate spherical diameter from largest lesion volume: V = (4/3)*pi*(d/2)^3 -> d = 2 * (3V / 4pi)^(1/3)
        largest_vol = lesion_sizes_cm3[0] if lesion_sizes_cm3 else total_tumor_cm3
        max_diameter_cm = round(2.0 * ((3.0 * largest_vol) / (4.0 * np.pi)) ** (1.0 / 3.0), 2)
    else:
        max_diameter_cm = 0.0
        
    # Milan Criteria Assessment
    # Solitary tumor <= 5 cm OR up to 3 tumors each <= 3 cm
    num_significant_lesions = len(lesion_sizes_cm3)
    if num_significant_lesions == 1 and max_diameter_cm <= 5.0:
        milan_status = "Within Milan Criteria (Solitary lesion <= 5.0 cm)"
        transplant_candidate = True
    elif 1 < num_significant_lesions <= 3 and max(lesion_sizes_cm3) <= 14.13: # (4/3)*pi*(1.5)^3 ~ 14.1 cm3
        milan_status = f"Within Milan Criteria ({num_significant_lesions} lesions, all <= 3.0 cm)"
        transplant_candidate = True
    elif num_significant_lesions == 0:
        milan_status = "No Active Tumor Detected"
        transplant_candidate = True
    else:
        milan_status = f"Exceeds Milan Criteria ({num_significant_lesions} lesion(s), max diameter ~{max_diameter_cm} cm)"
        transplant_candidate = False
        
    # TACE Planning & Vascular Targeting
    rha_involved = any(s in [5, 6, 7, 8] for s in [r["Segment ID"] for r in rows if r["Tumor Volume (cm3)"] >= 0.5])
    lha_involved = any(s in [2, 3, 4] for s in [r["Segment ID"] for r in rows if r["Tumor Volume (cm3)"] >= 0.5])
    caudate_involved = any(s == 1 for s in [r["Segment ID"] for r in rows if r["Tumor Volume (cm3)"] >= 0.5])
    
    if rha_involved and lha_involved:
        tace_plan = "Bilateral / Multi-segmental TACE (Requires staged Right & Left Hepatic Artery catheterization)"
        tace_category = "Bilobar"
    elif rha_involved:
        if len(involved_segments) <= 2:
            tace_plan = f"Superselective Subsegmental TACE targeting Right Hepatic Artery ({', '.join(involved_segments)})"
            tace_category = "Subsegmental Right"
        else:
            tace_plan = "Right Lobar TACE via Right Hepatic Artery trunk"
            tace_category = "Lobar Right"
    elif lha_involved:
        if len(involved_segments) <= 2:
            tace_plan = f"Superselective Subsegmental TACE targeting Left Hepatic Artery ({', '.join(involved_segments)})"
            tace_category = "Subsegmental Left"
        else:
            tace_plan = "Left Lobar TACE via Left Hepatic Artery trunk"
            tace_category = "Lobar Left"
    elif caudate_involved:
        tace_plan = "Selective Caudate Artery TACE (Direct caudate branches)"
        tace_category = "Caudate Selective"
    else:
        tace_plan = "No active arterial embolization target indicated"
        tace_category = "None"

    # Future Liver Remnant (FLR)
    flr_pct = round((total_healthy_cm3 / total_liver_cm3 * 100.0) if total_liver_cm3 > 0 else 100.0, 1)

    summary = {
        "total_liver_cm3": total_liver_cm3,
        "total_tumor_cm3": total_tumor_cm3,
        "total_healthy_cm3": total_healthy_cm3,
        "tumor_burden_pct": overall_burden,
        "future_liver_remnant_pct": flr_pct,
        "num_lesions": num_significant_lesions,
        "largest_lesion_cm3": lesion_sizes_cm3[0] if lesion_sizes_cm3 else 0.0,
        "estimated_max_diameter_cm": max_diameter_cm,
        "milan_criteria": milan_status,
        "transplant_candidate": transplant_candidate,
        "involved_segments": involved_segments,
        "clear_segments": clear_segments,
        "tace_plan": tace_plan,
        "tace_category": tace_category
    }
    
    return df, summary
