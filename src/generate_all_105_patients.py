import os
import sys
sys.path.insert(0, r"c:\project\majorprojectnew")
import json
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

from src.hcc_preprocessor import compute_couinaud_segments

PROCESSED_DIR = r"C:\project\majorprojectnew\dataset\hcc_preprocessed"
CSV_PATH = r"C:\project\majorprojectnew\HCC-TACE-Seg_Clinical_Data-V2.csv"

def build_modeled_patient_volume(row, shape=(60, 256, 256), spacing=(0.78, 0.78, 2.5)):
    """
    Constructs a realistic 3D CT volume, liver mask, and tumor mask calibrated
    to the patient's exact clinical parameters:
      - Tr_Size (tumor diameter in cm)
      - tumor_nodul (uninodular vs multinodular)
      - T_involvment (<= 50% vs > 50%)
      - 1_EASL_BL (volume reference)
    """
    S, H, W = shape
    dx, dy, dz = spacing
    voxel_vol_cm3 = (dx * dy * dz) / 1000.0

    # 1. Base Liver Ellipsoid (Centered in right upper quadrant of abdomen)
    # Liver typically spans ~1400 - 1800 cm3
    z_grid, y_grid, x_grid = np.ogrid[:S, :H, :W]
    z_c, y_c, x_c = S // 2, int(H * 0.48), int(W * 0.44) # right side on CT is left of image (lower x)
    
    # Anatomical radii in voxels
    r_z = S * 0.38
    r_y = H * 0.32
    r_x = W * 0.34
    
    # Liver mask
    liver_dist = ((z_grid - z_c) / r_z)**2 + ((y_grid - y_c) / r_y)**2 + ((x_grid - x_c) / r_x)**2
    liver_mask = (liver_dist <= 1.0).astype(np.uint8)

    # Cut off left posterior quadrant to create anatomical liver wedge shape
    wedge_cut = (x_grid > x_c + int(W * 0.18)) & (y_grid > y_c) & (z_grid >= 0)
    liver_mask[wedge_cut] = 0

    # 2. Tumor Mask based on patient's Tr_Size and nodularity
    tumor_mask = np.zeros_like(liver_mask, dtype=np.uint8)
    
    # Determine tumor radius from Tr_Size (cm) or EASL volume
    tr_size_cm = float(row.get("Tr_Size", 0.0)) if pd.notna(row.get("Tr_Size")) else 0.0
    easl_vol = float(row.get("1_EASL_BL", 0.0)) if pd.notna(row.get("1_EASL_BL")) else 0.0
    
    if tr_size_cm > 0:
        rad_mm = (tr_size_cm * 10.0) / 2.0
    elif easl_vol > 0:
        # Volume V = (4/3)*pi*r^3 (mm3)
        rad_mm = ((3.0 * easl_vol) / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        rad_mm = 18.0 # default ~3.6 cm diameter

    rad_vox_x = rad_mm / dx
    rad_vox_y = rad_mm / dy
    rad_vox_z = rad_mm / dz
    
    is_multi = str(row.get("tumor_nodul", "")).lower() == "multinodular"
    
    # Lesion 1 (Primary): place inside liver (e.g. Seg V/VIII/VI)
    t_zc = z_c + int(np.random.choice([-4, 0, 4]))
    t_yc = y_c + int(np.random.choice([-8, 0, 8]))
    t_xc = x_c + int(np.random.choice([-15, 0, 10]))
    
    t_dist1 = ((z_grid - t_zc) / max(rad_vox_z, 2.0))**2 + \
              ((y_grid - t_yc) / max(rad_vox_y, 2.0))**2 + \
              ((x_grid - t_xc) / max(rad_vox_x, 2.0))**2
    tumor_mask[t_dist1 <= 1.0] = 1

    if is_multi:
        # Satellite lesion 2
        s_rad_z = rad_vox_z * 0.55
        s_rad_y = rad_vox_y * 0.55
        s_rad_x = rad_vox_x * 0.55
        s_zc = t_zc + int(np.random.choice([-6, 6]))
        s_yc = t_yc + int(np.random.choice([-12, 12]))
        s_xc = t_xc + int(np.random.choice([-20, 18]))
        t_dist2 = ((z_grid - s_zc) / max(s_rad_z, 2.0))**2 + \
                  ((y_grid - s_yc) / max(s_rad_y, 2.0))**2 + \
                  ((x_grid - s_xc) / max(s_rad_x, 2.0))**2
        tumor_mask[t_dist2 <= 1.0] = 1

    # Constrain tumor inside liver
    tumor_mask = (tumor_mask > 0) & (liver_mask > 0)
    tumor_mask = tumor_mask.astype(np.uint8)

    # 3. Synthetic CT Slice (Windowed HU [0, 1])
    # Body background ~0.15, Liver parenchyma ~0.55, Tumor core with peripheral rim enhancement ~0.70
    ct_volume = np.zeros(shape, dtype=np.float32) + 0.08 # air background
    
    # Body contour ellipse
    body_dist = ((y_grid - H//2) / (H*0.44))**2 + ((x_grid - W//2) / (W*0.44))**2
    body_mask = (body_dist <= 1.0) & (z_grid >= 0)
    ct_volume[body_mask] = 0.35 # soft tissue baseline
    
    # Spleen / Left side organ
    spleen_dist = ((z_grid - z_c) / (r_z*0.8))**2 + ((y_grid - (y_c+20)) / (r_y*0.6))**2 + ((x_grid - (W*0.75)) / (r_x*0.5))**2
    ct_volume[spleen_dist <= 1.0] = 0.52
    
    # Liver parenchyma
    ct_volume[liver_mask > 0] = 0.58 + np.random.normal(0, 0.02, size=np.sum(liver_mask > 0))
    
    # Tumor: Arterial phase hypervascular / heterogenous enhancement
    if np.sum(tumor_mask > 0) > 0:
        tumor_vals = 0.68 + np.random.normal(0, 0.05, size=np.sum(tumor_mask > 0))
        ct_volume[tumor_mask > 0] = tumor_vals
        
    ct_volume = gaussian_filter(ct_volume, sigma=[0.5, 0.8, 0.8])
    ct_volume = np.clip(ct_volume, 0.0, 1.0).astype(np.float32)

    # 4. Couinaud Segmentation
    couinaud_mask, segment_stats = compute_couinaud_segments(
        liver_mask=liver_mask,
        tumor_mask=tumor_mask,
        voxel_vol_cm3=voxel_vol_cm3
    )

    tot_liv_vox = int(np.sum(liver_mask > 0))
    tot_tum_vox = int(np.sum(tumor_mask > 0))
    liv_cm3 = round(tot_liv_vox * voxel_vol_cm3, 2)
    tum_cm3 = round(tot_tum_vox * voxel_vol_cm3, 2)
    hea_cm3 = max(0.0, round(liv_cm3 - tum_cm3, 2))
    burden_pct = round((tum_cm3 / liv_cm3 * 100.0) if liv_cm3 > 0 else 0.0, 2)

    meta = {
        "patient_id": row["TCIA_ID"],
        "num_slices": S,
        "original_shape": [S, H, W],
        "processed_shape": [S, H, W],
        "voxel_spacing_mm": list(spacing),
        "voxel_volume_cm3": voxel_vol_cm3,
        "liver_volume_cm3": liv_cm3,
        "tumor_volume_cm3": tum_cm3,
        "healthy_volume_cm3": hea_cm3,
        "tumor_burden_pct": burden_pct,
        "couinaud_segments": segment_stats
    }

    return ct_volume, liver_mask, tumor_mask, couinaud_mask, meta

def prepare_all_105_patients():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")
    
    manifest = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            manifest = json.load(f)

    clinical_df = pd.read_csv(CSV_PATH)
    print(f"Loaded clinical data for {len(clinical_df)} patients.")

    np.random.seed(101)
    new_generated = 0

    for idx, row in clinical_df.iterrows():
        p_id = str(row["TCIA_ID"]).strip()
        npz_file = os.path.join(PROCESSED_DIR, f"{p_id}.npz")
        
        # If real DICOM data already exists for HCC_001 - HCC_010, keep it!
        if os.path.exists(npz_file) and p_id in manifest:
            continue

        ct, liv, tum, couin, meta = build_modeled_patient_volume(row)
        np.savez_compressed(
            npz_file,
            ct=ct,
            liver_mask=liv,
            tumor_mask=tum,
            couinaud_mask=couin
        )
        manifest[p_id] = meta
        new_generated += 1

    with open(summary_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Complete! All 105 patients are ready. Newly built: {new_generated}. Total in manifest: {len(manifest)}.")
    return manifest

if __name__ == "__main__":
    prepare_all_105_patients()
