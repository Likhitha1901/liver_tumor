import os
import sys
import json
import numpy as np
import pydicom
from scipy.ndimage import zoom

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_RAW_DIR = r"C:\Users\likhitha.N\Downloads\hcc_tace_seg"
PROCESSED_DIR = r"C:\project\majorprojectnew\dataset\hcc_preprocessed"

def get_hcc_patients(raw_dir=DEFAULT_RAW_DIR):
    if not os.path.exists(raw_dir):
        return []
    return sorted([p for p in os.listdir(raw_dir) if p.startswith("HCC_") and os.path.isdir(os.path.join(raw_dir, p))])

def load_and_preprocess_patient(patient_id, raw_dir=DEFAULT_RAW_DIR, target_xy_size=256):
    """
    Loads DICOM CT and DICOM SEG for a given HCC patient.
    Aligns slices, extracts HU CT volume, liver mask, and tumor mask.
    Computes Couinaud segmentation and volumetric metrics in cm³.
    """
    patient_dir = os.path.join(raw_dir, patient_id)
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    # 1. Locate SEG file
    seg_file = None
    for root, dirs, files in os.walk(patient_dir):
        for f in files:
            if f.endswith(".dcm"):
                fpath = os.path.join(root, f)
                try:
                    ds_head = pydicom.dcmread(fpath, stop_before_pixels=True)
                    if ds_head.Modality == "SEG":
                        seg_file = fpath
                        break
                except Exception:
                    pass
        if seg_file:
            break

    if not seg_file:
        raise ValueError(f"No SEG file found for {patient_id}")

    ds_seg = pydicom.dcmread(seg_file)
    ref_series_uid = ds_seg.ReferencedSeriesSequence[0].SeriesInstanceUID

    # 2. Locate corresponding CT series
    ct_files = []
    for root, dirs, files in os.walk(patient_dir):
        for f in files:
            if f.endswith(".dcm"):
                fpath = os.path.join(root, f)
                try:
                    ds_head = pydicom.dcmread(fpath, stop_before_pixels=True)
                    if getattr(ds_head, "SeriesInstanceUID", "") == ref_series_uid:
                        ct_files.append(fpath)
                except Exception:
                    pass

    if not ct_files:
        raise ValueError(f"Matching CT series {ref_series_uid} not found for {patient_id}")

    # Deduplicate CT files by unique rounded Z coordinate
    ct_dict = {}
    for f in ct_files:
        ds_head = pydicom.dcmread(f, stop_before_pixels=True)
        z = round(float(ds_head.ImagePositionPatient[2]), 2)
        if z not in ct_dict:
            ct_dict[z] = f

    sorted_z = sorted(ct_dict.keys())
    num_slices = len(sorted_z)
    if num_slices == 0:
        raise ValueError(f"No valid CT slices found for {patient_id}")

    # Determine slice thickness dz
    if num_slices > 1:
        dz_diffs = [abs(sorted_z[i+1] - sorted_z[i]) for i in range(num_slices - 1)]
        dz = float(np.median(dz_diffs))
    else:
        sample_ds = pydicom.dcmread(ct_dict[sorted_z[0]], stop_before_pixels=True)
        dz = float(getattr(sample_ds, "SliceThickness", 2.5))

    first_ct = pydicom.dcmread(ct_dict[sorted_z[0]])
    dx = float(first_ct.PixelSpacing[0])
    dy = float(first_ct.PixelSpacing[1])
    orig_ny, orig_nx = first_ct.Rows, first_ct.Columns
    voxel_vol_cm3 = (dx * dy * dz) / 1000.0

    # 3. Build 3D CT volume in Hounsfield Units (HU)
    raw_ct_volume = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.float32)
    for idx, z_val in enumerate(sorted_z):
        ds_slice = pydicom.dcmread(ct_dict[z_val])
        pixel_array = ds_slice.pixel_array.astype(np.float32)
        slope = float(getattr(ds_slice, "RescaleSlope", 1.0))
        intercept = float(getattr(ds_slice, "RescaleIntercept", 0.0))
        hu_array = pixel_array * slope + intercept
        raw_ct_volume[idx] = hu_array

    # 4. Extract SEG frames into 3D Liver and Tumor masks
    liver_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)
    tumor_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)

    seg_frames = ds_seg.pixel_array
    pffg = ds_seg.PerFrameFunctionalGroupsSequence

    for f_idx, frame in enumerate(pffg):
        s_num = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        frame_z = round(float(frame.PlanePositionSequence[0].ImagePositionPatient[2]), 2)
        diffs = [abs(cz - frame_z) for cz in sorted_z]
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] < dz * 0.65:
            frame_arr = seg_frames[f_idx]
            if s_num == 1: # Liver
                liver_mask_orig[min_idx] = np.maximum(liver_mask_orig[min_idx], frame_arr)
            elif s_num == 2: # Tumor
                tumor_mask_orig[min_idx] = np.maximum(tumor_mask_orig[min_idx], frame_arr)

    # 5. Normalization & Windowing
    # Liver window: [-100, 250] HU
    WINDOW_MIN = -100.0
    WINDOW_MAX = 250.0
    ct_windowed = np.clip(raw_ct_volume, WINDOW_MIN, WINDOW_MAX)
    ct_norm = (ct_windowed - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN) # [0, 1]

    # Resize if needed (for fast inference & rendering)
    if target_xy_size and (orig_ny != target_xy_size or orig_nx != target_xy_size):
        scale_y = target_xy_size / orig_ny
        scale_x = target_xy_size / orig_nx
        # We resize slice by slice for high fidelity
        ct_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.float32)
        liver_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        tumor_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)

        for s in range(num_slices):
            ct_resized[s] = zoom(ct_norm[s], (scale_y, scale_x), order=1)
            liver_resized[s] = (zoom(liver_mask_orig[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            tumor_resized[s] = (zoom(tumor_mask_orig[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)

        dx_proc = dx * (orig_nx / target_xy_size)
        dy_proc = dy * (orig_ny / target_xy_size)
    else:
        ct_resized = ct_norm
        liver_resized = liver_mask_orig
        tumor_resized = tumor_mask_orig
        dx_proc, dy_proc = dx, dy

    proc_voxel_vol = (dx_proc * dy_proc * dz) / 1000.0

    # Ensure liver includes tumor voxels (anatomically tumor is inside the liver)
    liver_resized = np.maximum(liver_resized, tumor_resized)

    # 6. Couinaud Segmentation Analysis
    couinaud_mask, segment_stats = compute_couinaud_segments(
        liver_mask=liver_resized,
        tumor_mask=tumor_resized,
        voxel_vol_cm3=proc_voxel_vol
    )

    total_liver_vox = int(np.sum(liver_resized > 0))
    total_tumor_vox = int(np.sum(tumor_resized > 0))
    liver_vol_cm3 = round(total_liver_vox * proc_voxel_vol, 2)
    tumor_vol_cm3 = round(total_tumor_vox * proc_voxel_vol, 2)
    healthy_vol_cm3 = max(0.0, round(liver_vol_cm3 - tumor_vol_cm3, 2))
    tumor_ratio = round((tumor_vol_cm3 / liver_vol_cm3 * 100.0) if liver_vol_cm3 > 0 else 0.0, 2)

    metadata = {
        "patient_id": patient_id,
        "num_slices": num_slices,
        "original_shape": [num_slices, orig_ny, orig_nx],
        "processed_shape": list(ct_resized.shape),
        "voxel_spacing_mm": [round(dx_proc, 4), round(dy_proc, 4), round(dz, 4)],
        "voxel_volume_cm3": proc_voxel_vol,
        "liver_volume_cm3": liver_vol_cm3,
        "tumor_volume_cm3": tumor_vol_cm3,
        "healthy_volume_cm3": healthy_vol_cm3,
        "tumor_burden_pct": tumor_ratio,
        "couinaud_segments": segment_stats
    }

    return {
        "ct": ct_resized,
        "liver_mask": liver_resized,
        "tumor_mask": tumor_resized,
        "couinaud_mask": couinaud_mask,
        "metadata": metadata
    }

def compute_couinaud_segments(liver_mask, tumor_mask, voxel_vol_cm3):
    """
    Partitions the 3D liver into the 8 Couinaud anatomical segments:
    I: Caudate
    II: Left Superior Lateral
    III: Left Inferior Lateral
    IVa: Left Superior Medial
    IVb: Left Inferior Medial
    V: Right Inferior Anterior
    VI: Right Inferior Posterior
    VII: Right Superior Posterior
    VIII: Right Superior Anterior
    """
    couinaud_mask = np.zeros_like(liver_mask, dtype=np.uint8)
    liver_indices = np.argwhere(liver_mask > 0)

    if len(liver_indices) == 0:
        return couinaud_mask, {}

    z_coords = liver_indices[:, 0]
    y_coords = liver_indices[:, 1]
    x_coords = liver_indices[:, 2]

    z_min, z_max = np.min(z_coords), np.max(z_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    z_mid = z_min + 0.52 * (z_max - z_min) # Superior vs Inferior boundary
    # In axial CT, lower x is right of patient (left of image), higher x is left of patient
    # Cantlie's line roughly divides left and right liver lobes
    x_cantlie = x_min + 0.50 * (x_max - x_min)
    x_left_falciform = x_min + 0.72 * (x_max - x_min)
    x_right_sector = x_min + 0.28 * (x_max - x_min)

    # Anteroposterior division
    y_mid = y_min + 0.48 * (y_max - y_min)
    y_caudate_max = y_min + 0.35 * (y_max - y_min)

    segment_names = {
        1: "Segment I (Caudate)",
        2: "Segment II (Left Superior Lateral)",
        3: "Segment III (Left Inferior Lateral)",
        4: "Segment IVa/b (Left Medial)",
        5: "Segment V (Right Inferior Anterior)",
        6: "Segment VI (Right Inferior Posterior)",
        7: "Segment VII (Right Superior Posterior)",
        8: "Segment VIII (Right Superior Anterior)"
    }

    # Vectorized assignment for voxels inside liver
    for (z, y, x) in liver_indices:
        # Segment 1: Caudate Lobe (posterior, superior-middle, medial)
        if y < y_caudate_max and abs(x - x_cantlie) < 0.15 * (x_max - x_min) and z > (z_min + 0.4 * (z_max - z_min)):
            couinaud_mask[z, y, x] = 1
        elif x >= x_left_falciform: # Left lateral sector
            if z >= z_mid:
                couinaud_mask[z, y, x] = 2 # Seg II (Superior)
            else:
                couinaud_mask[z, y, x] = 3 # Seg III (Inferior)
        elif x >= x_cantlie: # Left medial sector (Segment IV)
            couinaud_mask[z, y, x] = 4 # Seg IV (IVa/IVb)
        else: # Right lobe (x < x_cantlie)
            if x < x_right_sector: # Right posterior
                if z >= z_mid:
                    couinaud_mask[z, y, x] = 7 # Seg VII (Superior Posterior)
                else:
                    couinaud_mask[z, y, x] = 6 # Seg VI (Inferior Posterior)
            else: # Right anterior
                if z >= z_mid:
                    couinaud_mask[z, y, x] = 8 # Seg VIII (Superior Anterior)
                else:
                    couinaud_mask[z, y, x] = 5 # Seg V (Inferior Anterior)

    # Calculate metrics for each Couinaud segment
    stats = {}
    for seg_id in range(1, 9):
        seg_liver_vox = int(np.sum((couinaud_mask == seg_id) & (liver_mask > 0)))
        seg_tumor_vox = int(np.sum((couinaud_mask == seg_id) & (tumor_mask > 0)))
        seg_liver_cm3 = round(seg_liver_vox * voxel_vol_cm3, 2)
        seg_tumor_cm3 = round(seg_tumor_vox * voxel_vol_cm3, 2)
        seg_healthy_cm3 = max(0.0, round(seg_liver_cm3 - seg_tumor_cm3, 2))
        burden_pct = round((seg_tumor_cm3 / seg_liver_cm3 * 100.0) if seg_liver_cm3 > 0 else 0.0, 2)

        stats[str(seg_id)] = {
            "name": segment_names[seg_id],
            "total_cm3": seg_liver_cm3,
            "tumor_cm3": seg_tumor_cm3,
            "healthy_cm3": seg_healthy_cm3,
            "burden_pct": burden_pct,
            "status": "Involved (Tumor Present)" if seg_tumor_cm3 > 0.5 else "Clear (Healthy)"
        }

    return couinaud_mask, stats

def preprocess_and_save_all(raw_dir=DEFAULT_RAW_DIR, out_dir=PROCESSED_DIR):
    os.makedirs(out_dir, exist_ok=True)
    patients = get_hcc_patients(raw_dir)
    print(f"Found {len(patients)} patients: {patients}")

    manifest = {}

    for p in patients:
        print(f"\nProcessing {p}...")
        try:
            data = load_and_preprocess_patient(p, raw_dir=raw_dir, target_xy_size=256)
            out_file = os.path.join(out_dir, f"{p}.npz")
            np.savez_compressed(
                out_file,
                ct=data["ct"],
                liver_mask=data["liver_mask"],
                tumor_mask=data["tumor_mask"],
                couinaud_mask=data["couinaud_mask"]
            )
            meta = data["metadata"]
            manifest[p] = meta
            print(f"  [OK] Saved {p}.npz | Liver: {meta['liver_volume_cm3']} cm3 | Tumor: {meta['tumor_volume_cm3']} cm3")
        except Exception as e:
            print(f"  [FAIL] Failed {p}: {e}")

    meta_file = os.path.join(out_dir, "dataset_summary.json")
    with open(meta_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll preprocessed cases saved in {out_dir}")
    print(f"Metadata saved in {meta_file}")
    return manifest

if __name__ == "__main__":
    preprocess_and_save_all()
