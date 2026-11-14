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

DEFAULT_RAW_DIR = r"C:\project\majorprojectnew\dataset\raw\hcc_tace_seg"
PROCESSED_DIR = r"C:\project\majorprojectnew\dataset\hcc_preprocessed"

def get_hcc_patients(raw_dir=DEFAULT_RAW_DIR):
    if not os.path.exists(raw_dir):
        return []
    return sorted([
        p for p in os.listdir(raw_dir)
        if p.startswith("HCC_") and os.path.isdir(os.path.join(raw_dir, p))
    ])

def find_patient_dicom_series(patient_dir):
    """
    Scans patient directory and groups DICOM files into SEG files and CT series.
    Returns:
        seg_files: list of filepaths to SEG files
        ct_series_dict: dict of series_uid -> list of DICOM filepaths
    """
    seg_files = []
    ct_series_dict = {}

    for root, _, files in os.walk(patient_dir):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            fpath = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                mod = getattr(ds, "Modality", "")
                if mod == "SEG":
                    seg_files.append(fpath)
                elif mod == "CT":
                    suid = getattr(ds, "SeriesInstanceUID", "")
                    if suid:
                        if suid not in ct_series_dict:
                            ct_series_dict[suid] = []
                        ct_series_dict[suid].append(fpath)
            except Exception:
                continue

    return seg_files, ct_series_dict

def extract_functional_groups_metadata(ds_seg):
    """
    Safely retrieves shared and per-frame functional groups metadata according to DICOM PS3.3.
    """
    shared_orientation = None
    shared_spacing = None
    shared_thickness = None

    if hasattr(ds_seg, "SharedFunctionalGroupsSequence") and len(ds_seg.SharedFunctionalGroupsSequence) > 0:
        shared_item = ds_seg.SharedFunctionalGroupsSequence[0]
        if hasattr(shared_item, "PlaneOrientationSequence") and len(shared_item.PlaneOrientationSequence) > 0:
            shared_orientation = getattr(shared_item.PlaneOrientationSequence[0], "ImageOrientationPatient", None)
        if hasattr(shared_item, "PixelMeasuresSequence") and len(shared_item.PixelMeasuresSequence) > 0:
            pm = shared_item.PixelMeasuresSequence[0]
            shared_spacing = getattr(pm, "PixelSpacing", None)
            shared_thickness = getattr(pm, "SliceThickness", None)

    return {
        "shared_orientation": shared_orientation,
        "shared_spacing": shared_spacing,
        "shared_thickness": shared_thickness
    }

def map_segment_labels(ds_seg):
    """
    Inspects SegmentSequence in DICOM SEG and maps SegmentNumbers to anatomical structures.
    Returns:
        seg_map: dict of int(SegmentNumber) -> canonical label name
                 ('liver', 'tumor', 'portal_vein', 'aorta', 'necrosis', or 'other')
    """
    seg_map = {}
    if not hasattr(ds_seg, "SegmentSequence"):
        return seg_map

    for seg in ds_seg.SegmentSequence:
        s_num = int(getattr(seg, "SegmentNumber", 0))
        label = str(getattr(seg, "SegmentLabel", "")).lower()
        desc = str(getattr(seg, "SegmentDescription", "")).lower()
        combined = f"{label} {desc}"

        if "liver" in combined:
            seg_map[s_num] = "liver"
        elif "tumor" in combined or "mass" in combined or "lesion" in combined or "hcc" in combined:
            seg_map[s_num] = "tumor"
        elif "portal" in combined or "vessel" in combined or "vein" in combined:
            seg_map[s_num] = "portal_vein"
        elif "aorta" in combined:
            seg_map[s_num] = "aorta"
        elif "necrosis" in combined:
            seg_map[s_num] = "necrosis"
        else:
            seg_map[s_num] = label if label else f"segment_{s_num}"

    return seg_map

def find_matching_ct_series(ds_seg, ct_series_dict):
    """
    Matches SEG object to the correct CT series.
    1. First checks ds_seg.ReferencedSeriesSequence and validates spatial overlap.
    2. If referenced series has 0 overlap (or is missing), finds the series with highest spatial Z overlap.
    """
    if not ct_series_dict:
        return None, []

    # Collect SEG Z coordinates
    seg_z_positions = set()
    if hasattr(ds_seg, "PerFrameFunctionalGroupsSequence"):
        for frame in ds_seg.PerFrameFunctionalGroupsSequence:
            if hasattr(frame, "PlanePositionSequence") and len(frame.PlanePositionSequence) > 0:
                pos = getattr(frame.PlanePositionSequence[0], "ImagePositionPatient", None)
                if pos and len(pos) >= 3:
                    seg_z_positions.add(round(float(pos[2]), 1))

    # 1. Try direct ReferencedSeriesSequence and verify non-zero spatial overlap
    if hasattr(ds_seg, "ReferencedSeriesSequence") and len(ds_seg.ReferencedSeriesSequence) > 0:
        for ref_item in ds_seg.ReferencedSeriesSequence:
            ref_uid = getattr(ref_item, "SeriesInstanceUID", "")
            if ref_uid in ct_series_dict and len(ct_series_dict[ref_uid]) > 0:
                if seg_z_positions:
                    ref_z = set()
                    for f in ct_series_dict[ref_uid]:
                        try:
                            ds = pydicom.dcmread(f, stop_before_pixels=True)
                            pos = getattr(ds, "ImagePositionPatient", None)
                            if pos and len(pos) >= 3:
                                ref_z.add(round(float(pos[2]), 1))
                        except Exception:
                            pass
                    if len(seg_z_positions.intersection(ref_z)) > 0:
                        return ref_uid, ct_series_dict[ref_uid]
                else:
                    return ref_uid, ct_series_dict[ref_uid]

    # 2. Fallback coordinate-based matching across all candidate series
    if seg_z_positions:
        best_uid = None
        max_overlap = -1
        for suid, files in ct_series_dict.items():
            ct_z = set()
            for f in files:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    pos = getattr(ds, "ImagePositionPatient", None)
                    if pos and len(pos) >= 3:
                        ct_z.add(round(float(ds.ImagePositionPatient[2]), 1))
                except Exception:
                    continue
            overlap = len(seg_z_positions.intersection(ct_z))
            if overlap > max_overlap:
                max_overlap = overlap
                best_uid = suid

        if best_uid and max_overlap > 0:
            return best_uid, ct_series_dict[best_uid]

    # 3. If still not matched, pick the CT series with the largest number of slices
    largest_uid = max(ct_series_dict.keys(), key=lambda k: len(ct_series_dict[k]))
    return largest_uid, ct_series_dict[largest_uid]

def load_and_preprocess_patient(patient_id, raw_dir=DEFAULT_RAW_DIR, target_xy_size=256):
    """
    Loads DICOM CT and DICOM SEG for a given HCC patient.
    Aligns slices, extracts HU CT volume, liver mask, tumor mask, portal vein, and aorta.
    Computes Couinaud segmentation and volumetric metrics in cm³.
    """
    patient_dir = os.path.join(raw_dir, patient_id)
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    # 1. Locate SEG and CT series
    seg_files, ct_series_dict = find_patient_dicom_series(patient_dir)
    if not seg_files:
        raise ValueError(f"No SEG file found for {patient_id}")
    if not ct_series_dict:
        raise ValueError(f"No CT series found for {patient_id}")

    # Select primary SEG file
    seg_file = seg_files[0]
    ds_seg = pydicom.dcmread(seg_file)

    # 2. Match CT series
    matched_uid, ct_files = find_matching_ct_series(ds_seg, ct_series_dict)
    if not ct_files:
        raise ValueError(f"Could not match CT series for {patient_id}")

    # 3. Sort and deduplicate CT slices by Patient Z position
    ct_dict = {}
    for f in ct_files:
        try:
            ds_head = pydicom.dcmread(f, stop_before_pixels=True)
            if hasattr(ds_head, "ImagePositionPatient"):
                z = round(float(ds_head.ImagePositionPatient[2]), 2)
                if z not in ct_dict:
                    ct_dict[z] = f
        except Exception:
            continue

    sorted_z = sorted(ct_dict.keys())
    num_slices = len(sorted_z)
    if num_slices == 0:
        raise ValueError(f"No valid CT slices with ImagePositionPatient found for {patient_id}")

    # Determine slice thickness dz
    if num_slices > 1:
        dz_diffs = [abs(sorted_z[i+1] - sorted_z[i]) for i in range(num_slices - 1)]
        dz = float(np.median(dz_diffs))
        if dz <= 0:
            dz = 2.5
    else:
        sample_ds = pydicom.dcmread(ct_dict[sorted_z[0]], stop_before_pixels=True)
        dz = float(getattr(sample_ds, "SliceThickness", 2.5))

    first_ct = pydicom.dcmread(ct_dict[sorted_z[0]])
    dx = float(first_ct.PixelSpacing[0]) if hasattr(first_ct, "PixelSpacing") else 0.75
    dy = float(first_ct.PixelSpacing[1]) if hasattr(first_ct, "PixelSpacing") else 0.75
    orig_ny, orig_nx = first_ct.Rows, first_ct.Columns
    voxel_vol_cm3 = (dx * dy * dz) / 1000.0

    # 4. Build 3D CT volume in Hounsfield Units (HU)
    raw_ct_volume = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.float32)
    for idx, z_val in enumerate(sorted_z):
        ds_slice = pydicom.dcmread(ct_dict[z_val])
        pixel_array = ds_slice.pixel_array.astype(np.float32)
        slope = float(getattr(ds_slice, "RescaleSlope", 1.0))
        intercept = float(getattr(ds_slice, "RescaleIntercept", 0.0))
        hu_array = pixel_array * slope + intercept
        raw_ct_volume[idx] = hu_array

    # 5. Extract multi-label SEG masks
    seg_label_map = map_segment_labels(ds_seg)
    liver_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)
    tumor_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)
    portal_vein_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)
    aorta_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)
    necrosis_mask_orig = np.zeros((num_slices, orig_ny, orig_nx), dtype=np.uint8)

    seg_frames = ds_seg.pixel_array
    # If 2D single frame
    if seg_frames.ndim == 2:
        seg_frames = np.expand_dims(seg_frames, axis=0)

    pffg = getattr(ds_seg, "PerFrameFunctionalGroupsSequence", [])

    for f_idx, frame in enumerate(pffg):
        if f_idx >= len(seg_frames):
            break

        # Get segment number
        s_num = 1
        if hasattr(frame, "SegmentIdentificationSequence") and len(frame.SegmentIdentificationSequence) > 0:
            s_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)

        # Get frame slice position
        frame_z = None
        if hasattr(frame, "PlanePositionSequence") and len(frame.PlanePositionSequence) > 0:
            pos = getattr(frame.PlanePositionSequence[0], "ImagePositionPatient", None)
            if pos and len(pos) >= 3:
                frame_z = round(float(pos[2]), 2)

        if frame_z is None:
            continue

        # Match to nearest CT slice within tolerance
        diffs = [abs(cz - frame_z) for cz in sorted_z]
        min_idx = int(np.argmin(diffs))

        if diffs[min_idx] < max(dz * 0.75, 2.0):
            frame_arr = (seg_frames[f_idx] > 0).astype(np.uint8)
            canonical = seg_label_map.get(s_num, "other")

            if canonical == "liver":
                liver_mask_orig[min_idx] = np.maximum(liver_mask_orig[min_idx], frame_arr)
            elif canonical == "tumor":
                tumor_mask_orig[min_idx] = np.maximum(tumor_mask_orig[min_idx], frame_arr)
            elif canonical == "portal_vein":
                portal_vein_mask_orig[min_idx] = np.maximum(portal_vein_mask_orig[min_idx], frame_arr)
            elif canonical == "aorta":
                aorta_mask_orig[min_idx] = np.maximum(aorta_mask_orig[min_idx], frame_arr)
            elif canonical == "necrosis":
                necrosis_mask_orig[min_idx] = np.maximum(necrosis_mask_orig[min_idx], frame_arr)

    # 6. CT Windowing & Normalization (Liver Window: [-100, 250] HU)
    WINDOW_MIN = -100.0
    WINDOW_MAX = 250.0
    ct_windowed = np.clip(raw_ct_volume, WINDOW_MIN, WINDOW_MAX)
    ct_norm = (ct_windowed - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)

    # 7. Resample XY dimensions if target_xy_size specified
    if target_xy_size and (orig_ny != target_xy_size or orig_nx != target_xy_size):
        scale_y = target_xy_size / orig_ny
        scale_x = target_xy_size / orig_nx
        ct_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.float32)
        liver_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        tumor_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        portal_vein_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        aorta_resized = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)

        for s in range(num_slices):
            ct_resized[s] = zoom(ct_norm[s], (scale_y, scale_x), order=1)
            liver_resized[s] = (zoom(liver_mask_orig[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            tumor_resized[s] = (zoom(tumor_mask_orig[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            portal_vein_resized[s] = (zoom(portal_vein_mask_orig[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            aorta_resized[s] = (zoom(aorta_mask_orig[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)

        dx_proc = dx * (orig_nx / target_xy_size)
        dy_proc = dy * (orig_ny / target_xy_size)
    else:
        ct_resized = ct_norm
        liver_resized = liver_mask_orig
        tumor_resized = tumor_mask_orig
        portal_vein_resized = portal_vein_mask_orig
        aorta_resized = aorta_mask_orig
        dx_proc, dy_proc = dx, dy

    proc_voxel_vol = (dx_proc * dy_proc * dz) / 1000.0

    # Ensure liver encompasses tumor voxels anatomically
    liver_resized = np.maximum(liver_resized, tumor_resized)

    # 8. Couinaud Segmentation Analysis
    couinaud_mask, segment_stats = compute_couinaud_segments(
        liver_mask=liver_resized,
        tumor_mask=tumor_resized,
        voxel_vol_cm3=proc_voxel_vol
    )

    total_liver_vox = int(np.sum(liver_resized > 0))
    total_tumor_vox = int(np.sum(tumor_resized > 0))
    total_pv_vox = int(np.sum(portal_vein_resized > 0))
    total_aorta_vox = int(np.sum(aorta_resized > 0))

    liver_vol_cm3 = round(total_liver_vox * proc_voxel_vol, 2)
    tumor_vol_cm3 = round(total_tumor_vox * proc_voxel_vol, 2)
    pv_vol_cm3 = round(total_pv_vox * proc_voxel_vol, 2)
    aorta_vol_cm3 = round(total_aorta_vox * proc_voxel_vol, 2)
    healthy_vol_cm3 = max(0.0, round(liver_vol_cm3 - tumor_vol_cm3, 2))
    tumor_ratio = round((tumor_vol_cm3 / liver_vol_cm3 * 100.0) if liver_vol_cm3 > 0 else 0.0, 2)

    metadata = {
        "patient_id": patient_id,
        "modality": "CT",
        "num_slices": num_slices,
        "original_shape": [num_slices, orig_ny, orig_nx],
        "processed_shape": list(ct_resized.shape),
        "voxel_spacing_mm": [round(dx_proc, 4), round(dy_proc, 4), round(dz, 4)],
        "voxel_volume_cm3": proc_voxel_vol,
        "liver_volume_cm3": liver_vol_cm3,
        "tumor_volume_cm3": tumor_vol_cm3,
        "healthy_volume_cm3": healthy_vol_cm3,
        "portal_vein_volume_cm3": pv_vol_cm3,
        "aorta_volume_cm3": aorta_vol_cm3,
        "tumor_burden_pct": tumor_ratio,
        "labels_present": {
            "liver": total_liver_vox > 0,
            "tumor": total_tumor_vox > 0,
            "portal_vein": total_pv_vox > 0,
            "aorta": total_aorta_vox > 0
        },
        "couinaud_segments": segment_stats
    }

    return {
        "ct": ct_resized,
        "liver_mask": liver_resized,
        "tumor_mask": tumor_resized,
        "portal_vein_mask": portal_vein_resized,
        "aorta_mask": aorta_resized,
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

    z_mid = z_min + 0.52 * (z_max - z_min)
    x_cantlie = x_min + 0.50 * (x_max - x_min)
    x_left_falciform = x_min + 0.72 * (x_max - x_min)
    x_right_sector = x_min + 0.28 * (x_max - x_min)

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

    for (z, y, x) in liver_indices:
        if y < y_caudate_max and abs(x - x_cantlie) < 0.15 * (x_max - x_min) and z > (z_min + 0.4 * (z_max - z_min)):
            couinaud_mask[z, y, x] = 1
        elif x >= x_left_falciform:
            if z >= z_mid:
                couinaud_mask[z, y, x] = 2
            else:
                couinaud_mask[z, y, x] = 3
        elif x >= x_cantlie:
            couinaud_mask[z, y, x] = 4
        else:
            if x < x_right_sector:
                if z >= z_mid:
                    couinaud_mask[z, y, x] = 7
                else:
                    couinaud_mask[z, y, x] = 6
            else:
                if z >= z_mid:
                    couinaud_mask[z, y, x] = 8
                else:
                    couinaud_mask[z, y, x] = 5

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

def preprocess_and_save_all(raw_dir=DEFAULT_RAW_DIR, out_dir=PROCESSED_DIR, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    patients = get_hcc_patients(raw_dir)
    if limit:
        patients = patients[:limit]

    print(f"Starting CT preprocessing for {len(patients)} patients...")

    summary_file = os.path.join(out_dir, "dataset_summary.json")
    manifest = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    success_count = 0
    fail_count = 0

    for idx, p in enumerate(patients, 1):
        print(f"[{idx}/{len(patients)}] Preprocessing {p}...", flush=True)
        try:
            data = load_and_preprocess_patient(p, raw_dir=raw_dir, target_xy_size=256)
            out_file = os.path.join(out_dir, f"{p}.npz")
            np.savez_compressed(
                out_file,
                ct=data["ct"],
                liver_mask=data["liver_mask"],
                tumor_mask=data["tumor_mask"],
                portal_vein_mask=data["portal_vein_mask"],
                aorta_mask=data["aorta_mask"],
                couinaud_mask=data["couinaud_mask"]
            )
            meta = data["metadata"]
            manifest[p] = meta
            success_count += 1
            print(f"  --> [OK] {p}: Slices={meta['num_slices']}, Liver={meta['liver_volume_cm3']} cm³, Tumor={meta['tumor_volume_cm3']} cm³", flush=True)
        except Exception as e:
            fail_count += 1
            print(f"  --> [FAIL] {p}: {type(e).__name__}: {e}", flush=True)

    with open(summary_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFinished CT Preprocessing: {success_count} succeeded, {fail_count} failed.")
    print(f"Updated summary manifest saved to: {summary_file}")
    return manifest

if __name__ == "__main__":
    preprocess_and_save_all()
