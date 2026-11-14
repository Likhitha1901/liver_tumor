import os
import sys
import glob
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = r"C:\project\majorprojectnew"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hcc_preprocessor import compute_couinaud_segments

DEFAULT_MRI_RAW_DIR = r"C:\project\majorprojectnew\dataset\raw\mri_liverhccseg\nifti_and_segms"
MRI_PROCESSED_DIR = r"C:\project\majorprojectnew\dataset\mri_preprocessed"

def get_mri_patients(raw_dir=DEFAULT_MRI_RAW_DIR):
    """Returns sorted list of patient directory names in raw MRI folder."""
    if not os.path.exists(raw_dir):
        return []
    return sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith(".") and d != "__MACOSX"
    ])

def normalize_mri_intensity(volume, p_low=1.0, p_high=99.0):
    """
    Robust intensity normalization for MRI volumes using percentile clipping.
    """
    valid_voxels = volume[volume > 0]
    if len(valid_voxels) == 0:
        return np.zeros_like(volume, dtype=np.float32)

    v_min = float(np.percentile(valid_voxels, p_low))
    v_max = float(np.percentile(valid_voxels, p_high))
    if v_max <= v_min:
        v_max = float(np.max(volume))
        v_min = float(np.min(volume))

    clipped = np.clip(volume, v_min, v_max)
    if v_max > v_min:
        norm = (clipped - v_min) / (v_max - v_min)
    else:
        norm = np.zeros_like(clipped, dtype=np.float32)
    return norm.astype(np.float32)

def to_radiological_axial(nii_img):
    """
    Converts a NIfTI image to canonical RAS and reorients to standard
    radiological axial orientation: (num_slices, H, W)
    where:
      - slice index goes from Inferior to Superior
      - row index (top to bottom) goes from Anterior to Posterior
      - column index (left to right) goes from Patient Right to Patient Left
    """
    can_img = nib.as_closest_canonical(nii_img)
    data = can_img.get_fdata(dtype=np.float32)
    # can_img data is (X: Right->Left, Y: Posterior->Anterior, Z: Inferior->Superior)
    # Transpose to (Z, Y, X)
    axial = np.transpose(data, (2, 1, 0))
    # Flip Y (Anterior on top) and flip X (Patient Right on viewer's left)
    axial = axial[:, ::-1, ::-1]

    # Voxel spacing in (dz, dy, dx)
    zooms = can_img.header.get_zooms()
    dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])

    return axial, (dx, dy, dz)

def load_and_preprocess_mri_patient(patient_id, raw_dir=DEFAULT_MRI_RAW_DIR, target_xy_size=256):
    """
    Loads multi-phase dynamic MRI and multi-rater segmentations for an MRI patient.
    Extracts phases: art, pre, pv, del (using registered versions aligned to arterial phase).
    Loads liver and tumor masks for rater1, rater2, and consensus.
    Computes volumetric metrics and Couinaud segments.
    """
    patient_dir = os.path.join(raw_dir, patient_id)
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"MRI patient directory not found: {patient_dir}")

    # Find session date folder
    date_folders = [
        d for d in os.listdir(patient_dir)
        if os.path.isdir(os.path.join(patient_dir, d)) and not d.startswith(".")
    ]
    if not date_folders:
        raise FileNotFoundError(f"No date session directory in {patient_dir}")

    session_date = date_folders[0]
    session_dir = os.path.join(patient_dir, session_date)

    # 1. Load Reference Arterial Phase
    art_path = os.path.join(session_dir, "art.nii.gz")
    if not os.path.exists(art_path):
        raise FileNotFoundError(f"Missing primary arterial phase (art.nii.gz) for {patient_id}")

    art_nii = nib.load(art_path)
    ref_axial, (dx, dy, dz) = to_radiological_axial(art_nii)
    num_slices, orig_ny, orig_nx = ref_axial.shape

    # 2. Phase mapping (prefer co-registered phases that share the arterial grid)
    phase_files = {
        "art": "art.nii.gz",
        "pre": "art_pre.nii.gz" if os.path.exists(os.path.join(session_dir, "art_pre.nii.gz")) else "pre.nii.gz",
        "pv": "art_pv.nii.gz" if os.path.exists(os.path.join(session_dir, "art_pv.nii.gz")) else "pv.nii.gz",
        "del": "art_del.nii.gz" if os.path.exists(os.path.join(session_dir, "art_del.nii.gz")) else "del.nii.gz",
    }

    raw_phases = {}
    for phase_key, filename in phase_files.items():
        fpath = os.path.join(session_dir, filename)
        if os.path.exists(fpath):
            p_nii = nib.load(fpath)
            p_axial, _ = to_radiological_axial(p_nii)
            if p_axial.shape == ref_axial.shape:
                raw_phases[phase_key] = p_axial
            else:
                # If native unregistered shape differs, resample or fallback to arterial
                raw_phases[phase_key] = ref_axial.copy()
        else:
            raw_phases[phase_key] = ref_axial.copy()

    # 3. Load Liver Masks (Rater 1 and Rater 2)
    r1_liver_path = os.path.join(session_dir, "rater1_liver.nii.gz")
    r2_liver_path = os.path.join(session_dir, "rater2_liver.nii.gz")

    r1_liver = np.zeros_like(ref_axial, dtype=np.uint8)
    r2_liver = np.zeros_like(ref_axial, dtype=np.uint8)

    if os.path.exists(r1_liver_path):
        r1_nii = nib.load(r1_liver_path)
        r1_arr, _ = to_radiological_axial(r1_nii)
        if r1_arr.shape == ref_axial.shape:
            r1_liver = (r1_arr > 0).astype(np.uint8)

    if os.path.exists(r2_liver_path):
        r2_nii = nib.load(r2_liver_path)
        r2_arr, _ = to_radiological_axial(r2_nii)
        if r2_arr.shape == ref_axial.shape:
            r2_liver = (r2_arr > 0).astype(np.uint8)

    # Build consensus liver (union of raters, or whichever rater segmented)
    if np.sum(r1_liver) > 0 and np.sum(r2_liver) > 0:
        consensus_liver = np.maximum(r1_liver, r2_liver)
    elif np.sum(r1_liver) > 0:
        consensus_liver = r1_liver.copy()
    else:
        consensus_liver = r2_liver.copy()

    # 4. Load Tumor Masks (Multi-lesions for Rater 1 and Rater 2)
    r1_tumor = np.zeros_like(ref_axial, dtype=np.uint8)
    r2_tumor = np.zeros_like(ref_axial, dtype=np.uint8)

    r1_tumor_files = sorted(glob.glob(os.path.join(session_dir, "rater1_tumor*.nii.gz")))
    for tf in r1_tumor_files:
        t_nii = nib.load(tf)
        t_arr, _ = to_radiological_axial(t_nii)
        if t_arr.shape == ref_axial.shape:
            r1_tumor = np.maximum(r1_tumor, (t_arr > 0).astype(np.uint8))

    r2_tumor_files = sorted(glob.glob(os.path.join(session_dir, "rater2_tumor*.nii.gz")))
    for tf in r2_tumor_files:
        t_nii = nib.load(tf)
        t_arr, _ = to_radiological_axial(t_nii)
        if t_arr.shape == ref_axial.shape:
            r2_tumor = np.maximum(r2_tumor, (t_arr > 0).astype(np.uint8))

    # Build consensus tumor
    if np.sum(r1_tumor) > 0 and np.sum(r2_tumor) > 0:
        consensus_tumor = np.maximum(r1_tumor, r2_tumor)
    elif np.sum(r1_tumor) > 0:
        consensus_tumor = r1_tumor.copy()
    else:
        consensus_tumor = r2_tumor.copy()

    # Anatomical constraint: liver contains tumor
    consensus_liver = np.maximum(consensus_liver, consensus_tumor)

    # 5. Normalization & Resizing
    norm_phases = {}
    for pk, vol in raw_phases.items():
        norm_phases[pk] = normalize_mri_intensity(vol)

    if target_xy_size and (orig_ny != target_xy_size or orig_nx != target_xy_size):
        scale_y = target_xy_size / orig_ny
        scale_x = target_xy_size / orig_nx

        proc_phases = {}
        for pk, vol in norm_phases.items():
            proc_vol = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.float32)
            for s in range(num_slices):
                proc_vol[s] = zoom(vol[s], (scale_y, scale_x), order=1)
            proc_phases[pk] = proc_vol

        proc_liver = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        proc_tumor = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        proc_r1_liver = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        proc_r2_liver = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        proc_r1_tumor = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)
        proc_r2_tumor = np.zeros((num_slices, target_xy_size, target_xy_size), dtype=np.uint8)

        for s in range(num_slices):
            proc_liver[s] = (zoom(consensus_liver[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            proc_tumor[s] = (zoom(consensus_tumor[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            proc_r1_liver[s] = (zoom(r1_liver[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            proc_r2_liver[s] = (zoom(r2_liver[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            proc_r1_tumor[s] = (zoom(r1_tumor[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)
            proc_r2_tumor[s] = (zoom(r2_tumor[s], (scale_y, scale_x), order=0) > 0.5).astype(np.uint8)

        dx_proc = dx * (orig_nx / target_xy_size)
        dy_proc = dy * (orig_ny / target_xy_size)
    else:
        proc_phases = norm_phases
        proc_liver = consensus_liver
        proc_tumor = consensus_tumor
        proc_r1_liver = r1_liver
        proc_r2_liver = r2_liver
        proc_r1_tumor = r1_tumor
        proc_r2_tumor = r2_tumor
        dx_proc, dy_proc = dx, dy

    proc_voxel_vol = (dx_proc * dy_proc * dz) / 1000.0

    # 6. Couinaud Segmentation Analysis
    couinaud_mask, segment_stats = compute_couinaud_segments(
        liver_mask=proc_liver,
        tumor_mask=proc_tumor,
        voxel_vol_cm3=proc_voxel_vol
    )

    total_liver_vox = int(np.sum(proc_liver > 0))
    total_tumor_vox = int(np.sum(proc_tumor > 0))
    liver_vol_cm3 = round(total_liver_vox * proc_voxel_vol, 2)
    tumor_vol_cm3 = round(total_tumor_vox * proc_voxel_vol, 2)
    healthy_vol_cm3 = max(0.0, round(liver_vol_cm3 - tumor_vol_cm3, 2))
    tumor_ratio = round((tumor_vol_cm3 / liver_vol_cm3 * 100.0) if liver_vol_cm3 > 0 else 0.0, 2)

    metadata = {
        "patient_id": patient_id,
        "modality": "MRI",
        "session_date": session_date,
        "num_slices": num_slices,
        "original_shape": [num_slices, orig_ny, orig_nx],
        "processed_shape": [num_slices, target_xy_size, target_xy_size],
        "voxel_spacing_mm": [round(dx_proc, 4), round(dy_proc, 4), round(dz, 4)],
        "voxel_volume_cm3": proc_voxel_vol,
        "liver_volume_cm3": liver_vol_cm3,
        "tumor_volume_cm3": tumor_vol_cm3,
        "healthy_volume_cm3": healthy_vol_cm3,
        "tumor_burden_pct": tumor_ratio,
        "available_phases": list(proc_phases.keys()),
        "raters_available": {
            "rater1": bool(np.sum(proc_r1_liver) > 0),
            "rater2": bool(np.sum(proc_r2_liver) > 0),
            "rater1_tumor_count": len(r1_tumor_files),
            "rater2_tumor_count": len(r2_tumor_files)
        },
        "couinaud_segments": segment_stats
    }

    return {
        "phases": proc_phases,
        "liver_mask": proc_liver,
        "tumor_mask": proc_tumor,
        "rater1_liver": proc_r1_liver,
        "rater2_liver": proc_r2_liver,
        "rater1_tumor": proc_r1_tumor,
        "rater2_tumor": proc_r2_tumor,
        "couinaud_mask": couinaud_mask,
        "metadata": metadata
    }

def preprocess_and_save_all_mri(raw_dir=DEFAULT_MRI_RAW_DIR, out_dir=MRI_PROCESSED_DIR):
    """
    Processes all discovered MRI cases and saves `.npz` volume files and metadata manifest.
    """
    os.makedirs(out_dir, exist_ok=True)
    patients = get_mri_patients(raw_dir)
    print(f"Discovered {len(patients)} MRI patients in {raw_dir}:")
    print(patients)

    manifest = {}
    success_count = 0
    fail_count = 0

    for idx, p in enumerate(patients, 1):
        print(f"\n[{idx}/{len(patients)}] Preprocessing MRI patient {p}...", flush=True)
        try:
            data = load_and_preprocess_mri_patient(p, raw_dir=raw_dir, target_xy_size=256)
            out_file = os.path.join(out_dir, f"{p}.npz")
            np.savez_compressed(
                out_file,
                art=data["phases"]["art"],
                pre=data["phases"]["pre"],
                pv=data["phases"]["pv"],
                **{"del": data["phases"]["del"]},
                liver_mask=data["liver_mask"],
                tumor_mask=data["tumor_mask"],
                rater1_liver=data["rater1_liver"],
                rater2_liver=data["rater2_liver"],
                rater1_tumor=data["rater1_tumor"],
                rater2_tumor=data["rater2_tumor"],
                couinaud_mask=data["couinaud_mask"]
            )
            meta = data["metadata"]
            manifest[p] = meta
            success_count += 1
            print(f"  --> [OK] Saved {p}.npz | Slices: {meta['num_slices']} | Liver: {meta['liver_volume_cm3']} cm³ | Tumor: {meta['tumor_volume_cm3']} cm³", flush=True)
        except Exception as e:
            fail_count += 1
            print(f"  --> [FAIL] Error processing MRI {p}: {type(e).__name__}: {e}", flush=True)

    summary_file = os.path.join(out_dir, "mri_dataset_summary.json")
    with open(summary_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nMRI Preprocessing Complete: {success_count} succeeded, {fail_count} failed.")
    print(f"Summary manifest saved to: {summary_file}")
    return manifest

if __name__ == "__main__":
    preprocess_and_save_all_mri()
