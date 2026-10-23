import os
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\project\majorprojectnew\dataset"

OUTPUT_PATH = os.path.join(
    r"C:\project\majorprojectnew\outputs",
    "aligned_segmentations"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# VOLUME FOLDERS
# ============================================================

VOLUME_FOLDERS = [
    "volume_pt1",
    "volume_pt2",
    "volume_pt3",
    "volume_pt4",
    "volume_pt5"
]


# ============================================================
# FUNCTION TO FIND CT
# ============================================================

def find_ct(case_number):

    filename = f"volume-{case_number}.nii"

    for folder in VOLUME_FOLDERS:

        path = os.path.join(
            DATASET_PATH,
            folder,
            filename
        )

        if os.path.exists(path):
            return path

    return None


# ============================================================
# START
# ============================================================

print("=" * 60)
print("       ALIGNING COMPLETE DATASET")
print("=" * 60)

successful = 0
failed = 0

failed_cases = []


# ============================================================
# PROCESS CASES 0 - 50
# ============================================================

for case_number in range(51):

    print()
    print("-" * 60)
    print(f"Processing Case {case_number:03d}")
    print("-" * 60)

    # --------------------------------------------------------
    # FIND CT
    # --------------------------------------------------------

    ct_path = find_ct(case_number)

    if ct_path is None:

        print("ERROR: CT not found")

        failed += 1
        failed_cases.append(case_number)

        continue


    # --------------------------------------------------------
    # SEGMENTATION PATH
    # --------------------------------------------------------

    seg_path = os.path.join(
        DATASET_PATH,
        "segmentations",
        f"segmentation-{case_number}.nii"
    )

    if not os.path.exists(seg_path):

        print("ERROR: Segmentation not found")

        failed += 1
        failed_cases.append(case_number)

        continue


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        ct_img = nib.load(ct_path)
        seg_img = nib.load(seg_path)

        print("CT shape :", ct_img.shape)
        print("SEG shape:", seg_img.shape)


        # ----------------------------------------------------
        # ALIGN SEGMENTATION → CT
        # ----------------------------------------------------

        aligned_seg = resample_from_to(
            seg_img,
            ct_img,
            order=0
        )


        # ----------------------------------------------------
        # CHECK LABELS
        # ----------------------------------------------------

        aligned_data = aligned_seg.get_fdata()

        labels = np.unique(aligned_data)

        print("Labels after alignment:", labels)


        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        valid_labels = {0.0, 1.0, 2.0}

        if not set(labels).issubset(valid_labels):

            print(
                "WARNING: Unexpected labels:",
                labels
            )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_PATH,
            f"segmentation-{case_number}.nii.gz"
        )

        nib.save(
            aligned_seg,
            output_file
        )

        print("Saved:", output_file)

        successful += 1


    except Exception as e:

        print("FAILED:", e)

        failed += 1
        failed_cases.append(case_number)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("              ALIGNMENT SUMMARY")
print("=" * 60)

print("Total expected cases :", 51)
print("Successfully aligned :", successful)
print("Failed               :", failed)
print("Failed case numbers  :", failed_cases)

print()
print("Aligned masks saved in:")
print(OUTPUT_PATH)

print("=" * 60)