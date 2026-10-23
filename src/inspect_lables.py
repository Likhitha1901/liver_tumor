import os
import re
import nibabel as nib
import numpy as np

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\project\majorprojectnew\dataset"

VOLUME_FOLDERS = [
    "volume_pt1",
    "volume_pt2",
    "volume_pt3",
    "volume_pt4",
    "volume_pt5"
]

SEGMENTATION_FOLDER = os.path.join(
    DATASET_PATH,
    "segmentations"
)

# ============================================================
# FIND CT VOLUMES
# ============================================================

ct_files = {}

for folder in VOLUME_FOLDERS:

    folder_path = os.path.join(
        DATASET_PATH,
        folder
    )

    if not os.path.exists(folder_path):
        continue

    for filename in os.listdir(folder_path):

        match = re.match(
            r"volume-(\d+)\.nii$",
            filename
        )

        if match:

            number = int(match.group(1))

            ct_files[number] = os.path.join(
                folder_path,
                filename
            )


# ============================================================
# CHECK FIRST 51 PAIRS
# ============================================================

print("\n========================================")
print("       SEGMENTATION LABEL CHECK")
print("========================================")

print("\nTotal CT volumes found:", len(ct_files))

print("\nChecking CT + segmentation pairs...\n")


all_labels = set()

tumor_cases = []
no_tumor_cases = []

shape_mismatches = []


for number in sorted(ct_files):

    if number > 50:
        continue

    ct_path = ct_files[number]

    seg_path = os.path.join(
        SEGMENTATION_FOLDER,
        f"segmentation-{number}.nii"
    )

    # --------------------------------------------------------
    # Check segmentation exists
    # --------------------------------------------------------

    if not os.path.exists(seg_path):

        print(
            f"WARNING: Missing segmentation-{number}.nii"
        )

        continue

    # --------------------------------------------------------
    # Load CT
    # --------------------------------------------------------

    ct_img = nib.load(ct_path)

    ct_data = ct_img.get_fdata()

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

    seg_img = nib.load(seg_path)

    seg_data = seg_img.get_fdata()

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    if ct_data.shape != seg_data.shape:

        shape_mismatches.append(number)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = np.unique(seg_data)

    all_labels.update(labels.tolist())

    # --------------------------------------------------------
    # Liver
    # --------------------------------------------------------

    liver_voxels = np.sum(
        seg_data == 1
    )

    # --------------------------------------------------------
    # Tumour
    # --------------------------------------------------------

    tumor_voxels = np.sum(
        seg_data == 2
    )

    # --------------------------------------------------------
    # Tumour cases
    # --------------------------------------------------------

    if tumor_voxels > 0:

        tumor_cases.append(number)

    else:

        no_tumor_cases.append(number)

    print(
        f"Case {number:03d} | "
        f"CT shape: {ct_data.shape} | "
        f"Labels: {labels.tolist()} | "
        f"Liver: {liver_voxels:,} | "
        f"Tumour: {tumor_voxels:,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("             FINAL SUMMARY")
print("========================================")

print(
    "\nLabels found in dataset:",
    sorted(all_labels)
)

print(
    "\nTumour-containing cases:",
    len(tumor_cases)
)

print(
    "Cases without tumour:",
    len(no_tumor_cases)
)

print(
    "\nTumour case numbers:",
    tumor_cases
)

print(
    "\nShape mismatches:",
    shape_mismatches
)

print("\n========================================")
print("          INSPECTION COMPLETE")
print("========================================")