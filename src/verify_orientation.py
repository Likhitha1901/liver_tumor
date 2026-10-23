import os
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\project\majorprojectnew\dataset"

OUTPUT_PATH = r"C:\project\majorprojectnew\outputs"

# ============================================================
# SELECT ONE CASE
# ============================================================

CASE = 0

# ============================================================
# FIND CT FILE
# ============================================================

ct_path = None

for folder in [
    "volume_pt1",
    "volume_pt2",
    "volume_pt3",
    "volume_pt4",
    "volume_pt5"
]:

    possible_path = os.path.join(
        DATASET_PATH,
        folder,
        f"volume-{CASE}.nii"
    )

    if os.path.exists(possible_path):

        ct_path = possible_path
        break


# ============================================================
# SEGMENTATION PATH
# ============================================================

seg_path = os.path.join(
    DATASET_PATH,
    "segmentations",
    f"segmentation-{CASE}.nii"
)


print("========================================")
print("       ORIENTATION VERIFICATION")
print("========================================")

print("\nCT:")
print(ct_path)

print("\nSegmentation:")
print(seg_path)


# ============================================================
# LOAD ORIGINAL NIFTI
# ============================================================

ct_img = nib.load(ct_path)
seg_img = nib.load(seg_path)

print("\nOriginal CT shape:")
print(ct_img.shape)

print("Original SEG shape:")
print(seg_img.shape)


# ============================================================
# PRINT AFFINES
# ============================================================

print("\nCT affine:")
print(ct_img.affine)

print("\nSEG affine:")
print(seg_img.affine)


# ============================================================
# REORIENT BOTH TO RAS
# ============================================================

ct_ras = nib.as_closest_canonical(ct_img)
seg_ras = nib.as_closest_canonical(seg_img)

print("\n========================================")
print("AFTER REORIENTATION")
print("========================================")

print("\nCT RAS shape:")
print(ct_ras.shape)

print("SEG RAS shape:")
print(seg_ras.shape)


# ============================================================
# LOAD DATA
# ============================================================

ct_data = ct_ras.get_fdata(
    dtype=np.float32
)

seg_data = seg_ras.get_fdata()


# ============================================================
# CHECK SHAPES
# ============================================================

if ct_data.shape != seg_data.shape:

    print("\nERROR:")
    print("CT and segmentation shapes differ after reorientation.")

    print(
        "CT:",
        ct_data.shape
    )

    print(
        "SEG:",
        seg_data.shape
    )

    raise SystemExit


# ============================================================
# FIND BEST LIVER SLICE
# ============================================================

liver_mask = (
    seg_data == 1
)

liver_counts = np.sum(
    liver_mask,
    axis=(0, 1)
)

best_slice = int(
    np.argmax(liver_counts)
)

print(
    "\nBest liver slice:",
    best_slice
)

print(
    "Liver pixels:",
    liver_counts[best_slice]
)


# ============================================================
# EXTRACT SLICE
# ============================================================

ct_slice = ct_data[
    :, :,
    best_slice
]

liver_slice = liver_mask[
    :, :,
    best_slice
]


# ============================================================
# DISPLAY
# ============================================================

plt.figure(
    figsize=(9, 9)
)

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.contour(
    liver_slice,
    levels=[0.5],
    linewidths=2
)

plt.title(
    f"Orientation Check | "
    f"Case {CASE} | Slice {best_slice}"
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

save_path = os.path.join(
    OUTPUT_PATH,
    "orientation_check.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nSaved:")
print(save_path)

plt.show()