import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\project\majorprojectnew\dataset"
OUTPUT_PATH = r"C:\project\majorprojectnew\outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# CASE TO TEST
# ============================================================

CASE = 48


# ============================================================
# FIND CT
# ============================================================

ct_path = None

for part in [
    "volume_pt1",
    "volume_pt2",
    "volume_pt3",
    "volume_pt4",
    "volume_pt5"
]:
    possible = os.path.join(
        DATASET_PATH,
        part,
        f"volume-{CASE}.nii"
    )

    if os.path.exists(possible):
        ct_path = possible
        break


# ============================================================
# SEGMENTATION
# ============================================================

seg_path = os.path.join(
    DATASET_PATH,
    "segmentations",
    f"segmentation-{CASE}.nii"
)


print("=" * 60)
print("       SEGMENTATION ALIGNMENT")
print("=" * 60)

print("\nCT:")
print(ct_path)

print("\nSegmentation:")
print(seg_path)


if ct_path is None:
    raise FileNotFoundError("CT volume not found!")

if not os.path.exists(seg_path):
    raise FileNotFoundError("Segmentation not found!")


# ============================================================
# LOAD FILES
# ============================================================

ct_img = nib.load(ct_path)
seg_img = nib.load(seg_path)


print("\nOriginal CT shape:")
print(ct_img.shape)

print("Original segmentation shape:")
print(seg_img.shape)


# ============================================================
# PRINT AFFINES
# ============================================================

print("\nCT affine:")
print(ct_img.affine)

print("\nSegmentation affine:")
print(seg_img.affine)


# ============================================================
# RESAMPLE SEGMENTATION TO CT SPACE
# ============================================================

print("\nAligning segmentation to CT...")

aligned_seg = resample_from_to(
    seg_img,
    ct_img,
    order=0
)


# order=0 = nearest-neighbour
# IMPORTANT for segmentation labels


# ============================================================
# SAVE ALIGNED SEGMENTATION
# ============================================================

aligned_seg_path = os.path.join(
    OUTPUT_PATH,
    f"aligned_segmentation_{CASE}.nii.gz"
)

nib.save(
    aligned_seg,
    aligned_seg_path
)

print("\nAligned segmentation saved:")
print(aligned_seg_path)


# ============================================================
# LOAD ALIGNED DATA
# ============================================================

ct_data = ct_img.get_fdata(
    dtype=np.float32
)

seg_data = aligned_seg.get_fdata()


print("\nAligned segmentation shape:")
print(seg_data.shape)

print(
    "Labels:",
    np.unique(seg_data)
)


# ============================================================
# FIND LIVER
# ============================================================

liver_mask = (
    seg_data == 1
)


# ============================================================
# FIND SLICE WITH MAXIMUM LIVER
# ============================================================

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
# DISPLAY
# ============================================================

ct_slice = ct_data[
    :, :,
    best_slice
]

liver_slice = liver_mask[
    :, :,
    best_slice
]


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
    f"ALIGNED CT + LIVER | "
    f"Case {CASE} | Slice {best_slice}"
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# SAVE IMAGE
# ============================================================

image_path = os.path.join(
    OUTPUT_PATH,
    f"aligned_liver_check_{CASE}.png"
)

plt.savefig(
    image_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nVerification image saved:")
print(image_path)

plt.show()


print("\n" + "=" * 60)
print("ALIGNMENT COMPLETE")
print("=" * 60)