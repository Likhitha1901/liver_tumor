import os
import numpy as np
import nibabel as nib

# ============================================================
# VERIFY ALIGNED LIVER SEGMENTATIONS
# ============================================================

CT_DIR = "outputs/prepared"
SEG_DIR = "outputs/aligned_segmentations"

print("=" * 60)
print("      VERIFYING ALIGNED DATASET")
print("=" * 60)

# ------------------------------------------------------------
# Find files
# ------------------------------------------------------------

ct_files = sorted([
    f for f in os.listdir(CT_DIR)
    if f.endswith(".npy")
])

seg_files = sorted([
    f for f in os.listdir(SEG_DIR)
    if f.endswith(".nii.gz")
])

print(f"\nCT files found          : {len(ct_files)}")
print(f"Segmentation files     : {len(seg_files)}")

# ------------------------------------------------------------
# Check each case
# ------------------------------------------------------------

successful = 0
failed = 0

empty_masks = []
wrong_shape = []
wrong_labels = []

for i in range(51):

    ct_name = f"ct_{i:03d}.npy"
    seg_name = f"segmentation-{i}.nii.gz"

    ct_path = os.path.join(CT_DIR, ct_name)
    seg_path = os.path.join(SEG_DIR, seg_name)

    print("\n" + "-" * 60)
    print(f"Case {i:03d}")
    print(f"CT  : {ct_name}")
    print(f"SEG : {seg_name}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(ct_path):
        print("❌ CT missing")
        failed += 1
        continue

    if not os.path.exists(seg_path):
        print("❌ Segmentation missing")
        failed += 1
        continue

    try:

        # ----------------------------------------------------
        # Load CT (.npy)
        # ----------------------------------------------------

        ct_data = np.load(ct_path)

        # ----------------------------------------------------
        # Load segmentation (.nii.gz)
        # ----------------------------------------------------

        seg = nib.load(seg_path)
        seg_data = seg.get_fdata()

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        print(f"CT shape       : {ct_data.shape}")
        print(f"SEG shape      : {seg_data.shape}")

        if ct_data.shape != seg_data.shape:
            print("❌ SHAPE MISMATCH")

            wrong_shape.append(i)
            failed += 1
            continue

        # ----------------------------------------------------
        # Label check
        # ----------------------------------------------------

        labels = np.unique(seg_data)

        print(f"Labels         : {labels}")

        valid_labels = np.all(
            np.isin(labels, [0, 1, 2])
        )

        if not valid_labels:
            print("⚠ Unexpected labels")
            wrong_labels.append(i)

        # ----------------------------------------------------
        # Liver voxel check
        # ----------------------------------------------------

        liver_voxels = np.sum(seg_data == 1)

        print(f"Liver voxels   : {liver_voxels}")

        if liver_voxels == 0:
            print("❌ EMPTY LIVER MASK")

            empty_masks.append(i)
            failed += 1
            continue

        # ----------------------------------------------------
        # Segmentation affine information
        # ----------------------------------------------------

        print("Segmentation affine:")
        print(seg.affine)

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print("✅ Case verified")

        successful += 1

    except Exception as e:

        print(f"❌ ERROR: {e}")
        failed += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("             VERIFICATION SUMMARY")
print("=" * 60)

print(f"Total cases checked     : 51")
print(f"Successfully verified   : {successful}")
print(f"Failed                  : {failed}")

print(f"\nEmpty masks             : {empty_masks}")
print(f"Shape mismatches        : {wrong_shape}")
print(f"Unexpected labels       : {wrong_labels}")

print("\n" + "=" * 60)

if failed == 0:
    print("🎉 DATASET VERIFICATION PASSED")
else:
    print("⚠ DATASET NEEDS ATTENTION")

print("=" * 60)