import os
import re
import nibabel as nib
import numpy as np

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\project\majorprojectnew\dataset"

OUTPUT_PATH = r"C:\project\majorprojectnew\outputs\prepared"

os.makedirs(OUTPUT_PATH, exist_ok=True)

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
# FIND ALL CT VOLUMES
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

print("========================================")
print("       DATASET PREPARATION")
print("========================================")

print("\nCT volumes found:", len(ct_files))


# ============================================================
# PROCESS ALL MATCHED VOLUMES
# ============================================================

processed = 0
skipped = 0

for number in sorted(ct_files):

    # --------------------------------------------------------
    # Only process CT volumes having segmentation
    # --------------------------------------------------------

    ct_path = ct_files[number]

    seg_path = os.path.join(
        SEGMENTATION_FOLDER,
        f"segmentation-{number}.nii"
    )

    if not os.path.exists(seg_path):

        print(
            f"Skipping volume-{number}: "
            "segmentation not found"
        )

        skipped += 1
        continue

    print(
        f"\nProcessing case {number:03d}..."
    )

    # --------------------------------------------------------
    # LOAD CT
    # --------------------------------------------------------

    ct_img = nib.load(ct_path)

    ct_data = ct_img.get_fdata(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # LOAD SEGMENTATION
    # --------------------------------------------------------

    seg_img = nib.load(seg_path)

    seg_data = seg_img.get_fdata()

    # --------------------------------------------------------
    # CHECK SHAPE
    # --------------------------------------------------------

    if ct_data.shape != seg_data.shape:

        print(
            "ERROR: Shape mismatch!"
        )

        print(
            "CT:",
            ct_data.shape
        )

        print(
            "SEG:",
            seg_data.shape
        )

        skipped += 1
        continue

    # ========================================================
    # EXTRACT LIVER
    # Label 1 = Liver
    # ========================================================

    liver_mask = (
        seg_data == 1
    ).astype(np.uint8)

    # ========================================================
    # BASIC CT PREPROCESSING
    # ========================================================

    # Clip CT intensity to a useful abdominal window
    ct_data = np.clip(
        ct_data,
        -200,
        250
    )

    # Normalize to 0-1
    ct_min = ct_data.min()
    ct_max = ct_data.max()

    if ct_max > ct_min:

        ct_data = (
            ct_data - ct_min
        ) / (
            ct_max - ct_min
        )

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        os.path.join(
            OUTPUT_PATH,
            f"ct_{number:03d}.npy"
        ),
        ct_data
    )

    np.save(
        os.path.join(
            OUTPUT_PATH,
            f"liver_{number:03d}.npy"
        ),
        liver_mask
    )

    processed += 1

    print(
        f"CT shape: {ct_data.shape}"
    )

    print(
        f"Liver voxels: "
        f"{np.sum(liver_mask):,}"
    )

    print(
        "Saved successfully."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("       PREPARATION COMPLETE")
print("========================================")

print(
    f"CT volumes found : {len(ct_files)}"
)

print(
    f"Cases processed   : {processed}"
)

print(
    f"Cases skipped     : {skipped}"
)

print(
    f"\nPrepared data saved in:"
)

print(
    OUTPUT_PATH
)

print("========================================")