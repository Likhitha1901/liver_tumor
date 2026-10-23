import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PREPARED_PATH = r"C:\project\majorprojectnew\outputs\prepared"

OUTPUT_PATH = r"C:\project\majorprojectnew\outputs"

CSV_FILE = os.path.join(
    OUTPUT_PATH,
    "train_slices.csv"
)

# ============================================================
# FIND A SLICE THAT CONTAINS LIVER
# ============================================================

selected_case = None
selected_slice = None

with open(CSV_FILE, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        if int(row["has_liver"]) == 1:

            selected_case = int(row["case"])
            selected_slice = int(row["slice"])

            break


print("========================================")
print("       DATASET VISUALIZATION")
print("========================================")

print(
    "\nSelected case:",
    selected_case
)

print(
    "Selected slice:",
    selected_slice
)

# ============================================================
# LOAD CT
# ============================================================

ct_path = os.path.join(
    PREPARED_PATH,
    f"ct_{selected_case:03d}.npy"
)

liver_path = os.path.join(
    PREPARED_PATH,
    f"liver_{selected_case:03d}.npy"
)

ct_volume = np.load(ct_path)
liver_volume = np.load(liver_path)

# ============================================================
# EXTRACT SLICE
# ============================================================

ct_slice = ct_volume[
    :, :,
    selected_slice
]

liver_slice = liver_volume[
    :, :,
    selected_slice
]

print(
    "\nCT volume shape:",
    ct_volume.shape
)

print(
    "CT slice shape:",
    ct_slice.shape
)

print(
    "Liver pixels:",
    np.sum(liver_slice > 0)
)

# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.imshow(
    np.ma.masked_where(
        liver_slice == 0,
        liver_slice
    ),
    alpha=0.45
)

plt.title(
    f"CT + Ground Truth Liver | "
    f"Case {selected_case} | Slice {selected_slice}"
)

plt.axis("off")

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

save_path = os.path.join(
    OUTPUT_PATH,
    "sample_liver_overlay.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

print(
    "\nSaved visualization:"
)

print(save_path)

plt.show()