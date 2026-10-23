import os
import json
import csv
import numpy as np

# ============================================================
# PATHS
# ============================================================

PREPARED_PATH = r"C:\project\majorprojectnew\outputs\prepared"

SPLIT_FILE = r"C:\project\majorprojectnew\outputs\dataset_split.json"

OUTPUT_PATH = r"C:\project\majorprojectnew\outputs"

# ============================================================
# LOAD DATASET SPLIT
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)

# ============================================================
# FUNCTION TO CREATE SLICE INDEX
# ============================================================

def create_index(case_numbers, output_file):

    rows = []

    for case_number in case_numbers:

        ct_file = os.path.join(
            PREPARED_PATH,
            f"ct_{case_number:03d}.npy"
        )

        liver_file = os.path.join(
            PREPARED_PATH,
            f"liver_{case_number:03d}.npy"
        )

        if not os.path.exists(ct_file):
            print(f"Missing CT: {ct_file}")
            continue

        if not os.path.exists(liver_file):
            print(f"Missing liver mask: {liver_file}")
            continue

        # ----------------------------------------------------
        # Load only to determine slice information
        # ----------------------------------------------------

        liver = np.load(
            liver_file,
            mmap_mode="r"
        )

        num_slices = liver.shape[2]

        # ----------------------------------------------------
        # Create slice entries
        # ----------------------------------------------------

        for slice_index in range(num_slices):

            mask_slice = liver[:, :, slice_index]

            liver_pixels = np.sum(
                mask_slice > 0
            )

            # ------------------------------------------------
            # Keep all liver-containing slices
            # ------------------------------------------------

            if liver_pixels > 0:

                rows.append([
                    case_number,
                    slice_index,
                    1
                ])

            # ------------------------------------------------
            # Keep some background slices
            # Every 5th slice without liver
            # ------------------------------------------------

            elif slice_index % 5 == 0:

                rows.append([
                    case_number,
                    slice_index,
                    0
                ])

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "case",
            "slice",
            "has_liver"
        ])

        writer.writerows(rows)

    return len(rows)


# ============================================================
# CREATE TRAIN INDEX
# ============================================================

print("========================================")
print("       CREATING SLICE INDEX")
print("========================================")

train_count = create_index(
    split_data["train"],
    os.path.join(
        OUTPUT_PATH,
        "train_slices.csv"
    )
)

print(
    "\nTraining slices:",
    train_count
)

# ============================================================
# CREATE VALIDATION INDEX
# ============================================================

val_count = create_index(
    split_data["validation"],
    os.path.join(
        OUTPUT_PATH,
        "val_slices.csv"
    )
)

print(
    "Validation slices:",
    val_count
)

# ============================================================
# CREATE TEST INDEX
# ============================================================

test_count = create_index(
    split_data["test"],
    os.path.join(
        OUTPUT_PATH,
        "test_slices.csv"
    )
)

print(
    "Testing slices:",
    test_count
)

# ============================================================
# FINAL
# ============================================================

print("\n========================================")
print("       SLICE INDEX COMPLETE")
print("========================================")

print("\nCreated:")

print(
    os.path.join(
        OUTPUT_PATH,
        "train_slices.csv"
    )
)

print(
    os.path.join(
        OUTPUT_PATH,
        "val_slices.csv"
    )
)

print(
    os.path.join(
        OUTPUT_PATH,
        "test_slices.csv"
    )
)

print("========================================")