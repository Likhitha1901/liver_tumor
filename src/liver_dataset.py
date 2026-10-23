import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# PATHS
# ============================================================

PREPARED_PATH = r"C:\project\majorprojectnew\outputs\prepared"
OUTPUT_PATH = r"C:\project\majorprojectnew\outputs"


# ============================================================
# LIVER DATASET
# ============================================================

class LiverDataset(Dataset):

    def __init__(self, csv_file):

        self.samples = []

        csv_path = os.path.join(
            OUTPUT_PATH,
            csv_file
        )

        # ----------------------------------------------------
        # Read slice index
        # ----------------------------------------------------

        with open(csv_path, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                self.samples.append({
                    "case": int(row["case"]),
                    "slice": int(row["slice"]),
                    "has_liver": int(row["has_liver"])
                })

        print(
            f"Loaded {len(self.samples)} slices "
            f"from {csv_file}"
        )

    # ========================================================
    # NUMBER OF SAMPLES
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # LOAD ONE SAMPLE
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        case_number = sample["case"]
        slice_index = sample["slice"]

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        ct_path = os.path.join(
            PREPARED_PATH,
            f"ct_{case_number:03d}.npy"
        )

        liver_path = os.path.join(
            PREPARED_PATH,
            f"liver_{case_number:03d}.npy"
        )

        # ----------------------------------------------------
        # Load 3D volume
        # ----------------------------------------------------

        ct_volume = np.load(ct_path)

        liver_volume = np.load(liver_path)

        # ----------------------------------------------------
        # Extract axial slice
        #
        # Volume shape:
        # (512, 512, number_of_slices)
        #
        # Therefore:
        # [:, :, slice_index]
        # ----------------------------------------------------

        ct_slice = ct_volume[
            :, :,
            slice_index
        ]

        liver_slice = liver_volume[
            :, :,
            slice_index
        ]

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        ct_slice = ct_slice.astype(
            np.float32
        )

        liver_slice = liver_slice.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Add channel dimension
        #
        # Before:
        # (512, 512)
        #
        # After:
        # (1, 512, 512)
        # ----------------------------------------------------

        ct_slice = np.expand_dims(
            ct_slice,
            axis=0
        )

        liver_slice = np.expand_dims(
            liver_slice,
            axis=0
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            ct_slice
        )

        mask = torch.from_numpy(
            liver_slice
        )

        return image, mask


# ============================================================
# TEST THE DATASET
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       TESTING LIVER DATASET")
    print("========================================")

    # --------------------------------------------------------
    # Create training dataset
    # --------------------------------------------------------

    dataset = LiverDataset(
        "train_slices.csv"
    )

    print(
        "\nDataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Load first sample
    # --------------------------------------------------------

    image, mask = dataset[0]

    print("\nFirst sample:")

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    print(
        "Mask dtype:",
        mask.dtype
    )

    print(
        "Image minimum:",
        image.min().item()
    )

    print(
        "Image maximum:",
        image.max().item()
    )

    print(
        "Mask unique values:",
        torch.unique(mask).tolist()
    )

    print("\n========================================")
    print("       DATASET TEST COMPLETE")
    print("========================================")