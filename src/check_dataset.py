import os
import re

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = r"C:\project\majorprojectnew\dataset"

# ============================================================
# FIND CT VOLUMES
# ============================================================

ct_files = []

for part in ["volume_pt1", "volume_pt2", "volume_pt3", "volume_pt4", "volume_pt5"]:

    folder = os.path.join(DATASET_PATH, part)

    if not os.path.exists(folder):
        print(f"WARNING: Folder not found: {folder}")
        continue

    for filename in os.listdir(folder):

        if filename.lower().endswith(".nii"):

            match = re.match(r"volume-(\d+)\.nii$", filename)

            if match:
                number = int(match.group(1))

                ct_files.append({
                    "number": number,
                    "file": filename,
                    "folder": part
                })


# ============================================================
# FIND SEGMENTATIONS
# ============================================================

seg_folder = os.path.join(DATASET_PATH, "segmentations")

seg_files = []

for filename in os.listdir(seg_folder):

    if filename.lower().endswith(".nii"):

        match = re.match(r"segmentation-(\d+)\.nii$", filename)

        if match:
            number = int(match.group(1))

            seg_files.append({
                "number": number,
                "file": filename
            })


# ============================================================
# SORT
# ============================================================

ct_files.sort(key=lambda x: x["number"])
seg_files.sort(key=lambda x: x["number"])


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("        DATASET VERIFICATION")
print("========================================")

print("\nCT VOLUMES")
print("----------------------------------------")
print("Total CT volumes:", len(ct_files))

if ct_files:
    print("First CT:", ct_files[0]["file"])
    print("Last CT :", ct_files[-1]["file"])


print("\nSEGMENTATIONS")
print("----------------------------------------")
print("Total segmentation files:", len(seg_files))

if seg_files:
    print("First segmentation:", seg_files[0]["file"])
    print("Last segmentation :", seg_files[-1]["file"])


# ============================================================
# CHECK DUPLICATE CT NUMBERS
# ============================================================

ct_numbers = [x["number"] for x in ct_files]
seg_numbers = [x["number"] for x in seg_files]

duplicate_ct = sorted(
    set(n for n in ct_numbers if ct_numbers.count(n) > 1)
)

duplicate_seg = sorted(
    set(n for n in seg_numbers if seg_numbers.count(n) > 1)
)


print("\nDUPLICATES")
print("----------------------------------------")
print("Duplicate CT numbers:", duplicate_ct)
print("Duplicate segmentation numbers:", duplicate_seg)


# ============================================================
# CHECK MATCHING NUMBERS
# ============================================================

ct_set = set(ct_numbers)
seg_set = set(seg_numbers)

matching = sorted(ct_set & seg_set)
missing_seg = sorted(ct_set - seg_set)
extra_seg = sorted(seg_set - ct_set)


print("\nPAIRING CHECK")
print("----------------------------------------")
print("Matching numbers:", len(matching))
print("CT without segmentation:", missing_seg)
print("Extra segmentation files:", extra_seg)


# ============================================================
# SHOW CT FILE DISTRIBUTION
# ============================================================

print("\nCT DISTRIBUTION")
print("----------------------------------------")

for part in ["volume_pt1", "volume_pt2", "volume_pt3", "volume_pt4", "volume_pt5"]:

    part_files = [
        x for x in ct_files
        if x["folder"] == part
    ]

    if part_files:
        numbers = [x["number"] for x in part_files]

        print(
            f"{part}: {len(numbers)} files "
            f"({min(numbers)} - {max(numbers)})"
        )


print("\n========================================")
print("       VERIFICATION COMPLETE")
print("========================================")