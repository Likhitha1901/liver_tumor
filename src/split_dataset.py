import os
import random
import json

# ============================================================
# PATH
# ============================================================

PREPARED_PATH = r"C:\project\majorprojectnew\outputs\prepared"

OUTPUT_PATH = r"C:\project\majorprojectnew\outputs"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# FIND PREPARED CASES
# ============================================================

case_numbers = []

for filename in os.listdir(PREPARED_PATH):

    if filename.startswith("ct_") and filename.endswith(".npy"):

        number = int(
            filename.replace("ct_", "").replace(".npy", "")
        )

        liver_file = os.path.join(
            PREPARED_PATH,
            f"liver_{number:03d}.npy"
        )

        if os.path.exists(liver_file):
            case_numbers.append(number)

case_numbers.sort()

print("========================================")
print("       DATASET SPLITTING")
print("========================================")

print("\nTotal valid cases:", len(case_numbers))

# ============================================================
# SHUFFLE
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(case_numbers)

# ============================================================
# CALCULATE SPLIT SIZES
# ============================================================

total = len(case_numbers)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

# Remaining cases go to test
test_count = total - train_count - val_count

# ============================================================
# CREATE SPLITS
# ============================================================

train_cases = case_numbers[:train_count]

val_cases = case_numbers[
    train_count:train_count + val_count
]

test_cases = case_numbers[
    train_count + val_count:
]

# Sort for readability
train_cases.sort()
val_cases.sort()
test_cases.sort()

# ============================================================
# PRINT
# ============================================================

print("\nTRAINING SET")
print("----------------------------------------")
print("Number of cases:", len(train_cases))
print(train_cases)

print("\nVALIDATION SET")
print("----------------------------------------")
print("Number of cases:", len(val_cases))
print(val_cases)

print("\nTEST SET")
print("----------------------------------------")
print("Number of cases:", len(test_cases))
print(test_cases)

# ============================================================
# SAVE SPLIT
# ============================================================

split_data = {
    "random_seed": RANDOM_SEED,
    "train": train_cases,
    "validation": val_cases,
    "test": test_cases
}

split_file = os.path.join(
    OUTPUT_PATH,
    "dataset_split.json"
)

with open(split_file, "w") as f:
    json.dump(
        split_data,
        f,
        indent=4
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("       SPLIT COMPLETE")
print("========================================")

print("Training   :", len(train_cases))
print("Validation :", len(val_cases))
print("Testing    :", len(test_cases))

print("\nSplit saved to:")
print(split_file)

print("========================================")