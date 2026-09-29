from pathlib import Path
from collections import Counter

# Class names
class_names = {
    0: "D00",
    1: "D01",
    2: "D10",
    3: "D11",
    4: "D20",
    5: "D40",
    6: "D43",
    7: "D44",
    8: "D50"
}

def count_classes(labels_path):

    counts = Counter()

    label_files = sorted(labels_path.glob("*.txt"))

    for label_file in label_files:

        with open(label_file, "r") as file:

            for line in file:

                values = line.strip().split()

                if not values:
                    continue

                class_id = int(values[0])

                counts[class_id] += 1

    return counts

# Paths
train_labels = Path("dataset/labels/train")
val_labels = Path("dataset/labels/val")

# Count objects
train_counts = count_classes(train_labels)
val_counts = count_classes(val_labels)

print("TRAINING SET")
print("-" * 30)

for class_id, name in class_names.items():
    print(
        f"{name}: {train_counts[class_id]}"
    )

print("\nVALIDATION SET")
print("-" * 30)

for class_id, name in class_names.items():
    print(
        f"{name}: {val_counts[class_id]}"
    )

print("\nTOTAL")
print("-" * 30)

for class_id, name in class_names.items():

    total = (
        train_counts[class_id]
        + val_counts[class_id]
    )

    print(f"{name}: {total}")
