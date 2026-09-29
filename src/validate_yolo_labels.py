from pathlib import Path

# Valid class IDs
valid_class_ids = set(range(9))

def validate_labels(labels_path):

    label_files = sorted(labels_path.glob("*.txt"))

    total_labels = 0
    invalid_labels = 0

    for label_file in label_files:

        with open(label_file, "r") as file:

            for line_number, line in enumerate(file, start=1):

                values = line.strip().split()

                # Check number of values
                if len(values) != 5:
                    print(
                        f"Invalid format: "
                        f"{label_file.name}, line {line_number}"
                    )
                    invalid_labels += 1
                    continue

                try:
                    class_id = int(values[0])
                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                except ValueError:
                    print(
                        f"Invalid numbers: "
                        f"{label_file.name}, line {line_number}"
                    )
                    invalid_labels += 1
                    continue

                # Check class ID
                if class_id not in valid_class_ids:
                    print(
                        f"Invalid class ID: "
                        f"{label_file.name}, line {line_number}"
                    )
                    invalid_labels += 1
                    continue

                # Check coordinates
                if not (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):
                    print(
                        f"Invalid coordinates: "
                        f"{label_file.name}, line {line_number}"
                    )
                    invalid_labels += 1
                    continue

                total_labels += 1

    return len(label_files), total_labels, invalid_labels

# Training
train_files, train_objects, train_invalid = validate_labels(
    Path("dataset/labels/train")
)

# Validation
val_files, val_objects, val_invalid = validate_labels(
    Path("dataset/labels/val")
)

print("\nTRAIN")
print("-" * 30)
print("Label files:", train_files)
print("Valid objects:", train_objects)
print("Invalid objects:", train_invalid)

print("\nVALIDATION")
print("-" * 30)
print("Label files:", val_files)
print("Valid objects:", val_objects)
print("Invalid objects:", val_invalid)

print("\nOVERALL")
print("-" * 30)

if train_invalid == 0 and val_invalid == 0:
    print("YOLO label validation PASSED!")
else:
    print("YOLO label validation FAILED!")
