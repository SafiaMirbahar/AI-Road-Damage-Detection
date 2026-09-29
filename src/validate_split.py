from pathlib import Path

# Paths
images_train = Path("dataset/images/train")
labels_train = Path("dataset/labels/train")

images_val = Path("dataset/images/val")
labels_val = Path("dataset/labels/val")

def check_pairs(images_path, labels_path, split_name):

    images = list(images_path.glob("*.jpg"))
    labels = list(labels_path.glob("*.txt"))

    image_names = {image.stem for image in images}
    label_names = {label.stem for label in labels}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print(f"\n{split_name}")
    print("-" * 30)
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    return len(missing_labels) == 0 and len(missing_images) == 0

train_ok = check_pairs(
    images_train,
    labels_train,
    "TRAIN"
)

val_ok = check_pairs(
    images_val,
    labels_val,
    "VALIDATION"
)

if train_ok and val_ok:
    print("\nDataset validation PASSED!")
else:
    print("\nDataset validation FAILED!")
