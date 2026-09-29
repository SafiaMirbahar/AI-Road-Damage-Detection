from pathlib import Path
import random
import shutil

# Source paths
images_source = Path("dataset/train/images")
labels_source = Path("dataset/train/labels")

# Destination paths
images_train = Path("dataset/images/train")
images_val = Path("dataset/images/val")

labels_train = Path("dataset/labels/train")
labels_val = Path("dataset/labels/val")

# Create directories
for path in [images_train, images_val, labels_train, labels_val]:
    path.mkdir(parents=True, exist_ok=True)

# Get all images
images = sorted(images_source.glob("*.jpg"))

# Shuffle images randomly
random.seed(42)
random.shuffle(images)

# Calculate split
split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

print("Total images:", len(images))
print("Training images:", len(train_images))
print("Validation images:", len(val_images))

# Copy training files
for image in train_images:

    label = labels_source / f"{image.stem}.txt"

    shutil.copy2(image, images_train / image.name)
    shutil.copy2(label, labels_train / label.name)

# Copy validation files
for image in val_images:

    label = labels_source / f"{image.stem}.txt"

    shutil.copy2(image, images_val / image.name)
    shutil.copy2(label, labels_val / label.name)

print("\nDataset split complete!")
