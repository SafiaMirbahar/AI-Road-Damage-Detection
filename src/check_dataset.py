from pathlib import Path

images_path = Path("dataset/train/images")
labels_path = Path("dataset/train/labels")

# Find files
images = sorted(images_path.glob("*.jpg"))
labels = sorted(labels_path.glob("*.txt"))

print("Images:", len(images))
print("Labels:", len(labels))

# Compare filenames
image_names = {image.stem for image in images}
label_names = {label.stem for label in labels}

# Images without labels
missing_labels = image_names - label_names

# Labels without images
missing_images = label_names - image_names

print("Images without labels:", len(missing_labels))
print("Labels without images:", len(missing_images))

if missing_labels:
    print("\nMissing labels:")
    for name in sorted(missing_labels):
        print(name)

if missing_images:
    print("\nMissing images:")
    for name in sorted(missing_images):
        print(name)
