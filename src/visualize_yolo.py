from pathlib import Path
from PIL import Image, ImageDraw

# Paths
images_path = Path("dataset/train/images")
labels_path = Path("dataset/train/labels")

# Select one image
image_path = sorted(images_path.glob("*.jpg"))[0]

# Corresponding YOLO label
label_path = labels_path / f"{image_path.stem}.txt"

# Open image
image = Image.open(image_path).convert("RGB")

# Get image dimensions
image_width, image_height = image.size

# Drawing object
draw = ImageDraw.Draw(image)

# Read YOLO labels
with open(label_path, "r") as file:

    for line in file:

        values = line.strip().split()

        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert normalized values back to pixels
        x_center *= image_width
        y_center *= image_height
        box_width *= image_width
        box_height *= image_height

        # Calculate corners
        xmin = int(x_center - box_width / 2)
        ymin = int(y_center - box_height / 2)
        xmax = int(x_center + box_width / 2)
        ymax = int(y_center + box_height / 2)

        # Draw bounding box
        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline="red",
            width=3
        )

        # Draw class ID
        draw.text(
            (xmin, max(0, ymin - 20)),
            str(class_id),
            fill="red"
        )

# Save result
output_path = Path("results/yolo_annotated.jpg")
output_path.parent.mkdir(exist_ok=True)

image.save(output_path)

print("Image:", image_path.name)
print("Label:", label_path.name)
print("Saved:", output_path)
