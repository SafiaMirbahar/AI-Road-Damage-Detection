from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# Paths
images_path = Path("dataset/train/images")
annotations_path = Path("dataset/train/annotations/xmls")

# Select one XML file
xml_file = sorted(annotations_path.glob("*.xml"))[0]

# Parse XML
tree = ET.parse(xml_file)
root = tree.getroot()

# Get image filename
filename = root.find("filename").text

# Open image
image_path = images_path / filename
image = Image.open(image_path).convert("RGB")

# Create drawing object
draw = ImageDraw.Draw(image)

# Draw bounding boxes
for obj in root.findall("object"):

    class_name = obj.find("name").text

    box = obj.find("bndbox")

    xmin = int(box.find("xmin").text)
    ymin = int(box.find("ymin").text)
    xmax = int(box.find("xmax").text)
    ymax = int(box.find("ymax").text)

    # Draw rectangle
draw.rectangle(
        [xmin, ymin, xmax, ymax],
        outline="red",
        width=3
    )

    # Add class name
draw.text(
        (xmin, max(0, ymin - 20)),
        class_name,
        fill="red"
    )

# Save result
output_path = Path("results/annotated_image.jpg")
output_path.parent.mkdir(exist_ok=True)

image.save(output_path)

print("Saved:", output_path)
