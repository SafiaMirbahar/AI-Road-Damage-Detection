from pathlib import Path
from PIL import Image

# Dataset paths
images_path = Path("dataset/train/images")
annotations_path = Path("dataset/train/annotations/xmls")

# Find files
images = list(images_path.glob("*.jpg"))
xml_files = list(annotations_path.glob("*.xml"))

print("Number of images:", len(images))
print("Number of XML annotations:", len(xml_files))

# Inspect first image
if images:
    image = Image.open(images[0])

    print("\nFirst image:")
    print("Filename:", images[0].name)
    print("Width:", image.width)
    print("Height:", image.height)
    print("Format:", image.format)
