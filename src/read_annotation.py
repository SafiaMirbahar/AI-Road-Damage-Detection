from pathlib import Path
import xml.etree.ElementTree as ET

# Paths
annotations_path = Path("dataset/train/annotations/xmls")
images_path = Path("dataset/train/images")

# Get the first XML file
xml_file = sorted(annotations_path.glob("*.xml"))[0]

print("XML file:", xml_file.name)

# Parse XML
tree = ET.parse(xml_file)
root = tree.getroot()

# Image filename
filename = root.find("filename").text
print("Image:", filename)

# Image size
size = root.find("size")

width = int(size.find("width").text)
height = int(size.find("height").text)

print("Width:", width)
print("Height:", height)

# Read every object
for i, obj in enumerate(root.findall("object"), start=1):

    class_name = obj.find("name").text

    box = obj.find("bndbox")

    xmin = int(box.find("xmin").text)
    ymin = int(box.find("ymin").text)
    xmax = int(box.find("xmax").text)
    ymax = int(box.find("ymax").text)

    print(f"\nObject {i}")
    print("Class:", class_name)
    print("Bounding box:")
    print("xmin:", xmin)
    print("ymin:", ymin)
    print("xmax:", xmax)
    print("ymax:", ymax)
