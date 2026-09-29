from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

# Path to XML annotations
annotations_path = Path("dataset/train/annotations/xmls")

# Store class counts
class_counts = Counter()

# Read every XML file
for xml_file in annotations_path.glob("*.xml"):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find every object in this image
    for obj in root.findall("object"):

        class_name = obj.find("name").text
        class_counts[class_name] += 1

# Display results
print("Classes found:\n")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nTotal objects:", sum(class_counts.values()))
