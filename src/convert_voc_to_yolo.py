from pathlib import Path
import xml.etree.ElementTree as ET

# Class mapping
class_mapping = {
    "D00": 0,
    "D01": 1,
    "D10": 2,
    "D11": 3,
    "D20": 4,
    "D40": 5,
    "D43": 6,
    "D44": 7,
    "D50": 8
}

# Paths
annotations_path = Path("dataset/train/annotations/xmls")
labels_path = Path("dataset/train/labels")

# Create labels folder
labels_path.mkdir(parents=True, exist_ok=True)

# Find all XML files
xml_files = sorted(annotations_path.glob("*.xml"))

print("XML files found:", len(xml_files))

converted = 0
skipped = 0

# Process every XML file
for xml_file in xml_files:

    try:
        # Read XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image size
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        # Store YOLO annotations for this image
        yolo_lines = []

        # Process every object
        for obj in root.findall("object"):

            # Get class
            class_name = obj.find("name").text

            # Check class
            if class_name not in class_mapping:
                print(f"Unknown class in {xml_file.name}: {class_name}")
                continue

            class_id = class_mapping[class_name]

            # Get bounding box
            box = obj.find("bndbox")

            if box is None:
                print(f"No bounding box in {xml_file.name}")
                continue

            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)

            # Check coordinates
            if xmax <= xmin or ymax <= ymin:
                print(f"Invalid bounding box in {xml_file.name}")
                continue

            # Calculate width and height
            box_width = xmax - xmin
            box_height = ymax - ymin

            # Calculate center
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Normalize
            x_center /= width
            y_center /= height
            box_width /= width
            box_height /= height

            # Check normalized values
            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 <= box_width <= 1
                and 0 <= box_height <= 1
            ):
                print(f"Invalid normalized box in {xml_file.name}")
                continue

            # Create YOLO annotation
            yolo_line = (
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            yolo_lines.append(yolo_line)

        # Create output filename
        output_filename = xml_file.stem + ".txt"
        output_path = labels_path / output_filename

        # Write all annotations
        with open(output_path, "w") as file:
            file.write("\n".join(yolo_lines) + "\n")

        converted += 1

    except Exception as e:
        print(f"Error processing {xml_file.name}: {e}")
        skipped += 1

print("\nConversion complete!")
print("Converted:", converted)
print("Skipped:", skipped)