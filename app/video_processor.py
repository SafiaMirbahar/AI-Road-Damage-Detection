from pathlib import Path
import tempfile

import cv2


def process_video(model, uploaded_video):
    if model is None:
        return None

    # Create a permanent temporary folder
    temp_dir = tempfile.mkdtemp()

    input_path = Path(temp_dir) / "input_video.mp4"
    output_path = Path(temp_dir) / "output_video.mp4"

    # Save uploaded video
    input_path.write_bytes(uploaded_video.getbuffer())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        return None

    frame_count = 0
    detection_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        results = model(
            frame,
            conf=0.05,
            iou=0.45,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:
            detection_count += len(result.boxes)

        annotated_frame = result.plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    return str(output_path)
