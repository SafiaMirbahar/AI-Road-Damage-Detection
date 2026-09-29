from pathlib import Path

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_image(image, confidence):

    if image is None:
        return None, "No image uploaded."

    results = model(
        image,
        conf=confidence,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    annotated_image = result.plot()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            class_name = model.names[class_id]

            detections.append(
                f"{class_name} — {conf:.2f}"
            )

    if detections:

        summary = (
            f"Detections: {len(detections)}\n\n"
            + "\n".join(detections)
        )

    else:

        summary = "No road damage detected."

    return annotated_image, summary

# ============================================================
# VIDEO DETECTION
# ============================================================

def detect_video(video_path, confidence):

    if video_path is None:
        return None

    video_path = str(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Temporary AVI file
    temp_output = results_dir / "annotated_temp.avi"

    # Final browser-compatible MP4
    final_output = results_dir / "annotated_video.mp4"

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        print("ERROR: Could not create video writer.")
        return None

    frame_count = 0
    detection_count = 0

    print("Processing video...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        results = model(
            frame,
            conf=confidence,
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

    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")

    if frame_count == 0:
        temp_output.unlink(missing_ok=True)
        return None

    # ---------------------------------------------------------
    # Convert AVI → H.264 MP4
    # ---------------------------------------------------------

    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(temp_output),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_output)
    ]

    conversion = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    temp_output.unlink(missing_ok=True)

    if conversion.returncode != 0:
        print("FFmpeg conversion failed:")
        print(conversion.stderr)
        return None

    if not final_output.exists():
        print("ERROR: Final video was not created.")
        return None

    print(f"Video saved to: {final_output}")

    return str(final_output)

# ============================================================
# LIVE WEBCAM DETECTION
# ============================================================

def detect_webcam(frame, confidence):

    if frame is None:
        return None

    results = model(
        frame,
        conf=confidence,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes and labels
    annotated_frame = result.plot()

    return annotated_frame

# ============================================================
# GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="Road Damage Detection"
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # Road Damage Detection

        Detect and locate road damage from images, videos, or a
        live camera feed.
        """
    )

    # --------------------------------------------------------
    # IMAGE DETECTION
    # --------------------------------------------------------

    with gr.Tab("Image"):

        gr.Markdown(
            """
            ### Image Detection

            Upload a road image to identify damaged areas.
            """
        )

        with gr.Row():

            with gr.Column():

                image_input = gr.Image(
                    type="numpy",
                    label="Road Image"
                )

                confidence_image = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=0.05,
                    step=0.05,
                    label="Detection Confidence"
                )

                image_button = gr.Button(
                    "Detect Damage",
                    variant="primary"
                )

            with gr.Column():

                image_output = gr.Image(
                    label="Detection Result",
                    width=650
                )

                image_info = gr.Textbox(
                    label="Detection Details",
                    interactive=False
                )

        image_button.click(
            fn=detect_image,
            inputs=[
                image_input,
                confidence_image
            ],
            outputs=[
                image_output,
                image_info
            ]
        )

    # --------------------------------------------------------
    # VIDEO DETECTION
    # --------------------------------------------------------

    with gr.Tab("Video"):

        gr.Markdown(
            """
            ### Video Detection

            Upload a road video. Each frame will be analyzed and
            the detected damage will be shown in the processed video.
            """
        )

        with gr.Row():

            with gr.Column():

                video_input = gr.Video(
                    label="Road Video"
                )

                confidence_video = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=0.05,
                    step=0.05,
                    label="Detection Confidence"
                )

                video_button = gr.Button(
                    "Process Video",
                    variant="primary"
                )

            with gr.Column():

                video_output = gr.Video(
                    label="Detection Result",
                    width=700,
                    height=400
                )

        video_button.click(
            fn=detect_video,
            inputs=[
                video_input,
                confidence_video
            ],
            outputs=video_output
        )

    # --------------------------------------------------------
    # LIVE CAMERA
    # --------------------------------------------------------

    with gr.Tab("Live Camera"):

        gr.Markdown(
            """
            ### Live Camera Detection

            Start your camera and point it toward the road.
            Detected damage will be marked on the live video.
            """
        )

        confidence_webcam = gr.Slider(
            minimum=0.05,
            maximum=0.90,
            value=0.05,
            step=0.05,
            label="Detection Confidence"
        )

        with gr.Row():

            with gr.Column():

                webcam_input = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    label="Camera"
                )

            with gr.Column():

                webcam_output = gr.Image(
                    label="Detection Result"
                )

        webcam_input.stream(
            fn=detect_webcam,
            inputs=[
                webcam_input,
                confidence_webcam
            ],
            outputs=webcam_output
        )

# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(theme=gr.themes.Soft())
