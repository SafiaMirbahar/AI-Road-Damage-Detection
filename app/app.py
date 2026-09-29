import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide",
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))

model = load_model()

if model is None:
    st.warning("Model file not found. Add your trained weights to the `models` folder as `best.pt`.")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def run_detection(model, image_array, confidence):
    if model is None:
        st.warning(
            "No trained model found at models/best.pt."
        )
        return None, []

    results = model(
        image_array,
        conf=confidence,
        iou=0.45,
        imgsz=640,
        verbose=False,
    )

    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names.get(class_id, f"Class_{class_id}")
            confidence_value = float(box.conf[0].item())
            detections.append({
                "class": class_name,
                "confidence": confidence_value,
            })

    annotated = result.plot() if result is not None else image_array
    if len(annotated.shape) == 3 and annotated.shape[2] == 3:
        annotated = annotated[:, :, ::-1]

    return annotated, detections


def process_video(model, uploaded_video):

    if model is None:
        st.error("Model not found.")
        return None

    temp_dir = tempfile.mkdtemp()

    input_path = Path(temp_dir) / "input.mp4"
    raw_output_path = Path(temp_dir) / "raw_output.mp4"
    final_output_path = Path(temp_dir) / "annotated_video.mp4"

    # Save uploaded video
    input_path.write_bytes(uploaded_video.getbuffer())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        st.error("Unable to open uploaded video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary OpenCV video
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        st.error("Could not create output video.")
        return None

    progress_bar = st.progress(0)

    frame_count = 0
    total_detections = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Run YOLO
        results = model(
            frame,
            conf=0.05,
            iou=0.45,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Count detections
        if result.boxes is not None:
            total_detections += len(result.boxes)

        # Draw bounding boxes
        annotated_frame = result.plot()

        # Save annotated frame
        writer.write(annotated_frame)

        # Update progress
        if total_frames > 0:
            progress_bar.progress(
                min(frame_count / total_frames, 1.0)
            )

    cap.release()
    writer.release()

    progress_bar.empty()

    st.success(
        f"Video processing complete: "
        f"{frame_count} frames processed."
    )

    st.info(
        f"Total detections across video: "
        f"{total_detections}"
    )

    # Convert OpenCV video to browser-compatible H.264
    try:

        import subprocess

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final_output_path)
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        st.success("Video converted to browser-compatible format.")

        return str(final_output_path)

    except Exception as e:

        st.error(f"Video conversion failed: {e}")

        return str(raw_output_path)


def display_detections(detections):
    if not detections:
        st.info("No road damage detected.")
        return

    st.subheader("Detected Damage")

    for detection in detections:
        st.write(
            f"**{detection['class']}** — "
            f"Confidence: {detection['confidence']:.1%}"
        )

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("Road Damage Detection")
st.caption("AI-based road damage detection using YOLO11n")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.01,
    max_value=0.90,
    value=0.05,
    step=0.01,
)

image_tab, video_tab, camera_tab = st.tabs(
    ["Upload Image", "Upload Video", "Live Camera"]
)

# --------------------------------------------------
# Image detection
# --------------------------------------------------

with image_tab:
    uploaded_image = st.file_uploader(
        "Choose a road image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_array = np.array(image)

        annotated, detections = run_detection(
            model,
            image_array,
            confidence,
        )

        if annotated is not None:
            st.image(
                annotated,
                caption="Detection Result",
                use_container_width=True,
            )

        display_detections(detections)

# --------------------------------------------------
# Video detection
# --------------------------------------------------

with video_tab:

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.info(
            "Processing video frame-by-frame. "
            "The annotated video will appear below."
        )

        output_path = process_video(
            model,
            uploaded_video
        )

        if output_path:

            st.subheader("Detected Road Damage")

            st.video(output_path)

# --------------------------------------------------
# Live camera
# --------------------------------------------------

with camera_tab:
    camera_image = st.camera_input("Capture a road image")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image_array = np.array(image)

        annotated, detections = run_detection(
            model,
            image_array,
            confidence,
        )

        if annotated is not None:
            st.image(
                annotated,
                caption="Camera Detection",
                use_container_width=True,
            )

        display_detections(detections)
