# AI-Based Real-Time Road Damage Detection

An AI-based computer vision project that uses **YOLO11n** to detect and localize road damage from images, videos, and live camera input.

The project covers the complete pipeline from dataset preprocessing and annotation conversion to model training, evaluation, and deployment-ready application development.

---

## Overview

Road damage inspection is often performed manually and can be difficult to scale. This project explores an automated object-detection approach for identifying road-damage instances from visual data.

### Project Pipeline

```text
RDD2022 Dataset
      ↓
Pascal VOC XML
      ↓
YOLO Annotation Conversion
      ↓
Dataset Validation
      ↓
Train / Validation Split
      ↓
YOLO11n Training
      ↓
Model Evaluation
      ↓
Gradio Application
      ↓
Image / Video / Live Camera Detection
```

---

## Key Features

* Road-damage object detection using YOLO11n
* Image detection with bounding boxes and confidence scores
* Video detection with annotated output
* Live-camera detection
* Adjustable confidence threshold
* Custom Pascal VOC → YOLO annotation conversion
* Dataset and annotation validation
* Class-distribution analysis
* Baseline and class-balancing experiments
* Interactive Gradio interface

---

## Dataset

The project uses the **RDD2022 India subset**.

* **1,530 images**
* **4,524 annotated objects**
* **9 road-damage classes**
* **80/20 train-validation split**
* **1,224 training images**
* **306 validation images**

The dataset was found to have significant class imbalance, with D40 being the dominant class and several classes containing fewer than 10 objects.

Raw dataset files are excluded from this repository because of their size.

---

## Model

**YOLO11n (YOLO11 Nano)** was selected as the primary model because the project targets real-time detection and requires a lightweight model suitable for practical deployment.

### Training Configuration

| Parameter  | Value           |
| ---------- | --------------- |
| Model      | YOLO11n         |
| Epochs     | 50              |
| Image Size | 640 × 640       |
| Batch Size | 16              |
| GPU        | NVIDIA Tesla T4 |
| Optimizer  | AdamW           |
| Pretrained | Yes             |

Training was performed using **Google Colab** with GPU acceleration.

---

## Results

Two experiments were conducted using the same validation set.

### Baseline

| Metric    | Result |
| --------- | -----: |
| Precision |  0.655 |
| Recall    |  0.174 |
| mAP@50    |  0.208 |
| mAP@50–95 |  0.091 |

### Controlled Oversampling

A second experiment increased the exposure of minority-class training examples while keeping the validation set unchanged.

| Metric    | Result |
| --------- | -----: |
| Precision |  0.508 |
| Recall    |  0.407 |
| mAP@50    |  0.369 |
| mAP@50–95 |  0.146 |

The second experiment improved recall and mAP compared with the baseline, while precision decreased.

Because several classes have extremely small validation samples, class-specific results must be interpreted carefully.

---

## Application

The trained model was integrated into a **Gradio-based application** supporting:

**Image Detection**

Upload an image and receive detected damage classes, bounding boxes, and confidence scores.

**Video Detection**

Upload a video and process it frame-by-frame to generate an annotated, browser-compatible video.

**Live Camera Detection**

Use a camera feed for road-damage detection.

### Application Workflow

```text
Image / Video / Camera
          ↓
       Gradio
          ↓
       YOLO11n
          ↓
  Road Damage Detection
          ↓
 Annotated Result
```

OpenCV is used for video processing, while FFmpeg/imageio-ffmpeg is used to produce browser-compatible video output.

---

## Technologies

* Python
* YOLO11n
* Ultralytics
* PyTorch
* OpenCV
* NumPy
* Pillow
* Gradio
* FFmpeg
* Jupyter Notebook
* Google Colab

---

## Project Structure

```text
AI-Road-Damage-Detection/
│
├── app/                 # Gradio application
├── backend/             # FastAPI backend
├── frontend/            # Web frontend
├── dataset/             # Dataset configuration
├── DOCUMENTATION/       # Research documentation
├── models/              # Local model weights
├── notebooks/           # Analysis notebooks
├── results/             # Sample results
├── src/                 # Preprocessing & validation scripts
├── training/            # Training notebook
│
├── requirements.txt
├── README.md
├── README_DEPLOYMENT.md
└── .gitignore
```

Model weights and raw dataset files are not included in the public repository.

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/SafiaMirbahar/AI-Road-Damage-Detection.git
cd AI-Road-Damage-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained model at:

```text
models/best.pt
```

Run the application:

```bash
python app/gradio_app.py
```

The application will normally be available at:

```text
http://127.0.0.1:7860
```

---

## Documentation

A detailed research/case-study document is included in:

```text
DOCUMENTATION/
```

It contains the complete methodology, dataset analysis, preprocessing, experiments, evaluation, confusion-matrix analysis, limitations, and research discussion.

For deployment instructions:

**[Deployment Guide](README_DEPLOYMENT.md)**

---

## Limitations

The main limitation is the severe class imbalance in the dataset.

Some validation classes contain only one to four examples, making their class-specific metrics statistically unstable. The confusion matrix also shows missed detections and confusion between visually similar damage categories.

Therefore, the project should be considered a **working detection prototype**, not a universally accurate road-damage detector.

---

## Future Work

* Expand the dataset, especially for rare classes
* Improve minority-class representation
* Investigate additional balancing strategies
* Experiment with larger YOLO models
* Improve small-object detection
* Test on additional geographic regions
* Develop a stronger real-time streaming pipeline
* Integrate GPS-based road-damage location and warning

---

## Author

**Safia Mirbahar**

Computer Science Student | Aspiring AI Engineer / Full-Stack AI Developer

[GitHub](https://github.com/SafiaMirbahar)
