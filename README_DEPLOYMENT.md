# Road Damage Detection Deployment Guide

This project is split into:

- Frontend: Vercel-hosted static site
- Backend: Render or Railway-hosted Python API

## 1) Backend on Render or Railway

### Files
- `backend/main.py`
- `backend/requirements.txt`

### Deployment steps

#### Render
1. Push your project to GitHub.
2. Create a new Web Service in Render.
3. Set the root directory to `backend`.
4. Use the following build command:
   ```bash
   pip install -r requirements.txt
   ```
5. Use the start command:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```
6. Add the model file `models/best.pt` to your deployment or upload it via the project repository.

#### Railway
1. Create a new project in Railway.
2. Link your GitHub repo.
3. Set the service root to `backend`.
4. Use the default Python environment.
5. Add the model file to the project if needed.
6. Expose port `8000` or use the Railway-provided port.

### Health check
Visit:
```text
https://your-backend-url/health
```

You should receive a JSON response like:
```json
{"status":"ok","model":"best.pt"}
```

## 2) Frontend on Vercel

### Files
- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

### Deployment steps
1. Push the project to GitHub.
2. Import the repo in Vercel.
3. Set the root directory to `frontend`.
4. Vercel will serve the static files automatically.
5. In the browser app, set the backend URL:
   ```javascript
   window.__BACKEND_URL__ = 'https://your-backend-url';
   ```
   or update the constant in `frontend/app.js`.

## 3) Important Notes

- The YOLO model is stored in the `models` folder and is required by the backend.
- Webcam access is not supported in the basic static frontend, but the image-based detection flow works well in production.
- For real live video or webcam detection, the backend would need a dedicated streaming API or a proper Gradio deployment environment.

## 4) Local Testing

From the project root, run:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open the frontend files locally with a static web server or VS Code Live Server.

## 5) Running Gradio App Locally

The project includes a Gradio-based interactive application for road damage detection with a user-friendly interface.

### Prerequisites
Ensure Gradio is installed:
```bash
pip install gradio
```

### Running the Gradio App
From the project root, run:
```bash
python app/gradio_app.py
```

The app will start on:
```
Local URL: http://127.0.0.1:7860
```

### Features
- **Image Detection**: Upload images to detect road damage with bounding boxes
- **Video Detection**: Process video files and download annotated results
- **Webcam Detection**: Real-time detection from your webcam feed
- **Confidence Threshold**: Adjust detection sensitivity with a slider

### Output
- Annotated images with bounding boxes and confidence scores
- Detection summary showing damage type and confidence levels
- Processed videos saved to `results/annotated_video.mp4`

### Model Requirements
The app requires the trained YOLO model at `models/best.pt`. Ensure this file exists before running.
