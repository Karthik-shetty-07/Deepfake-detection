# 🛡️ Hybrid Deepfake Detection System

An enterprise-ready, high-accuracy deepfake detection pipeline utilizing a hybrid approach that combines frame-level Spatial CNNs, Frequency Domain (FFT/DCT) Analysis, and Temporal Consistency Modeling (Transformers).

Optimized with a **lazy-loading model architecture** and **asynchronous background execution** to run efficiently within memory-constrained environments (e.g., Render Free Tier).

---

## 🌟 Key Features

- 🧠 **Multi-Layer Hybrid ML Pipeline**:
  - **Spatial CNN (XceptionNet / EfficientNet-B4)**: Detects frame-level texture blending boundaries, color discrepancies, and facial manipulation anomalies.
  - **Frequency Domain Analyzer (FFT/DCT)**: Exposes high-frequency grid patterns and GAN fingerprints invisible to the human eye, alongside color-channel inconsistencies.
  - **Temporal Consistency (Transformer)**: Inspects sequence dynamics, eye blinking frequencies, and frame-to-frame flow anomalies.
- ⚡ **Production-Ready Optimization**:
  - **Lazy Loading**: PyTorch models and heavy libraries (`torch`, `timm`, `facenet-pytorch`) are initialized on-demand only when a video is processed, reducing server startup memory usage to a minimum.
  - **Non-blocking Event Loop**: Synchronous ML inference tasks run inside a worker thread pool, keeping the async FastAPI event loop open to handle real-time status polling requests.
- 🎨 **Modern User Interface**: A premium, dark-themed responsive frontend utilizing glassmorphic components, intersection animations, real-time status-tracking progress rings, and detailed analytical breakdown bars.
- 🔒 **Privacy-Preserving**: Direct streaming uploads with prompt background cleaning. Uploaded videos are permanently deleted immediately after processing.

---

## 🛠️ Technology Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **ML / Deep Learning**: [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models)
- **Computer Vision**: [OpenCV Headless](https://opencv.org/)
- **Face Detection**: [MTCNN](https://github.com/timesler/facenet-pytorch) (high precision) / [MediaPipe](https://google.github.io/mediapipe/) (optimized fallback)
- **Web App**: Vanilla HTML5, CSS3 Custom Properties (Variables), Modern ES6 JavaScript

---

## 📂 Project Structure

```
├── backend/
│   ├── main.py              # Main FastAPI Application & Routing
│   ├── config.py            # System Configuration & Threshold limits
│   ├── video_processor.py   # Frame extraction, face alignment & cropping
│   ├── requirements.txt     # Python production dependencies
│   ├── train.py             # Custom model fine-tuning CLI script
│   ├── weights/             # Directory for local model checkpoints (.pth)
│   └── models/
│       ├── __init__.py      # Lazy load routing hooks (using __getattr__)
│       ├── cnn_detector.py      # Spatial classification CNN backbones
│       ├── frequency_analyzer.py # FFT/DCT spectral profile & edge analysis
│       ├── temporal_model.py    # Transformer-based consistency evaluator
│       └── ensemble.py          # Adaptive score fusion & calibration
├── frontend/
│   ├── index.html           # Landing page UI
│   ├── styles.css           # Modern Glassmorphic CSS Theme
│   ├── app.js               # API communications, state management & polling
│   └── vercel.json          # Vercel static hosting routing setup
├── .gitattributes           # Git Large File Storage (LFS) file filters
├── .gitignore               # Ignored system files and local uploads
├── render.yaml              # Production Render deployment blueprint
└── README.md
```

---

## 🚀 Getting Started

### 📋 Prerequisites

- **Python**: Version `3.10` or higher
- **Git LFS**: Installed (to pull model checkpoints)

---

### 📥 1. Clone & Set Up Git LFS

Ensure you have Git Large File Storage installed before cloning the repository to successfully download the pretrained model weights:

```bash
# Verify or install Git LFS
git lfs install

# Clone the repository
git clone https://github.com/Karthik-shetty-07/Deepfake-detection.git
cd Deepfake-detection
```

---

### ⚙️ 2. Backend Installation & Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On Windows (CMD):
   .venv\Scripts\activate.bat
   # On Linux/macOS:
   source .venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the API locally:
   ```bash
   python main.py
   ```
   The backend API will start on **`http://localhost:8000`**.

---

### 🌐 3. Frontend Setup

The frontend is a static web application and can be opened directly or served using any static web server.

To serve it using Python's built-in HTTP server:
```bash
cd ../frontend
python -m http.server 5500
```
Open your browser and navigate to **`http://localhost:5500`**.

---

## 📡 API Reference

### 1. Health Status
Check if the API is running and monitor model initialization states.

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "version": "2.0.0",
    "models_loaded": false,
    "gpu_available": false
  }
  ```

### 2. Analyze Video
Upload a video file for processing. It returns a background task ID.

- **URL**: `/api/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (Video format: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, max `50MB`)
- **Response**:
  ```json
  {
    "task_id": "335d0e90-21e2-4a89-9d32-cd828e4b6d75",
    "status": "pending",
    "message": "Video uploaded successfully. Processing started."
  }
  ```

### 3. Track Task Status
Poll the processing state and current progress percentage of the analysis task.

- **URL**: `/api/status/{task_id}`
- **Method**: `GET`
- **Response (Processing)**:
  ```json
  {
    "task_id": "335d0e90-21e2-4a89-9d32-cd828e4b6d75",
    "status": "processing",
    "progress": 0.3,
    "message": "Analyzing 22 faces with CNN...",
    "result": null,
    "created_at": "2026-06-30T18:02:39.123456",
    "completed_at": null
  }
  ```

### 4. Fetch Analysis Result
Retrieve detailed classification metrics after task completion.

- **URL**: `/api/result/{task_id}`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "task_id": "335d0e90-21e2-4a89-9d32-cd828e4b6d75",
    "status": "completed",
    "result": {
      "final_score": 74.8,
      "classification": "most_probably_deepfake",
      "classification_label": "🔶 Most Probably Deepfake - Strong manipulation indicators found",
      "confidence": 55.1,
      "component_scores": {
        "cnn": 49.9,
        "frequency": 100.0,
        "temporal": 63.7
      },
      "processing_time": 63.74,
      "frames_analyzed": 22,
      "video_info": {
        "duration": 61.0,
        "frame_count": 732,
        "resolution": "768x432"
      }
    }
  }
  ```

---

## 📊 Detection Classifications

The final ensemble score determines the video's authenticity rating:

| Score Range | Badge | Classification | Description |
|---|---|---|---|
| **< 45%** | `✅` | **Real** | No anomalous features detected. Authentic video. |
| **45% - 72%** | `⚠️` | **Low Confidence Deepfake** | Minimal frequency irregularities detected. Use caution. |
| **72% - 88%** | `🔶` | **Most Probably Deepfake** | Strong signs of facial manipulation. |
| **> 88%** | `🔴` | **Definitely a Deepfake** | High confidence anomalies across all components. |

---

## ☁️ Production Deployment

### Backend (Render Blueprint)
This repository contains a pre-configured `render.yaml` blueprint. The configuration has been explicitly tuned to prevent OOM errors:
- Starts Gunicorn **without** `--preload` to enable lazy model initialization across worker processes.
- Memory restrictions are controlled by managing the video upload limits via `MAX_FILE_SIZE_MB`.

To deploy, connect your repository to Render using the **Blueprint** option.

### Frontend (Vercel)
The frontend is optimized for static hosting providers like Vercel. Standard cache controls, security headers, and rewrite configurations are defined in `frontend/vercel.json`.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
