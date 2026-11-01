# 🛣️ RoadGuard AI — Smart Road Damage & Hazard Detection

An AI-powered road damage detection system that identifies potholes, cracks, and manholes using YOLOv8 object detection.

## 📌 Project Overview

RoadGuard AI uses computer vision and deep learning to detect road surface damage and potential hazards.

The system supports:
- Image-based road damage detection
- Recorded video detection
- Live webcam detection
- Bounding-box visualization
- Class labels and confidence scores
- FPS measurement

## 🎯 Detection Classes

| Class ID | Class |
|---|---|
| 0 | Pothole |
| 1 | Crack |
| 2 | Manhole |

## 🧰 Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- VS Code

## 📂 Dataset

**Dataset:** Road Damage Dataset: Potholes, Cracks and Manholes

**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)

### Dataset Statistics

| Description | Count |
|---|---:|
| Unique images | 1,999 |
| Annotated objects | 4,701 |
| Training images | 1,406 |
| Validation images | 292 |
| Testing images | 301 |

### Class Distribution

| Class | Objects |
|---|---:|
| Pothole | 1,253 |
| Crack | 2,498 |
| Manhole | 950 |

The dataset was cleaned, duplicate images were excluded, and a group-aware train-validation-test split was performed.

## 🧠 Models Implemented

Two YOLOv8n models were trained and compared.

### Model 1 — YOLOv8n from Scratch

- Random initialization
- 3 training epochs
- Image size: 416 × 416
- Batch size: 1
- CPU training

### Model 2 — Pretrained YOLOv8n

- Initialized from pretrained YOLOv8n detector weights
- Fine-tuned on the road damage dataset
- 3 training epochs
- Image size: 416 × 416
- Batch size: 1
- CPU training

## 📊 Model Comparison

Held-out test set results:

| Metric | Scratch | Pretrained |
|---|---:|---:|
| Precision | 0.0018 | 0.2646 |
| Recall | 0.1859 | 0.2911 |
| mAP@50 | 0.0023 | 0.1790 |
| mAP@50–95 | 0.0004 | 0.0782 |
| Inference time/image | 41.3 ms | 59.9 ms |
| Training time | 34.3 min | 41.35 min |
| Model size | 6.2 MB | 6.5 MB |

The pretrained model performed better than the scratch model on the held-out test set. However, its overall detection performance remains limited.

## 🎥 Demonstrations

### Recorded Video Detection

The system was tested on three recorded videos.

| Video | Frames | Detections | FPS |
|---|---:|---:|---:|
| demo1.mp4 | 142 | 4 | 10.47 |
| demo2.mp4 | 39 | 0 | 9.67 |
| demo3.mp4 | 21 | 0 | 6.12 |

### Live Webcam Detection

Implemented features:
- Fullscreen webcam display
- Real-time object detection
- Bounding-box visualization
- FPS display
- Processed video recording

Observed webcam performance: approximately 11–13 FPS on CPU.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RoadGuard-AI.git
cd RoadGuard-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

### Recorded Video Detection

```powershell
python src/detect_video.py
```

Enter the path of your video when prompted.

### Live Webcam Detection

```powershell
python src/detect_webcam.py
```

Press `Q` to stop webcam detection.

## 📁 Project Structure

```text
RoadGuard-AI/
│
├── dataset/
│   ├── raw/
│   └── yolo_dataset/
│
├── models/
│
├── src/
│   ├── train_yolo_scratch.py
│   ├── train_yolo_pretrained.py
│   ├── evaluate_yolo_test.py
│   ├── visualize_yolo_test.py
│   ├── detect_video.py
│   └── detect_webcam.py
│
├── results/
│   ├── yolo_training/
│   ├── yolo_test_evaluation/
│   ├── yolo_predictions/
│   ├── video_demo/
│   ├── webcam_demo/
│   └── FINAL_EVALUATION.md
│
├── requirements.txt
└── README.md
```

## ⚠️ Limitations

- Current test-set detection performance is limited.
- Crack detection requires improvement.
- Video detections were inconsistent across the three tested videos.
- Webcam testing did not confirm successful road-hazard detection.
- CPU inference is slower than GPU inference.
- Model 2 uses pretrained full-detector weights, not only a pretrained backbone with a newly initialized detection head.

## 🚀 Future Improvements

- Increase training epochs.
- Improve annotation quality and dataset diversity.
- Add more road images from different geographical regions.
- Improve crack detection performance.
- Evaluate on direct road-camera footage.
- Optimize inference speed.

## 👨‍💻 Author

**Khaja Mainuddin**

AI & Data Science Graduate

---

*RoadGuard AI — Smart Road Damage & Hazard Detection*
