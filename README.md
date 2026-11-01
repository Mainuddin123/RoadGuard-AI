# 🚧 RoadGuard AI — Smart Road Damage & Hazard Detection

**An AI-powered road monitoring prototype using YOLOv8 to detect potholes, road cracks, and manholes from images and videos.**

<p align="center">
  <strong>Computer Vision | Deep Learning | Object Detection | Python | Streamlit</strong>
</p>

<p align="center">
  <a href="https://roadguard-ai-ieilzx9wrrjzvrk89vfrei.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open RoadGuard AI">
  </a>
</p>

---

## 🌐 Live Demo

**[🚀 Launch RoadGuard AI](https://roadguard-ai-ieilzx9wrrjzvrk89vfrei.streamlit.app/)**

**GitHub Repository:** [Mainuddin123/RoadGuard-AI](https://github.com/Mainuddin123/RoadGuard-AI)

---

## 📌 Project Overview

RoadGuard AI is a computer vision-based application developed to identify road-related hazards using a YOLOv8 object detection model.

The application analyzes uploaded road images and videos to identify three categories:

- 🕳️ Potholes
- 🛣️ Road Cracks
- 🚧 Manholes

It generates annotated outputs with bounding boxes and confidence scores, along with detection summaries and downloadable results through an interactive Streamlit application.

The project demonstrates an end-to-end deep learning workflow, including dataset preparation, model training, evaluation, application development, and cloud deployment.

## 🎯 Project Objectives

- Automate the identification of visible road hazards.
- Implement YOLOv8 for object detection.
- Support image-based and video-based analysis.
- Visualize detected objects using bounding boxes.
- Display confidence scores and detection statistics.
- Develop an interactive web application for road hazard analysis.

## ✨ Key Features

### 1. Image Detection

- Upload road images in JPG and PNG formats.
- Detect potholes, cracks, and manholes.
- Display annotated images with bounding boxes.
- Show object-level confidence scores.
- View detection counts and inference time.
- Download annotated images.

### 2. Video Detection

- Upload road videos for frame-based analysis.
- Process video frames using YOLOv8.
- Display detected road hazards.
- Generate processed video output.
- Show processing time and FPS.
- Download processed videos.

### 3. Model Configuration

- Adjustable confidence threshold.
- Configurable inference image size.
- CPU-based model inference.

### 4. Interactive Dashboard

- Professional dark-themed interface.
- Separate Image Detection and Video Detection modules.
- Model configuration panel.
- Project information page.

## 🧠 Model Architecture

The project uses **YOLOv8n (You Only Look Once)**, a lightweight object detection model initialized with pretrained weights and fine-tuned for three road hazard classes.

| Component | Details |
|---|---|
| Model | YOLOv8n |
| Task | Object Detection |
| Framework | Ultralytics |
| Initialization | Pretrained YOLOv8 detector |
| Training Epochs | 5 |
| Input Image Size | 416 × 416 |
| Training Device | CPU |
| Detection Classes | 3 |

### Detection Classes

| Class ID | Class Name |
|---|---|
| 0 | Pothole |
| 1 | Crack |
| 2 | Manhole |

## 📊 Dataset Information

The project uses the **Road Damage Dataset: Potholes, Cracks and Manholes**.

The dataset was converted into YOLO annotation format and divided into training, validation, and testing subsets.

| Dataset Split | Images | Annotated Objects |
|---|---:|---:|
| Training | 1,406 | 3,263 |
| Validation | 292 | 719 |
| Testing | 301 | 709 |
| **Total** | **1,999** | **4,691** |

### Dataset Preparation and Validation

- YOLO annotation format validation.
- Image and label consistency checks.
- Bounding-box boundary analysis.
- Training, validation, and test split verification.
- Dataset integrity checks.

## 📈 Model Evaluation

The trained model was evaluated on a separate held-out test dataset containing 301 images and 709 annotated objects.

### Overall Test Performance

| Evaluation Metric | Result |
|---|---:|
| Precision | 0.2127 |
| Recall | 0.2793 |
| mAP@50 | 0.1806 |
| mAP@50–95 | 0.0706 |
| Inference Speed | 31.5 ms/image |

### Class-wise Evaluation

| Class | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Pothole | 0.231 | 0.128 | 0.0944 |
| Crack | 0.222 | 0.0794 | 0.0757 |
| Manhole | 0.185 | 0.630 | 0.3720 |

**Performance note:** The current model has limited detection performance, particularly for potholes and cracks. These metrics are the measured results from the held-out test dataset, not estimated or claimed accuracy.

Further training, dataset improvement, and model optimization are required before practical deployment.

## 🏗️ Project Architecture

```text
RoadGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── results/
│   └── yolo_training/
│       └── model2_final/
│           └── weights/
│               └── best.pt
│
├── dataset/
│   └── yolo_dataset/
│       ├── images/
│       ├── labels/
│       └── data.yaml
│
└── src/
    └── analyze_boundary_errors.py
```

**Note:** This is a simplified representation of the main project structure. Raw dataset files and some local development artifacts are not included in the GitHub repository.

## ⚙️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | YOLOv8 |
| Computer Vision | OpenCV |
| Deep Learning Framework | PyTorch |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Model Evaluation | Precision, Recall, mAP |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 🔄 Application Workflow

```text
          Road Image / Video
                  |
                  v
           Upload Interface
                  |
                  v
          Image / Video Input
                  |
                  v
          YOLOv8 Detection
                  |
                  v
          Object Detection
                  |
          +-------+-------+
          |       |       |
          v       v       v
       Pothole  Crack  Manhole
          |       |       |
          +-------+-------+
                  |
                  v
         Bounding Box Annotation
                  |
                  v
          Detection Statistics
                  |
                  v
          Download Results
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.11
- Git
- VS Code or another Python IDE

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mainuddin123/RoadGuard-AI.git
```

### Step 2: Navigate to the Project Directory

```bash
cd RoadGuard-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser.

**Important:** The trained model checkpoint must be available at the model path configured in `app.py`.

## 🖥️ How to Use

1. Open the RoadGuard AI application.
2. Select Image Detection or Video Detection.
3. Upload a road image or video.
4. Adjust the confidence threshold and image size if required.
5. Run the detection process.
6. Review the annotated output and detection statistics.
7. Download the processed image or video.

## ⚠️ Known Limitations

The current implementation is an experimental prototype. Its limitations include:

1. **Missed detections:** Some visible road damage may not be identified.
2. **Low-confidence predictions:** Certain detections have low confidence scores.
3. **False positives:** Normal road textures may sometimes be classified as road hazards.
4. **Overlapping bounding boxes:** Multiple boxes may identify the same damaged region.
5. **Limited training:** The model was trained for only five epochs.
6. **Dataset limitations:** Dataset size and class distribution can affect generalization.
7. **Video summary inconsistency:** In some tests, displayed detection boxes did not match the total detection count in the summary.

### Future Improvements

- Increase dataset size and diversity.
- Train for additional epochs.
- Optimize model hyperparameters.
- Improve pothole and crack detection.
- Apply suitable data augmentation and class balancing.
- Investigate duplicate detections and non-maximum suppression.
- Improve video detection counting and reporting.
- Evaluate on more real-world road conditions.
- Explore advanced object detection architectures.

## 🔮 Future Scope

Potential extensions of RoadGuard AI include:

- Real-time road hazard detection.
- GPS-based hazard location tracking.
- Road condition monitoring dashboards.
- Automated road maintenance reporting.
- Integration with smart city infrastructure.

These are proposed enhancements and are not currently implemented.

## 👨‍💻 Developer

**Khaja Mainuddin**

B.Tech — Artificial Intelligence and Data Science

- **GitHub:** [Mainuddin123](https://github.com/Mainuddin123)
- **LinkedIn:** [Khaja Mainuddin](https://www.linkedin.com/in/khaja-mainuddin-sk-958436348)
- **Live Project:** [RoadGuard AI](https://roadguard-ai-ieilzx9wrrjzvrk89vfrei.streamlit.app/)

---

## 📄 Disclaimer

RoadGuard AI is an educational and experimental computer vision project developed to demonstrate object detection for road hazard identification.

Its predictions should not be treated as verified road safety assessments or used as the sole basis for road maintenance decisions. Further model development, validation, and real-world testing are necessary before practical use.

---

<p align="center">
  <strong>🚧 RoadGuard AI — Making Road Monitoring Smarter with Computer Vision</strong>
</p>
