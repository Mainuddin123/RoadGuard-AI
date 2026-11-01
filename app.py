
from pathlib import Path
from io import BytesIO
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "results" / "yolo_training" / "model2_final" / "weights" / "best.pt"

CLASS_NAMES = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}


st.set_page_config(
    page_title="RoadGuard AI | Smart Road Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL DARK UI + ANIMATED RAINBOW GRADIENT
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {
    color-scheme: dark;
}

/* GLOBAL BACKGROUND */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #080d18 !important;
    color: #f1f5ff !important;
}

/* REMOVE WHITE TOP SPACE */

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: #080d18 !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #101a2b !important;
    border-right: 1px solid #2b3c55;
}

[data-testid="stSidebar"] * {
    color: #e5edff;
}

/* ANIMATED RAINBOW TITLE */

.gradient-title {
    font-size: clamp(32px, 4vw, 48px);
    font-weight: 800;
    letter-spacing: 1px;

    background: linear-gradient(
        90deg,
        #00eaff,
        #438bff,
        #a855f7,
        #ff4fa3,
        #ff9d43,
        #faff5c,
        #00eaff
    );

    background-size: 300% 300%;

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: rainbowMove 6s ease infinite;
}

@keyframes rainbowMove {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }

}

/* HERO SECTION */

.hero {
    padding: 35px;

    border: 1px solid #344966;
    border-radius: 20px;

    background: linear-gradient(
        120deg,
        #11233e,
        #172b48,
        #10383b
    );

    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);

    margin-bottom: 30px;
}

.subtitle {
    color: #d5e2f7 !important;
    font-size: 16px;
    margin-top: 8px;
}

/* SECTION HEADINGS */

.section-title {
    color: #ffffff !important;
    font-size: 23px;
    font-weight: 800;
    margin-top: 18px;
}

h1, h2, h3, h4, p, label {
    color: #edf3ff;
}

/* STATUS BADGE */

.status {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;

    background: #123c31;
    color: #63f0b1;

    font-size: 13px;
    font-weight: 700;
}

/* INFORMATION CARDS */

.info-card {
    background: #142239;
    border: 1px solid #344966;
    border-radius: 15px;
    padding: 20px;
    min-height: 110px;
}

.info-label {
    color: #c0cde0;
    font-size: 14px;
}

.info-value {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
}

/* BUTTONS */

div.stButton > button,
div.stDownloadButton > button,
.stLinkButton a {

    background: linear-gradient(
        100deg,
        #087fca,
        #6554db,
        #a044c6
    ) !important;

    color: #ffffff !important;

    border: 1px solid #657eff !important;
    border-radius: 11px !important;

    min-height: 46px;

    font-weight: 700 !important;

    transition: all 0.25s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
.stLinkButton a:hover {

    transform: translateY(-2px);

    box-shadow: 0 5px 20px rgba(90, 115, 255, 0.35);

    border-color: #ffffff !important;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #121f33 !important;
    border: 1px dashed #6683a9 !important;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stFileUploader"] * {
    color: #e5edff !important;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #142239;
    border: 1px solid #344966;
    padding: 17px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #b9c9df !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* TABLE */

[data-testid="stDataFrame"] {
    border: 1px solid #344966;
    border-radius: 10px;
}

/* FOOTER */

.footer {
    color: #b9c9df;
    text-align: center;
    font-size: 13px;

    border-top: 1px solid #344966;

    padding-top: 22px;
    margin-top: 45px;
}

/* FILE UPLOADER - HIGH VISIBILITY */

[data-testid="stFileUploader"] {
    background: #142239 !important;
    border: 1px solid #52749f !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

[data-testid="stFileUploader"] section {
    background: #1b2b43 !important;
    border: 1px dashed #6683a9 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(
        100deg,
        #087fca,
        #6554db
    ) !important;
    color: #ffffff !important;
    border: 1px solid #70aaff !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(
        100deg,
        #079fe8,
        #8055f5
    ) !important;
    color: #ffffff !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #dce8ff !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD TRAINED YOLO MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    return YOLO(str(model_path))


if not MODEL_PATH.exists():

    st.error("Trained YOLO model checkpoint not found.")

    st.code(
        "results/yolo_training/model2_pretrained/weights/best.pt"
    )

    st.info(
        "Make sure the trained checkpoint is included "
        "in your project repository."
    )

    st.stop()


try:

    model = load_model(MODEL_PATH)

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# 4. HEADER
# ============================================================

st.html("""
<div class="hero">
    <div class="gradient-title">
        🛣️ ROADGUARD AI
    </div>

    <div class="subtitle">
        Smart Road Damage &amp; Hazard Detection using Deep Learning
    </div>

    <br>

    <span class="status">
        ● YOLOv8 MODEL ACTIVE
    </span>
</div>
""")



# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🛣️ RoadGuard AI")

    st.caption("Smart Road Intelligence Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Image Detection",
            "Video Detection",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### ⚙️ Model Configuration")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.05,
        step=0.05
    )

    image_size = st.select_slider(
        "Image Size",
        options=[320, 416, 512, 640],
        value=416
    )

    st.divider()

    st.markdown("### Model Details")

    st.caption("Architecture: YOLOv8n")
    st.caption("Initialization: Pretrained detector")
    st.caption("Device: CPU")
    st.caption("Detection Classes: 3")


# ============================================================
# 6. DETECTION SUMMARY FUNCTION
# ============================================================

def show_detection_summary(result, elapsed):

    boxes = result.boxes

    total = len(boxes)

    class_counts = {
        "Pothole": 0,
        "Crack": 0,
        "Manhole": 0
    }

    records = []

    for box in boxes:

        class_id = int(box.cls[0].item())

        confidence_score = float(
            box.conf[0].item()
        )

        label = CLASS_NAMES.get(
            class_id,
            f"Class {class_id}"
        )

        class_counts[label] = (
            class_counts.get(label, 0) + 1
        )

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        records.append({

            "Detected Object": label,

            "Confidence": round(
                confidence_score, 4
            ),

            "X1": round(x1, 1),
            "Y1": round(y1, 1),
            "X2": round(x2, 1),
            "Y2": round(y2, 1)

        })

    st.markdown("### Detection Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Detections", total)

    col2.metric(
        "Potholes",
        class_counts["Pothole"]
    )

    col3.metric(
        "Cracks",
        class_counts["Crack"]
    )

    col4.metric(
        "Manholes",
        class_counts["Manhole"]
    )

    st.caption(
        f"Inference Time: {elapsed * 1000:.2f} ms"
    )

    if records:

        st.markdown("### Detection Details")

        dataframe = pd.DataFrame(records)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No road hazards detected. "
            "Try another image or adjust the confidence threshold."
        )


# ============================================================
# 7. IMAGE DETECTION PAGE
# ============================================================

if page == "Image Detection":

    st.markdown(
        '<div class="section-title">📷 Road Image Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a road image to identify potholes, cracks and manholes."
    )

    uploaded_image = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        try:

            image = Image.open(
                uploaded_image
            ).convert("RGB")

        except Exception:

            st.error("Unable to read the uploaded image.")
            st.stop()

        st.markdown("### Original Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🔍 DETECT ROAD HAZARDS",
            use_container_width=True
        ):

            with st.spinner(
                "AI is analyzing your road image..."
            ):

                image_array = cv2.cvtColor(
                    np.array(image),
                    cv2.COLOR_RGB2BGR)

                start = time.perf_counter()

                result = model.predict(
                    source=image_array,
                    conf=confidence,
                    imgsz=image_size,
                    device="cpu",
                    verbose=False
                )[0]

                elapsed = (
                    time.perf_counter() - start
                )

                annotated_bgr = result.plot()

                annotated_rgb = cv2.cvtColor(
                    annotated_bgr,
                    cv2.COLOR_BGR2RGB
                )

            st.success(
                "Road image analysis completed!"
            )

            st.markdown("### AI Detection Result")

            st.image(
                annotated_rgb,
                use_container_width=True
            )

            show_detection_summary(
                result,
                elapsed
            )

            image_buffer = BytesIO()

            Image.fromarray(
                annotated_rgb
            ).save(
                image_buffer,
                format="PNG"
            )

            st.download_button(
                "⬇️ DOWNLOAD ANNOTATED IMAGE",
                data=image_buffer.getvalue(),
                file_name="roadguard_detection.png",
                mime="image/png",
                use_container_width=True
            )


# ============================================================
# 8. VIDEO DETECTION PAGE
# ============================================================

elif page == "Video Detection":

    st.markdown(
        '<div class="section-title">🎥 Road Video Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a recorded road video to detect road hazards frame by frame."
    )

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=["mp4", "mov", "avi"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "▶️ START VIDEO DETECTION",
            use_container_width=True
        ):

            input_path = None
            output_path = None

            try:

                suffix = (
                    Path(uploaded_video.name).suffix.lower()
                    or ".mp4"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as input_file:

                    input_file.write(
                        uploaded_video.getvalue()
                    )

                    input_path = input_file.name

                cap = cv2.VideoCapture(
                    input_path
                )

                if not cap.isOpened():

                    st.error(
                        "Unable to open uploaded video."
                    )

                    st.stop()

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                video_fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if video_fps <= 0:
                    video_fps = 20.0

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as output_file:

                    output_path = output_file.name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps,
                    (width, height)
                )

                if not writer.isOpened():

                    cap.release()

                    st.error(
                        "Unable to create processed video."
                    )

                    st.stop()

                progress = st.progress(0)

                status = st.empty()

                preview = st.empty()

                frame_count = 0

                total_detections = 0

                start = time.perf_counter()

                try:

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        result = model.predict(
                            source=frame,
                            conf=confidence,
                            imgsz=image_size,
                            device="cpu",
                            verbose=False
                        )[0]

                        total_detections += len(
                            result.boxes
                        )

                        annotated_frame = result.plot()

                        writer.write(
                            annotated_frame
                        )

                        frame_count += 1

                        if (
                            frame_count % 5 == 0
                            or frame_count == total_frames
                        ):

                            preview_frame = cv2.cvtColor(
                                annotated_frame,
                                cv2.COLOR_BGR2RGB
                            )

                            preview.image(
                                preview_frame,
                                channels="RGB",
                                use_container_width=True
                            )

                        if total_frames > 0:

                            progress.progress(
                                min(
                                    frame_count / total_frames,
                                    1.0
                                )
                            )

                        status.write(
                            f"Processed {frame_count} / "
                            f"{total_frames} frames"
                        )

                finally:

                    cap.release()
                    writer.release()

                elapsed = (
                    time.perf_counter() - start
                )

                average_fps = (
                    frame_count / elapsed
                    if elapsed > 0 else 0
                )

                st.success(
                    "Video detection completed!"
                )

                st.markdown(
                    "### Video Detection Summary"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    frame_count
                )

                col2.metric(
                    "Total Detections",
                    total_detections
                )

                col3.metric(
                    "Processing FPS",
                    f"{average_fps:.2f}"
                )

                st.caption(
                    f"Processing Time: {elapsed:.2f} seconds"
                )

                with open(
                    output_path,
                    "rb"
                ) as output_file:

                    video_bytes = output_file.read()

                st.markdown(
                    "### Processed Video"
                )

                st.video(video_bytes)

                st.download_button(
                    "⬇️ DOWNLOAD PROCESSED VIDEO",
                    data=video_bytes,
                    file_name="roadguard_processed.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )

            finally:

                for temporary_path in [
                    input_path,
                    output_path
                ]:

                    if temporary_path:

                        try:

                            Path(
                                temporary_path
                            ).unlink(
                                missing_ok=True
                            )

                        except OSError:
                            pass


# ============================================================
# 9. ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">📚 About RoadGuard AI</div>',
        unsafe_allow_html=True
    )

    st.write("""
    RoadGuard AI is an object detection application developed
    to identify road surface damage and hazards using deep learning.

    It uses a fine-tuned YOLOv8n model to identify three road
    damage categories.
    """)

    st.markdown("### Detection Categories")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="info-card">
        <div class="info-value">🕳️ Pothole</div>
        <div class="info-label">
            Road surface cavities
        </div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="info-card">
        <div class="info-value">〰️ Crack</div>
        <div class="info-label">
            Road surface fractures
        </div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="info-card">
        <div class="info-value">⭕ Manhole</div>
        <div class="info-label">
            Road manhole structures
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Information")

    st.markdown("""
    - Architecture: YOLOv8n
    - Initialization: Pretrained detector weights
    - Fine-tuning: Road damage dataset
    - Input image size: Configurable
    - Inference device: CPU
    """)

    st.markdown("### Dataset Information")

    st.write(
        "Road Damage Dataset: Potholes, Cracks and Manholes"
    )

    st.link_button(
        "VIEW DATASET ON KAGGLE",
        "https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
    )

    st.markdown("### Model Evaluation")

    st.info("""
    Held-out test results for the pretrained model:

    - Precision: 0.2646
    - Recall: 0.2911
    - mAP@50: 0.1790
    - mAP@50–95: 0.0782

    The model outperformed the scratch model in testing,
    but further improvement is needed before real-world deployment.
    """)

    st.warning(
        "This application is a research prototype and should not "
        "be used as the sole basis for road safety decisions."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.html("""
<div class="footer">
    <b>ROADGUARD AI</b>
    <br>
    Smart Road Damage &amp; Hazard Detection
    <br>
    Developed by Khaja Mainuddin
    <br><br>
    Powered by YOLOv8 | Python | OpenCV | Streamlit
</div>
""")

