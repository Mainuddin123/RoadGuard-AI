
from pathlib import Path
import cv2
import time
import numpy as np
import ctypes
from ultralytics import YOLO

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "results"
    / "yolo_training"
    / "model2_pretrained"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "results" / "webcam_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "roadguard_webcam_output.mp4"

# ==========================================
# 2. LOAD MODEL
# ==========================================

print("\nROADGUARD AI - FULLSCREEN WEBCAM DETECTION")

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    raise SystemExit

model = YOLO(str(MODEL_PATH))

# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==========================================
# 4. VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    20,
    (width, height)
)

if not writer.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    raise SystemExit

# ==========================================
# 5. FULLSCREEN WINDOW
# ==========================================

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

window_name = "RoadGuard AI - Live Detection"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("\nWebcam started successfully.")
print("Press Q to stop detection.")

# ==========================================
# 6. DETECTION LOOP
# ==========================================

frame_count = 0
total_detections = 0

start_time = time.time()

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        # Run YOLO detection
        result = model.predict(
            source=frame,
            conf=0.05,
            imgsz=416,
            device="cpu",
            verbose=False
        )[0]

        detections = len(result.boxes)
        total_detections += detections
        frame_count += 1

        # Draw bounding boxes and labels
        annotated_frame = result.plot()

        # Calculate FPS
        elapsed = time.time() - start_time

        fps = (
            frame_count / elapsed
            if elapsed > 0 else 0
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Save original-size annotated frame
        writer.write(annotated_frame)

        # ==================================
        # 7. FIT FRAME TO FULLSCREEN
        # ==================================

        frame_h, frame_w = annotated_frame.shape[:2]

        scale = min(
            screen_width / frame_w,
            screen_height / frame_h
        )

        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)

        resized_frame = cv2.resize(
            annotated_frame,
            (new_w, new_h)
        )

        # Create fullscreen background
        display_frame = np.zeros(
            (screen_height, screen_width, 3),
            dtype=np.uint8
        )

        x_offset = (screen_width - new_w) // 2
        y_offset = (screen_height - new_h) // 2

        display_frame[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = resized_frame

        cv2.imshow(
            window_name,
            display_frame
        )

        # Stop using Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopping webcam...")
            break

finally:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# ==========================================
# 8. FINAL RESULTS
# ==========================================

processing_time = time.time() - start_time

average_fps = (
    frame_count / processing_time
    if processing_time > 0 else 0
)

print("\nWEBCAM DETECTION COMPLETED")

print(f"Frames processed: {frame_count}")
print(f"Total detections: {total_detections}")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Average FPS: {average_fps:.2f}")

print(f"\nSaved video: {OUTPUT_VIDEO}")
