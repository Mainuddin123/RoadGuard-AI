
from pathlib import Path
import cv2
import time
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "results" / "yolo_training" / "model2_pretrained" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "results" / "video_demo"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load trained model
print("\nROADGUARD AI - VIDEO DETECTION")

model = YOLO(str(MODEL_PATH))

# Get input video path
video_path = input("\nEnter the video file path: ").strip().strip('"')

input_path = Path(video_path)

if not input_path.exists():
    print("ERROR: Video file not found.")
    raise SystemExit

# Create a unique output filename
video_name = input_path.stem
OUTPUT_VIDEO = OUTPUT_DIR / f"{video_name}_output.mp4"

# Open video
cap = cv2.VideoCapture(str(input_path))

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    print("ERROR: Cannot create output video.")
    cap.release()
    raise SystemExit

print("\nProcessing video...")
print(f"Video dimensions: {width} x {height}")
print(f"Total frames: {total_frames}")

frame_count = 0
total_detections = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Run object detection
        result = model.predict(
            source=frame,
            conf=0.10,
            imgsz=416,
            device="cpu",
            verbose=False
        )[0]

        # Count detections
        detections = len(result.boxes)
        total_detections += detections

        # Draw bounding boxes and labels
        annotated_frame = result.plot()

        frame_count += 1

        # Calculate live FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            annotated_frame,
            f"FPS: {current_fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Save processed frame
        writer.write(annotated_frame)

        # Display video
        cv2.imshow("RoadGuard AI - Video Detection", annotated_frame)

        if frame_count % 30 == 0:
            print(
                f"Processed {frame_count}/{total_frames} frames | "
                f"Detections: {total_detections}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped by user.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

end_time = time.time()
processing_time = end_time - start_time
average_fps = frame_count / processing_time if processing_time > 0 else 0

print("\nVIDEO DETECTION COMPLETED")
print(f"Frames processed: {frame_count}")
print(f"Total detections: {total_detections}")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Average processing FPS: {average_fps:.2f}")
print(f"Saved video: {OUTPUT_VIDEO}")
