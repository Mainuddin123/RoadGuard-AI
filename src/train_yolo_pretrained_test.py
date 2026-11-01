
from pathlib import Path
from ultralytics import YOLO
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset/yolo_dataset/data.yaml"

OUTPUT_DIR = ROOT / "results/yolo_benchmark"

print("ROADGUARD AI - YOLO PRETRAINED BENCHMARK")

model = YOLO("yolov8n.pt")

start_time = time.time()

results = model.train(
    data=str(DATA_YAML),
    epochs=1,
    imgsz=640,
    batch=1,
    device="cpu",
    workers=0,
    fraction=0.02,
    project=str(OUTPUT_DIR),
    name="pretrained_test",
    exist_ok=True,
    plots=True,
    verbose=True
)

elapsed = time.time() - start_time

print("\nBENCHMARK COMPLETED")
print("Training time:", round(elapsed / 60, 2), "minutes")
print("Output directory:", OUTPUT_DIR)
