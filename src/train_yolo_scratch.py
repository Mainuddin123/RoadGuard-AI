
from pathlib import Path
from ultralytics import YOLO
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset/yolo_dataset/data.yaml"

OUTPUT_DIR = ROOT / "results" / "yolo_training"

print("\nROADGUARD AI - MODEL 1")
print("YOLOv8n - Random Initialization")

# YAML architecture: no pretrained weights
model = YOLO("yolov8n.yaml")

start_time = time.time()

model.train(
    data=str(DATA_YAML),
    epochs=3,
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    fraction=1.0,
    project=str(OUTPUT_DIR),
    name="model1_scratch",
    exist_ok=True,
    pretrained=False,
    patience=5,
    seed=42,
    plots=True,
    save=True,
    verbose=True
)

elapsed = time.time() - start_time

print("\n================================")
print("MODEL 1 TRAINING COMPLETED")
print("================================")

print("Training time:", round(elapsed / 60, 2), "minutes")

print(
    "Best checkpoint:",
    OUTPUT_DIR / "model1_scratch" / "weights" / "best.pt"
)
