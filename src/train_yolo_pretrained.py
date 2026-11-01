
from pathlib import Path
from ultralytics import YOLO
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset/yolo_dataset/data.yaml"

OUTPUT_DIR = ROOT / "results" / "yolo_training"

print("\nROADGUARD AI - MODEL 2")
print("YOLOv8n - Pretrained")

model = YOLO("yolov8n.pt")

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
    name="model2_pretrained",
    exist_ok=True,
    pretrained=True,
    patience=5,
    seed=42,
    plots=True,
    save=True,
    verbose=True
)

elapsed = time.time() - start_time

print("\n================================")
print("MODEL 2 TRAINING COMPLETED")
print("================================")

print("Training time:", round(elapsed / 60, 2), "minutes")

print(
    "Best checkpoint:",
    OUTPUT_DIR / "model2_pretrained" / "weights" / "best.pt"
)
