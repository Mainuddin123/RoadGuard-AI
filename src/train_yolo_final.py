
from pathlib import Path
from ultralytics import YOLO
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset" / "yolo_dataset" / "data.yaml"
OUTPUT_DIR = ROOT / "results" / "yolo_training"

print("\n====================================")
print("ROADGUARD AI - FINAL MODEL TRAINING")
print("YOLOv8n - Pretrained Fine-tuning")
print("====================================")

model = YOLO("yolov8n.pt")

start_time = time.time()

model.train(
    data=str(DATA_YAML),
    epochs=5,
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    fraction=1.0,
    project=str(OUTPUT_DIR),
    name="model2_final",
    exist_ok=True,
    pretrained=True,
    patience=5,
    seed=42,
    plots=True,
    save=True,
    verbose=True
)

elapsed = time.time() - start_time

best_model = OUTPUT_DIR / "model2_final" / "weights" / "best.pt"

print("\n====================================")
print("FINAL MODEL TRAINING COMPLETED")
print("====================================")
print(f"Training time: {elapsed / 60:.2f} minutes")
print(f"Best checkpoint: {best_model}")
print(f"Checkpoint exists: {best_model.exists()}")
