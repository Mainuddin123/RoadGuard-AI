
from pathlib import Path
from ultralytics import YOLO
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset" / "yolo_dataset" / "data.yaml"

OUTPUT_DIR = ROOT / "results" / "yolo_training"

# Load the existing trained checkpoint
CHECKPOINT = (
    OUTPUT_DIR
    / "model2_pretrained"
    / "weights"
    / "best.pt"
)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

print("\n================================")
print("ROADGUARD AI - MODEL 2")
print("YOLOv8n - Additional Fine-tuning")
print("================================")

print("Loading existing trained checkpoint...")

model = YOLO(str(CHECKPOINT))

start_time = time.time()

model.train(
    data=str(DATA_YAML),

    # Additional training
    epochs=5,

    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    fraction=1.0,

    # Separate output directory
    project=str(OUTPUT_DIR),
    name="model2_finetune_5epochs",
    exist_ok=False,

    pretrained=True,
    patience=5,
    seed=42,

    plots=True,
    save=True,
    verbose=True
)

elapsed = time.time() - start_time

print("\n================================")
print("ADDITIONAL TRAINING COMPLETED")
print("================================")

print("Training time:", round(elapsed / 60, 2), "minutes")

print(
    "New best checkpoint:",
    OUTPUT_DIR
    / "model2_finetune_5epochs"
    / "weights"
    / "best.pt"
)
