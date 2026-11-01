
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset" / "yolo_dataset" / "data.yaml"

MODEL_PATH = (
    ROOT / "results" / "yolo_training"
    / "model2_finetune_5epochs"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "results" / "yolo_test_evaluation"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

print("Loading fine-tuned model...")
model = YOLO(str(MODEL_PATH))

print("\nEvaluating on held-out TEST dataset...")

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    plots=True,
    project=str(OUTPUT_DIR),
    name="finetuned",
    exist_ok=True
)

results = {
    "Model": "YOLOv8n Fine-tuned",
    "Precision": metrics.box.mp,
    "Recall": metrics.box.mr,
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map
}

print("\n========== TEST RESULTS ==========")

for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

pd.DataFrame([results]).to_csv(
    OUTPUT_DIR / "finetuned_test_metrics.csv",
    index=False
)

print("\nTest evaluation completed.")
