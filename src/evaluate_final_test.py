
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "results/yolo_training/model2_final/weights/best.pt"
)

DATA_YAML = ROOT / "dataset/yolo_dataset/data.yaml"

print("ROADGUARD AI - FINAL TEST EVALUATION")

model = YOLO(str(MODEL_PATH))

results = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    plots=True,
    project=str(ROOT / "results/yolo_test_evaluation"),
    name="final_model",
    exist_ok=True
)

print("\nFINAL HELD-OUT TEST RESULTS")
print("Precision:", round(results.box.mp, 4))
print("Recall:", round(results.box.mr, 4))
print("mAP50:", round(results.box.map50, 4))
print("mAP50-95:", round(results.box.map, 4))
