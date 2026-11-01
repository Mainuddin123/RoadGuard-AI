
from pathlib import Path
from ultralytics import YOLO
import csv
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset/yolo_dataset/data.yaml"

MODEL_PATHS = {
    "Scratch": ROOT / "results/yolo_training/model1_scratch/weights/best.pt",
    "Pretrained": ROOT / "results/yolo_training/model2_pretrained/weights/best.pt"
}

OUTPUT_DIR = ROOT / "results/yolo_test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = []

for model_name, model_path in MODEL_PATHS.items():

    print("\n================================")
    print("TESTING:", model_name)
    print("================================")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))

    start_time = time.time()

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=416,
        batch=1,
        device="cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name=model_name.lower(),
        exist_ok=True,
        verbose=True
    )

    elapsed = time.time() - start_time

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    print("\nTEST RESULTS:", model_name)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("mAP50:", round(map50, 4))
    print("mAP50-95:", round(map5095, 4))
    print("Evaluation time:", round(elapsed, 2), "seconds")

    summary.append([
        model_name,
        precision,
        recall,
        map50,
        map5095,
        elapsed
    ])

summary_file = OUTPUT_DIR / "test_comparison.csv"

with summary_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50_95",
        "Evaluation_Time_Seconds"
    ])

    writer.writerows(summary)

print("\n================================")
print("BOTH MODELS TESTED")
print("================================")
print("Comparison saved:", summary_file)
