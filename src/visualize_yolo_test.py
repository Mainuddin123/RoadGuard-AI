
from pathlib import Path
from ultralytics import YOLO
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = ROOT / "dataset/yolo_dataset/images/test"

MODEL_PATHS = {
    "Scratch": ROOT / "results/yolo_training/model1_scratch/weights/best.pt",
    "Pretrained": ROOT / "results/yolo_training/model2_pretrained/weights/best.pt"
}

OUTPUT_DIR = ROOT / "results/yolo_predictions"

images = sorted([
    path for path in TEST_IMAGE_DIR.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

if not images:
    raise RuntimeError("No test images found.")

# Select 6 evenly distributed test images
indices = np.linspace(
    0, len(images) - 1, 6, dtype=int
)

selected_images = [images[i] for i in indices]

print("ROADGUARD AI - TEST PREDICTION VISUALIZATION")
print("Selected test images:", len(selected_images))

for model_name, model_path in MODEL_PATHS.items():

    print(f"\nLoading {model_name} model...")

    model = YOLO(str(model_path))

    results = model.predict(
        source=[str(path) for path in selected_images],
        imgsz=416,
        conf=0.10,
        device="cpu",
        save=True,
        project=str(OUTPUT_DIR),
        name=model_name.lower(),
        exist_ok=True,
        verbose=False
    )

    total_detections = 0

    for result in results:

        count = len(result.boxes)
        total_detections += count

        print(
            f"{Path(result.path).name}: "
            f"{count} detections"
        )

    print(
        f"{model_name} total detections:",
        total_detections
    )

print("\nVISUALIZATION COMPLETED")
print("Output directory:", OUTPUT_DIR)
