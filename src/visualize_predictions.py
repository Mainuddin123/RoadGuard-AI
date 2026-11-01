
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageDraw

from road_dataset import RoadDamageDataset
from build_model import build_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"

OUTPUT_DIR = ROOT / "results" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

NUM_IMAGES = 6

CONFIDENCE_THRESHOLD = 0.01

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack",
    3: "Manhole"
}

COLORS = {
    1: "red",
    2: "lime",
    3: "cyan"
}

MODEL_CONFIGS = {
    "Scratch": {
        "type": "scratch",
        "checkpoint": MODEL_DIR / "mobilenet_scratch_final_best.pth"
    },
    "Pretrained": {
        "type": "pretrained",
        "checkpoint": MODEL_DIR / "mobilenet_pretrained_final_best.pth"
    }
}

torch.set_num_threads(4)


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

dataset = RoadDamageDataset("test")

print("ROADGUARD AI - PREDICTION VISUALIZATION")


# ==========================================
# 3. VISUALIZATION FUNCTION
# ==========================================

def draw_boxes(draw, boxes, labels, scores=None):

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = map(float, box)

        class_id = int(labels[index])

        class_name = CLASS_NAMES.get(class_id, "Unknown")

        color = COLORS.get(class_id, "white")

        text = class_name

        if scores is not None:
            text += f" {float(scores[index]):.3f}"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            text,
            fill=color
        )


# ==========================================
# 4. LOAD AND TEST MODELS
# ==========================================

for model_name, config in MODEL_CONFIGS.items():

    print(f"\nLoading {model_name} model...")

    model = build_model(config["type"])

    checkpoint = torch.load(
        config["checkpoint"],
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)

    # Lower internal threshold for diagnostic visualization
    model.roi_heads.score_thresh = 0.001

    model.eval()

    model_output = OUTPUT_DIR / model_name.lower()

    model_output.mkdir(exist_ok=True)

    for index in range(min(NUM_IMAGES, len(dataset))):

        image_tensor, target = dataset[index]

        with torch.inference_mode():
            prediction = model([image_tensor.to(DEVICE)])[0]

        image_array = (
            image_tensor.permute(1, 2, 0)
            .mul(255)
            .byte()
            .numpy()
        )

        original = Image.fromarray(image_array)

        # Ground-truth visualization
        gt_image = original.copy()

        gt_draw = ImageDraw.Draw(gt_image)

        draw_boxes(
            gt_draw,
            target["boxes"].tolist(),
            target["labels"].tolist()
        )

        # Prediction visualization
        pred_image = original.copy()

        pred_draw = ImageDraw.Draw(pred_image)

        pred_boxes = prediction["boxes"].cpu()
        pred_labels = prediction["labels"].cpu()
        pred_scores = prediction["scores"].cpu()

        keep = pred_scores >= CONFIDENCE_THRESHOLD

        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        draw_boxes(
            pred_draw,
            pred_boxes.tolist(),
            pred_labels.tolist(),
            pred_scores.tolist()
        )

        # Combine side by side
        combined = Image.new(
            "RGB",
            (original.width * 2, original.height)
        )

        combined.paste(gt_image, (0, 0))
        combined.paste(pred_image, (original.width, 0))

        output_path = (
            model_output / f"test_image_{index + 1}.jpg"
        )

        combined.save(output_path, quality=95)

        print(
            f"{model_name}: Image {index + 1} saved | "
            f"Predictions: {len(pred_boxes)}"
        )

    del model


print("\nVISUALIZATION COMPLETED")
print("Output directory:", OUTPUT_DIR)
