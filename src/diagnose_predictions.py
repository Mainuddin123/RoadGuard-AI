
import torch
from pathlib import Path
from collections import Counter

from road_dataset import RoadDamageDataset
from build_model import build_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"

DEVICE = torch.device("cpu")

NUM_IMAGES = 10

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack",
    3: "Manhole"
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

THRESHOLDS = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]


# ==========================================
# 2. LOAD TEST DATASET
# ==========================================

test_dataset = RoadDamageDataset("test")

num_images = min(NUM_IMAGES, len(test_dataset))

print("\n========================================")
print(" ROADGUARD AI - PREDICTION DIAGNOSIS")
print("========================================")

print("Test dataset:", len(test_dataset))
print("Images inspected:", num_images)
print("Device:", DEVICE)


# ==========================================
# 3. DIAGNOSE EACH MODEL
# ==========================================

for model_name, config in MODEL_CONFIGS.items():

    print("\n\n========================================")
    print("MODEL:", model_name)
    print("========================================")

    checkpoint_path = config["checkpoint"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Build architecture
    model = build_model(config["type"])

    # Load trained weights
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    # Disable internal confidence filtering.
    # Diagnostic only; does not modify checkpoint weights.
    model.roi_heads.score_thresh = 0.0

    model.eval()

    total_raw_predictions = 0

    threshold_counts = {
        threshold: 0
        for threshold in THRESHOLDS
    }

    predicted_class_counts = Counter()

    all_scores = []

    # ======================================
    # 4. TEST IMAGES
    # ======================================

    with torch.inference_mode():

        for index in range(num_images):

            image, target = test_dataset[index]

            image = image.to(DEVICE)

            prediction = model([image])[0]

            boxes = prediction["boxes"].cpu()
            labels = prediction["labels"].cpu()
            scores = prediction["scores"].cpu()

            total_raw_predictions += len(scores)

            print(f"\nImage {index + 1}/{num_images}")

            print("Ground-truth objects:", len(target["boxes"]))
            print("Raw predictions:", len(scores))

            if len(scores) == 0:

                print("No predictions returned.")

                continue

            max_score = scores.max().item()

            min_score = scores.min().item()

            print(f"Maximum confidence: {max_score:.6f}")
            print(f"Minimum confidence: {min_score:.6f}")

            # Count predicted classes
            for label in labels.tolist():

                predicted_class_counts[
                    CLASS_NAMES.get(label, f"Unknown {label}")
                ] += 1

            # Save confidence values
            all_scores.extend(scores.tolist())

            # Threshold analysis
            for threshold in THRESHOLDS:

                count = int(
                    (scores >= threshold).sum().item()
                )

                threshold_counts[threshold] += count

            # Show top 5 predictions
            top_count = min(5, len(scores))

            top_indices = torch.argsort(
                scores,
                descending=True
            )[:top_count]

            print("\nTop predictions:")

            for prediction_index in top_indices:

                label = labels[prediction_index].item()

                score = scores[prediction_index].item()

                class_name = CLASS_NAMES.get(
                    label,
                    "Unknown"
                )

                box = boxes[prediction_index].tolist()

                print(
                    f"{class_name} | "
                    f"Confidence: {score:.4f} | "
                    f"Box: {[round(v, 1) for v in box]}"
                )

    # ======================================
    # 5. MODEL SUMMARY
    # ======================================

    print("\n========== MODEL SUMMARY ==========")

    print("Model:", model_name)

    print("Images inspected:", num_images)

    print("Total raw predictions:", total_raw_predictions)

    print("\nPredicted class distribution:")

    for class_name in CLASS_NAMES.values():

        print(
            f"{class_name}: "
            f"{predicted_class_counts[class_name]}"
        )

    print("\nConfidence threshold analysis:")

    for threshold in THRESHOLDS:

        print(
            f"Threshold {threshold:.3f}: "
            f"{threshold_counts[threshold]} predictions"
        )

    if all_scores:

        print("\nOverall confidence statistics:")

        print(
            "Average confidence:",
            round(sum(all_scores) / len(all_scores), 6)
        )

        print(
            "Maximum confidence:",
            round(max(all_scores), 6)
        )

    else:

        print("\nNo raw prediction scores available.")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n========================================")
print(" PREDICTION DIAGNOSIS COMPLETED")
print("========================================")
