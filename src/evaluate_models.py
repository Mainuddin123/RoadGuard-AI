
import csv
import time
from pathlib import Path
from collections import defaultdict

import torch
from torchvision.ops import box_iou

from road_dataset import RoadDamageDataset
from build_model import build_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

IOU_THRESHOLD = 0.50
CONFIDENCE_THRESHOLD = 0.50

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack",
    3: "Manhole"
}

MODEL_CONFIGS = {
    "Scratch": MODEL_DIR / "mobilenet_scratch_final_best.pth",
    "Pretrained": MODEL_DIR / "mobilenet_pretrained_final_best.pth"
}

torch.set_num_threads(4)


# ==========================================
# 2. TEST DATASET
# ==========================================

test_dataset = RoadDamageDataset("test")

print("ROADGUARD AI - MODEL EVALUATION")
print("Test images:", len(test_dataset))
print("Device:", DEVICE)


# ==========================================
# 3. LOAD MODEL
# ==========================================

def load_model(checkpoint_path, model_type):

    model = build_model(model_type)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, model_name):

    counts = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0, "IoU": []}
    )

    total_inference_time = 0.0

    print(f"\nEvaluating {model_name}...")

    for index in range(len(test_dataset)):

        image, target = test_dataset[index]

        image = image.to(DEVICE)

        ground_boxes = target["boxes"]
        ground_labels = target["labels"]

        start_time = time.perf_counter()

        with torch.inference_mode():
            prediction = model([image])[0]

        inference_time = time.perf_counter() - start_time

        total_inference_time += inference_time

        pred_boxes = prediction["boxes"].cpu()
        pred_labels = prediction["labels"].cpu()
        pred_scores = prediction["scores"].cpu()

        # Apply confidence threshold
        keep = pred_scores >= CONFIDENCE_THRESHOLD

        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # Match predictions separately for each class
        for class_id, class_name in CLASS_NAMES.items():

            gt_indices = torch.where(
                ground_labels == class_id
            )[0]

            pred_indices = torch.where(
                pred_labels == class_id
            )[0]

            gt_boxes = ground_boxes[gt_indices]

            gt_matched = set()

            # Highest-confidence predictions first
            if len(pred_indices) > 0:

                order = torch.argsort(
                    pred_scores[pred_indices],
                    descending=True
                )

                pred_indices = pred_indices[order]

            for pred_index in pred_indices:

                if len(gt_boxes) == 0:
                    counts[class_name]["FP"] += 1
                    continue

                predicted_box = pred_boxes[pred_index].unsqueeze(0)

                ious = box_iou(predicted_box, gt_boxes)[0]

                # Exclude ground truths already matched
                for matched_index in gt_matched:
                    ious[matched_index] = -1

                best_iou, best_index = torch.max(ious, dim=0)

                if best_iou.item() >= IOU_THRESHOLD:

                    counts[class_name]["TP"] += 1
                    counts[class_name]["IoU"].append(
                        best_iou.item()
                    )

                    gt_matched.add(best_index.item())

                else:
                    counts[class_name]["FP"] += 1

            # Count unmatched ground-truth objects
            counts[class_name]["FN"] += (
                len(gt_boxes) - len(gt_matched)
            )

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(test_dataset)} images"
            )

    # ======================================
    # 5. CALCULATE METRICS
    # ======================================

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    print("\n========== EVALUATION RESULTS ==========")

    for class_name in CLASS_NAMES.values():

        result = counts[class_name]

        tp = result["TP"]
        fp = result["FP"]
        fn = result["FN"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        mean_iou = (
            sum(result["IoU"]) / len(result["IoU"])
            if result["IoU"] else 0.0
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_ious.extend(result["IoU"])

        print(f"\n{class_name}")
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )

    overall_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )

    overall_iou = (
        sum(all_ious) / len(all_ious)
        if all_ious else 0.0
    )

    total_time = total_inference_time

    average_inference_time = total_time / len(test_dataset)

    fps = (
        1 / average_inference_time
        if average_inference_time > 0 else 0.0
    )

    checkpoint_path = MODEL_CONFIGS[model_name]
    model_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    print("\n========== OVERALL METRICS ==========")

    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"Mean IoU: {overall_iou:.4f}")

    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Average inference time: {average_inference_time:.4f} seconds")
    print(f"Inference FPS: {fps:.2f}")
    print(f"Checkpoint size: {model_size_mb:.2f} MB")

    return {
        "model": model_name,
        "precision": overall_precision,
        "recall": overall_recall,
        "mean_iou": overall_iou,
        "inference_time": total_time,
        "average_inference_time": average_inference_time,
        "fps": fps,
        "model_size_mb": model_size_mb
    }


# ==========================================
# 6. EVALUATE BOTH MODELS
# ==========================================

results = []

for model_name, checkpoint_path in MODEL_CONFIGS.items():

    model_type = (
        "scratch" if model_name == "Scratch"
        else "pretrained"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = load_model(
        checkpoint_path,
        model_type
    )

    result = evaluate_model(model, model_name)

    results.append(result)

    del model


# ==========================================
# 7. SAVE COMPARISON
# ==========================================

output_file = RESULT_DIR / "model_comparison.csv"

with output_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


print("\n========================================")
print("MODEL EVALUATION COMPLETED")
print("========================================")

print("Comparison saved:", output_file)
