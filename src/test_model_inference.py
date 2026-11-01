
import torch

from build_model import build_model
from road_dataset import RoadDamageDataset, CLASS_NAMES

print("Loading Model 1...")

model = build_model("scratch")
model.eval()

dataset = RoadDamageDataset("validation")
image, target = dataset[0]

print("Input image shape:", image.shape)

print("\nRunning inference...")

with torch.no_grad():
    prediction = model([image])[0]

print("\n===== MODEL INFERENCE RESULTS =====")
print("Predicted boxes:", prediction["boxes"].shape)
print("Predicted labels:", prediction["labels"].tolist())
print("Confidence scores:", prediction["scores"].tolist())

print("\nPredicted objects:")

for label, score, box in zip(
    prediction["labels"],
    prediction["scores"],
    prediction["boxes"]
):
    class_id = label.item()

    if class_id in CLASS_NAMES:
        print(
            f"{CLASS_NAMES[class_id]} | "
            f"Confidence: {score.item():.4f} | "
            f"Box: {box.tolist()}"
        )

print("\nInference test completed.")
