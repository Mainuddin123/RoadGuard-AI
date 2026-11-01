
from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

print("\nROADGUARD AI - YOLO MODEL CHECK")

# Model 1: Random initialization
print("\nLoading Model 1 - Scratch...")

model1 = YOLO("yolov8n.yaml")

print("Model 1 architecture:", model1.model.__class__.__name__)

# Model 2: Pretrained initialization
print("\nLoading Model 2 - Pretrained...")

model2 = YOLO("yolov8n.pt")
# Ensure all parameters are trainable for fine-tuning
for parameter in model2.model.parameters():
    parameter.requires_grad = True

print("Model 2 architecture:", model2.model.__class__.__name__)

# Display model information
print("\nMODEL COMPARISON")

for name, model in [
    ("Scratch", model1),
    ("Pretrained", model2)
]:

    parameters = sum(
        p.numel() for p in model.model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.model.parameters()
        if p.requires_grad
    )

    print(f"\n{name}")
    print("Parameters:", f"{parameters:,}")
    print("Trainable parameters:", f"{trainable:,}")

print("\nBOTH YOLO MODELS LOADED SUCCESSFULLY")
