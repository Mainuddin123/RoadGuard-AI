
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from road_dataset import RoadDamageDataset, detection_collate
from build_model import build_model


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = "pretrained"

EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.005

# Initial pipeline test limits
MAX_TRAIN_BATCHES = 5
MAX_VAL_BATCHES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# 2. REPRODUCIBILITY
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# 3. DATASET AND DATALOADERS
# ==================================================

train_dataset = RoadDamageDataset("train")
val_dataset = RoadDamageDataset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=detection_collate,
    num_workers=0
)


# ==================================================
# 4. BUILD MODEL 2
# ==================================================

print("\nLoading Model 2...")

model = build_model(MODEL_TYPE)
model.to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)


# ==================================================
# 5. MOVE DATA TO DEVICE
# ==================================================

def move_batch(images, targets):

    images = [
        image.to(DEVICE)
        for image in images
    ]

    targets = [
        {
            key: value.to(DEVICE)
            for key, value in target.items()
        }
        for target in targets
    ]

    return images, targets


# ==================================================
# 6. VALIDATION LOSS
# ==================================================

def calculate_validation_loss():

    # Torchvision Faster R-CNN returns losses in train mode.
    # Gradients remain disabled during validation.
    model.train()

    total_loss = 0.0
    batches_processed = 0

    with torch.no_grad():

        for batch_index, (images, targets) in enumerate(val_loader):

            if batch_index >= MAX_VAL_BATCHES:
                break

            images, targets = move_batch(images, targets)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            total_loss += loss.item()
            batches_processed += 1

    if batches_processed == 0:
        raise RuntimeError("No validation batches processed.")

    return total_loss / batches_processed


# ==================================================
# 7. OUTPUT FILES
# ==================================================

history_file = RESULT_DIR / "model2_training_history.csv"

checkpoint_file = MODEL_DIR / "model2_best.pth"

best_loss = float("inf")


# ==================================================
# 8. TRAINING INFORMATION
# ==================================================

print("\n========================================")
print("       ROADGUARD AI - MODEL 2")
print("========================================")

print("Architecture: Faster R-CNN ResNet50-FPN")
print("Initialization: ImageNet-pretrained backbone")
print("Device:", DEVICE)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Training batch limit:", MAX_TRAIN_BATCHES)
print("Validation batch limit:", MAX_VAL_BATCHES)

start_time = time.time()


# ==================================================
# 9. TRAINING LOOP
# ==================================================

with history_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "training_loss",
        "validation_loss",
        "epoch_time_seconds"
    ])

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        batches_processed = 0

        epoch_start = time.time()

        print(f"\nStarting epoch {epoch + 1}/{EPOCHS}")

        for batch_index, (images, targets) in enumerate(train_loader):

            if batch_index >= MAX_TRAIN_BATCHES:
                break

            images, targets = move_batch(images, targets)

            # Forward pass
            loss_dict = model(images, targets)

            # Combine Faster R-CNN losses
            loss = sum(loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            # Update model parameters
            optimizer.step()

            running_loss += loss.item()
            batches_processed += 1

            print(
                f"Training batch "
                f"{batch_index + 1}/{MAX_TRAIN_BATCHES} | "
                f"Loss: {loss.item():.4f}"
            )

        if batches_processed == 0:
            raise RuntimeError("No training batches processed.")

        train_loss = running_loss / batches_processed

        # Validation
        val_loss = calculate_validation_loss()

        scheduler.step()

        epoch_time = time.time() - epoch_start

        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            epoch_time
        ])

        file.flush()

        print("\n========== EPOCH RESULTS ==========")

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        # Save best checkpoint
        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_type": MODEL_TYPE,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss
                },
                checkpoint_file
            )

            print("Model 2 checkpoint saved successfully.")


# ==================================================
# 10. FINAL RESULTS
# ==================================================

total_time = time.time() - start_time

print("\n========================================")
print("         MODEL 2 TEST COMPLETED")
print("========================================")

print(f"Total training time: {total_time:.2f} seconds")
print(f"Best validation loss: {best_loss:.4f}")

print("Training history:", history_file)
print("Model checkpoint:", checkpoint_file)

print("\nModel 2 pipeline test completed.")
