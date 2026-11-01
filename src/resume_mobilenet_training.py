
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from road_dataset import RoadDamageDataset, detection_collate
from build_model import build_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Train one model at a time: scratch or pretrained

MODEL_TYPE = "pretrained"

TARGET_EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 0.005

DEVICE = torch.device("cpu")

SEED = 42
MAX_GRAD_NORM = 5.0

torch.set_num_threads(4)

checkpoint_file = (
    MODEL_DIR / f"mobilenet_{MODEL_TYPE}_final_best.pth"
)

latest_file = (
    MODEL_DIR / f"mobilenet_{MODEL_TYPE}_latest.pth"
)

history_file = (
    RESULT_DIR / f"mobilenet_{MODEL_TYPE}_continued_history.csv"
)


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# 3. DATASET
# ==========================================

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


# ==========================================
# 4. BUILD AND RESTORE MODEL
# ==========================================

if not checkpoint_file.exists():
    raise FileNotFoundError(checkpoint_file)

print("\nLoading existing checkpoint...")

model = build_model(MODEL_TYPE)

checkpoint = torch.load(
    checkpoint_file,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)

if "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

start_epoch = checkpoint["epoch"]
best_loss = checkpoint["validation_loss"]

if latest_file.exists():
    latest = torch.load(
        latest_file,
        map_location=DEVICE,
        weights_only=False
    )

    if latest["epoch"] > start_epoch:
        model.load_state_dict(latest["model_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        start_epoch = latest["epoch"]
        best_loss = min(best_loss, latest["validation_loss"])

print("Model:", MODEL_TYPE)
print("Resuming after epoch:", start_epoch)
print("Target total epochs:", TARGET_EPOCHS)


# ==========================================
# 5. MOVE DATA
# ==========================================

def move_batch(images, targets):

    images = [image.to(DEVICE) for image in images]

    targets = [
        {key: value.to(DEVICE) for key, value in target.items()}
        for target in targets
    ]

    return images, targets


# ==========================================
# 6. VALIDATION
# ==========================================

def calculate_validation_loss():

    model.train()

    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images, targets = move_batch(images, targets)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            total_loss += loss.item()
            batches += 1

    if batches == 0:
        raise RuntimeError("Validation dataset is empty.")

    return total_loss / batches


# ==========================================
# 7. HISTORY FILE
# ==========================================

write_header = not history_file.exists()

with history_file.open(
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if write_header:
        writer.writerow([
            "epoch",
            "training_loss",
            "validation_loss",
            "epoch_time_seconds"
        ])

    # ======================================
    # 8. TRAINING LOOP
    # ======================================

    total_start = time.time()

    for epoch in range(start_epoch, TARGET_EPOCHS):

        model.train()

        running_loss = 0.0
        batches = 0

        epoch_start = time.time()

        print(f"\nStarting epoch {epoch + 1}/{TARGET_EPOCHS}")

        for batch_index, (images, targets) in enumerate(train_loader):

            images, targets = move_batch(images, targets)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

            optimizer.step()

            running_loss += loss.item()
            batches += 1

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Batch {batch_index + 1}/{len(train_loader)}"
                    f" | Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / batches

        print("\nRunning validation...")

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

        print("\nEPOCH RESULTS")
        print("Training loss:", round(train_loss, 4))
        print("Validation loss:", round(val_loss, 4))
        print("Epoch time:", round(epoch_time, 2), "seconds")

        state = {
            "epoch": epoch + 1,
            "model_type": MODEL_TYPE,
            "architecture": "Faster R-CNN MobileNetV3-Large-FPN",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_loss": val_loss
        }

        # Always save latest checkpoint
        torch.save(state, latest_file)

        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(state, checkpoint_file)
            print("Best checkpoint updated.")

    print("\nCONTINUED TRAINING COMPLETED")
    print("Total time:", round(time.time() - total_start, 2))
    print("Best validation loss:", round(best_loss, 4))
    print("Latest checkpoint:", latest_file)
    print("Best checkpoint:", checkpoint_file)
