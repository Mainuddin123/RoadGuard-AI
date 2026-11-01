
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"

scratch_file = RESULT_DIR / "mobilenet_scratch_final_history.csv"
pretrained_file = RESULT_DIR / "mobilenet_pretrained_final_history.csv"

scratch = pd.read_csv(scratch_file)
pretrained = pd.read_csv(pretrained_file)

plt.figure(figsize=(10, 6))

plt.plot(
    scratch["epoch"],
    scratch["training_loss"],
    marker="o",
    label="Scratch - Training"
)

plt.plot(
    scratch["epoch"],
    scratch["validation_loss"],
    marker="o",
    label="Scratch - Validation"
)

plt.plot(
    pretrained["epoch"],
    pretrained["training_loss"],
    marker="s",
    label="Pretrained - Training"
)

plt.plot(
    pretrained["epoch"],
    pretrained["validation_loss"],
    marker="s",
    label="Pretrained - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RoadGuard AI - Training and Validation Loss")
plt.legend()
plt.grid(True)

output_path = RESULT_DIR / "mobilenet_training_loss_comparison.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Loss graph saved:", output_path)
