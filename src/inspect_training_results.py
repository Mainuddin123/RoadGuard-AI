
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"

files = {
    "Model 1 - Scratch":
        "mobilenet_scratch_final_history.csv",

    "Model 2 - Pretrained":
        "mobilenet_pretrained_final_history.csv"
}

print("\nROADGUARD AI - TRAINING HISTORY")

for model_name, filename in files.items():

    file_path = RESULT_DIR / filename

    print("\n================================")
    print(model_name)
    print("================================")

    if not file_path.exists():
        print("History file not found:", file_path)
        continue

    df = pd.read_csv(file_path)

    print(df.to_string(index=False))

    print("\nTraining loss:", df["training_loss"].iloc[-1])
    print("Validation loss:", df["validation_loss"].iloc[-1])
    print("Epochs completed:", len(df))


print("\nHISTORY INSPECTION COMPLETED")
