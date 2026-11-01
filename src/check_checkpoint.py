from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "models" / "model1_best.pth"

print("ROADGUARD AI - CHECKPOINT VERIFICATION")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError("Model checkpoint not found!")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint loaded successfully!")
print("Model type:", checkpoint["model_type"])
print("Epoch:", checkpoint["epoch"])
print("Validation loss:", checkpoint["validation_loss"])
print("Number of saved parameters:", len(checkpoint["model_state_dict"]))

print("\nCHECKPOINT VERIFICATION COMPLETED")