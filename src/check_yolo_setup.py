
from pathlib import Path
import importlib.util
import ultralytics
print("Ultralytics version:", ultralytics.__version__)
ROOT = Path(__file__).resolve().parent.parent

print("ROADGUARD AI - YOLO SETUP CHECK")

# Check Ultralytics
installed = importlib.util.find_spec("ultralytics")

print("\nUltralytics installed:", installed is not None)

# Check dataset
manifest = ROOT / "dataset/annotations/final_split_manifest.csv"

print("Split manifest exists:", manifest.exists())

# Check cleaned annotations
labels = ROOT / "dataset/annotations/primary_yolo_clipped"

print("Cleaned YOLO labels exist:", labels.exists())

if labels.exists():
    print("Number of annotation files:", len(list(labels.glob("*.txt"))))

# Check original images
images = ROOT / "dataset/raw/primary/data/images"

print("Original images exist:", images.exists())

if images.exists():
    print("Number of image files:", len(list(images.glob("*.jpg"))))

print("\nSetup check completed.")
