
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset/raw/primary/data/images"

hash_groups = defaultdict(list)

for image_path in IMAGE_DIR.glob("*"):
    if not image_path.is_file():
        continue

    file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
    hash_groups[file_hash].append(image_path.name)

duplicates = {
    key: names
    for key, names in hash_groups.items()
    if len(names) > 1
}

print("\n===== EXACT DUPLICATE IMAGE CHECK =====")
print("Total images:", len(list(IMAGE_DIR.glob("*"))))
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicates))

for names in list(duplicates.values())[:10]:
    print(names)

print("\nExact duplicate check completed.")
