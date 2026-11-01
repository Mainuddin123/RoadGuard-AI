
from pathlib import Path
from datetime import datetime
import re
import csv

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset/annotations/clean_manifest.csv"

def get_timestamp(filename):
    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h"
        r"(\d{2})m(\d{2})s(\d+)",
        filename,
        re.IGNORECASE
    )

    if not match:
        return None

    y, mo, d, h, mi, s, ms = match.groups()

    return datetime(
        int(y), int(mo), int(d),
        int(h), int(mi), int(s),
        int(ms.ljust(3, "0")[:3]) * 1000
    )

with MANIFEST.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

timestamped = []

for row in rows:
    name = row["image"]

    if not name.lower().startswith("vlcsnap-"):
        continue

    timestamp = get_timestamp(name)

    if timestamp:
        timestamped.append((timestamp, name))

timestamped.sort()

print("\n===== TIMESTAMPED VIDEO SEQUENCES =====")
print("Timestamped VLC snapshots:", len(timestamped))

for threshold in [10, 30, 60]:
    groups = []

    for item in timestamped:
        if not groups:
            groups.append([item])
            continue

        gap = (item[0] - groups[-1][-1][0]).total_seconds()

        if gap <= threshold:
            groups[-1].append(item)
        else:
            groups.append([item])

    print(f"\nGap threshold: {threshold} seconds")
    print("Sequence groups:", len(groups))

    for index, group in enumerate(groups, 1):
        start = group[0][0]
        end = group[-1][0]

        print(
            f"Group {index}: {len(group)} images | "
            f"{start} to {end}"
        )

print("\nAnalysis completed.")
