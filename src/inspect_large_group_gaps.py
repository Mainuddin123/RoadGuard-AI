
from pathlib import Path
from datetime import datetime
import csv
import re

ROOT = Path(__file__).resolve().parent.parent
GROUP_FILE = ROOT / "dataset/annotations/sequence_groups.csv"

def get_timestamp(filename):
    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h"
        r"(\d{2})m(\d{2})s(\d+)",
        filename
    )

    if not match:
        return None

    y, mo, d, h, mi, s, ms = match.groups()

    return datetime(
        int(y), int(mo), int(d),
        int(h), int(mi), int(s),
        int(ms.ljust(3, "0")[:3]) * 1000
    )

with GROUP_FILE.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for group_id in [
    "VLCSNAP_TIMESTAMPED_015",
    "VLCSNAP_TIMESTAMPED_035"
]:
    items = []

    for row in rows:
        if row["group_id"] == group_id:
            timestamp = get_timestamp(row["image"])

            if timestamp:
                items.append((timestamp, row["image"]))

    items.sort()

    print(f"\n===== {group_id} =====")
    print("Images:", len(items))

    gaps = []

    for previous, current in zip(items, items[1:]):
        seconds = (current[0] - previous[0]).total_seconds()

        if seconds > 10:
            gaps.append((seconds, previous[1], current[1]))

    gaps.sort(reverse=True)

    print("Gaps greater than 10 seconds:", len(gaps))

    for seconds, first, second in gaps[:15]:
        print(f"{seconds:.2f}s | {first} -> {second}")
