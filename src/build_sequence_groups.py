
from pathlib import Path
from datetime import datetime
import csv
import re
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = ROOT / "dataset/annotations/clean_manifest.csv"
OUTPUT = ROOT / "dataset/annotations/sequence_groups.csv"

GAP_LIMIT_SECONDS = 60


def identify_image(filename):
    # Numbered VLC frames
    if re.fullmatch(r"vlcsnap-\d+", Path(filename).stem, re.I):
        return "numbered_sequence", None

    # Timestamped VLC snapshots
    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h"
        r"(\d{2})m(\d{2})s(\d+)",
        filename,
        re.I
    )

    if match:
        y, mo, d, h, mi, s, ms = match.groups()
        timestamp = datetime(
            int(y), int(mo), int(d),
            int(h), int(mi), int(s),
            int(ms.ljust(3, "0")[:3]) * 1000
        )
        return "vlcsnap_timestamped", timestamp

    # Date-and-time filenames
    match = re.search(r"(\d{8})_(\d{6})", filename)

    if match:
        timestamp = datetime.strptime(
            match.group(1) + match.group(2),
            "%Y%m%d%H%M%S"
        )
        return "date_timestamped", timestamp

    return "unknown", None


with MANIFEST.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

families = {
    "vlcsnap_timestamped": [],
    "date_timestamped": []
}

numbered = []
unknown = []

for row in rows:
    family, timestamp = identify_image(row["image"])

    if family == "numbered_sequence":
        numbered.append(row)

    elif family == "unknown":
        unknown.append(row)

    else:
        families[family].append((timestamp, row))


grouped_rows = []


def add_group(group_id, group_rows, method):
    for row in group_rows:
        grouped_rows.append({
            "image": row["image"],
            "annotation": row["annotation"],
            "group_id": group_id,
            "group_method": method
        })


# Keep all numbered frames together.
if numbered:
    add_group(
        "NUMBERED_SEQUENCE_001",
        numbered,
        "numbered_sequence"
    )


# Group timestamped images separately by filename family.
for family, items in families.items():
    items.sort(key=lambda item: item[0])

    current_group = []
    group_number = 0

    for timestamp, row in items:

        if not current_group:
            current_group = [(timestamp, row)]
            continue

        previous_timestamp = current_group[-1][0]
        gap = (timestamp - previous_timestamp).total_seconds()

        if gap <= GAP_LIMIT_SECONDS:
            current_group.append((timestamp, row))

        else:
            group_number += 1

            add_group(
                f"{family.upper()}_{group_number:03d}",
                [item[1] for item in current_group],
                f"{family}_60sec_gap"
            )

            current_group = [(timestamp, row)]

    if current_group:
        group_number += 1

        add_group(
            f"{family.upper()}_{group_number:03d}",
            [item[1] for item in current_group],
            f"{family}_60sec_gap"
        )


# Keep unrecognized filenames separate and flag them.
for index, row in enumerate(unknown, start=1):
    add_group(
        f"UNKNOWN_{index:03d}",
        [row],
        "manual_review_required"
    )


# Save the grouping manifest.
with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image", "annotation", "group_id", "group_method"
        ]
    )
    writer.writeheader()
    writer.writerows(grouped_rows)


group_sizes = Counter(
    row["group_id"] for row in grouped_rows
)

print("\n===== SEQUENCE GROUP MANIFEST =====")
print("Manifest images:", len(rows))
print("Grouped images:", len(grouped_rows))
print("Numbered sequence images:", len(numbered))
print("Unknown images:", len(unknown))
print("Total sequence groups:", len(group_sizes))
print("Largest group:", max(group_sizes.values()))
print("Singleton groups:", sum(size == 1 for size in group_sizes.values()))
print("Output:", OUTPUT)

print("\nGroups by method:")
print(Counter(row["group_method"] for row in grouped_rows))
