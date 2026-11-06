"""Create a non-destructive, reviewable dataset layout.

Original videos are never moved or renamed. Archive labels use its ground-truth
annotation files. Videos from datafall/datanofall are placed in uncertain
because their folder names are known to be unreliable.
"""
from pathlib import Path
import shutil
import csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reviewed"
for label in ("fall", "nofall", "uncertain"):
    (OUT / label).mkdir(parents=True, exist_ok=True)

rows = []

def unique_target(label, source):
    target = OUT / label / source.name
    if not target.exists(): return target
    i = 2
    while True:
        candidate = target.with_name(f"{target.stem}_{i}{target.suffix}")
        if not candidate.exists(): return candidate
        i += 1

def archive_annotation(video):
    for folder in ("Annotation_files", "Annotations_files"):
        p = video.parent.parent / folder / f"{video.stem}.txt"
        if p.exists(): return p
    return None

archive = ROOT / "archive (1)"
for video in sorted(archive.rglob("*.avi")):
    label = "fall" if archive_annotation(video) else "nofall"
    target = unique_target(label, video)
    shutil.copy2(video, target)
    rows.append([str(video), label, "archive annotation" if label == "fall" else "no matching annotation"])

for source_root in (ROOT / "datafall", ROOT / "datanofall"):
    for video in sorted(source_root.rglob("*.mp4")):
        target = unique_target("uncertain", video)
        shutil.copy2(video, target)
        rows.append([str(video), "uncertain", "folder label not trusted; manual review required"])

with (OUT / "review_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "label", "reason"])
    writer.writerows(rows)

print(f"Created: {OUT}")
for label in ("fall", "nofall", "uncertain"):
    print(f"{label}: {sum(1 for r in rows if r[1] == label)}")
print(f"Manifest: {OUT / 'review_manifest.csv'}")
