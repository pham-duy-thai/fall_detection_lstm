"""Convert the supplied Fall/No_Fall keypoint CSV dataset to LSTM sequences.

The dataset already contains pose keypoints, so YOLO extraction is not needed.
"""
import csv
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CSV_DATASET_ROOT", "fallandnofall/split_dataset"))
OUT = Path(os.environ.get("CSV_SEQUENCE_DIR", "csv_sequences"))
WINDOW = int(os.environ.get("SEQUENCE_WINDOW", "40"))
STRIDE = int(os.environ.get("SEQUENCE_STRIDE", "5"))

NAMES = {"Nose": 0, "Left Shoulder": 5, "Right Shoulder": 6,
         "Left Hip": 11, "Right Hip": 12}

def read_features(path):
    frames = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["Keypoint"].strip()
            if name not in NAMES:
                continue
            frame = int(row["Frame"])
            frames.setdefault(frame, {})[name] = (float(row["X"]), float(row["Y"]))
    features = []
    for frame in sorted(frames):
        p = frames[frame]
        if any(name not in p for name in NAMES):
            features.append((90.0, 0.0, 0.0))
            continue
        nose = np.array(p["Nose"])
        shoulder = (np.array(p["Left Shoulder"]) + np.array(p["Right Shoulder"])) / 2
        hip = (np.array(p["Left Hip"]) + np.array(p["Right Hip"])) / 2
        d = hip - shoulder
        features.append((np.degrees(np.arctan2(d[1], d[0])),
                         np.linalg.norm(d), np.linalg.norm(nose - hip)))
    return np.asarray(features, dtype=np.float32)

def build_split(split):
    xs, ys = [], []
    for label, folder in ((1, "Fall"), (0, "No_Fall")):
        source = ROOT / split / folder / "Keypoints_CSV"
        files = sorted(source.glob("*.csv"))
        print(f"{split}/{folder}: {len(files)} CSV files")
        for i, path in enumerate(files, 1):
            f = read_features(path)
            for start in range(0, max(0, len(f) - WINDOW + 1), STRIDE):
                seq = f[start:start + WINDOW]
                if len(seq) == WINDOW:
                    xs.append(seq); ys.append(label)
            if i % 250 == 0:
                print(f"  processed {i}/{len(files)}")
    if not xs:
        raise RuntimeError(f"No sequences generated for {split}")
    x, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{split}.npz", X=x, y=y)
    print(f"saved {split}: X={x.shape}, labels={np.bincount(y)}")

for split in ("train", "val", "test"):
    build_split(split)
