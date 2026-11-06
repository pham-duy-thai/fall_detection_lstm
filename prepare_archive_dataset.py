"""Convert the LE2I archive into the keypoint format used by this project.

Videos with a matching annotation file are labelled fall. Videos without an
annotation file are labelled nofall. The archive annotations contain the fall
frame range and bounding boxes, not pose keypoints; YOLOv8-Pose generates the
17 keypoints here.
"""
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive (1)"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
POSE = YOLO(str(ROOT / "yolov8n-pose.pt"))


def annotation_for(video: Path):
    # In the archive Videos and Annotation_files/Annotations_files are siblings.
    for name in ("Annotation_files", "Annotations_files"):
        candidate = video.parent.parent / name / f"{video.stem}.txt"
        if candidate.exists():
            return candidate
    return None


def extract(video: Path, destination: Path):
    cap = cv2.VideoCapture(str(video))
    points = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = POSE(frame, verbose=False)[0]
        if result.keypoints is not None and len(result.keypoints.xy):
            kp = result.keypoints.xy[0].cpu().numpy()
            points.append(kp if kp.shape == (17, 2) else np.zeros((17, 2)))
        else:
            points.append(np.zeros((17, 2)))
    cap.release()
    np.save(destination, np.asarray(points, dtype=np.float32))
    return len(points)


videos = sorted(ARCHIVE.rglob("*.avi"))
if not videos:
    raise SystemExit(f"No AVI files found in: {ARCHIVE}")

fall_count = nofall_count = 0
for index, video in enumerate(videos, 1):
    is_fall = annotation_for(video) is not None
    label = "fall" if is_fall else "nofall"
    if is_fall:
        fall_count += 1
    else:
        nofall_count += 1
    output = OUT / f"keypoints_{label}_archive_{index:04d}.npy"
    if output.exists():
        print(f"[{index}/{len(videos)}] exists: {output.name}")
        continue
    frames = extract(video, output)
    print(f"[{index}/{len(videos)}] {label}: {video.name} -> {frames} frames")

print(f"Done. videos={len(videos)}, fall={fall_count}, nofall={nofall_count}")
print("Next: run generate_sequences.py, then train_fall_lstm.py")
