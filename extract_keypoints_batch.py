import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

input_dir = os.environ.get("KEYPOINT_INPUT_DIR", "videos")
output_dir = os.environ.get("KEYPOINT_OUTPUT_DIR", "data")
os.makedirs(output_dir, exist_ok=True)

def extract_keypoints(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    all_kps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xy
            try:
                arr = kps[0].cpu().numpy()
                if arr.shape != (17, 2):
                    raise ValueError("Invalid shape")
                all_kps.append(arr)
            except:
                all_kps.append(np.zeros((17, 2)))
        else:
            all_kps.append(np.zeros((17, 2)))

    cap.release()
    try:
        np.save(save_path, np.array(all_kps))
        print(f"✅ Saved: {save_path}")
    except Exception as e:
        print(f"❌ Error saving {save_path}: {e}")

for root, _, files in os.walk(input_dir):
  for file in files:
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        path = os.path.join(root, file)
        stem = os.path.splitext(file)[0].lower()
        parent = os.path.basename(root).lower()
        if parent in ("fall", "nofall"):
            stem = parent + "_" + stem
        if stem.startswith("fall"):
            save = os.path.join(output_dir, "keypoints_fall_" + os.path.splitext(file)[0] + ".npy")
        elif stem.startswith("nofall"):
            save = os.path.join(output_dir, "keypoints_nofall_" + os.path.splitext(file)[0] + ".npy")
        else:
            continue
        extract_keypoints(path, save)
