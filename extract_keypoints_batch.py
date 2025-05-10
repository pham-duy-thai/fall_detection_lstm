import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

input_dir = "videos"
output_dir = "data"
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

for file in os.listdir(input_dir):
    if file.endswith(".mp4"):
        path = os.path.join(input_dir, file)
        if file.startswith("fall"):
            save = os.path.join(output_dir, file.replace(".mp4", "").replace("fall", "keypoints_fall") + ".npy")
        elif file.startswith("nofall"):
            save = os.path.join(output_dir, file.replace(".mp4", "").replace("nofall", "keypoints_nofall") + ".npy")
        else:
            continue
        extract_keypoints(path, save)
