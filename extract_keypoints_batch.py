import os
import cv2
import numpy as np
from ultralytics import YOLO

# بارگذاری مدل pose YOLO
model = YOLO("yolov8n-pose.pt")

# مسیرها
input_dir = "videos"
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
                    raise ValueError("Invalid keypoint shape")
                all_kps.append(arr)
            except:
                all_kps.append(np.zeros((17, 2)))
        else:
            all_kps.append(np.zeros((17, 2)))

    cap.release()
    try:
        keypoints_array = np.array(all_kps)
        np.save(save_path, keypoints_array)
        print(f"✅ Saved: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save {save_path}: {e}")

# اسکن همه ویدیوهای mp4
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_dir, filename)

        if filename.startswith("fall"):
            save_name = filename.replace(".mp4", ".npy").replace("fall", "keypoints_fall")
        elif filename.startswith("nofall"):
            save_name = filename.replace(".mp4", ".npy").replace("nofall", "keypoints_nofall")
        else:
            continue  # نادیده گرفتن ویدیوهایی که اسمشون مناسب نیست

        save_path = os.path.join(output_dir, save_name)
        extract_keypoints(video_path, save_path)
