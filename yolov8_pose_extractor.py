import cv2
import numpy as np
from ultralytics import YOLO
import os

# مدل YOLO Pose
model = YOLO("yolov8n-pose.pt")

def extract_keypoints_from_video(video_path="videos/input.mp4", save_path="keypoints.npy"):
    print(f"🔍 Loading video from: {video_path}")

    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    print("✅ Video opened successfully.")
    print("🔄 Starting keypoint extraction...")

    all_keypoints = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, verbose=False)

        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xy
            if len(kps) > 0:
                all_keypoints.append(kps[0].cpu().numpy())
            else:
                all_keypoints.append(np.zeros((17, 2)))
        else:
            all_keypoints.append(np.zeros((17, 2)))

    cap.release()

    if len(all_keypoints) == 0:
        print("⚠️ No keypoints extracted.")
    else:
        np.save(save_path, np.array(all_keypoints))
        print(f"✅ Keypoints saved to {save_path} ({len(all_keypoints)} frames processed)")

# اجرای تابع
extract_keypoints_from_video()
