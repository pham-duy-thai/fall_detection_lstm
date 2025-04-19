from ultralytics import YOLO
import cv2
import numpy as np
import os

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # بررسی وجود keypoints
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xy
            if len(kps) > 0:
                k = kps[0].cpu().numpy()
                if k.shape == (17, 2):
                    all_keypoints.append(k)
                    continue

        # اگر keypoints وجود نداشت یا ناقص بود → آرایه صفر جایگزین
        all_keypoints.append(np.zeros((17, 2)))

    cap.release()

    all_keypoints_np = np.array(all_keypoints)
    np.save(save_path, all_keypoints_np)
    print(f"✅ Keypoints saved to {save_path} ({len(all_keypoints_np)} frames processed)")

# اجرای تابع
extract_keypoints_from_video()
