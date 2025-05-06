from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("yolov8n-pose.pt")

def extract_keypoints_from_video(video_path, save_path):
    print(f"📥 Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xy
            if len(kps) > 0:
                keypoints.append(kps[0].cpu().numpy())
                continue

        keypoints.append(np.zeros((17, 2)))

    cap.release()
    np.save(save_path, np.array(keypoints))
    print(f"✅ Keypoints saved to {save_path}")

extract_keypoints_from_video("videos/fall.mp4", "keypoints_fall.npy")
extract_keypoints_from_video("videos/nofall.mp4", "keypoints_nofall.npy")
