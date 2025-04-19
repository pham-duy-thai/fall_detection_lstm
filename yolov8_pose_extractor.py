import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def extract_keypoints_from_video(video_path="Videos/input.mp4", save_path="keypoints.npy"):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints.xy

        if len(keypoints) > 0:
            kps = keypoints[0].cpu().numpy()
            all_keypoints.append(kps)
        else:
            all_keypoints.append(np.zeros((17, 2)))

    cap.release()
    np.save(save_path, np.array(all_keypoints))
    print(f"Keypoints saved to {save_path}")
