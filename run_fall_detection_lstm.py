import cv2
import numpy as np
import torch
import time
import requests
from ultralytics import YOLO
from lstm_model import LSTMClassifier

# تنظیمات
model_path = "lstm_fall_model_fallandnofall.pth"
video_path = "videos_test/test_fall1.mp4"  # اگر use_webcam=False باشه از این استفاده میشه
use_webcam = True
camera_index = 0
window_size = 40
backend_url = "http://127.0.0.1:8000"
camera_id = "camera_01"
person_id = "unknown"
alert_cooldown_seconds = 30
status_interval_seconds = 1

# دستگاه اجرا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚙️ Running on device:", device)

# بارگذاری مدل‌ها
pose_model = YOLO("yolov8n-pose.pt")
model = LSTMClassifier(input_dim=3, hidden_dim=64, output_dim=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

sequence = []
cap = cv2.VideoCapture(camera_index if use_webcam else video_path, cv2.CAP_DSHOW if use_webcam else cv2.CAP_ANY)
if use_webcam and not cap.isOpened():
    raise RuntimeError(f"Không mở được webcam index {camera_index}. Hãy thử camera_index = 1 hoặc kiểm tra quyền Camera của Windows.")
last_alert_time = 0.0
last_status_time = 0.0


def send_fall_event(confidence):
    """Send one fall event to the backend, without stopping detection if API is offline."""
    global last_alert_time
    now = time.time()
    if now - last_alert_time < alert_cooldown_seconds:
        return

    payload = {
        "person_id": person_id,
        "camera_id": camera_id,
        "confidence": float(confidence),
        "message": "Phát hiện người ngã",
    }
    try:
        response = requests.post(f"{backend_url}/api/events", json=payload, timeout=3)
        response.raise_for_status()
        last_alert_time = now
        print("📨 Fall event sent to backend")
    except requests.RequestException as exc:
        print(f"⚠️ Cannot send event to backend: {exc}")


def send_detector_status(state, confidence, fps=0):
    try:
        requests.post(f"{backend_url}/api/status", json={
            "state": state, "confidence": float(confidence),
            "fps": float(fps), "camera_id": camera_id,
        }, timeout=1)
    except requests.RequestException:
        pass

def extract_features(kp):
    try:
        nose = kp[0]
        l_shoulder = kp[5]
        r_shoulder = kp[6]
        l_hip = kp[11]
        r_hip = kp[12]
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip = (l_hip + r_hip) / 2
        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]
        angle = np.degrees(np.arctan2(dy, dx))
        torso = np.linalg.norm(mid_hip - mid_shoulder)
        head_dist = np.linalg.norm(nose - mid_hip)
        return [angle, torso, head_dist]
    except:
        return [90, 0, 0]

# اجرای پردازش
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        kp = results[0].keypoints.xy[0]
        feat = extract_features(kp.cpu().numpy())
        sequence.append(feat)

        if len(sequence) >= window_size:
            window = np.array(sequence[-window_size:])
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                probabilities = torch.softmax(out, dim=1)[0]
                pred = torch.argmax(probabilities).item()
                confidence = probabilities[pred].item()
            state = "FALL" if pred == 1 and confidence >= 0.80 else "WARNING" if confidence >= 0.60 else "NORMAL"
            now = time.time()
            if now - last_status_time >= status_interval_seconds:
                send_detector_status(state, confidence)
                last_status_time = now
            if pred == 1:
                send_fall_event(confidence)
                cv2.putText(annotated, "FALL DETECTED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(annotated, "NO FALL", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
