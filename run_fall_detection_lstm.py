import cv2
import numpy as np
import torch
from ultralytics import YOLO
from lstm_model import LSTMClassifier

# تنظیمات
model_path = "lstm_fall_model.pth"
video_path = "videos_test/test_fall1.mp4"  # اگر use_webcam=False باشه از این استفاده میشه
use_webcam = True
window_size = 40

# دستگاه اجرا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚙️ Running on device:", device)

# بارگذاری مدل‌ها
pose_model = YOLO("yolov8n-pose.pt")
model = LSTMClassifier(input_dim=3, hidden_dim=64, output_dim=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

sequence = []
cap = cv2.VideoCapture(0 if use_webcam else video_path)

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
                pred = torch.argmax(out, dim=1).item()
            if pred == 1:
                cv2.putText(annotated, "FALL DETECTED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(annotated, "NO FALL", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
