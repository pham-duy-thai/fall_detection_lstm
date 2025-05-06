import cv2
import numpy as np
import torch
from ultralytics import YOLO
from lstm_model import LSTMClassifier

# تنظیمات اصلی
model_path = "lstm_fall_model.pth"
video_path = "videos/nofall_18.mp4"  # یا مسیر ویدیو تست
use_webcam = False                # True برای تست با وب‌کم
window_size = 40

# انتخاب device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# بارگذاری مدل‌ها
print("📦 Loading YOLO pose model and LSTM classifier...")
pose_model = YOLO("yolov8n-pose.pt")
model = LSTMClassifier(input_dim=3, hidden_dim=64, output_dim=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Models loaded.")

# ویدیو یا وب‌کم
cap = cv2.VideoCapture(0 if use_webcam else video_path)
sequence = []

# تابع استخراج ویژگی‌ها از keypoints
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

# شروع پردازش فریم‌ها
print("▶️ Starting detection...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ End of video or cannot read frame.")
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
                cv2.putText(annotated, "No Fall", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection finished.")
