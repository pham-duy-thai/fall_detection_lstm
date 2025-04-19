import cv2
import numpy as np
import torch
from ultralytics import YOLO
from lstm_model import LSTMClassifier

# === Config ===
model_path = "lstm_fall_model.pth"
video_path = "videos/input.mp4"  # اسم پوشه به صورت lowercase باشه
use_webcam = True               # True = استفاده از وب‌کم
window_size = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Models ===
pose_model = YOLO("yolov8n-pose.pt")
model = LSTMClassifier(input_dim=2, hidden_dim=64, output_dim=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

sequence = []
cap = cv2.VideoCapture(0 if use_webcam else video_path)

print("🔄 Running fall detection... Press 'q' to exit.")

def extract_features(kp):
    try:
        s = kp[5]  # Shoulder
        h = kp[11] # Hip
        dx = h[0] - s[0]
        dy = h[1] - s[1]
        angle = np.degrees(np.arctan2(dy, dx))
        dist = np.linalg.norm(h - s)
        return [angle, dist]
    except:
        return [90, 0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame)
    annotated = results[0].plot()

    # فقط با اولین فرد در تصویر کار کن (نتیجه‌ی مطمئن‌تر)
    if results[0].keypoints is not None:
        kp = results[0].keypoints.xy[0]  # فقط نفر اول
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

    cv2.imshow("Fall Detection LSTM", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
