import numpy as np
import os

# ---------- تابع استخراج ویژگی از keypoints ----------
def extract_features_from_kp(kp):
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
        return [90, 0, 0]  # fallback in case of bad data

# ---------- تنظیمات ----------
data_dir = "data"
window_size = 40
stride = 5

all_sequences = []
all_labels = []

# ---------- پردازش فایل‌های .npy ----------
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".npy"):
        continue

    label = 1 if "keypoints_fall" in filename else 0
    file_path = os.path.join(data_dir, filename)

    try:
        keypoints = np.load(file_path)
        features = [extract_features_from_kp(kp) for kp in keypoints]

        for i in range(0, len(features) - window_size + 1, stride):
            window = features[i:i+window_size]
            all_sequences.append(window)
            all_labels.append(label)
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# ---------- ذخیره خروجی ----------
X = np.array(all_sequences)
y = np.array(all_labels)

np.save("sequences.npy", X)
np.save("labels.npy", y)

# ---------- گزارش نهایی ----------
unique, counts = np.unique(y, return_counts=True)
print("✅ Label distribution:", dict(zip(unique, counts)))
print(f"✅ Saved {len(X)} sequences and {len(y)} labels to sequences.npy and labels.npy")
