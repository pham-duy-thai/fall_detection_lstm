import numpy as np
import os

keypoint_file = "keypoints.npy"

if not os.path.exists(keypoint_file):
    print("❌ Error: keypoints.npy file not found.")
    exit()

try:
    keypoints = np.load(keypoint_file)
    print(f"✅ Loaded {len(keypoints)} keypoints from '{keypoint_file}'")
except Exception as e:
    print(f"❌ Failed to load keypoints: {e}")
    exit()

def extract_features_from_kp(kp):
    try:
        s = kp[5]
        h = kp[11]
        dx = h[0] - s[0]
        dy = h[1] - s[1]
        angle = np.degrees(np.arctan2(dy, dx))
        dist = np.linalg.norm(h - s)
        return [angle, dist]
    except:
        return [90, 0]

features = [extract_features_from_kp(kp) for kp in keypoints]
features = np.array(features)
print(f"✅ Extracted {len(features)} feature vectors.")

window_size = 30
stride = 5
sequences = []
labels = []

for i in range(0, len(features) - window_size, stride):
    window = features[i:i+window_size]
    sequences.append(window)
    labels.append(0)  # Label 0 = Not Fall

if not sequences:
    print("⚠️ No sequences were created. Try adjusting window size or stride.")
    exit()

# ذخیره نهایی
np.save("sequences.npy", np.array(sequences))
np.save("labels.npy", np.array(labels))
print(f"✅ Saved {len(sequences)} sequences and {len(labels)} labels.")
print("✅ Sequence generation completed successfully.")
