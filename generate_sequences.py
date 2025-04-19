import numpy as np

keypoints = np.load("keypoints.npy")

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

window_size = 30
stride = 5
sequences = []
labels = []

for i in range(0, len(features) - window_size, stride):
    window = features[i:i+window_size]
    sequences.append(window)
    labels.append(0)  # فعلاً برچسب همه صفر

np.save("sequences.npy", np.array(sequences))
np.save("labels.npy", np.array(labels))
print("Sequences and labels saved.")
