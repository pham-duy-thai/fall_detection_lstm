import numpy as np
import os

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
        return [90, 0, 0]

def build_sequences(path, label, window=40, stride=5):
    try:
        kps = np.load(path)
        feats = [extract_features_from_kp(kp) for kp in kps]
        seqs, labels = [], []
        for i in range(0, len(feats) - window, stride):
            seqs.append(feats[i:i+window])
            labels.append(label)
        return np.array(seqs), np.array(labels)
    except Exception as e:
        print(f"❌ Failed {path}: {e}")
        return np.array([]), np.array([])

X_all, y_all = [], []
data_dir = os.environ.get("KEYPOINT_OUTPUT_DIR", "data")
sequence_output = os.environ.get("SEQUENCE_OUTPUT", "sequences.npy")
label_output = os.environ.get("LABEL_OUTPUT", "labels.npy")
for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)
    if file.startswith("keypoints_fall"):
        X, y = build_sequences(path, 1)
    elif file.startswith("keypoints_nofall"):
        X, y = build_sequences(path, 0)
    else:
        continue
    if len(X):
        X_all.append(X)
        y_all.append(y)

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)
np.save(sequence_output, X_all)
np.save(label_output, y_all)

print(f"✅ Total sequences: {len(X_all)}")
print(f"✅ Label distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")
