import os
import numpy as np

fall_path = "features/fall"
nofall_path = "features/nofall"

X = []
y = []

# توالی‌های سقوط = برچسب 1
for file in os.listdir(fall_path):
    if file.endswith(".npy"):
        path = os.path.join(fall_path, file)
        data = np.load(path)
        X.append(data)
        y.append(1)

# توالی‌های نُفول = برچسب 0
for file in os.listdir(nofall_path):
    if file.endswith(".npy"):
        path = os.path.join(nofall_path, file)
        data = np.load(path)
        X.append(data)
        y.append(0)

X = np.array(X)
y = np.array(y)

np.save("sequences.npy", X)
np.save("labels.npy", y)

print("✅ Saved sequences.npy and labels.npy")
print(f"📦 X shape: {X.shape} | y shape: {y.shape}")
