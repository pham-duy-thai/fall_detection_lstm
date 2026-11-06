import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import LSTMClassifier

DATA = Path(os.environ.get("CSV_SEQUENCE_DIR", "csv_sequences"))
OUT = os.environ.get("CSV_MODEL_OUTPUT", "lstm_fall_model_fallandnofall.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(split):
    d = np.load(DATA / f"{split}.npz")
    return torch.tensor(d["X"], dtype=torch.float32), torch.tensor(d["y"], dtype=torch.long)

X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
model = LSTMClassifier(input_dim=3, hidden_dim=64, output_dim=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def accuracy(x, y):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(x, y), batch_size=256):
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += int((pred == yb).sum()); total += len(yb)
    return 100 * correct / total

best = -1
for epoch in range(20):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb.to(device)), yb.to(device)).backward(); opt.step()
    val = accuracy(X_val, y_val)
    print(f"epoch {epoch + 1}/20 - val accuracy: {val:.2f}%")
    if val > best:
        best = val; torch.save(model.state_dict(), OUT)

model.load_state_dict(torch.load(OUT, map_location=device))
print(f"best val accuracy: {best:.2f}%")
print(f"test accuracy: {accuracy(X_test, y_test):.2f}%")
print(f"saved: {OUT}")
