import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from lstm_model import LSTMClassifier
import os

# === FallDataset class ===
class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Load data ===
if not os.path.exists("sequences.npy") or not os.path.exists("labels.npy"):
    print("❌ Error: sequences.npy or labels.npy file not found.")
    exit()

try:
    X = np.load("sequences.npy")
    y = np.load("labels.npy")
    print(f"✅ Loaded sequences: {X.shape}, labels: {y.shape}")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit()

# === Split and prepare DataLoader ===
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=16, shuffle=True)
    print("✅ Training data prepared.")
except Exception as e:
    print(f"❌ Failed to prepare dataset: {e}")
    exit()

# === Define model ===
model = LSTMClassifier(input_dim=2, hidden_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Training ===
try:
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        print(f"✅ Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")
    print("✅ Training completed successfully.")
except Exception as e:
    print(f"❌ Training failed: {e}")
    exit()

# === Save model ===
try:
    torch.save(model.state_dict(), "lstm_fall_model.pth")
    print("✅ Model saved as 'lstm_fall_model.pth'")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
