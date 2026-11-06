import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from lstm_model import LSTMClassifier

# Dataset definition
class FallDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load data
print("📥 Loading sequences and labels...")
sequence_path = os.environ.get("SEQUENCE_INPUT", "sequences.npy")
label_path = os.environ.get("LABEL_INPUT", "labels.npy")
model_path = os.environ.get("MODEL_OUTPUT", "lstm_fall_model.pth")
X = np.load(sequence_path)
y = np.load(label_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=16, shuffle=True)

# Define model
model = LSTMClassifier(input_dim=3, hidden_dim=64, output_dim=2)

# Load existing model if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(model_path) and os.environ.get("RESUME_TRAINING", "0") == "1":
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("🔁 Previous model loaded. Continuing training...")

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("🚀 Training started...\n")
for epoch in range(20):
    model.train()
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total * 100
    print(f"📚 Epoch {epoch+1}/20 | Loss: {loss.item():.4f} | Train Accuracy: {train_acc:.2f}%")

# Evaluate
model.eval()
correct, total = 0, 0
for i in range(len(X_test)):
    input_tensor = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
    label = y_test[i]
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()
    if pred == label:
        correct += 1
    total += 1

acc = correct / total * 100
print(f"\n📊 Final Test Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), model_path)
print("✅ Model saved to lstm_fall_model.pth")
