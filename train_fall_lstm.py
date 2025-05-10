import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from lstm_model import LSTMClassifier

# کلاس دیتاست
class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# بارگذاری داده‌ها
print("📥 Loading sequences and labels...")
X = np.load("sequences.npy")
y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=16, shuffle=True)

# تعیین device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚙️ Using device:", device)

# تعریف مدل و پارامترها
model = LSTMClassifier(input_dim=3, hidden_dim=64, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# آموزش
print("🚀 Training started...\n")
for epoch in range(20):
    model.train()
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total * 100
    print(f"📚 Epoch {epoch+1}/20 | Loss: {loss.item():.4f} | Train Accuracy: {acc:.2f}%")

# تست
model.eval()
correct, total = 0, 0
for i in range(len(X_test)):
    input_tensor = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
    label = torch.tensor(y_test[i]).to(device)
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1)
    if pred == label:
        correct += 1
    total += 1

acc = correct / total * 100
print(f"\n📊 Final Test Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "lstm_fall_model.pth")
print("✅ Model saved to lstm_fall_model.pth")
