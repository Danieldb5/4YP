
import numpy as np
import torch
from torch import nn
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent

# Load tipping and non-tipping trajectories
tip = loadmat(BASE / "ETH_tips.mat")["X"]
notip = loadmat(BASE / "ETH_notips.mat")["X"]

# Use first 95 observations and all market variables
X = np.concatenate([tip[:, :95, :], notip[:, :95, :]]).astype("float32")
y = np.r_[np.ones(len(tip)), np.zeros(len(notip))].astype("int64")

print("Dataset shape:", X.shape)

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalise each market variable using training data
mu = Xtr.mean(axis=(0,1), keepdims=True)
sd = Xtr.std(axis=(0,1), keepdims=True) + 1e-8

Xtr = torch.tensor(((Xtr-mu)/sd).transpose(0,2,1))
Xte = torch.tensor(((Xte-mu)/sd).transpose(0,2,1))
ytr = torch.tensor(ytr, dtype=torch.long)
yte = torch.tensor(yte, dtype=torch.long)

# Multivariate CNN
model = nn.Sequential(
    nn.Conv1d(Xtr.shape[1],64,3,padding=1), nn.ReLU(),
    nn.AvgPool1d(2),
    nn.Conv1d(64,64,3,padding=1), nn.ReLU(),
    nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    nn.Linear(64,2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(20):
    model.train()

    for batch in torch.randperm(len(Xtr)).split(32):
        loss = loss_fn(model(Xtr[batch]), ytr[batch])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Test
model.eval()
with torch.no_grad():
    pred = torch.cat([
        model(batch).argmax(1) for batch in Xte.split(256)
    ])

print("Accuracy:", (pred == yte).float().mean().item())

# Save trained CNN
torch.save({
    "model_state": model.state_dict(),
    "mean": mu,
    "std": sd,
    "features": Xtr.shape[1]
}, BASE / "ETH_CNN.pt")

print("CNN saved!")
