
import numpy as np
import torch
from torch import nn
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve().parent.parent / "Data" / "largesample_case1data.mat"
data = loadmat(FILE)

X = data["ts_analysis"].T
y = data["label"].ravel()

# Observations the CNN sees
a, b = 580, 680
X = X[:, a:b]

# Select 50,000 tipping and 50,000 non-tipping
rng = np.random.default_rng(42)
ids = np.r_[
    rng.choice(np.where(y == 0)[0], 50000, replace=False),
    rng.choice(np.where(y == 1)[0], 50000, replace=False)
]

X, y = X[ids].astype("float32"), y[ids]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# NORMALISE EACH TRAJECTORY INDIVIDUALLY
def normalise(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd

X_train = torch.tensor(normalise(X_train)[:, None, :])
X_test = torch.tensor(normalise(X_test)[:, None, :])

# CNN
model = nn.Sequential(
    nn.Conv1d(1, 64, 3, padding=1),
    nn.ReLU(),
    nn.AvgPool1d(2),

    nn.Conv1d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.AvgPool1d(2),

    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(64, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(20):
    model.train()
    total_loss = 0

    for batch in torch.randperm(len(X_train)).split(128):
        xb = X_train[batch]
        yb = torch.tensor(y_train[batch], dtype=torch.long)

        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)

    print(f"Epoch {epoch+1}: Loss {total_loss/len(X_train):.4f}")

# Testing
model.eval()
predictions = []

with torch.no_grad():
    for xb in X_test.split(256):
        predictions.append(model(xb).argmax(dim=1))

predictions = torch.cat(predictions).numpy()
print("Accuracy:", np.mean(predictions == y_test))

# Save separately from your original CNN
torch.save({
    "model_state": model.state_dict(),
    "normalisation": "per_trajectory",
    "start": a,
    "end": b
}, Path(__file__).resolve().parent / "huang_cnn_scaled.pt")

print("Scaled CNN model saved!")
