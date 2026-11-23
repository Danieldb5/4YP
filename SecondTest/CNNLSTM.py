
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

# Number of observations the CNN sees
a, b = 480, 580
print(a, b)
X = X[:, a:b]

# Select 500 tipping and 500 non-tipping trajectories
ids = np.r_[np.where(y == 0)[0][:50000],
            np.where(y == 1)[0][:50000]]

X, y = X[ids].astype("float32"), y[ids]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

mu, sd = X_train.mean(), X_train.std() + 1e-8
X_train = torch.tensor(((X_train-mu)/sd)[:, None, :])
X_test = torch.tensor(((X_test-mu)/sd)[:, None, :])


class TipLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        output, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])

model = TipLSTM()

# CNN
"""
model = nn.Sequential(
    nn.Conv1d(1, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AvgPool1d(2),

    nn.Conv1d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AvgPool1d(2),

    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(64, 2)
)
"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(20):
    model.train()

    for i in range(0, len(X_train), 32):
        xb = X_train[i:i+32]
        yb = torch.tensor(y_train[i:i+32], dtype=torch.long)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Testing
model.eval()

with torch.no_grad():
    probabilities = torch.softmax(model(X_test), dim=1)
    predictions = probabilities.argmax(dim=1).numpy()

print("Accuracy:", np.mean(predictions == y_test))

"""
torch.save({
    "model_state": model.state_dict(),
    "mean": float(mu),
    "std": float(sd),
    "start": a,
    "end": b
}, Path(__file__).resolve().parent / "huang_cnn_gradcam580.pt")

print("CNN model saved!")
"""