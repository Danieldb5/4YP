
import numpy as np
import torch
from torch import nn
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
OBS = 75

tip = loadmat(BASE / "ETH_tips.mat")["X"]
notip = loadmat(BASE / "ETH_notips.mat")["X"]

X = np.concatenate([tip[:, :OBS, :], notip[:, :OBS, :]]).astype("float32")
y = np.r_[np.ones(len(tip)), np.zeros(len(notip))].astype("int64")

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

mu = Xtr.mean((0,1), keepdims=True)
sd = Xtr.std((0,1), keepdims=True) + 1e-6

Xtr = torch.tensor((Xtr-mu)/sd)
Xte = torch.tensor((Xte-mu)/sd)
ytr = torch.tensor(ytr, dtype=torch.long)

class VAECNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.enc = nn.Linear(features*5, 64)
        self.dec = nn.Linear(32, features*5)

        self.cnn = nn.Sequential(
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(),
            nn.Conv1d(64,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64,2)
        )

    def forward(self, x):
        b, t, f = x.shape
        w = x.reshape(b, t//5, 5*f)
        mean, logvar = self.enc(w).chunk(2,-1)
        z = mean + torch.randn_like(mean)*torch.exp(0.5*logvar)
        recon = self.dec(z)
        pred = self.cnn(mean.transpose(1,2))
        return recon, pred, mean, logvar

model = VAECNN(Xtr.shape[2])
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)

for epoch in range(30):
    model.train()
    losses = []

    for batch in torch.randperm(len(Xtr)).split(32):
        xb, yb = Xtr[batch], ytr[batch]
        recon, pred, mean, logvar = model(xb)

        loss = nn.functional.cross_entropy(pred,yb)
        loss += 0.01*nn.functional.mse_loss(
            recon, xb.reshape(len(xb),-1,5*xb.shape[2])
        )
        loss += 0.001*(-0.5*(1+logvar-mean**2-logvar.exp()).mean())

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Epoch {epoch+1}: Loss {np.mean(losses):.4f}")

model.eval()
with torch.no_grad():
    pred = torch.cat([
        model(xb)[1].argmax(1) for xb in Xte.split(128)
    ]).numpy()

print("Improved VAE-CNN Accuracy:", np.mean(pred == yte))

torch.save({
    "model_state": model.state_dict(),
    "mean": mu,
    "std": sd,
    "features": Xtr.shape[2],
    "observations": OBS
}, BASE / "ETH_VAECNN_improved.pt")

print("Improved model saveds!")
