
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
d = loadmat(BASE.parent / "Data/largesample_case1data.mat")
X, y = d["ts_analysis"].T, d["label"].ravel()

# Use the same 100 observations as your CNN
rng = np.random.default_rng(42)
ids = np.r_[rng.choice(np.where(y==0)[0], 5000, False),
            rng.choice(np.where(y==1)[0], 5000, False)]
X = X[ids, 480:580].astype("float32")
y = y[ids]

X = (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-8)
train, test = train_test_split(np.arange(len(y)), test_size=0.2,
                               stratify=y, random_state=42)
x = torch.tensor(X[:,None,:])

# Existing CNN
cnn = nn.Sequential(
    nn.Conv1d(1,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.Conv1d(64,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64,2)
)
ck = torch.load(BASE/"huang_cnn_scaled.pt", weights_only=True)
cnn.load_state_dict(ck["model_state"])
cnn.eval()

# VAE encodes 10 windows of 10 observations; LSTM models their sequence
class VAELSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(10, 8)   # 4 means + 4 log-variances
        self.dec = nn.Linear(4, 10)
        self.lstm = nn.LSTM(4, 16, batch_first=True)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        w = x.reshape(-1, 10, 10)
        mu, lv = self.enc(w).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        recon = self.dec(z)
        predicted = self.out(self.lstm(mu)[0])
        return recon, predicted, mu, lv

vae = VAELSTM()
opt = torch.optim.Adam(vae.parameters(), lr=0.001)

# Train only on non-tipping examples
normal = train[y[train] == 0]

for epoch in range(20):
    vae.train()
    for batch in torch.randperm(len(normal)).split(128):
        xb = x[normal[batch]]
        recon, predicted, mu, lv = vae(xb)
        kl = -0.5*(1+lv-mu**2-lv.exp()).mean()
        loss = ((recon-xb.reshape(-1,10,10))**2).mean()
        loss += 0.01*kl + ((predicted[:,:-1]-mu[:,1:])**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# CNN classifications and VAE reconstruction errors
cnn.eval()
vae.eval()
with torch.no_grad():
    p = torch.cat([cnn(v).softmax(1)[:,1] for v in x[test].split(256)]).numpy()
    errors = torch.cat([
        ((vae(v)[0]-v.reshape(-1,10,10))**2).mean((1,2))
        for v in x[test].split(256)
    ]).numpy()

actual = y[test]
pred = (p >= 0.5).astype(int)

groups = {
    "TP": (actual==1)&(pred==1),
    "TN": (actual==0)&(pred==0),
    "FP": (actual==0)&(pred==1),
    "FN": (actual==1)&(pred==0)
}

plt.boxplot([errors[g] for g in groups.values()],
            tick_labels=list(groups))
plt.ylabel("VAE reconstruction error")
plt.title("CNN prediction groups vs VAE anomaly score")
plt.show()

for name, g in groups.items():
    print(name, "Count:", g.sum(),
          "Mean error:", errors[g].mean() if g.any() else np.nan)
