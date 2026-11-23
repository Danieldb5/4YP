
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from scipy.io import loadmat
from pathlib import Path

BASE = Path(__file__).resolve().parent
d = loadmat(BASE.parent / "Data/largesample_case1data.mat")
X, y = d["ts_analysis"].T, d["label"].ravel()

model = nn.Sequential(
    nn.Conv1d(1,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.Conv1d(64,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64,2)
)
ck = torch.load(BASE/"huang_cnn_gradcam680.pt", weights_only=True)
model.load_state_dict(ck["model_state"])
model.eval()

ids = np.r_[np.where(y==0)[0][:500], np.where(y==1)[0][:500]]
accuracy = []

for a in range(0,581,10):
    x = torch.tensor(((X[ids,a:a+100]-ck["mean"])/ck["std"])[:,None,:], dtype=torch.float32)
    with torch.no_grad():
        pred = torch.cat([model(b).argmax(1) for b in x.split(256)]).numpy()
    accuracy.append(np.mean(pred == y[ids]))

plt.plot(range(0,581,10),accuracy)
plt.axhline(0.5,color="red",linestyle="--")
plt.xlabel("Starting observation")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
