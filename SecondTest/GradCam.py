
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from scipy.io import loadmat
from pathlib import Path

BASE = Path(__file__).resolve().parent
data = loadmat(BASE.parent / "Data" / "largesample_case1data.mat")
X, y = data["ts_analysis"].T, data["label"].ravel()

model = nn.Sequential(
    nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 2)
)

ckpt = torch.load(BASE / "huang_cnn_gradcam580.pt", weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()

i = 70400
x = X[i, ckpt["start"]:ckpt["end"]]
inp = torch.tensor(((x - ckpt["mean"]) / ckpt["std"])[None, None, :], dtype=torch.float32)

features = {}
model[3].register_forward_hook(lambda m, a, o: features.update(A=o))
scores = model(inp)
features["A"].retain_grad()
scores[0, 0].backward()

A = features["A"]
cam = torch.relu((A.grad.mean(2, keepdim=True) * A).sum(1)).detach().numpy()[0]
cam = np.interp(np.linspace(0, len(cam)-1, len(x)), np.arange(len(cam)), cam)
cam /= cam.max() + 1e-8

plt.figure(figsize=(10, 4))
plt.plot(x, color="black")
plt.plot(np.arange(len(x)), cam * x.max(), color="red", alpha=0.5)


plt.title(f"Trajectory {i} | Label {int(y[i])}")
plt.xlabel("Observation")
plt.ylabel("System state")
plt.show()
