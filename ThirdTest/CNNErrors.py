
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from scipy.io import loadmat
from pathlib import Path

BASE = Path(__file__).resolve().parent
data = loadmat(BASE.parent / "Data" / "largesample_case1data.mat")
X, y = data["ts_analysis"].T, data["label"].ravel()

model = nn.Sequential(
    nn.Conv1d(1,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.Conv1d(64,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64,2)
)

ck = torch.load(BASE / "huang_cnn_scaled.pt",
                map_location="cpu", weights_only=True)
model.load_state_dict(ck["model_state"])
model.eval()

a, b = ck["start"], ck["end"]
rng = np.random.default_rng(42)

# Select 500 trajectories from each class
ids = np.r_[
    rng.choice(np.where(y == 0)[0], 500, replace=False),
    rng.choice(np.where(y == 1)[0], 500, replace=False)
]

raw = X[ids, a:b].astype("float32")
scaled = (raw-raw.mean(1,keepdims=True))/(raw.std(1,keepdims=True)+1e-8)
inputs = torch.tensor(scaled[:,None,:])

# CNN predictions
with torch.no_grad():
    scores = torch.cat([
        model(batch).softmax(1)[:,1]
        for batch in inputs.split(256)
    ]).numpy()

pred = (scores >= 0.5).astype(int)
actual = y[ids]

groups = {
    "TP": np.where((actual == 1) & (pred == 1))[0],
    "TN": np.where((actual == 0) & (pred == 0))[0],
    "FP": np.where((actual == 0) & (pred == 1))[0],
    "FN": np.where((actual == 1) & (pred == 0))[0]
}

for name, group in groups.items():
    print(f"{name}: {len(group)}")
    print("Trajectory IDs:", ids[group][:10])

# Grad-CAM
features = {}
model[3].register_forward_hook(lambda m, inp, out: features.update(A=out))

def get_cam(j, target):
    model.zero_grad()
    output = model(inputs[j:j+1])
    A = features["A"]
    A.retain_grad()
    output[0,target].backward()

    cam = torch.relu(
        (A.grad.mean(2,keepdim=True)*A).sum(1)
    ).detach().numpy()[0]

    cam = np.interp(
        np.linspace(0,len(cam)-1,b-a),
        np.arange(len(cam)),cam
    )
    return cam/(cam.max()+1e-8)

# Plot 3 individual trajectories per group
for name, group in groups.items():
    for j in group[:3]:
        cam = get_cam(j, int(pred[j]))
        t = np.arange(a,b)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(t, raw[j], color="black")
        ax.set_xlabel("Observation")
        ax.set_ylabel("System state")

        ax2 = ax.twinx()
        ax2.plot(t, cam, color="red", alpha=0.7)
        ax2.set_ylim(0,1.05)
        ax2.set_ylabel("Grad-CAM")

        plt.title(f"{name} | ID {ids[j]} | P(Tip)={scores[j]:.2f}")
        plt.show()

# Compare mean and variance of all four groups
fig, axes = plt.subplots(2,2,figsize=(12,8))
colours = {"TP":"green","TN":"blue","FP":"red","FN":"orange"}

for name, group in groups.items():
    if len(group) == 0:
        continue

    values = raw[group]
    target = 1 if name in ("TP","FP") else 0
    cams = np.array([get_cam(j,target) for j in group])

    axes[0,0].plot(np.arange(a,b), values.mean(0),
                   label=name, color=colours[name])
    axes[0,1].plot(np.arange(a,b), values.var(0),
                   label=name, color=colours[name])
    axes[1,0].plot(np.arange(a,b), cams.mean(0),
                   label=name, color=colours[name])
    axes[1,1].plot(np.arange(a,b), cams.var(0),
                   label=name, color=colours[name])

for ax, title in zip(axes.flat, [
    "Mean trajectory", "Trajectory variance",
    "Mean Grad-CAM", "Grad-CAM variance"
]):
    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
