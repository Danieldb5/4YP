
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
    nn.Conv1d(1,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.Conv1d(64,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64,2)
)

ckpt = torch.load(BASE / "huang_cnn_scaled.pt",
                  map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()

i = 70400
x = X[i, ckpt["start"]:ckpt["end"]].astype("float32")

# Same per-trajectory normalisation used during training
inp = torch.tensor(((x-x.mean())/(x.std()+1e-8))[None,None,:])

features = {}
model[3].register_forward_hook(lambda m,a,o: features.update(A=o))

scores = model(inp)
A = features["A"]
A.retain_grad()
scores[0,0].backward()  # 0 = No Tip, 1 = Tip

cam = torch.relu(
    (A.grad.mean(2,keepdim=True)*A).sum(1)
).detach().numpy()[0]

cam = np.interp(np.linspace(0,len(cam)-1,len(x)),
                np.arange(len(cam)),cam)
cam /= cam.max()+1e-8

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x, color="black")
ax.set_ylabel("System state")

ax2 = ax.twinx()
ax2.plot(cam, color="red", alpha=0.7)
ax2.set_ylabel("Grad-CAM relevance")
ax2.set_ylim(0,1.05)

p = torch.softmax(scores,dim=1)[0].detach().numpy()
plt.title(f"Trajectory {i} | Actual {int(y[i])} | Predicted {p.argmax()} | P(Tip) {p[1]:.2f}")
ax.set_xlabel("Observation")
plt.show()
