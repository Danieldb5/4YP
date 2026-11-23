
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from scipy.io import loadmat
from pathlib import Path

BASE = Path(__file__).resolve().parent
FILE = BASE.parent / "Data" / "largesample_case1data.mat"
data = loadmat(FILE)
X, y = data["ts_analysis"].T, data["label"].ravel()

model = nn.Sequential(
    nn.Conv1d(1,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.Conv1d(64,64,3,padding=1), nn.ReLU(), nn.AvgPool1d(2),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64,2)
)

ck = torch.load(BASE / "huang_cnn_gradcam580.pt", weights_only=True)
model.load_state_dict(ck["model_state"])
model.eval()

rng = np.random.default_rng(42)
ids = np.r_[rng.choice(np.where(y==0)[0],500,False),
            rng.choice(np.where(y==1)[0],500,False)]

features = {}
model[3].register_forward_hook(lambda m,a,o: features.update(A=o))
sums = np.zeros((2,2,ck["end"]-ck["start"]))

cams = np.zeros((2, 2, 50, ck["end"] - ck["start"]))
counts = [0, 0]

for n, i in enumerate(ids):
    x = X[i,ck["start"]:ck["end"]].astype("float32")
    inp = torch.tensor(((x-ck["mean"])/ck["std"])[None,None,:])
    scores = model(inp)
    A = features["A"]
    A.retain_grad()

    for c in (0,1):
        model.zero_grad()
        scores[0,c].backward(retain_graph=True)
        cam = torch.relu((A.grad.mean(2,keepdim=True)*A).sum(1)).detach().numpy()[0]
        cam = np.interp(np.linspace(0,len(cam)-1,len(x)),np.arange(len(cam)),cam)
        sums[int(y[i]),c] += cam/(cam.max()+1e-8)
        A.grad = None

    if (n+1)%10 == 0:
        print(f"{n+1}/100 complete", flush=True)

for actual in (0,1):
    for target in (0,1):
        plt.plot(sums[actual,target],
                 label=f"Actual {actual}, CAM {target}")

plt.xlabel("Observation")
plt.ylabel("Summed CAM relevance")
plt.legend()
plt.grid()
plt.show()
