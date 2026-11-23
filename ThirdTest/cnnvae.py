
import numpy as np
import torch
from torch import nn
from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve().parent.parent / "Data/largesample_case1data.mat"
data = loadmat(FILE)
X, y = data["ts_analysis"].T, data["label"].ravel()

ids = np.r_[np.where(y==0)[0][:5000], np.where(y==1)[0][:5000]]
X, y = X[ids,580:680].astype("float32"), y[ids]
X = (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-8)

Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
Xtr = torch.tensor(Xtr[:,None,:])
Xte = torch.tensor(Xte[:,None,:])
ytr = torch.tensor(ytr,dtype=torch.long)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),
                                 nn.AvgPool1d(2),nn.Flatten(),nn.Linear(800,16))
        self.dec = nn.Linear(8,100)
        self.cls = nn.Linear(8,2)

    def forward(self,x):
        mu, logvar = self.enc(x).chunk(2,dim=1)
        z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
        return self.dec(z), self.cls(mu), mu, logvar

model = VAE()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(20):
    model.train()
    for batch in torch.randperm(len(Xtr)).split(128):
        xb, yb = Xtr[batch], ytr[batch]
        recon, pred, mu, logvar = model(xb)
        kl = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
        loss = nn.functional.mse_loss(recon,xb[:,0]) + nn.functional.cross_entropy(pred,yb) + 0.01*kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch+1, loss.item())

model.eval()
with torch.no_grad():
    pred = model(Xte)[1].argmax(1).numpy()
print("Accuracy:",np.mean(pred==yte))
