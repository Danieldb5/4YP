import torch
from model import VAELSTM
from data import X_train, y_train

vae = VAELSTM()
opt = torch.optim.Adam(vae.parameters(), lr=0.001)

normal = X_train[y_train == 0]

for epoch in range(20):
    for xb in normal.split(128):
        recon, future, mu, lv = vae(xb)
        w = xb.reshape(-1, 10, 10)

        loss = ((recon-w)**2).mean()
        loss += 0.01*(-0.5*(1+lv-mu**2-lv.exp()).mean())
        loss += ((future[:,:-1]-mu[:,1:])**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(epoch+1, loss.item())

torch.save(vae.state_dict(), "huang_vaelstm.pt")
