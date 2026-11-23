
import torch
from torch import nn

class VAELSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(10, 8)
        self.dec = nn.Linear(4, 10)
        self.lstm = nn.LSTM(4, 16, batch_first=True)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        w = x.reshape(-1, 10, 10)
        mu, lv = self.enc(w).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        recon = self.dec(z)
        future = self.out(self.lstm(mu)[0])
        return recon, future, mu, lv
