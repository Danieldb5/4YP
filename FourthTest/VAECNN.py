
import torch
import numpy as np
from torch import nn
from pathlib import Path
from model import VAELSTM
from data import X_train, X_test, y_train, y_test

BASE = Path(__file__).resolve().parent

# Load your trained VAE encoder
vae = VAELSTM()
vae.load_state_dict(torch.load(BASE / "huang_vaelstm.pt",
                               map_location="cpu", weights_only=True))
vae.eval()

# Encode trajectories into 4 channels x 10 windows
with torch.no_grad():
    Z_train = vae.enc(X_train.reshape(-1,10,10))[:,:,:4].transpose(1,2)
    Z_test = vae.enc(X_test.reshape(-1,10,10))[:,:,:4].transpose(1,2)

# CNN operating on VAE features
model = nn.Sequential(
    nn.Conv1d(4,32,3,padding=1), nn.ReLU(),
    nn.Conv1d(32,64,3,padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64*10,2)
)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
y_train = torch.tensor(y_train, dtype=torch.long)

# Train CNN
for epoch in range(20):
    model.train()

    for batch in torch.randperm(len(Z_train)).split(128):
        pred = model(Z_train[batch])
        loss = loss_fn(pred, y_train[batch])

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Test
model.eval()
with torch.no_grad():
    pred = torch.cat([
        model(batch).argmax(1)
        for batch in Z_test.split(256)
    ]).numpy()

print("VAE-CNN Accuracy:", np.mean(pred == y_test))

torch.save(model.state_dict(), BASE / "huang_vaecnn.pt")
print("VAE-CNN saved!")
