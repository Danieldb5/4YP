
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import VAELSTM
from data import X_test, y_test

BASE = Path(__file__).resolve().parent

# Load trained VAE-LSTM
vae = VAELSTM()
vae.load_state_dict(torch.load(BASE / "huang_vaelstm.pt",
                               map_location="cpu", weights_only=True))
vae.eval()

# Calculate anomaly scores
results = []

with torch.no_grad():
    for x in X_test.split(256):
        recon, future, mu, lv = vae(x)
        w = x.reshape(-1, 10, 10)

        reconstruction = ((vae.dec(mu)-w)**2).mean((1,2))
        temporal = ((future[:,:-1]-mu[:,1:])**2).mean((1,2))
        decoded = vae.dec(future[:,:-1])
        lin = torch.linalg.vector_norm(decoded-w[:,1:], dim=2).sum(1)

        results.append(torch.stack([reconstruction, temporal, lin], dim=1))

scores = torch.cat(results).numpy()
y = np.asarray(y_test).ravel()

# Compare Tip vs No Tip
names = ["VAE reconstruction", "LSTM temporal", "Lin prediction"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for i, name in enumerate(names):
    axes[i].boxplot(
        [scores[y==0, i], scores[y==1, i]],
        tick_labels=["No Tip", "Tip"]
    )
    axes[i].set_title(name)
    axes[i].set_ylabel("Error")

plt.tight_layout()
plt.show()

# Print results
for i, name in enumerate(names):
    print(name)
    print("No Tip:", round(scores[y==0, i].mean(), 4))
    print("Tip:", round(scores[y==1, i].mean(), 4))
