
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

BASE = Path(__file__).resolve().parent
data = loadmat(BASE / "ETH_tips.mat")

X = data["X"]
names = [str(n[0]) for n in data["feature_names"].ravel()]
price = names.index("midpoint")

print("Total trajectories:", len(X))
print("Shape:", X.shape)

# Plot 5 trajectories, one at a time
for i in range(min(3, len(X))):
    plt.figure(figsize=(10, 4))
    plt.plot(range(-100, 10), X[i, :, price])
    plt.axvline(0, color="red", linestyle="--", label="Jump window starts")
    plt.title(f"Trajectory {i+1} of {len(X)}")
    plt.xlabel("Observations relative to jump")
    plt.ylabel("ETH Price")
    plt.legend()
    plt.grid()
    plt.show()
