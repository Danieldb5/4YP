from pathlib import Path
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import random

# Your MATLAB file
FILE = Path(__file__).resolve().parent.parent / "Data" / "largesample_case1data.mat"

# Load MATLAB data
data = loadmat(FILE)

X = data['ts_analysis'].T
y = data['label'].ravel()



rng = np.random.default_rng(42)

plt.figure(figsize=(10, 5))

for label, colour in [(0, "blue"), (1, "red")]:
    ids = rng.choice(np.where(y == label)[0], 1000, replace=False)
    values = X[ids, 580:680]
    plt.plot(np.arange(580, 680), values.sum(axis=0),
             color=colour, label=f"{'Tip' if label else 'No Tip'}")

plt.xlabel("Observation")
plt.ylabel("Summed system state")
plt.title("100 Huang Trajectories (1 Tip, 1 No Tip)")
plt.legend()
plt.grid()
plt.show()
