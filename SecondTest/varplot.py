from pathlib import Path
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve().parent.parent / "Data" / "largesample_case1data.mat"

data = loadmat(FILE)

X = data['ts_analysis'].T
y = data['label'].ravel()

rng = np.random.default_rng(42)

plt.figure(figsize=(10, 5))
for label, colour in [(0, "blue"), (1, "red")]:
    ids = rng.choice(np.where(y == label)[0], 1000, replace=False)
    values = X[ids, 580:680]

    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)

    t = np.arange(580, 680)
    name = "Tip" if label else "No Tip"

    plt.plot(t, mean, color=colour, label=name)
    plt.fill_between(
        t, mean - std, mean + std,
        color=colour, alpha=0.2
    )

plt.xlabel("Observation")
plt.ylabel("System state")
plt.title("Mean Huang Trajectories with Standard Deviation")
plt.legend()
plt.grid()
plt.show()
"""
for label, colour in [(0, "blue"), (1, "red")]:
    ids = rng.choice(np.where(y == label)[0], 1000, replace=False)

    values = X[ids, 580:680]

    # Variance across 1000 trajectories at each observation
    variance = np.var(values, axis=0)

    plt.plot(
        np.arange(580, 680),
        variance,
        color=colour,
        label=f"{'Tip' if label else 'No Tip'}"
    )

plt.xlabel("Observation")
plt.ylabel("Variance of system state")
plt.title("Variance Across 1000 Huang Trajectories")
plt.legend()
plt.grid()
plt.show()
"""