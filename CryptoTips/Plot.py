
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "data" / "ETH_1sec.csv"
df = pd.read_csv(FILE)

x = df["midpoint"].to_numpy()

length = 100
trajectories = np.array([
    x[i:i+length] for i in range(0, len(x)-length+1, length)
])

print("Trajectories shape:", trajectories.shape)

for i in range(min(20, len(trajectories))):
    plt.plot(trajectories[i], label=f"Trajectory {i}")

plt.xlabel("Observation (seconds)")
plt.ylabel("ETH price")
plt.legend()
plt.show()
