
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from pathlib import Path

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE.parent / "data" / "ETH_1sec.csv")

price = df["midpoint"].to_numpy()
seconds, threshold = 5, 6
before, after = 100, 5
print("Total seconds:", len(df))

# Find price movements of 20+ within 5 observations
change = np.abs(price[seconds:] - price[:-seconds])
events = np.where(change >= threshold)[0]

# Avoid counting the same jump repeatedly
tips = []
for i in events:
    if i >= before and i + seconds + after < len(df):
        if not tips or i - tips[-1] > seconds:
            tips.append(i)

# Extract 100 before + 5 during jump + 5 after
columns = df.select_dtypes(include="number").columns.drop(
    "Unnamed: 0", errors="ignore"
)

X = np.array([
    df.iloc[i-before:i+seconds+after][columns].to_numpy()
    for i in tips
])

savemat(BASE / "ETH_tips.mat", {
    "X": X,
    "tip_indices": np.array(tips),
    "tip_times": df["system_time"].iloc[tips].astype(str).to_numpy(),
    "feature_names": np.array(columns.tolist(), dtype=object)
})

print("Total tips detected:", len(tips))
print("Data shape:", X.shape)

# Plot first 5 detected events
for i in range(min(5, len(tips))):
    plt.figure(figsize=(9, 3))
    plt.plot(np.arange(-before, seconds+after), X[i,:,columns.get_loc("midpoint")])
    plt.axvline(0, color="red", linestyle="--", label="Jump window starts")
    plt.title(f"ETH Event {i+1}")
    plt.xlabel("Observations relative to jump start")
    plt.ylabel("ETH Price")
    plt.legend()
    plt.show()
