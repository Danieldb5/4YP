
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from pathlib import Path

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE.parent / "data" / "BTC_1sec.csv")

price = df["midpoint"].to_numpy()
seconds, threshold = 5, 180
before, after = 100, 5

# Detect price jumps of 20+ within 5 observations
change = np.abs(price[seconds:] - price[:-seconds])
events = np.where(change >= threshold)[0]

# Remove repeated detections of the same jump
tips = []
for i in events:
    if i >= before and i + seconds + after <= len(df):
        if not tips or i - tips[-1] > seconds:
            tips.append(i)

# Save 100 before + 5 during + 5 after
columns = df.select_dtypes(include="number").columns.drop(
    "Unnamed: 0", errors="ignore"
)

X = np.array([
    df.iloc[i-before:i+seconds+after][columns].to_numpy()
    for i in tips
])

# Save to separate BTC file
savemat(BASE / "BTC_tips.mat", {
    "X": X,
    "tip_indices": np.array(tips),
    "tip_times": df["system_time"].iloc[tips].astype(str).to_numpy(),
    "feature_names": np.array(columns.tolist(), dtype=object)
})

print("Total BTC tips:", len(tips))
print("Data shape:", X.shape)

# Plot first 5 events
for i in range(min(5, len(tips))):
    plt.figure(figsize=(9, 3))
    plt.plot(np.arange(-100, 10), X[i, :, columns.get_loc("midpoint")])
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"BTC Event {i+1}")
    plt.xlabel("Observations relative to jump start")
    plt.ylabel("BTC Price")
    plt.grid()
    plt.show()
