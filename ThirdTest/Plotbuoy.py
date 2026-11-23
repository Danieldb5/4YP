
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "Data" / "data" / "minibuoy" / "lothianwest" / "mb12.csv"

data = pd.read_csv(FILE)
print(data.head())

time = pd.to_datetime(data["DATA"], dayfirst=True)
values = data["G"]

plt.figure(figsize=(12, 5))
plt.plot(time, values, linewidth=0.5)
plt.xlabel("Date / Time")
plt.ylabel("G")
plt.title("Mangrove Mini Buoy 12 - Sensor Data")
plt.grid()
plt.tight_layout()
plt.show()
