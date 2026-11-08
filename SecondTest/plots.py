from pathlib import Path
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import random

# Your MATLAB file
FILE = Path(__file__).resolve().parent.parent / "Data" / "largesample_case1data.mat"

# Load MATLAB data
data = loadmat(FILE)

x = data['ts_analysis'].T
L = data['label'].ravel()


n= random.randint(1, 100000)
print(n)
n = 90000
plt.plot(x[n, 580 : 680])
print(L[n])
plt.show()