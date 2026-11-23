
import numpy as np
import torch
from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve().parent.parent / "Data" / "largesample_case1data.mat"
data = loadmat(FILE)

X, y = data["ts_analysis"].T, data["label"].ravel()

ids = np.r_[np.where(y==0)[0][:5000],
            np.where(y==1)[0][:5000]]

X, y = X[ids,480:580].astype("float32"), y[ids]
X = (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-8)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = torch.tensor(X_train[:,None,:])
X_test = torch.tensor(X_test[:,None,:])
