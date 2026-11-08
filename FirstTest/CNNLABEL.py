print(1)

from pathlib import Path
from itertools import chain
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / 'extracted_data'
TS_FILE = DATA / 'ts_analysis.csv'
LABEL_FILE = DATA / 'label.csv'
CACHE = BASE / 'huang_label_cnn_cache_v2.npz'
OUTPUT = BASE / 'cnn_tip_predictions.csv'
MODEL = BASE / 'huang_label_cnn.pt'
PER_CLASS = 500
N_POINTS = 512
EPOCHS = 25
BATCH_SIZE = 32
SEED = 42
REUSE_CACHE = True
PRINT_EVERY = 20


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def header_like(row):
    if len(row) < 3:
        return False
    try:
        return all(float(row[j]) == j for j in range(min(8, len(row))))
    except ValueError:
        return True


def load_data():
    if REUSE_CACHE and CACHE.exists():
        with np.load(CACHE) as d:
            if len(d['y']) == 2 * PER_CLASS and d['X'].shape[1] == N_POINTS:
                log('Using cached training data')
                return d['X'], d['y'], d['ids']
    if not TS_FILE.exists() or not LABEL_FILE.exists():
        raise FileNotFoundError(f'Expected {TS_FILE} and {LABEL_FILE}')
    # Read labels as text first: the header is itself '0', so it looks like a label.
    label_values = pd.read_csv(LABEL_FILE, header=None).iloc[:, 0]
    label_values = pd.to_numeric(label_values, errors='coerce').dropna().to_numpy(dtype=np.int64)
    labels = label_values
    log(f'Raw numeric label entries: {len(labels):,}')
    log(f'Labels: {len(labels)}; 0={(labels==0).sum()}; 1={(labels==1).sum()}')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be 0/1')
    if min((labels==0).sum(), (labels==1).sum()) < PER_CLASS:
        raise ValueError('Not enough examples of both classes; reduce PER_CLASS')
    # Detect the transposed CSV BEFORE selecting IDs, so numeric label header
    # can be removed safely based on the number of trajectory columns.
    with TS_FILE.open('r', encoding='utf-8-sig', newline='') as probe:
        first_row = next(csv.reader(probe), None)
    if first_row is None:
        raise ValueError('Empty time-series CSV')
    width = len(first_row)
    if width == len(labels) - 1 and labels[0] == 0:
        log('Detected extra numeric label header 0; removing it')
        labels = labels[1:]
    log(f'Corrected labels: {len(labels):,}; 0={(labels==0).sum()}; 1={(labels==1).sum()}')
    if width != len(labels):
        raise ValueError(f'Expected {len(labels):,} trajectory columns, but found {width:,}. Check extraction.')
    rng = np.random.default_rng(SEED)
    ids = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == c), PER_CLASS, replace=False)
        for c in (0, 1)
    ]))
    log(f'Selected {len(ids)} balanced trajectories')
    with TS_FILE.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError('Time-series CSV is empty')
        has_header = header_like(first)
        width = len(first)
        log(f'First row has {width:,} columns; numeric header={has_header}')
        rows = reader if has_header else chain([first], reader)
        # In this extraction each column is a trajectory.
        if width == len(labels):
            log('Detected TRANSPOSED CSV: trajectories are columns')
            # For transposed data, each row is a time point. Select only needed columns.
            offset = 0
            sample_rows = []
            count = 0
            for row in rows:
                if len(row) != width:
                    raise ValueError(f'Inconsistent CSV width at time row {count}: {len(row)} vs {width}')
                sample_rows.append([float(row[int(i)+offset]) for i in ids])
                count += 1
                if count % PRINT_EVERY == 0:
                    log(f'Read {count} time rows')
            if not sample_rows:
                raise ValueError('No time rows found')
            raw = np.asarray(sample_rows, dtype=np.float32).T
            log(f'Loaded {raw.shape[0]} trajectories x {raw.shape[1]} time points')
        else:
            log('Detected ROW-ORIENTED CSV: trajectories are rows')
            chosen = {}
            wanted = set(ids.tolist())
            count = 0
            for i, row in enumerate(rows):
                count = i + 1
                if i in wanted:
                    chosen[i] = np.asarray(row, dtype=np.float32)
                    if len(chosen) % 100 == 0:
                        log(f'Found {len(chosen)}/{len(ids)} selected trajectories')
                if count % 10000 == 0:
                    log(f'Scanned {count:,} rows')
                if len(chosen) == len(ids):
                    break
            if len(chosen) != len(ids):
                raise ValueError(f'Found {len(chosen)}/{len(ids)} rows. CSV has ~{count} rows but labels have {len(labels)}; check extraction orientation.')
            lengths = {len(v) for v in chosen.values()}
            if len(lengths) != 1:
                raise ValueError('Selected time series have inconsistent lengths')
            raw = np.stack([chosen[int(i)] for i in ids])
    if raw.shape[1] < 2:
        raise ValueError('Not enough time points')
    positions = np.linspace(0, raw.shape[1] - 1, N_POINTS).astype(int)
    X = raw[:, positions].astype(np.float32)
    y = labels[ids]
    if not np.isfinite(X).all():
        raise ValueError('Non-finite time-series values detected')
    np.savez_compressed(CACHE, X=X, y=y, ids=ids)
    log(f'Saved cache: {CACHE.name}; X shape {X.shape}')
    return X, y, ids


def main():
    import torch
    from torch import nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    torch.manual_seed(SEED)
    X, y, ids = load_data()
    ix = np.arange(len(y))
    train, test = train_test_split(ix, test_size=.2, stratify=y, random_state=SEED)
    train, val = train_test_split(train, test_size=.2, stratify=y[train], random_state=SEED)
   
    plt.plot(range(len(X[0])), X[0])
    plt.show()

    mu = float(X[train].mean())
    sd = float(X[train].std() + 1e-8)
    X = ((X-mu)/sd)[:, None, :]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Training on {device}: train={len(train)}, val={len(val)}, test={len(test)}')
    model = nn.Sequential(
        nn.Conv1d(1,16,7,padding=3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32,64,5,padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        nn.Flatten(), nn.Linear(64,1)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32)[:,None])
    best = float('inf')
    weights = None
    rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in np.array_split(rng.permutation(train), max(1,int(np.ceil(len(train)/BATCH_SIZE)))):
            opt.zero_grad()
            loss = loss_fn(model(xt[batch].to(device)), yt[batch].to(device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xt[val].to(device)), yt[val].to(device)).item()
        if vl < best:
            best = vl
            weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        log(f'Epoch {epoch+1}/{EPOCHS}: train={np.mean(losses):.4f} val={vl:.4f}')
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(xt[test].to(device))).cpu().numpy().ravel()
    pred = (p >= .5).astype(int)
    print('\nHeld-out test results:')
    print(classification_report(y[test], pred, zero_division=0))
    print('Confusion matrix:\n', confusion_matrix(y[test], pred))
    print('ROC-AUC:', round(roc_auc_score(y[test], p),4))
    pd.DataFrame({'trajectory_id':ids[test], 'actual_label':y[test],
                  'p_tip_cnn':p, 'predicted_label':pred}).to_csv(OUTPUT,index=False)
    torch.save({'model_state':weights, 'mean':mu,'std':sd,'n_points':N_POINTS}, MODEL)
    log(f'DONE: {OUTPUT.name} and {MODEL.name}')
    log('CNN scores classify Huang labels; they are not calibrated physical tipping probabilities.')


if __name__ == '__main__':
    main()
