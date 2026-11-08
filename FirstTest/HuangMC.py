
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS - CHANGE THESE ONLY
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_FOLDER = BASE / "extracted_data"

TS_FILE = DATA_FOLDER / "ts_analysis.csv"
LABEL_FILE = DATA_FOLDER / "label.csv"

SAMPLE_FILE = BASE / "huang_random_sample.csv"
OUTPUT_FILE = BASE / "mcresults.csv"

N_TRAJECTORIES = 100
N_SIMULATIONS = 1000

D = 0.008
EPS = 1.25
HORIZON = 1.0
DT = 0.01

# Absolute model time of the selected final observation.
# 0.0 allows a test run but must be verified for research.
FORECAST_TIME = 0.0

SEED = 42
REUSE_SAMPLE = True

# Print progress while reading the large CSV.
PRINT_EVERY = 10000

# ============================================================


def log(message):
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True
    )


def forcing(t):
    return 1.5 * (
        np.tanh(1.5 * EPS * t) + 1.0
    )


def monte_carlo(x0, t0, rng):
    """
    Estimate probability of crossing the moving unstable
    equilibrium within HORIZON model-time units.
    """

    x = np.full(N_SIMULATIONS, x0, dtype=float)
    tipped = x >= 1.0 - forcing(t0)

    n_steps = int(np.ceil(HORIZON / DT))

    for j in range(n_steps):

        t = t0 + j * DT
        step_dt = min(DT, HORIZON - j * DT)

        if step_dt <= 0:
            break

        active = np.flatnonzero(~tipped)

        if active.size == 0:
            break

        drift = (
            x[active] + forcing(t)
        ) ** 2 - 1.0

        noise = np.sqrt(
            2.0 * D * step_dt
        ) * rng.standard_normal(active.size)

        x[active] += drift * step_dt + noise

        boundary = 1.0 - forcing(t + step_dt)

        tipped[active] = x[active] >= boundary

    p = float(tipped.mean())

    se = np.sqrt(
        p * (1.0 - p) / N_SIMULATIONS
    )

    return p, se


def random_sample_csv(path, n, rng):
    """
    Reservoir sampling:
    select n random rows without loading the full CSV.
    """

    log(f"Scanning {path.name}")

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}"
        )

    reservoir = []
    row_ids = []
    total = 0

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            # Skip a numeric column-index header such as:
            # 0,1,2,3,...
            if total == 0 and len(row) > 2:
                try:
                    values = np.asarray(
                        row, dtype=float
                    )

                    if np.array_equal(
                        values,
                        np.arange(len(values))
                    ):
                        log("Skipping column-index header")
                        continue
                except ValueError:
                    pass

            if total < n:
                reservoir.append(row)
                row_ids.append(total)
            else:
                k = rng.integers(0, total + 1)

                if k < n:
                    reservoir[k] = row
                    row_ids[k] = total

            total += 1

            if total % PRINT_EVERY == 0:
                log(
                    f"Scanned {total:,} trajectories"
                )

    if total == 0:
        raise ValueError("Time-series CSV is empty")

    log(
        f"Finished scanning {total:,} trajectories"
    )

    return row_ids, reservoir, total


def load_labels(path, selected_ids):
    """
    Read labels only for selected original row IDs.
    """

    log("Loading matching labels")

    wanted = set(selected_ids)
    result = {}

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.reader(f)

        first = next(reader, None)

        if first is None:
            raise ValueError("Label CSV is empty")

        # Handle a header named label or labels.
        header = first[0].strip().lower()

        if header in ("label", "labels"):
            rows = reader
        else:
            rows = iter([first] + list(reader))

        for i, row in enumerate(rows):
            if i in wanted:
                result[i] = row[0]

            if len(result) == len(wanted):
                break

    if len(result) != len(wanted):
        raise ValueError(
            "Could not match every selected trajectory "
            "to a label. Check CSV headers and row counts."
        )

    return result


def create_sample(rng):

    ids, rows, total = random_sample_csv(
        TS_FILE,
        N_TRAJECTORIES,
        rng
    )

    labels = load_labels(
        LABEL_FILE,
        ids
    )

    records = []

    for row_id, row in zip(ids, rows):

        values = np.asarray(
            row, dtype=float
        )

        if not np.all(np.isfinite(values)):
            log(f"Skipping invalid trajectory {row_id}")
            continue

        record = {
            "trajectory_id": row_id,
            "label": labels[row_id]
        }

        for j, value in enumerate(values):
            record[str(j)] = value

        records.append(record)

    sample = pd.DataFrame(records)

    sample.to_csv(
        SAMPLE_FILE,
        index=False
    )

    log(
        f"Saved reusable sample: {SAMPLE_FILE.name}"
    )

    return sample


def load_sample(rng):

    if REUSE_SAMPLE and SAMPLE_FILE.exists():
        log("Reusing saved random sample")
        sample = pd.read_csv(SAMPLE_FILE)

        if len(sample) != N_TRAJECTORIES:
            log(
                "Saved sample size differs from settings; "
                "creating a new sample"
            )
            return create_sample(rng)

        return sample

    return create_sample(rng)


def main():

    start = time.time()

    log("Starting Huang Monte Carlo analysis")

    log(f"Data folder: {DATA_FOLDER}")
    log(f"Time series exists: {TS_FILE.exists()}")
    log(f"Labels exist: {LABEL_FILE.exists()}")

    if not TS_FILE.exists():
        raise FileNotFoundError(TS_FILE)

    if not LABEL_FILE.exists():
        raise FileNotFoundError(LABEL_FILE)

    if DT <= 0 or HORIZON <= 0:
        raise ValueError(
            "DT and HORIZON must be positive"
        )

    if N_SIMULATIONS <= 0:
        raise ValueError(
            "N_SIMULATIONS must be positive"
        )

    rng = np.random.default_rng(SEED)

    sample = load_sample(rng)

    log(
        f"Loaded {len(sample)} random trajectories"
    )

    log(
        f"Forecast time: {FORECAST_TIME}"
    )

    log(
        f"Monte Carlo futures per trajectory: "
        f"{N_SIMULATIONS}"
    )

    log("Starting Monte Carlo simulations")

    results = []

    metadata = {
        "trajectory_id",
        "label"
    }

    time_columns = [
        c for c in sample.columns
        if c not in metadata
        and str(c).isdigit()
    ]

    time_columns = sorted(
        time_columns,
        key=lambda c: int(c)
    )

    if not time_columns:
        raise ValueError(
            "No numerical time-series columns found"
        )

    last_column = time_columns[-1]

    for i, (_, row) in enumerate(sample.iterrows()):

        x0 = float(row[last_column])

        if not np.isfinite(x0):
            continue

        p, se = monte_carlo(
            x0,
            FORECAST_TIME,
            rng
        )

        result = row.to_dict()

        result.update({
            "x0": x0,
            "model_time": FORECAST_TIME,
            "z0": x0 + forcing(FORECAST_TIME) + 1.0,
            "p_tip_mc": p,
            "mc_se": se,
            "horizon": HORIZON,
            "D": D,
            "eps": EPS,
            "n_sim": N_SIMULATIONS
        })

        results.append(result)

        if (
            (i + 1) % 10 == 0
            or i + 1 == len(sample)
        ):
            elapsed = time.time() - start

            log(
                f"Forecasted {i + 1}/{len(sample)} "
                f"| elapsed {elapsed:.1f}s"
            )

    log("Saving results")

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    log(
        f"DONE: {OUTPUT_FILE}"
    )

    log(
        f"Total runtime: "
        f"{time.time() - start:.2f} seconds"
    )

    print(
        output[
            [
                "trajectory_id",
                "label",
                "x0",
                "p_tip_mc",
                "mc_se"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
