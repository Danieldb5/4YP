
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

# =========================================
# SETTINGS
# =========================================

BASE = Path(__file__).resolve().parent

INPUT = BASE / "mcresults.csv"
OUTPUT = BASE / "probability_comparison.csv"
ERROR_OUTPUT = BASE / "largest_errors.csv"

TOP_N = 25       # Number of high-error trajectories
DZ = 0.025       # State grid spacing
DT = 0.005       # PDE time step
Z_MIN = -7.0     # Lower computational boundary
Z_MAX = 2.0      # Tipping boundary

# =========================================


def log(message):
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True
    )


def forcing(t, eps):
    return 1.5 * (np.tanh(1.5 * eps * t) + 1)


def forcing_speed(t, eps):
    return (
        2.25 * eps
        / np.cosh(1.5 * eps * t)**2
    )


def probability_grid(t0, horizon, D, eps):
    """
    Solve backward Kolmogorov PDE.

    Returns tipping probability at each
    possible starting state z.
    """

    n_cells = int(np.ceil(
        (Z_MAX - Z_MIN) / DZ
    ))

    z = np.linspace(Z_MIN, Z_MAX, n_cells + 1)
    h = z[1] - z[0]

    interior = z[1:-1]
    n = len(interior)

    # Terminal condition:
    # No future time remains to tip.
    q = np.zeros(len(z))
    q[-1] = 1.0

    steps = int(np.ceil(horizon / DT))
    dt = horizon / steps

    for j in range(steps - 1, -1, -1):

        t = t0 + j * dt

        # Tracking-lag drift
        a = (
            interior**2
            - 2 * interior
            + forcing_speed(t, eps)
        )

        # Discrete backward generator
        plus = D / h**2 + np.maximum(a, 0) / h
        minus = D / h**2 + np.maximum(-a, 0) / h

        diagonal = 1 + dt * (plus + minus)

        # Tridiagonal implicit matrix
        A = np.zeros((3, n))

        A[0, 1:] = -dt * plus[:-1]
        A[1, :] = diagonal
        A[2, :-1] = -dt * minus[1:]

        rhs = q[1:-1].copy()

        # Absorbing tipping boundary: q(2,t)=1
        rhs[-1] += dt * plus[-1]

        q[1:-1] = solve_banded(
            (1, 1),
            A,
            rhs,
            check_finite=False
        )

    return z, np.clip(q, 0, 1)


def main():

    log("Loading Monte Carlo results")

    df = pd.read_csv(INPUT)

    required = [
        "trajectory_id", "x0", "model_time",
        "p_tip_mc", "D", "eps", "horizon"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Input CSV is empty")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[required[1:]].to_numpy()).all():
        raise ValueError("Input contains invalid numbers")

    if (df["D"] <= 0).any() or (df["horizon"] <= 0).any():
        raise ValueError("D and horizon must be positive")

    log(f"Loaded {len(df)} trajectories")

    df["p_tip_pde"] = np.nan

    # One PDE solve per unique model configuration.
    groups = df.groupby(
        ["model_time", "horizon", "D", "eps"],
        sort=False
    ).groups

    for number, (settings, indices) in enumerate(
        groups.items(), start=1
    ):

        t0, horizon, D, eps = map(float, settings)

        log(
            f"Solving PDE group {number}/{len(groups)} "
            f"for {len(indices)} trajectories"
        )

        z_grid, p_grid = probability_grid(
            t0, horizon, D, eps
        )

        z0 = (
            df.loc[indices, "x0"].to_numpy()
            + forcing(t0, eps)
            + 1
        )

        # Interpolate PDE probability at each state.
        p = np.interp(
            z0, z_grid, p_grid,
            left=0.0,
            right=1.0
        )

        df.loc[indices, "p_tip_pde"] = p

    # =========================================
    # COMPARE PROBABILITIES
    # =========================================

    df["signed_error"] = (
        df["p_tip_pde"] - df["p_tip_mc"]
    )

    df["absolute_error"] = df["signed_error"].abs()

    # Identify observations already over the boundary.
    z0_all = (
        df["x0"].to_numpy()
        + forcing(
            df["model_time"].to_numpy(),
            df["eps"].to_numpy()
        )
        + 1
    )

    df["already_over_boundary"] = z0_all >= 2
    df["below_pde_grid"] = z0_all < Z_MIN

    # Sort largest errors first.
    df = df.sort_values(
        "absolute_error",
        ascending=False
    )

    # Save all trajectories and probabilities.
    df.to_csv(OUTPUT, index=False)

    # Save the worst cases with original time series.
    worst = df.head(TOP_N)
    worst.to_csv(ERROR_OUTPUT, index=False)

    log(f"Saved: {OUTPUT.name}")
    log(f"Saved: {ERROR_OUTPUT.name}")

    log(
        "Mean absolute error: "
        f"{df['absolute_error'].mean():.5f}"
    )

    log(
        "Maximum absolute error: "
        f"{df['absolute_error'].max():.5f}"
    )

    log(
        "Already over boundary: "
        f"{df['already_over_boundary'].sum()}"
    )

    log(
        "Below PDE grid: "
        f"{df['below_pde_grid'].sum()}"
    )

    print("\nLargest errors:")
    print(
        worst[
            [
                "trajectory_id",
                "p_tip_mc",
                "p_tip_pde",
                "absolute_error"
            ]
        ].to_string(index=False)
    )

    log("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
