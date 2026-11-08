
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat

# Locate the original 718 MB dataset
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "largesample_case1data.mat"
OUTPUT_DIR = Path(__file__).resolve().parent / "extracted_data"

OUTPUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 10000


def export_csv(name, array):
    """Convert a numeric array to a CSV file."""
    array = np.asarray(array)

    if array.dtype.kind not in "biufc":
        print(f"Skipping non-numeric variable: {name}")
        return

    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        # Flatten remaining dimensions into columns
        array = array.reshape(array.shape[0], -1)

    output = OUTPUT_DIR / f"{name}.csv"

    print(f"Converting {name}, shape {array.shape}")

    for start in range(0, array.shape[0], CHUNK_SIZE):
        chunk = array[start:start + CHUNK_SIZE]

        pd.DataFrame(chunk).to_csv(
            output,
            mode="w" if start == 0 else "a",
            header=start == 0,
            index=False
        )

    print(f"Saved: {output.name}")


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    print("Reading 718 MB MATLAB dataset...")

    if h5py.is_hdf5(INPUT_FILE):
        # MATLAB v7.3 format
        with h5py.File(INPUT_FILE, "r") as f:

            def process(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return

                if obj.dtype.kind not in "biufc":
                    print(f"Skipping unsupported: {name}")
                    return

                safe_name = name.replace("/", "_")
                output = OUTPUT_DIR / f"{safe_name}.csv"

                print(f"Converting {name}, shape {obj.shape}")

                if obj.ndim == 0:
                    export_csv(safe_name, obj[()])
                    return

                if obj.ndim == 1:
                    rows = obj.shape[0]
                    for start in range(0, rows, CHUNK_SIZE):
                        chunk = obj[start:start + CHUNK_SIZE]
                        pd.DataFrame(chunk).to_csv(
                            output,
                            mode="w" if start == 0 else "a",
                            header=start == 0,
                            index=False
                        )
                else:
                    rows = obj.shape[0]
                    for start in range(0, rows, CHUNK_SIZE):
                        chunk = obj[start:start + CHUNK_SIZE]
                        chunk = chunk.reshape(chunk.shape[0], -1)

                        pd.DataFrame(chunk).to_csv(
                            output,
                            mode="w" if start == 0 else "a",
                            header=start == 0,
                            index=False
                        )

                print(f"Saved: {output.name}")

            f.visititems(process)

    else:
        # Standard MATLAB format
        data = loadmat(INPUT_FILE)

        for name, value in data.items():
            if not name.startswith("__"):
                export_csv(name, value)

    print("\nDONE! CSV files saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
