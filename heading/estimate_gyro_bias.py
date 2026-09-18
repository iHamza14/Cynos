#!/usr/bin/env python3
"""Estimate stationary phone gyroscope bias.

Usage:
  python estimate_gyro_bias.py stationary.csv
  python estimate_gyro_bias.py stationary.csv --gyro-cols "gyro_1" "gyro_2" "gyro_3"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMMON_GROUPS = [
    ["GYROSCOPE X", "GYROSCOPE Y", "GYROSCOPE Z"],
    ["GYROSCOPE X (rad/s)", "GYROSCOPE Y (rad/s)", "GYROSCOPE Z (rad/s)"],
    ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"],
    ["Gyroscope X", "Gyroscope Y", "Gyroscope Z"],
    ["gyro_1", "gyro_2", "gyro_3"],
]


def find_gyro_columns(df, requested):
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Missing gyro columns {missing}. Available: {list(df.columns)}")
        return requested

    normalized = {str(c).strip().lower(): c for c in df.columns}
    for group in COMMON_GROUPS:
        found = [normalized.get(name.strip().lower()) for name in group]
        if all(c is not None for c in found):
            return found

    candidates = [c for c in df.columns if "gyro" in str(c).lower()]
    if len(candidates) == 3:
        return candidates
    raise ValueError(
        "Could not identify exactly 3 gyro columns. Pass --gyro-cols COL_X COL_Y COL_Z. "
        f"Available columns: {list(df.columns)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--gyro-cols", nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, default=Path("gyro_bias_stationary.json"))
    parser.add_argument("--trim-quantile", type=float, default=0.01)
    args = parser.parse_args()

    if not 0 <= args.trim_quantile < 0.5:
        parser.error("--trim-quantile must be in [0, 0.5)")

    df = pd.read_csv(args.csv, encoding='latin1')
    cols = find_gyro_columns(df, args.gyro_cols)
    gyro = df[cols].apply(pd.to_numeric, errors="coerce")
    gyro = gyro.replace([np.inf, -np.inf], np.nan).dropna()

    if len(gyro) < 10:
        raise SystemExit(f"Only {len(gyro)} valid rows; need at least 10.")

    # Remove extreme tails per axis to reduce isolated spikes.
    kept = gyro.copy()
    for col in cols:
        low = gyro[col].quantile(args.trim_quantile)
        high = gyro[col].quantile(1 - args.trim_quantile)
        kept = kept[(kept[col] >= low) & (kept[col] <= high)]

    if len(kept) < 10:
        raise SystemExit("Too few rows after trimming; lower --trim-quantile.")

    median = kept.median()
    mean = kept.mean()
    std = kept.std(ddof=1)

    result = {
        "source_csv": str(args.csv),
        "gyro_columns_axis_order": [str(c) for c in cols],
        "units": "same as source columns (typically rad/s)",
        "valid_rows_before_trim": int(len(gyro)),
        "valid_rows_after_trim": int(len(kept)),
        "trim_quantile_each_tail": args.trim_quantile,
        "bias_median": {str(c): float(median[c]) for c in cols},
        "mean": {str(c): float(mean[c]) for c in cols},
        "std_sample": {str(c): float(std[c]) for c in cols},
        "note": "Subtract bias_median from corresponding gyro axes. This estimates stationary offset only.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
