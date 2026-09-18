"""
Build blackout windows, fit scalers, save artifacts.

Split strategy:
  - Chronological 80/20 split on bins
  - Train windows from train_bins only, test windows from test_bins only
  - No overlap between test windows of the same duration

Scaler strategy:
  - Fit on RAW (unfiltered) accel and gyro
  - Scalers are applied to augmented data at train time — accept the small
    distribution shift, or drop standardization entirely in favor of an
    in-model BatchNorm.
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bin_builder import bin_dataset
from data.window_builder import build_blackout_windows


# ── Config ──────────────────────────────────────────────────────────────
DEFAULT_S_PATH = ('../Synchronised V abd S datasets/Categorised IOVNB Dataset/'
                  'Vw (Driver E)/Vw04/S-Vw4.csv')
DEFAULT_V_PATH = ('../Synchronised V abd S datasets/Categorised IOVNB Dataset/'
                  'Vw (Driver E)/Vw04/V-Vw4.csv')

TRAIN_BLACKOUT_DURATIONS = [15, 30, 60]
TEST_BLACKOUT_DURATIONS  = [15, 30, 60]
TRAIN_STEP_SEC           = 15      # overlapping training windows OK
TEST_STEP_SEC_FACTOR     = 1.0     # test step = window duration (no overlap)
SPLIT_FRACTION           = 0.8


def sanity_check_windows(windows, name):
    """Assert windows are structurally valid. Raises on failure."""
    assert len(windows) > 0, f"{name}: no windows generated"

    w0 = windows[0]
    ctx = w0['context']
    blk = w0['blackout']
    gt  = w0['ground_truth']

    # Field presence
    for k in ('raw_accel', 'raw_accel_raw', 'raw_gyro', 'speed_seq',
              'heading_seq', 'vr_seed', 'gps_age_at_entry'):
        assert k in ctx, f"{name}: missing context.{k}"
    for k in ('raw_accel', 'raw_accel_raw', 'raw_gyro', 'shock_mask'):
        assert k in blk, f"{name}: missing blackout.{k}"
    for k in ('cum_disp_m', 'speeds_ms', 'headings_rad', 'total_distance_m'):
        assert k in gt, f"{name}: missing ground_truth.{k}"

    # Velocity units: vr_seed (m/s) vs GT speeds (m/s) should match at t=0
    vr = float(ctx['vr_seed'])
    gt_v0 = float(gt['speeds_ms'][0])
    assert abs(vr - gt_v0) < 5.0, (
        f"{name}: vr_seed={vr:.2f} vs gt.speeds_ms[0]={gt_v0:.2f} "
        f"— units mismatch or seed error"
    )

    # Temporal contiguity: last context time and first blackout time must be
    # consecutive (window_builder doesn't store abs times, so check by length)
    assert len(gt['speeds_ms']) == w0['blackout_dur_sec'] * 10

    # No NaNs in critical arrays
    for arr_name, arr in [
        ('context.raw_accel', ctx['raw_accel']),
        ('blackout.raw_accel', blk['raw_accel']),
        ('ground_truth.cum_disp_m', gt['cum_disp_m']),
        ('ground_truth.speeds_ms', gt['speeds_ms']),
    ]:
        assert not np.isnan(arr).any(), f"{name}: NaN in {arr_name}"

    print(f"  [{name}] sanity OK  "
          f"(N={len(windows)}, vr_seed={vr:.2f} m/s, "
          f"gt_v0={gt_v0:.2f} m/s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_path', default=DEFAULT_S_PATH)
    parser.add_argument('--v_path', default=DEFAULT_V_PATH)
    parser.add_argument('--out_dir', default='data')
    args = parser.parse_args()

    if not os.path.exists(args.s_path) or not os.path.exists(args.v_path):
        print(f"Warning: datasets not found at:\n  {args.s_path}\n  {args.v_path}")
        return

    # ── Load ─────────────────────────────────────────────────────────
    df_s = pd.read_csv(args.s_path, encoding='latin-1')
    df_s.columns = df_s.columns.str.strip()
    df_v = pd.read_csv(args.v_path, encoding='latin-1')
    df_v.columns = df_v.columns.str.strip()

    bins = bin_dataset(df_s, df_v)
    print(f"Built {len(bins)} bins")

    # ── Chronological split ─────────────────────────────────────────
    split_idx = int(len(bins) * SPLIT_FRACTION)
    train_bins = bins[:split_idx]
    test_bins  = bins[split_idx:]
    print(f"Split: {len(train_bins)} train bins, {len(test_bins)} test bins")

    # ── Build windows ───────────────────────────────────────────────
    print("Generating training windows (15s, 30s, 60s)...")
    train_windows = build_blackout_windows(
        train_bins,
        blackout_durations=TRAIN_BLACKOUT_DURATIONS,
        step_sec=TRAIN_STEP_SEC,
    )

    print("Generating test windows (15s, 30s, 60s, non-overlapping)...")
    test_windows = []
    for dur in TEST_BLACKOUT_DURATIONS:
        test_windows += build_blackout_windows(
            test_bins,
            blackout_durations=[dur],
            step_sec=int(dur * TEST_STEP_SEC_FACTOR),
        )

    # ── Sanity ──────────────────────────────────────────────────────
    sanity_check_windows(train_windows, 'train')
    sanity_check_windows(test_windows,  'test')

    # ── Scalers ─────────────────────────────────────────────────────
    # Fit on RAW (unfiltered) data — that's what augmentation will rotate.
    # Standardization must be applied AFTER augmentation at train time;
    # accept the distribution shift, or move normalization into the model.
    all_raw_accel = np.vstack([w['blackout']['raw_accel_raw'] for w in train_windows])
    all_raw_gyro  = np.vstack([w['blackout']['raw_gyro']      for w in train_windows])

    scalers = {
        'raw_accel': StandardScaler().fit(all_raw_accel),
        'raw_gyro':  StandardScaler().fit(all_raw_gyro),
    }

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train_blackout_windows.pkl'), 'wb') as f:
        pickle.dump(train_windows, f)
    with open(os.path.join(args.out_dir, 'test_blackout_windows.pkl'), 'wb') as f:
        pickle.dump(test_windows, f)
    with open(os.path.join(args.out_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    print(f"\nSaved to {args.out_dir}/:")
    print(f"  train_blackout_windows.pkl  ({len(train_windows)} windows)")
    print(f"  test_blackout_windows.pkl   ({len(test_windows)} windows)")
    print(f"  scalers.pkl                 (raw_accel, raw_gyro)")


if __name__ == '__main__':
    main()