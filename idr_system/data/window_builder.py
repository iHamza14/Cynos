"""
Blackout window builder.

Slices bins into context + blackout pairs, applies quality gates, and
stores everything the dataset needs to produce IDRModel batches.

Key conventions:
  - vr_seed is stored in m/s (converted from km/h)
  - Accel (not gyro) receives the anti-vibration filter
  - ORIENTATION Yaw channel is not used (F3: unreliable)
  - Context = 30s, matching the MTN's design receptive field
"""

import numpy as np

from data.anti_vibration import compute_shock_mask, apply_anti_vibration_filter
from data.bin_builder import cumulative_displacement, total_distance_m


# ── Config ──────────────────────────────────────────────────────────────
CONTEXT_SEC        = 30
BLACKOUT_DURATIONS = [15, 30, 60]
BLACKOUT_STEP_SEC  = 30
MIN_SPEED_KMPH     = 5
MAX_GPS_ACC        = 10
MIN_GPS_SATS       = 6
HZ                 = 10
VIBRATION_CUTOFF   = 2.5


def build_blackout_windows(bins,
                           blackout_durations=BLACKOUT_DURATIONS,
                           context_sec=CONTEXT_SEC,
                           step_sec=BLACKOUT_STEP_SEC):
    windows = []
    total_steps = len(bins)

    context_steps = context_sec * HZ
    step_size = step_sec * HZ

    for blackout_dur_sec in blackout_durations:
        blackout_steps = blackout_dur_sec * HZ
        window_len_steps = context_steps + blackout_steps

        for i in range(0, total_steps - window_len_steps + 1, step_size):
            context_bins  = bins[i : i + context_steps]
            blackout_bins = bins[i + context_steps : i + window_len_steps]

            # ── Structural guards ────────────────────────────────────
            if len(context_bins) != context_steps: continue
            if len(blackout_bins) != blackout_steps: continue

            times = [b['abs_sec'] for b in context_bins + blackout_bins]
            if np.max(np.diff(times)) > 0.15: continue

            # ── Quality gates ────────────────────────────────────────
            bad_ctx = sum(
                1 for b in context_bins
                if b['gps_acc'] > MAX_GPS_ACC or b['gps_sats'] < MIN_GPS_SATS
            )
            if bad_ctx / context_steps > 0.05: continue

            bad_blk = sum(
                1 for b in blackout_bins
                if b['gps_acc'] > MAX_GPS_ACC or b['gps_sats'] < MIN_GPS_SATS
            )
            if bad_blk / blackout_steps > 0.05: continue

            avg_speed_kmh = np.mean([b['v_odo_speed'] * 3.6 for b in blackout_bins])
            if avg_speed_kmh < MIN_SPEED_KMPH: continue

            # ── Turn fraction (F5) ───────────────────────────────────
            blk_yaw_rates = np.array(
                [b.get('v_yaw_rate_deg_s', 0.0) for b in blackout_bins]
            )
            turn_fraction = (
                np.sum(np.abs(blk_yaw_rates) >= 2.0) / len(blk_yaw_rates)
            )

            # ── Context pack ─────────────────────────────────────────
            last_ctx = context_bins[-1]
            ctx_accel_raw = np.stack([b['raw_accel'] for b in context_bins])
            ctx_gyro_raw  = np.stack([b['raw_gyro']  for b in context_bins])
            ctx_accel_filt = apply_anti_vibration_filter(
                ctx_accel_raw, sample_rate_hz=HZ,
                cutoff_hz=VIBRATION_CUTOFF, order=4,
            )

            context = {
                'raw_accel_raw':  ctx_accel_raw,       # unfiltered (FFT, MTN)
                'raw_accel':      ctx_accel_filt,      # filtered (EKF)
                'raw_gyro':       ctx_gyro_raw,        # never filtered
                'gravity':        np.stack([b['gravity'] for b in context_bins]),
                'speed_seq':      np.array([b['v_odo_speed'] for b in context_bins]),
                'heading_seq':    np.array([b['v_heading_rad'] for b in context_bins]),
                'vr_seed':        float(last_ctx['v_odo_speed']),   # m/s
                'gps_age_at_entry': float(last_ctx.get('mobile_gps_age_sec', 0.0)),
                'shock_mask':     compute_shock_mask(ctx_accel_raw),
            }

            # ── Blackout pack ────────────────────────────────────────
            blk_accel_raw = np.stack([b['raw_accel'] for b in blackout_bins])
            blk_gyro_raw  = np.stack([b['raw_gyro']  for b in blackout_bins])
            blk_accel_filt = apply_anti_vibration_filter(
                blk_accel_raw, sample_rate_hz=HZ,
                cutoff_hz=VIBRATION_CUTOFF, order=4,
            )

            blackout = {
                'raw_accel_raw': blk_accel_raw,
                'raw_accel':     blk_accel_filt,
                'raw_gyro':      blk_gyro_raw,
                'gravity':       np.stack([b['gravity'] for b in blackout_bins]),
                'yaw_rate_proxy': blk_yaw_rates,
                'shock_mask':    compute_shock_mask(blk_accel_raw),
            }

            # ── Ground truth ─────────────────────────────────────────
            gt_cum_disp = cumulative_displacement(blackout_bins)
            gt_speeds   = np.array([b['v_odo_speed'] for b in blackout_bins])
            gt_headings = np.array([b['v_heading_rad'] for b in blackout_bins])

            # Per-step velocity delta in m/s (matches gt_speeds units)
            gt_delta_v = np.zeros(blackout_steps)
            gt_delta_v[0] = gt_speeds[0] - context['vr_seed']
            if blackout_steps > 1:
                gt_delta_v[1:] = np.diff(gt_speeds)

            ground_truth = {
                'cum_disp_m':        gt_cum_disp,
                'speeds_ms':         gt_speeds,
                'headings_rad':      gt_headings,
                'delta_v':           gt_delta_v,
                'total_distance_m':  total_distance_m(blackout_bins),
                'start_lat': float(blackout_bins[0]['v_lat']),
                'start_lon': float(blackout_bins[0]['v_lon']),
                'end_lat':   float(blackout_bins[-1]['v_lat']),
                'end_lon':   float(blackout_bins[-1]['v_lon']),
            }

            windows.append({
                'context':          context,
                'blackout':         blackout,
                'ground_truth':     ground_truth,
                'blackout_dur_sec': blackout_dur_sec,
                'context_dur_sec':  context_sec,
                'window_start_sec': context_bins[0]['abs_sec'],
                'turn_fraction':    turn_fraction,
            })

    print(f"  Generated {len(windows)} windows across {blackout_durations}s")
    for d in blackout_durations:
        n = sum(1 for w in windows if w['blackout_dur_sec'] == d)
        print(f"    {d:3d}s blackout: {n} windows")
    return windows