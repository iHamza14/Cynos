"""
Blackout Window Generator
Extends data_pipeline.py with evaluation windows for dead reckoning scoring.

Window structure:
  [---- context (10s) ----|---- blackout (N s) ----]
  GPS visible              GPS masked (model uses IMU only)
  
Ground truth positions stored separately — never fed to model during eval.
"""

import numpy as np
import pandas as pd
import pickle
import os
from data_pipeline import bin_dataset, extract_features, haversine_delta

# ── Config ─────────────────────────────────────────────────────────────────────
CONTEXT_SEC     = 10          # warm-up seconds before blackout (GPS visible)
BLACKOUT_DURATIONS = [15 , 30, 60]   # seconds — covers both hackathon constraints
BLACKOUT_STEP   = 30          # slide every 30s to get varied blackout scenarios
MIN_SPEED_KMPH  = 5           # reject stationary blackout windows
MAX_GPS_ACC     = 10          # quality gate on context window
MIN_GPS_SATS    = 6

# ── Cumulative displacement along a trajectory ─────────────────────────────────
def cumulative_displacement(bins):
    """
    Returns (N, 2) array of north/east displacement from bins[0] at each step.
    Uses vehicle lat/lon as ground truth — never phone GPS.
    """
    disps = [[0.0, 0.0]]
    for i in range(1, len(bins)):
        d = haversine_delta(
            bins[i-1]['v_lat'], bins[i-1]['v_lon'],
            bins[i]['v_lat'],   bins[i]['v_lon']
        )
        # Accumulate from origin
        disps.append([disps[-1][0] + d[0], disps[-1][1] + d[1]])
    return np.array(disps)   # shape (N, 2)

def total_distance_m(bins):
    """Haversine arc length of the ground truth trajectory in metres."""
    dist = 0.0
    for i in range(1, len(bins)):
        d = haversine_delta(
            bins[i-1]['v_lat'], bins[i-1]['v_lon'],
            bins[i]['v_lat'],   bins[i]['v_lon']
        )
        dist += np.linalg.norm(d)
    return dist

# ── Quality helpers ────────────────────────────────────────────────────────────
def context_is_clean(context_bins):
    """All context bins must have good GPS — model initialises from these."""
    for b in context_bins:
        if b['gps_acc'] > MAX_GPS_ACC or b['gps_sats'] < MIN_GPS_SATS:
            return False
    return True

def blackout_is_moving(blackout_bins):
    """Reject near-stationary blackout windows — drift % is meaningless at 0 km/h."""
    avg_speed = np.mean([b['v_odo_speed'] * 3.6 for b in blackout_bins])
    return avg_speed >= MIN_SPEED_KMPH

def context_is_contiguous(bins):
    for i in range(1, len(bins)):
        if bins[i]['abs_sec'] - bins[i-1]['abs_sec'] != 1:
            return False
    return True

# ── Main builder ───────────────────────────────────────────────────────────────
def build_blackout_windows(bins, blackout_durations=BLACKOUT_DURATIONS,
                           context_sec=CONTEXT_SEC, step_sec=BLACKOUT_STEP):
    """
    Returns list of blackout window dicts.
    
    Each window:
      context  — what the model sees to initialise state (GPS available)
      blackout — what the model receives during dropout (IMU only)
      gt_*     — ground truth labels (never fed to model, used for scoring only)
    """
    windows = []
    total_sec = len(bins)

    for blackout_dur in blackout_durations:
        window_len = context_sec + blackout_dur

        for i in range(0, total_sec - window_len + 1, step_sec):
            context_bins  = bins[i : i + context_sec]
            blackout_bins = bins[i + context_sec : i + window_len]

            # ── Structural guards ──────────────────────────────────────────
            if len(context_bins)  != context_sec:  continue
            if len(blackout_bins) != blackout_dur: continue
            if not context_is_contiguous(context_bins + blackout_bins): continue
            if not context_is_clean(context_bins):  continue
            if not blackout_is_moving(blackout_bins): continue

            # ── Context pack (model input at t=0 of blackout) ─────────────
            last_ctx = context_bins[-1]
            context = {
                # Statistical features over the 10-second warm-up
                'acc_feat':  np.stack([b['acc_feat']  for b in context_bins]),   # (10, 18)
                'gyro_feat': np.stack([b['gyro_feat'] for b in context_bins]),   # (10, 18)
                'mtn_input': np.stack([b['mtn_input'] for b in context_bins]),   # (10, 6)
                'raw_accel': np.stack([b['raw_accel'] for b in context_bins]),   # (10, 3)
                'gravity':   np.stack([b['gravity']   for b in context_bins]),   # (10, 3)

                # Seed values — last known state before blackout
                'vr_seed':         float(last_ctx['mobile_gps_speed']),          # teacher-force seed
            }

            # ── Blackout pack (model input during dropout — no GPS) ────────
            blackout = {
                'acc_feat':  np.stack([b['acc_feat']  for b in blackout_bins]),  # (N, 18)
                'gyro_feat': np.stack([b['gyro_feat'] for b in blackout_bins]),  # (N, 18)
                'mtn_input': np.stack([b['mtn_input'] for b in blackout_bins]),  # (N, 6)
                'raw_accel': np.stack([b['raw_accel'] for b in blackout_bins]),  # (N, 3)
                'raw_gyro':  np.stack([b.get('raw_gyro', [0,0,0]) for b in blackout_bins]),  # (N, 3)
                'gravity':   np.stack([b['gravity']   for b in blackout_bins]),  # (N, 3)
            }

            # ── Ground truth (scoring only — never seen by model) ──────────
            gt_cum_disp   = cumulative_displacement(blackout_bins)               # (N, 2) metres N/E
            gt_speeds     = np.array([b['v_odo_speed'] for b in blackout_bins]) # (N,) m/s
            gt_headings   = np.array([np.radians(b['v_heading']) for b in blackout_bins])  # (N,) rad
            
            # The speed change must be per-second, not the mean of 100Hz micro-changes!
            gt_delta_v    = np.zeros(blackout_dur)
            gt_delta_v[0] = gt_speeds[0] - context['vr_seed']
            if blackout_dur > 1:
                gt_delta_v[1:] = np.diff(gt_speeds)
            
            total_dist    = total_distance_m(blackout_bins)                      # scalar metres

            # Per-timestep drift constraint reference
            # Hackathon: <5m over 50m  OR  <100m over 1km
            # Store for scorer to compute: drift[t] / cumulative_dist[t]
            cumulative_dist_per_step = np.array([
                total_distance_m(blackout_bins[:t+1]) for t in range(blackout_dur)
            ])

            ground_truth = {
                'cum_disp_m':          gt_cum_disp,           # (N, 2) — compare DR pos here
                'speeds_ms':           gt_speeds,             # (N,)
                'headings_rad':        gt_headings,           # (N,)
                'delta_v':             gt_delta_v,            # (N,)
                'total_distance_m':    total_dist,            # scalar
                'cumulative_dist_m':   cumulative_dist_per_step,  # (N,) for % drift calc
                # Endpoints for plotting
                'start_lat': float(blackout_bins[0]['v_lat']),
                'start_lon': float(blackout_bins[0]['v_lon']),
                'end_lat':   float(blackout_bins[-1]['v_lat']),
                'end_lon':   float(blackout_bins[-1]['v_lon']),
            }

            windows.append({
                'context':          context,
                'blackout':         blackout,
                'ground_truth':     ground_truth,
                'blackout_dur_sec': blackout_dur,
                'context_dur_sec':  context_sec,
                'window_start_sec': bins[i]['abs_sec'],
            })

    print(f"  Generated {len(windows)} blackout windows across durations {blackout_durations}s")
    # Breakdown by duration
    for d in blackout_durations:
        n = sum(1 for w in windows if w['blackout_dur_sec'] == d)
        print(f"    {d:3d}s blackout: {n} windows")
    return windows

# ── Scoring helper (use this at eval time, not during training) ────────────────
def score_blackout_window(window, dr_positions_ne):
    """
    dr_positions_ne : (N, 2) numpy array of model's north/east position estimates
                      relative to blackout start, one per second.
    
    Returns dict of drift metrics matching hackathon constraint language.
    """
    gt   = window['ground_truth']['cum_disp_m']          # (N, 2)
    dist = window['ground_truth']['cumulative_dist_m']   # (N,)
    total_dist = window['ground_truth']['total_distance_m']

    # Euclidean error at each timestep (metres)
    errors = np.linalg.norm(dr_positions_ne - gt, axis=1)  # (N,)

    # Drift as % of distance travelled at that moment
    pct_drift = np.divide(errors, dist, out=np.zeros_like(errors), where=dist!=0) * 100

    return {
        'mean_error_m':       float(errors.mean()),
        'max_error_m':        float(errors.max()),         # worst case — use for constraint check
        'endpoint_error_m':   float(errors[-1]),
        'mean_pct_drift':     float(pct_drift.mean()),
        'max_pct_drift':      float(pct_drift.max()),
        'total_distance_m':   float(total_dist),
        'constraint_pass_5m_50m':   bool(errors.max() < 5   and total_dist >= 40),
        'constraint_pass_100m_1km': bool(errors.max() < 100 and total_dist >= 800),
        'per_step_errors_m':  errors.tolist(),             # for position plot
    }

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    s_path = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'
    v_path = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv'

    import pandas as pd
    df_s = pd.read_csv(s_path, encoding='latin-1')
    df_s.columns = df_s.columns.str.strip()
    df_v = pd.read_csv(v_path, encoding='latin-1')
    df_v.columns = df_v.columns.str.strip()

    bins = bin_dataset(df_s, df_v)

    # Chronological split — blackout windows come from TEST bins only
    # You never evaluate DR on data the model trained on
    split_idx = int(len(bins) * 0.8)
    train_bins = bins[:split_idx]
    test_bins = bins[split_idx:]

    print("Generating Training Windows (Fixed 60s for batching)...")
    train_windows = build_blackout_windows(train_bins, blackout_durations=[60], step_sec=15)
    
    print("Generating Test Windows (15s, 30s, 60s)...")
    test_windows = build_blackout_windows(test_bins, blackout_durations=[15, 30, 60], step_sec=30)

    # Fit StandardScaler on training data's blackout sections
    from sklearn.preprocessing import StandardScaler
    
    all_acc = np.vstack([w['blackout']['acc_feat'] for w in train_windows])
    all_gyro = np.vstack([w['blackout']['gyro_feat'] for w in train_windows])
    all_mtn = np.vstack([w['blackout']['mtn_input'] for w in train_windows])
    all_raw = np.vstack([w['blackout']['raw_accel'] for w in train_windows])
    all_grav = np.vstack([w['blackout']['gravity'] for w in train_windows])
    
    scalers = {
        'acc': StandardScaler().fit(all_acc),
        'gyro': StandardScaler().fit(all_gyro),
        'mtn': StandardScaler().fit(all_mtn),
        'raw': StandardScaler().fit(all_raw),
        'grav': StandardScaler().fit(all_grav)
    }

    os.makedirs('data', exist_ok=True)
    with open('data/train_blackout_windows.pkl', 'wb') as f:
        pickle.dump(train_windows, f)
    with open('data/test_blackout_windows.pkl', 'wb') as f:
        pickle.dump(test_windows, f)
    with open('data/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
        
    print(f"Saved {len(train_windows)} Train windows and {len(test_windows)} Test windows + Scalers to data/")