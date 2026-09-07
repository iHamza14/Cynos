import numpy as np
import pandas as pd
import pickle
import os
from helper import bin_dataset, cumulative_displacement,total_distance_m
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
CONTEXT_SEC     = 10          # warm-up seconds before blackout (GPS visible)
BLACKOUT_DURATIONS = [15 , 30, 60]   # seconds
BLACKOUT_STEP   = 30          # slide every 30s to get varied blackout scenarios
MIN_SPEED_KMPH  = 5           # reject stationary blackout windows
MAX_GPS_ACC     = 10          # quality gate on context window
MIN_GPS_SATS    = 6
HZ=10

# ── Main 10Hz Builder ──────────────────────────────────────────────────────────
def build_blackout_windows(bins, blackout_durations=BLACKOUT_DURATIONS,
                                context_sec=CONTEXT_SEC, step_sec=BLACKOUT_STEP):
    windows = []
    total_steps = len(bins)
    
    # Convert seconds to 10Hz step counts
    context_steps = context_sec * HZ
    step_size = step_sec * HZ

    for blackout_dur_sec in blackout_durations:
        blackout_steps = blackout_dur_sec * HZ
        window_len_steps = context_steps + blackout_steps

        for i in range(0, total_steps - window_len_steps + 1, step_size):
            context_bins  = bins[i : i + context_steps]
            blackout_bins = bins[i + context_steps : i + window_len_steps]

            # ── 1. Structural Guards ──
            if len(context_bins) != context_steps: continue
            if len(blackout_bins) != blackout_steps: continue
            
            # Check 10Hz continuity (timestamps should jump by exactly 0.1s. Allow 0.15s max drift)
            times = [b['abs_sec'] for b in context_bins + blackout_bins]
            if np.max(np.diff(times)) > 0.15: continue
            
            # ── 2. Quality Guards ──
            bad_gps = sum(1 for b in context_bins if b['gps_acc'] > MAX_GPS_ACC or b['gps_sats'] < MIN_GPS_SATS)
            if bad_gps > 0: continue
            
            avg_speed = np.mean([b['v_odo_speed'] * 3.6 for b in blackout_bins])
            if avg_speed < MIN_SPEED_KMPH: continue

            # ── 3. Context Pack (Model input at t=0 of blackout) ──
            last_ctx = context_bins[-1]
            context = {
                'raw_accel': np.stack([b['raw_accel'] for b in context_bins]),   # Shape: (100, 3)
                'raw_gyro':  np.stack([b['raw_gyro']  for b in context_bins]),   # Shape: (100, 3)
                'gravity':   np.stack([b['gravity']   for b in context_bins]),   # Shape: (100, 3)
                'vr_seed':   float(last_ctx['mobile_gps_speed']),
            }

            # ── 4. Blackout Pack (Model input during dropout) ──
            blackout = {
                'raw_accel': np.stack([b['raw_accel'] for b in blackout_bins]),  # Shape: (N, 3)
                'raw_gyro':  np.stack([b['raw_gyro']  for b in blackout_bins]),  # Shape: (N, 3)
                'gravity':   np.stack([b['gravity']   for b in blackout_bins]),  # Shape: (N, 3)
            }

            # ── 5. Ground Truth (For scoring) ──
            gt_cum_disp   = cumulative_displacement(blackout_bins)               # Shape: (N, 2)
            gt_speeds     = np.array([b['v_odo_speed'] for b in blackout_bins])  # Shape: (N,)
            gt_headings   = np.array([np.radians(b['v_heading']) for b in blackout_bins])  # Shape: (N,)
            
            # 10Hz per-step acceleration/deceleration
            gt_delta_v    = np.zeros(blackout_steps)
            gt_delta_v[0] = gt_speeds[0] - context['vr_seed']
            if blackout_steps > 1:
                gt_delta_v[1:] = np.diff(gt_speeds)
            
            # Vectorized fast cumulative distance calculation
            cum_dist_array = np.zeros(blackout_steps)
            if blackout_steps > 1:
                lats = np.radians([b['v_lat'] for b in blackout_bins])
                lons = np.radians([b['v_lon'] for b in blackout_bins])
                R = 6371000.0
                d_lats = np.diff(lats)
                d_lons = np.diff(lons)
                mid_lats = (lats[:-1] + lats[1:]) / 2.0
                step_dists = np.sqrt((d_lats * R)**2 + (d_lons * R * np.cos(mid_lats))**2)
                cum_dist_array[1:] = np.cumsum(step_dists)

            ground_truth = {
                'cum_disp_m':          gt_cum_disp,           
                'speeds_ms':           gt_speeds,             
                'headings_rad':        gt_headings,           
                'delta_v':             gt_delta_v,            
                'total_distance_m':    total_distance_m(blackout_bins),            
                'cumulative_dist_m':   cum_dist_array,  
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
            })

    print(f"  Generated {len(windows)} 10Hz blackout windows across durations {blackout_durations}s")
    for d in blackout_durations:
        n = sum(1 for w in windows if w['blackout_dur_sec'] == d)
        print(f"    {d:3d}s blackout: {n} windows")
    return windows

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
    
    all_raw_accel = np.vstack([w['blackout']['raw_accel'] for w in train_windows])
    all_raw_gyro = np.vstack([w['blackout']['raw_gyro'] for w in train_windows])
    all_gravity = np.vstack([w['blackout']['gravity'] for w in train_windows])

    scalers = {
        'raw_accel': StandardScaler().fit(all_raw_accel),
        'raw_gyro': StandardScaler().fit(all_raw_gyro),
        'gravity': StandardScaler().fit(all_gravity)
    }

    os.makedirs('data', exist_ok=True)
    with open('data/train_blackout_windows.pkl', 'wb') as f:
        pickle.dump(train_windows, f)
    with open('data/test_blackout_windows.pkl', 'wb') as f:
        pickle.dump(test_windows, f)
    with open('data/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    print(f"Saved {len(train_windows)} Train windows and {len(test_windows)} Test windows + Scalers to data/")