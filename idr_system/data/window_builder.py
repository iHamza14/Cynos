import numpy as np
from data.anti_vibration import compute_shock_mask

# ── Config ─────────────────────────────────────────────────────────────────────
CONTEXT_SEC     = 10          # warm-up seconds before blackout (GPS visible)
BLACKOUT_DURATIONS = [15 , 30, 60]   # seconds
BLACKOUT_STEP   = 30          # slide every 30s to get varied blackout scenarios
MIN_SPEED_KMPH  = 5           # reject stationary blackout windows
MAX_GPS_ACC     = 10          # quality gate on context window
MIN_GPS_SATS    = 6
HZ=10

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
            
            # ── 2. Quality Guards (IDR Loosened) ──
            # Context allow <= 5% bad samples
            bad_gps_ctx = sum(1 for b in context_bins if b['gps_acc'] > MAX_GPS_ACC or b['gps_sats'] < MIN_GPS_SATS)
            if bad_gps_ctx / context_steps > 0.05: continue
            
            # Blackout quality gate: skip if >5% of blackout bins have bad GPS 
            bad_gps_blk = sum(1 for b in blackout_bins if b['gps_acc'] > MAX_GPS_ACC or b['gps_sats'] < MIN_GPS_SATS)
            if bad_gps_blk / blackout_steps > 0.05: continue
            
            avg_speed = np.mean([b['v_odo_speed'] * 3.6 for b in blackout_bins])
            if avg_speed < MIN_SPEED_KMPH: continue

            # Calculate turn fraction (F5)
            # Gentle curve or sharp turn = |yaw rate| >= 2 deg/s
            blk_yaw_rates = np.array([b.get('v_yaw_rate_deg_s', 0) for b in blackout_bins])
            turn_fraction = np.sum(np.abs(blk_yaw_rates) >= 2.0) / len(blk_yaw_rates)

            # ── 3. Context Pack (Model input at t=0 of blackout) ──
            last_ctx = context_bins[-1]
            ctx_accel = np.stack([b['raw_accel'] for b in context_bins])
            
            context = {
                'raw_accel': ctx_accel,
                'raw_gyro':  np.stack([b['raw_gyro']  for b in context_bins]),
                'gravity':   np.stack([b['gravity']   for b in context_bins]),
                'orientation_yaw_vehicle_rad': np.array([b.get('orientation_yaw_vehicle_rad', 0) for b in context_bins]),
                'speed_seq': np.array([b['v_odo_speed'] for b in context_bins]),
                'heading_seq': np.array([np.radians(b['v_heading']) for b in context_bins]),
                'vr_seed':   float(last_ctx['mobile_gps_speed']),
                'initial_yaw': float(last_ctx.get('orientation_yaw_vehicle_rad', 0)),
                'gps_age_at_entry': float(last_ctx.get('mobile_gps_age_sec', 0)),
                'remount_confidence': 1.0, # Placeholder for remount detector output
                'shock_mask': compute_shock_mask(ctx_accel)
            }

            # ── 4. Blackout Pack (Model input during dropout) ──
            blk_accel = np.stack([b['raw_accel'] for b in blackout_bins])
            
            blackout = {
                'raw_accel': blk_accel,
                'raw_gyro':  np.stack([b['raw_gyro']  for b in blackout_bins]),
                'gravity':   np.stack([b['gravity']   for b in blackout_bins]),
                'orientation_yaw_vehicle_rad': np.array([b.get('orientation_yaw_vehicle_rad', 0) for b in blackout_bins]),
                'yaw_rate_proxy': blk_yaw_rates,
                'shock_mask': compute_shock_mask(blk_accel)
            }

            # ── 5. Ground Truth (For scoring) ──
            # We use the existing metrics from the user's code, but import them dynamically.
            from data.bin_builder import cumulative_displacement, total_distance_m
            
            gt_cum_disp   = cumulative_displacement(blackout_bins)
            gt_speeds     = np.array([b['v_odo_speed'] for b in blackout_bins])
            gt_headings   = np.array([np.radians(b['v_heading']) for b in blackout_bins])
            
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
                'gps_age_seq': np.array([b.get('mobile_gps_age_sec', 0) for b in blackout_bins]),
            }

            windows.append({
                'context':          context,
                'blackout':         blackout,
                'ground_truth':     ground_truth,
                'blackout_dur_sec': blackout_dur_sec,
                'context_dur_sec':  context_sec,
                'window_start_sec': context_bins[0]['abs_sec'],
                'turn_fraction':    turn_fraction
            })

    print(f"  Generated {len(windows)} 10Hz blackout windows across durations {blackout_durations}s")
    for d in blackout_durations:
        n = sum(1 for w in windows if w['blackout_dur_sec'] == d)
        print(f"    {d:3d}s blackout: {n} windows")
    return windows
