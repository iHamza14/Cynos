import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

# ── Load ──────────────────────────────────────────────────────────────────────
pkl_path = sys.argv[1] if len(sys.argv) > 1 else './data/train_windows.pkl'

with open(pkl_path, 'rb') as f:
    windows = pickle.load(f)

print(f"\n{'='*55}")
print(f"  PKL: {os.path.basename(pkl_path)}")
print(f"  Total windows : {len(windows)}")
print(f"{'='*55}")

# ── 1. Schema inspection ──────────────────────────────────────────────────────
w0 = windows[0]
print("\n[1] Keys and shapes in window[0]:")
for k, v in w0.items():
    if isinstance(v, np.ndarray):
        print(f"    {k:<25} ndarray  shape={v.shape}  dtype={v.dtype}")
    elif isinstance(v, (float, int)):
        print(f"    {k:<25} scalar   value={v:.4f}")
    else:
        print(f"    {k:<25} {type(v).__name__}  = {v}")

# ── 2. Global stats ───────────────────────────────────────────────────────────
all_disp   = np.stack([w['target_disp'] for w in windows])          # (N, 2)
all_seed   = np.array([w['vr_seed'] for w in windows])              # (N,)
all_dv_max = np.array([w['delta_v_seq'].max() for w in windows])
all_dv_min = np.array([w['delta_v_seq'].min() for w in windows])
displacements_m = np.linalg.norm(all_disp, axis=1)                  # Euclidean dist

print(f"\n[2] Dataset-level stats ({len(windows)} windows):")
print(f"    Displacement (m)   mean={displacements_m.mean():.1f}  "
      f"std={displacements_m.std():.1f}  "
      f"min={displacements_m.min():.1f}  max={displacements_m.max():.1f}")
print(f"    vr_seed (m/s)      mean={all_seed.mean():.2f}  "
      f"std={all_seed.std():.2f}")
print(f"    delta_v peak       max={all_dv_max.max():.3f}  min={all_dv_min.min():.3f}")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(f'Dataset Inspection: {os.path.basename(pkl_path)}  '
             f'({len(windows)} windows)', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- 3a. Displacement distribution ---
ax = fig.add_subplot(gs[0, 0])
ax.hist(displacements_m, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_title('Total Displacement per Window (m)')
ax.set_xlabel('metres'); ax.set_ylabel('count')

# --- 3b. North vs East scatter ---
ax = fig.add_subplot(gs[0, 1])
ax.scatter(all_disp[:, 1], all_disp[:, 0], alpha=0.4, s=10, c='tomato')
ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)
ax.set_title('Target Displacement: North vs East')
ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
ax.set_aspect('equal', adjustable='datalim')

# --- 3c. vr_seed distribution ---
ax = fig.add_subplot(gs[0, 2])
ax.hist(all_seed * 3.6, bins=40, color='mediumseagreen', edgecolor='white', linewidth=0.5)
ax.set_title('Starting Speed (km/h)')
ax.set_xlabel('km/h'); ax.set_ylabel('count')

# --- 3d. Single window: raw_accel over time ---
w = windows[0]
T = w['raw_accel'].shape[0]
t = np.arange(T)
ax = fig.add_subplot(gs[1, 0])
labels = ['X', 'Y', 'Z']
for i, l in enumerate(labels):
    ax.plot(t, w['raw_accel'][:, i], label=l)
ax.set_title('Window[0]: Raw Accel (m/s²)')
ax.set_xlabel('seconds'); ax.legend(fontsize=8)

# --- 3e. Single window: gravity over time ---
ax = fig.add_subplot(gs[1, 1])
for i, l in enumerate(labels):
    ax.plot(t, w['gravity'][:, i], label=l)
ax.set_title('Window[0]: Gravity (m/s²)')
ax.set_xlabel('seconds'); ax.legend(fontsize=8)

# --- 3f. Single window: delta_v target ---
ax = fig.add_subplot(gs[1, 2])
ax.plot(t, w['delta_v_seq'].squeeze(), color='darkorange')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_title('Window[0]: Δv target (m/s per sec)')
ax.set_xlabel('seconds')

# --- 3g. Single window: vr_train (odometry speed) ---
ax = fig.add_subplot(gs[2, 0])
ax.plot(t, w['vr_train'].squeeze() * 3.6, color='purple')
ax.set_title('Window[0]: Odometry Speed (km/h)')
ax.set_xlabel('seconds')

# --- 3h. Single window: heading target ---
ax = fig.add_subplot(gs[2, 1])
heading_deg = np.degrees(w['heading_target_rad']) if 'heading_target_rad' in w \
              else np.degrees(w['heading_target_rad'])
ax.plot(t, heading_deg, color='teal')
ax.set_title('Window[0]: Vehicle Heading (°)')
ax.set_xlabel('seconds')

# --- 3i. acc_feat heatmap (first window) ---
ax = fig.add_subplot(gs[2, 2])
feat_names = ['std','max','min','rms','skew','kurt'] * 3
im = ax.imshow(w['acc_feat'].T, aspect='auto', cmap='RdBu_r')
ax.set_title('Window[0]: acc_feat heatmap (18 × T)')
ax.set_xlabel('seconds'); ax.set_ylabel('feature index')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

out_path = os.path.join(os.path.dirname(pkl_path) or '.', 'inspection.png')
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\n[3] Plot saved → {out_path}")
plt.show()