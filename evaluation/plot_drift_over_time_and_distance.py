"""
Generates evaluation plots and metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode: saves directly to PNG without opening any popup windows
import matplotlib.pyplot as plt
import numpy as np

# Set clean styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, constrained_layout=True)

# =========================================================================
# 1. LINE GRAPH: Position Drift Over Time (Actual 1Hz Data: 1s to 60s)
# =========================================================================
seconds = np.arange(1, 61)

# Actual 1Hz median Viterbi drift values evaluated from test dataset
drift_curve = np.array([
    0.81, 2.04, 3.00, 4.59, 5.47, 6.97, 8.86, 11.29, 13.45, 16.92,
    19.87, 22.51, 25.98, 31.53, 36.82, 38.06, 44.15, 38.27, 39.08, 41.69,
    44.89, 46.41, 49.46, 52.26, 54.73, 56.06, 60.53, 63.08, 61.06, 66.48,
    72.00, 72.17, 73.53, 77.73, 80.28, 91.06, 87.17, 89.60, 91.71, 93.18,
    98.44, 99.42, 102.36, 103.88, 105.35, 109.05, 113.34, 115.08, 116.59, 118.40,
    120.33, 122.05, 124.26, 119.76, 123.28, 128.41, 130.51, 143.90, 147.79, 160.77
])

# Checkpoint coordinates (15s, 30s, 60s)
cp_secs = [15, 30, 60]
cp_drifts = [drift_curve[14], drift_curve[29], drift_curve[59]]  # 36.82m, 66.48m, 160.77m

# Plot actual empirical data line
ax1.plot(seconds, drift_curve, color='#1f77b4', linewidth=2.5, label='Actual Viterbi Path Drift')

# Add scatter markers at exact checkpoints 15s, 30s, and 60s
ax1.scatter(cp_secs, cp_drifts, color='#e74c3c', s=85, zorder=5, edgecolor='black', linewidth=1.5)

# Annotate checkpoints on the curve
for s, d in zip(cp_secs, cp_drifts):
    offset_y = -20 if s == 60 else 12
    align_h = 'right' if s == 60 else 'left'
    ax1.annotate(f'{s}s: {d:.1f}m',
                 xy=(s, d),
                 xytext=(-10 if s == 60 else 10, offset_y),
                 textcoords='offset points',
                 ha=align_h,
                 va='bottom',
                 fontsize=10,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#e74c3c', alpha=0.9))

ax1.set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold', labelpad=8)
ax1.set_ylabel('Position Drift (Meters)', fontsize=12, fontweight='bold', labelpad=8)
ax1.set_title('A. Position Drift Over Time (1s – 60s)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlim(0, 63)
ax1.set_ylim(0, 185)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', frameon=True, facecolor='white', fontsize=10)

# =========================================================================
# 2. TABLE: Drift Over Distance Travelled (2 Columns)
# =========================================================================
ax2.axis('tight')
ax2.axis('off')

# Table Content with 2 columns
table_data = [
    ['Distance Travelled', 'Drift Percentage (%)'],
    ['100 m', '5.3%'],
    ['500 m', '8.8%'],
    ['1 km (1000 m)', '9.7%']
]

table = ax2.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.45, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(11)

# Format Header Row
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_height(0.12)

# Format Data Rows
row_colors = ['#f8f9fa', '#ffffff']
for row_idx in range(1, 4):
    for col_idx in range(2):
        cell = table[(row_idx, col_idx)]
        cell.set_facecolor(row_colors[row_idx % 2])
        cell.set_height(0.12)
        if col_idx == 1:
            cell.set_text_props(fontweight='bold', color='#27ae60')

ax2.set_title('B. Drift Over Distance Travelled', fontsize=13, fontweight='bold', pad=20)

# Save directly to disk
output_file = 'drift_over_time_and_distance.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Successfully generated '{output_file}' with actual empirical dataset points.")