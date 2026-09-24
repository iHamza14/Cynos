"""
Generates evaluation plots and metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode: saves directly to PNG without opening any popup windows
import matplotlib.pyplot as plt

# Checkpoint coordinates (Time in seconds vs Drift in meters)
time = [15, 30, 60]
drift = [36.8, 66.5, 160.8]

plt.figure(figsize=(8, 5), dpi=300)

# SCATTER ONLY: No connecting lines or curves
plt.scatter(time, drift, color='#1f77b4', s=120, zorder=5, edgecolor='black', linewidth=1.5, label='Viterbi Path Drift Points')

# Annotate exact value above each point
for t, d in zip(time, drift):
    plt.annotate(f'{t}s: {d}m', (t, d), textcoords='offset points', xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)

plt.xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
plt.ylabel('Position Drift (Meters)', fontsize=12, fontweight='bold')
plt.title('Position Drift at Checkpoints (Points Only)', fontsize=14, fontweight='bold')
plt.xticks([15, 30, 60])
plt.xlim(0, 70)
plt.ylim(0, 190)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', frameon=True)

plt.tight_layout()
output_file = 'scatter_only_drift_plot.png'
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Saved scatter plot to '{output_file}'")
