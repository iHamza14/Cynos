"""
Generates evaluation plots and metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode: directly saves PNG without popup windows
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#d0d0d0'
plt.rcParams['axes.linewidth'] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, constrained_layout=True)

# =========================================================================
# PLOT 1: Position Drift vs SIH Target (15s, 30s, 60s)
# =========================================================================
horizons = [
    '15s Blackout\n(Travel ~270m)',
    '30s Blackout\n(Travel ~528m)',
    '60s Blackout\n(Travel ~1077m)'
]

sih_target = [27.0, 52.8, 107.7]      # SIH Target (<10% Drift)
viterbi_e2e = [36.8, 66.5, 160.8]     # Model Performance (Viterbi Path)
isolated_speed = [26.0, 48.0, 61.0]   # Isolated Speed Model

x = np.arange(len(horizons))
width = 0.26

r1 = ax1.bar(x - width, sih_target, width, label='SIH Target (< 10% Drift)', color='#2ca02c', alpha=0.9, edgecolor='#1b611b')
r2 = ax1.bar(x, viterbi_e2e, width, label='Model Median Drift (Viterbi E2E)', color='#1f77b4', alpha=0.9, edgecolor='#124a73')
r3 = ax1.bar(x + width, isolated_speed, width, label='Isolated Speed Model', color='#ff7f0e', alpha=0.9, edgecolor='#b35504')

ax1.set_ylabel('Position Drift (Meters)', fontsize=12, fontweight='bold', labelpad=8)
ax1.set_title('A. Position Drift vs. SIH Target by Horizon', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(horizons, fontsize=10.5)
ax1.set_ylim(0, 195)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

# Bar value labels with offset to avoid overlap
for rect in r1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}m', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
for i, rect in enumerate(r2):
    h = rect.get_height()
    offset = 12 if i == 0 else 3
    ax1.annotate(f'{h:.1f}m', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, offset), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
for rect in r3:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}m', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# =========================================================================
# PLOT 2: Benchmark Compliance / Pass Rate (< 10% and < 20%)
# =========================================================================
labels = ['15s Blackout', '30s Blackout', '60s Blackout']
pass_10 = [46.2, 34.6, 42.3]  # Meet strict < 10% threshold
pass_20 = [50.0, 57.7, 57.7]  # Within < 20% usable trajectory

y = np.arange(len(labels))
bar_height = 0.35

b1 = ax2.barh(y + bar_height/2, pass_10, bar_height, label='Strict Pass (< 10% Drift)', color='#2ecc71', alpha=0.9, edgecolor='#1e8449')
b2 = ax2.barh(y - bar_height/2, pass_20, bar_height, label='Usable Path (< 20% Drift)', color='#3498db', alpha=0.9, edgecolor='#21618c')

ax2.set_xlabel('% of Test Windows Passing Benchmark', fontsize=12, fontweight='bold', labelpad=8)
ax2.set_title('B. Benchmark Compliance Rate on Test Windows', fontsize=13, fontweight='bold', pad=12)
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=10.5)
ax2.set_xlim(0, 75)
ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
ax2.grid(True, linestyle='--', alpha=0.4, axis='x')

# Value labels on horizontal bars
for bar in b1:
    w = bar.get_width()
    ax2.annotate(f' {w:.1f}%', xy=(w, bar.get_y() + bar.get_height()/2), xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9.5, fontweight='bold')
for bar in b2:
    w = bar.get_width()
    ax2.annotate(f' {w:.1f}%', xy=(w, bar.get_y() + bar.get_height()/2), xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9.5, fontweight='bold')

# Save directly to PNG
output_file = 'benchmark_table_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Chart saved to '{output_file}'")