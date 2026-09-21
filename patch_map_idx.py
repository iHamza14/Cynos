import re

with open("plot_map_trajectory.py", "r") as f:
    content = f.read()

content = content.replace("item = test_items[0]", "item = test_items[12]")
content = content.replace(
    'plt.title("Map Trajectory: Dead-Reckoning (Speed Model + True Heading)", fontsize=14)',
    'plt.title("Map Trajectory for Window 13 (Speed Model + True Heading)", fontsize=14)'
)

with open("plot_map_trajectory.py", "w") as f:
    f.write(content)
