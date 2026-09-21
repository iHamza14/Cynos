import re

with open("train/models/dvse.py", "r") as f:
    content = f.read()

# Replace forward signature
content = content.replace("def forward(self, acc_window, gyro_window, v_r, v_0=None, hz=10):", "def forward(self, acc_window, gyro_window, v_r, v_0=None, hz=10, autoregressive=False):")

# Modify NCN and Physics to be step-by-step if autoregressive
# Wait, actually let's just rewrite the DVSEModel forward pass safely
