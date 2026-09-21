import re

with open("train/train_speed_model.py", "r") as f:
    content = f.read()

# Change chunk size
content = content.replace("chunk_size = 10 * hz", "chunk_size = 5 * hz")
# Change T
content = content.replace("T = 10", "T = 5")
# Change comment
content = content.replace("Remodel dataset into strict 10s chunks (T=10)", "Remodel dataset into strict 5s chunks (T=5)")

with open("train/train_speed_model.py", "w") as f:
    f.write(content)
