import re

with open("train/train_speed_model.py", "r") as f:
    content = f.read()

# 1. Dataset __init__ and is_train
content = content.replace(
    "def __init__(self, items, scalers, hz=10):",
    "def __init__(self, items, scalers, hz=10, is_train=False):\n        self.is_train = is_train"
)

# 2. Random Rotation and vr_seq logic in __getitem__
old_getitem = """        raw_accel = item["blackout"]["raw_accel"]
        raw_gyro = item["blackout"]["raw_gyro"]

        acc_scaled = self.acc_scaler.transform(raw_accel)
        gyro_scaled = self.gyro_scaler.transform(raw_gyro)

        speeds = item["ground_truth"]["speeds_ms"]  # [600]
        vr_seed = item["context"]["vr_seed_ms"]

        # Calculate 1Hz targets and V_r sequence
        T = len(speeds) // self.hz
        target_speeds = np.zeros(T, dtype=np.float32)
        target_delta_v = np.zeros(T, dtype=np.float32)
        vr_seq = np.zeros((T, 1), dtype=np.float32)

        v_prev = vr_seed
        for i in range(T):
            target_speeds[i] = speeds[(i + 1) * self.hz - 1]  # speed at the end of the second
            target_delta_v[i] = target_speeds[i] - v_prev
            vr_seq[i, 0] = v_prev
            v_prev = target_speeds[i]"""

new_getitem = """        raw_accel = item["blackout"]["raw_accel"]
        raw_gyro = item["blackout"]["raw_gyro"]

        if self.is_train:
            from scipy.spatial.transform import Rotation
            R = Rotation.random().as_matrix().astype(np.float32)
            raw_accel = raw_accel @ R.T
            raw_gyro = raw_gyro @ R.T

        acc_scaled = self.acc_scaler.transform(raw_accel)
        gyro_scaled = self.gyro_scaler.transform(raw_gyro)

        speeds = item["ground_truth"]["speeds_ms"]  # [600]
        vr_seed = item["context"]["vr_seed_ms"]

        # Calculate 1Hz targets and V_r sequence
        T = len(speeds) // self.hz
        target_speeds = np.zeros(T, dtype=np.float32)
        target_delta_v = np.zeros(T, dtype=np.float32)
        vr_seq = np.zeros((T, 1), dtype=np.float32)

        v_prev = vr_seed
        for i in range(T):
            target_speeds[i] = speeds[(i + 1) * self.hz - 1]  # speed at the end of the second
            target_delta_v[i] = target_speeds[i] - v_prev
            # PER PAPER: vr_seq is exactly the vr_seed scalar passed at every step!
            vr_seq[i, 0] = vr_seed
            v_prev = target_speeds[i]"""

content = content.replace(old_getitem, new_getitem)

# 3. Fix Datasets initialization
content = content.replace(
    "train_dataset = DVSEDataset(train_items, scalers)",
    "train_dataset = DVSEDataset(train_items, scalers, is_train=True)"
)
content = content.replace(
    "val_dataset = DVSEDataset(val_items, scalers)",
    "val_dataset = DVSEDataset(val_items, scalers, is_train=False)"
)
content = content.replace(
    "test_dataset = DVSEDataset(test_items, scalers)",
    "test_dataset = DVSEDataset(test_items, scalers, is_train=False)"
)

# 4. Remove autoregressive=True/False flags from evaluate and train
content = content.replace("autoregressive=True", "")
content = content.replace("autoregressive=False", "")
content = content.replace("v_0=v_0, )", "v_0=v_0)")

# 5. Optimizer and Scheduler
old_optim = "optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)"
new_optim = """optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)"""
content = content.replace(old_optim, new_optim)

# 6. Step scheduler
content = content.replace(
    "print(\n            f\"Epoch {epoch:03d}/{args.epochs}",
    "scheduler.step()\n\n        print(\n            f\"Epoch {epoch:03d}/{args.epochs}"
)

with open("train/train_speed_model.py", "w") as f:
    f.write(content)
