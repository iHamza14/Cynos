import re

with open("train/train_speed_model.py", "r") as f:
    content = f.read()

new_dataset_class = """class DVSEDataset(Dataset):
    \"\"\"Enterprise class definition for DVSEDataset remodeled for 10s windows.\"\"\"

    def __init__(self, items, scalers, hz=10, is_train=False):
        self.is_train = is_train
        self.hz = hz
        self.acc_scaler = scalers["raw_accel"]
        self.gyro_scaler = scalers["raw_gyro"]
        
        # Remodel dataset into strict 10s chunks (T=10)
        self.windows = []
        for item in items:
            raw_accel = item["blackout"]["raw_accel"]
            raw_gyro = item["blackout"]["raw_gyro"]
            speeds = item["ground_truth"]["speeds_ms"]
            vr_seed_start = item["context"]["vr_seed_ms"]
            
            total_samples = len(raw_accel)
            chunk_size = 10 * hz
            
            for k in range(total_samples // chunk_size):
                start_idx = k * chunk_size
                end_idx = start_idx + chunk_size
                
                acc_chunk = raw_accel[start_idx:end_idx]
                gyro_chunk = raw_gyro[start_idx:end_idx]
                speed_chunk = speeds[start_idx:end_idx]
                
                if k == 0:
                    v_seed = vr_seed_start
                else:
                    v_seed = speeds[start_idx - 1]
                    
                self.windows.append({
                    "acc": acc_chunk,
                    "gyro": gyro_chunk,
                    "speeds": speed_chunk,
                    "v_seed": v_seed
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # 1. Prepare raw tensors
        raw_accel = torch.tensor(window["acc"], dtype=torch.float32)
        raw_gyro = torch.tensor(window["gyro"], dtype=torch.float32)

        # 2. Apply random rotation to RAW tensors if training (using Section 7 spec)
        if self.is_train:
            import math
            from models.dvse_physics import apply_random_rotation_augmentation
            raw_accel, raw_gyro = apply_random_rotation_augmentation(raw_accel, raw_gyro, max_angle_rad=math.pi)

        # 3. Convert back to numpy for scaling, then back to tensor
        acc_scaled = self.acc_scaler.transform(raw_accel.numpy())
        gyro_scaled = self.gyro_scaler.transform(raw_gyro.numpy())

        speeds = window["speeds"]
        vr_seed = window["v_seed"]

        T = 10
        target_speeds = np.zeros(T, dtype=np.float32)
        target_delta_v = np.zeros(T, dtype=np.float32)
        vr_seq = np.zeros((T, 1), dtype=np.float32)

        v_prev = vr_seed
        for i in range(T):
            target_speeds[i] = speeds[(i + 1) * self.hz - 1]
            target_delta_v[i] = target_speeds[i] - v_prev
            vr_seq[i, 0] = vr_seed  # Scalar value repeated
            v_prev = target_speeds[i]

        return {
            "acc_window": torch.tensor(acc_scaled, dtype=torch.float32),
            "gyro_window": torch.tensor(gyro_scaled, dtype=torch.float32),
            "vr_seq": torch.tensor(vr_seq, dtype=torch.float32),
            "v_0": torch.tensor([vr_seed], dtype=torch.float32),
            "target_speeds": torch.tensor(target_speeds, dtype=torch.float32),
            "target_delta_v": torch.tensor(target_delta_v, dtype=torch.float32),
        }
"""

# Replace the old DVSEDataset with the new one
old_class_start = content.find("class DVSEDataset(Dataset):")
old_class_end = content.find("def smooth_l1_loss(pred, target):")

if old_class_start != -1 and old_class_end != -1:
    content = content[:old_class_start] + new_dataset_class + "\n\n" + content[old_class_end:]
    
with open("train/train_speed_model.py", "w") as f:
    f.write(content)
