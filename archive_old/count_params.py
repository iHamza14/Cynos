"""
Utility module: count_params.py.
"""

import torch
from train.models import DVSEModel, GyroTCN

def print_model_stats(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)
    print(f"| {name} | {total_params:,} | {size_mb:.2f} MB |")
    return total_params, size_mb

print("| Model Component | Total Parameters | Estimated Size (FP32) |")
print("| :--- | :--- | :--- |")
dvse = DVSEModel()
gyro = GyroTCN()

p1, s1 = print_model_stats(dvse.mtn, "MTN (Motion Transformation Net)")
p2, s2 = print_model_stats(dvse.ncn, "NCN (Noise Compensation Net)")
p3, s3 = print_model_stats(gyro, "Gyro TCN (Heading Model)")

total_p = p1 + p2 + p3
total_s = s1 + s2 + s3
print("| **Total On-Device Payload** | **{:,}** | **{:.2f} MB** |".format(total_p, total_s))
