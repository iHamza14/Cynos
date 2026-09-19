import os
import sys
from pathlib import Path

import torch

# Add models directory to path
sys.path.append(os.path.abspath("models"))
from models import DVSEModel, GyroTCN


def export_models():
    """Executes core logic for export_models."""
    out_dir = Path("onnx_models")
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cpu")

    # --- Export Gyro TCN ---
    print("Exporting Gyro TCN...")
    gyro_model = GyroTCN().to(device)
    gyro_ckpt = torch.load(
        "gyro_tcn_output/best_gyro_tcn_processed.pt", map_location=device, weights_only=False
    )
    gyro_model.load_state_dict(gyro_ckpt["model_state"])
    gyro_model.eval()

    # Dummy input: [Batch, Time, Features] = [1, 100, 3]
    dummy_gyro = torch.randn(1, 100, 3, dtype=torch.float32)

    torch.onnx.export(
        gyro_model,
        dummy_gyro,
        out_dir / "gyro_tcn.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["gyro_scaled"],
        output_names=["yaw_rate_norm"],
        dynamic_axes={"gyro_scaled": {0: "batch_size"}, "yaw_rate_norm": {0: "batch_size"}},
    )
    print("Saved Gyro TCN to onnx_models/gyro_tcn.onnx")

    # --- Export DVSE ---
    print("Exporting DVSE Model...")
    dvse_model = DVSEModel().to(device)
    dvse_model.load_state_dict(
        torch.load("dvse_output/best_dvse.pt", map_location=device, weights_only=False)
    )
    dvse_model.eval()

    # Dummy inputs: [1, 100, 3] for IMU, [1, 10, 1] for VR sequence, [1, 1] for initial velocity
    dummy_acc_window = torch.randn(1, 100, 3, dtype=torch.float32)
    dummy_gyro_window = torch.randn(1, 100, 3, dtype=torch.float32)
    dummy_vr_seq = torch.randn(1, 10, 1, dtype=torch.float32)
    dummy_v0 = torch.randn(1, 1, dtype=torch.float32)

    # Note: DVSE forward has `hz=10` as a default kwarg.
    # In ONNX export, we can just pass the positional arguments.
    torch.onnx.export(
        dvse_model,
        (dummy_acc_window, dummy_gyro_window, dummy_vr_seq, dummy_v0),
        out_dir / "dvse.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["acc_window", "gyro_window", "vr_seq", "v0"],
        output_names=["delta_v", "v", "N_disturbance", "angles"],
        dynamic_axes={
            "acc_window": {0: "batch_size"},
            "gyro_window": {0: "batch_size"},
            "vr_seq": {0: "batch_size"},
            "v0": {0: "batch_size"},
            "delta_v": {0: "batch_size"},
            "v": {0: "batch_size"},
        },
    )
    print("Saved DVSE to onnx_models/dvse.onnx")

    print("\nONNX Export Complete (FP32 Precision preserved)")


if __name__ == "__main__":
    export_models()
