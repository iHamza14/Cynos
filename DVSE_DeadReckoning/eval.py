
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DeadReckoningDataset
from models import NoiseNetwork


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_stage1():
    dataset = DeadReckoningDataset(
        "data/test_blackout_windows.pkl",
        "data/scalers.pkl",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = NoiseNetwork().to(DEVICE)

    state = torch.load(
        "checkpoints/stage1_noise_net.pth",
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    results = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            (
                acc,
                gyro,
                gravity,
                gt_speed,
                vr_seed,
                gt_dv,
                target_disp,
                heading_target,
                start_heading,
            ) = batch

            acc = acc.to(DEVICE)
            gyro = gyro.to(DEVICE)
            gt_speed = gt_speed.to(DEVICE)
            gt_dv = gt_dv.to(DEVICE)
            vr_seed = vr_seed.to(DEVICE).reshape(1, 1, 1)

            T = gt_speed.shape[1]

                        
            # -------------------------------------------------
            # 1. Autoregressive prediction
            # -------------------------------------------------
            predicted_speeds = []
            predicted_dvs = []

            previous_speed = vr_seed
            hidden = None

            for t in range(T):
                acc_t = acc[:, t:t + 1, :]
                gyro_t = gyro[:, t:t + 1, :]

                # Model predicts delta-v using current IMU
                # and previous predicted velocity.
                dv_t, hidden = model(
                    acc_t,
                    gyro_t,
                    previous_speed,
                    hidden,
                )

                predicted_dvs.append(dv_t)

                # Integrate predicted delta-v.
                previous_speed = previous_speed + dv_t

                predicted_speeds.append(previous_speed)

            pred_dv = torch.cat(predicted_dvs, dim=1)
            pred_speed = torch.cat(predicted_speeds, dim=1)
            # Teacher-forced diagnostic
            gt_prev_speed = torch.cat(
                [vr_seed, gt_speed[:, :-1, :]],
                dim=1,
            )

            tf_pred_dv, _ = model(
                acc,
                gyro,
                gt_prev_speed,
            )
            def mae(pred, target):
                return F.l1_loss(pred, target).item()

            tf_pred_dv_mae = mae(tf_pred_dv, gt_dv)
            # -------------------------------------------------
            # 3. Constant-speed baseline
            # -------------------------------------------------
            constant_speed = vr_seed.expand_as(gt_speed)

            # -------------------------------------------------
            # 4. Metrics
            # -------------------------------------------------

            def rmse(pred, target):
                return torch.sqrt(
                    F.mse_loss(pred, target)
                ).item()

            # Integrate predicted delta-v from initial speed.
            # Compare against ground-truth speeds.
            pred_endpoint_speed = pred_speed[:, -1, :]
            gt_endpoint_speed = gt_speed[:, -1, :]

            result = {
                "window": idx,
                "steps": T,
                "delta_v_mae": mae(pred_dv, gt_dv),
                "delta_v_rmse": rmse(pred_dv, gt_dv),
                "speed_mae": mae(pred_speed, gt_speed),
                "speed_rmse": rmse(pred_speed, gt_speed),
                "endpoint_speed_error": mae(
                    pred_endpoint_speed,
                    gt_endpoint_speed,
                ),
                "constant_speed_mae": mae(
                    constant_speed,
                    gt_speed,
                ),
                "teacher_forced_dv_mae": 
                    tf_pred_dv_mae
                ,
            }

            results.append(result)

    # -----------------------------------------------------
    # Aggregate results by blackout duration
    # -----------------------------------------------------
    durations = {
        150: "15s",
        300: "30s",
        600: "60s",
    }

    print("\n=== Stage 1 Evaluation ===")
    print(f"Device: {DEVICE}")
    print(f"Test windows: {len(results)}")

    for steps, label in durations.items():
        group = [r for r in results if r["steps"] == steps]

        if not group:
            print(f"\n{label}: No windows found")
            continue

        print(f"\n--- {label} ({len(group)} windows) ---")

        metrics = [
            "delta_v_mae",
            "delta_v_rmse",
            "speed_mae",
            "speed_rmse",
            "endpoint_speed_error",
            "constant_speed_mae",
            "teacher_forced_dv_mae",
        ]

        for metric in metrics:
            values = [r[metric] for r in group]
            print(
                f"{metric:25s}: "
                f"{np.mean(values):.4f} "
                f"+/- {np.std(values):.4f}"
            )

    return results


if __name__ == "__main__":
    evaluate_stage1()