with open("train/train_tcn_delta_v.py", "r") as f:
    code = f.read()

import_str = "import numpy as np"
new_import = "import scipy.signal\nimport numpy as np"
code = code.replace(import_str, new_import)

getitem_old = """        x = np.concatenate(fields, axis=1).astype(np.float32)  # (700, 9)"""
getitem_new = """        x = np.concatenate(fields, axis=1).astype(np.float32)  # (700, 9)
        # Apply Causal High-Pass Filter (cutoff=0.01 Hz) to raw_accel (cols 0,1,2)
        sos = scipy.signal.butter(1, 0.01, 'hp', fs=10, output='sos')
        x[:, :3] = scipy.signal.sosfilt(sos, x[:, :3], axis=0)"""
code = code.replace(getitem_old, getitem_new)

loss_old = """            loss = loss_fn(pred_dv, dv) + loss_fn(pred_v, true_v)"""
loss_new = """            # Aggressive penalty on cumulative velocity and final drift
            loss_dv = loss_fn(pred_dv, dv)
            loss_v = loss_fn(pred_v, true_v)
            loss_v_final = loss_fn(pred_v[:, -1], true_v[:, -1])
            loss = loss_dv + loss_v * 20.0 + loss_v_final * 50.0"""
code = code.replace(loss_old, loss_new)

with open("train/train_tcn_delta_v.py", "w") as f:
    f.write(code)

