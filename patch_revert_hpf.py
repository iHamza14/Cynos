with open("train/train_tcn_delta_v.py", "r") as f:
    code = f.read()

bad_hpf_code = """        # Apply Causal High-Pass Filter (cutoff=0.01 Hz) to raw_accel (cols 0,1,2)
        sos = scipy.signal.butter(1, 0.01, 'hp', fs=10, output='sos')
        x[:, :3] = scipy.signal.sosfilt(sos, x[:, :3], axis=0)"""
code = code.replace(bad_hpf_code, "")

with open("train/train_tcn_delta_v.py", "w") as f:
    f.write(code)

with open("train/diagnose_tcn_delta_v.py", "r") as f:
    code = f.read()

bad_hpf_diag = """        sos = scipy.signal.butter(1, 0.01, 'hp', fs=10, output='sos')
        x[:, :3] = scipy.signal.sosfilt(sos, x[:, :3], axis=0)"""
code = code.replace(bad_hpf_diag, "")

with open("train/diagnose_tcn_delta_v.py", "w") as f:
    f.write(code)

