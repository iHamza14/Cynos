with open("train/diagnose_tcn_delta_v.py", "r") as f:
    code = f.read()

import_str = "import numpy as np"
new_import = "import scipy.signal\nimport numpy as np"
code = code.replace(import_str, new_import)

getitem_old = """        x = np.concatenate([self.scalers[k].transform(np.concatenate([w['context'][k], w['blackout'][k]], axis=0))
                            for k in ('raw_accel', 'raw_gyro', 'gravity')], axis=1).astype(np.float32)"""
getitem_new = """        x = np.concatenate([self.scalers[k].transform(np.concatenate([w['context'][k], w['blackout'][k]], axis=0))
                            for k in ('raw_accel', 'raw_gyro', 'gravity')], axis=1).astype(np.float32)
        sos = scipy.signal.butter(1, 0.01, 'hp', fs=10, output='sos')
        x[:, :3] = scipy.signal.sosfilt(sos, x[:, :3], axis=0)"""
code = code.replace(getitem_old, getitem_new)

with open("train/diagnose_tcn_delta_v.py", "w") as f:
    f.write(code)

