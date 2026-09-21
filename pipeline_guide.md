# The Complete IO-VNBD Pipeline Guide

This is your permanent cheat sheet. It explains what the essential files do and the exact order you must run them to go from raw data to a finished Android model.

---

## The Core Structure (What Everything Does)

### 1. `data/` (The Data Factory)
*   **`generate_dataset.py`**: *(Executable)* The script that reads your raw CSV logs, chops them into non-overlapping 60-second blackout windows, and compiles them.
*   **`dvse_dataset.py`**: *(Background)* You never run this directly. It’s a PyTorch helper class that tells the training scripts how to load the `.pkl` files generated above.

### 2. `train/models/` (The Brains)
*   *(Background)* You never run these files directly. This folder holds the architectural blueprints (the math) for your Neural Networks. `dvse.py` builds the speed network, and `dvse_physics.py` holds the dead-reckoning physics equations.

### 3. `train/` (The Engine Room)
*   **`train_speed_model.py`**: *(Executable)* Trains the DVSE model to predict forward speed from accelerometer data.
*   **`train_heading_model.py`**: *(Executable)* Trains the Gyro-TCN model to predict yaw rate (turning) from gyroscope data.
*   **`map_matcher.py`**: *(Executable/Integration)* The Viterbi algorithm. It takes the messy dead-reckoning coordinates and mathematically snaps them to the real-world road network.
*   **`export_to_mobile.py`**: *(Executable)* Converts your trained PyTorch models (`.pt`) into lightweight ONNX models (`.onnx`) so the Android team can plug them into the mobile app.

---

## How to Run the Pipeline (Execution Order)

To build everything from scratch, open your terminal at the root of your project (`SIH/IO-VNBD`) and run these exact commands in order:

### Step 1: Generate the Training Data
This builds the `train`, `test`, and `val` `.pkl` files.
```bash
uv run python3 data/generate_dataset.py
```

### Step 2: Train the Speed Model
Move into the `train` folder and train the DVSE network for 40 epochs. It will save the weights to `train/dvse_output/best_dvse.pt`.
```bash
cd train
uv run python3 train_speed_model.py --data-dir ../data --epochs 40
```

### Step 3: Train the Heading (Gyro) Model
While still in the `train` folder, train the Gyroscope model. It will save the weights to `train/gyro_tcn_output/best_gyro_tcn_processed.pt`.
```bash
uv run python3 train_heading_model.py --data-dir ../data --epochs 50
```

### Step 4: Export to Android
Once both models are fully trained, run the exporter. This bundles the `.pt` weights into `.onnx` files and drops them into `train/onnx_models/` for your mobile team.
```bash
uv run python3 export_to_mobile.py
```
