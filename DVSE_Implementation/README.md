# Deep Vehicle Speed Estimator (DVSE) Implementation

This folder contains the complete end-to-end PyTorch implementation of the Deep Vehicle Speed Estimator based on your reference paper architecture.

## Overview of Components
1. **`models.py`**: Contains the PyTorch definitions for the GRU Noise Network, the TCN Motion Transformation Network (MTN), the physics integration layer, and the master DVSE module.
2. **`data_sync.py`**: A robust data-engineering pipeline that aligns timestamps, resamples the IMU and GPS data to a common frequency, applies GNSS quality filtering, extracts the required 36 structural IMU features (min, max, std, rms, skew, kurtosis), and generates 1-second continuous windows.
3. **`dataset.py`**: The PyTorch DataLoader that parses the 1-second chunks and returns batches of 10-second sequences.
4. **`train.py`**: The master training loop, heavily optimized with the 1-second-shifted Loss Matching mechanism mentioned in the paper.

## How to Run

1. **Pre-process Data**:
   Ensure you have `scipy`, `pandas`, and `torch` installed. 
   Run the data synchronization first to build the training windows:
   ```bash
   python data_sync.py
   ```

2. **Train the Model**:
   After the `processed_windows.csv` is created in the `data/` directory, you can begin training:
   ```bash
   python train.py
   ```

## Note on Dependencies
You will need to install `scipy` if you haven't already:
```bash
pip install scipy
```
