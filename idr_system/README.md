# Intelligent Dead Reckoning System (v4)

Self-contained Inertial Navigation System (INS) that uses smartphone IMU data to estimate vehicle position and velocity during GNSS outages.

## Overview
This system includes a multi-scale Temporal Convolutional Network (MTN) to estimate orientation alignment, a Noise Compensation Net (GRU) to estimate velocity errors, and a Differentiable Extended Kalman Filter (EKF) to fuse the neural predictions with physics.

## Architecture
- **Module 0**: Preprocessing (Yaw convention, anti-vibration, shock mask)
- **Module 1**: Motion Transformation Net (MTN)
- **Module 2**: Noise Compensation Net
- **Module 3**: Differentiable EKF
- **Module 4**: Remount Detector

## Critical Guarantees
- Single source of truth for rotation math (`models/rotation_utils.py`)
- Analytic Jacobians for the EKF to prevent overconfidence
- Yaw convention correction handled per-device with a remount detector

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy scipy pandas pytest pyyaml
```

## Testing
Run the validation and test scripts:
```bash
python scripts/validate_yaw_convention.py
pytest tests/
```
