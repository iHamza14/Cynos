import pickle, numpy as np
with open("data/train_blackout_windows.pkl", "rb") as f:
    w = pickle.load(f)[0]
    
print("Context Accel Y:", w["context"]["raw_accel"][:5, 1])
print("GT Delta V:", w["ground_truth"]["delta_v_ms"][:5])
print("Seed:", w["context"]["vr_seed_ms"])
