import pickle
with open("data/train_blackout_windows.pkl", "rb") as f:
    train = pickle.load(f)
lags = [w['ground_truth']['debug']['lag_applied'] for w in train]
print("Min lag:", min(lags), "Max lag:", max(lags))
