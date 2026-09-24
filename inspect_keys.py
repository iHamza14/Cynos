import pickle
with open('data/test_blackout_windows.pkl', 'rb') as f:
    data = pickle.load(f)

w = data[0]
print("Context keys:", w['context'].keys())
print("Ground truth keys:", w['ground_truth'].keys())
