import pickle
with open('data/test_blackout_windows.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Loaded {len(data)} windows.")
if len(data) > 0:
    print(data[0].keys())
    print("Metadata:", data[0].get('metadata'))
    print("Keys in blackout:", data[0]['blackout'].keys())
    # See if there's an 'E' dataset identifier
    for i, w in enumerate(data[:10]):
        print(f"Window {i}: {w.get('metadata')}")
