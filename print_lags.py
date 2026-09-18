import pickle
train = pickle.load(open('data/train_blackout_windows.pkl', 'rb'))
lags = [w['ground_truth']['debug']['lag_applied'] for w in train]
from collections import Counter
print(Counter(lags))
