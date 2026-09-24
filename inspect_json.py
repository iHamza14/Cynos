import json
with open('web_trajectory.json', 'r') as f:
    data = json.load(f)
w1 = data[1] # array index 1
gt = w1['ground_truth']
vit = w1['viterbi']
print("GT points:")
for p in gt[:8]: print(p)
print("Vit points:")
for p in vit[:8]: print(p)
