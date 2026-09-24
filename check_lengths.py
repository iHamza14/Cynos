import json
with open('web_trajectory.json', 'r') as f:
    data = json.load(f)
for w in data[:10]:
    print(f"Window {w['window_id']}: Context {len(w['context'])}, GT {len(w['ground_truth'])}, Pred {len(w['inference'])}, Vit {len(w['viterbi'])}")
