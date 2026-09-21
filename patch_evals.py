import os

scripts = ["evaluate_checkpoints.py", "plot_predictions.py", "plot_map_trajectory.py"]

for script in scripts:
    with open(script, "r") as f:
        content = f.read()
        
    content = content.replace("for k in range(6):", "for k in range(12):")
    content = content.replace("for k in range(6): # 6 leaps of 10 seconds", "for k in range(12): # 12 leaps of 5 seconds")
    
    content = content.replace("k*100 : (k+1)*100", "k*50 : (k+1)*50")
    
    content = content.replace("torch.full((1, 10, 1)", "torch.full((1, 5, 1)")
    
    with open(script, "w") as f:
        f.write(content)
