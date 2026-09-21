import re

with open("train/train_speed_model.py", "r") as f:
    content = f.read()

# Fix eval-only
content = content.replace(
    '''if args.eval_only:
        print(f"Loading best model from {best_path} for evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        val_loss, val_dv, val_v = evaluate(model, val_loader, device)''',
    '''if args.eval_only:
        print(f"Loading best model from {best_path} for evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        val_loss, val_dv, val_v = evaluate(model, test_loader, device)'''
)

with open("train/train_speed_model.py", "w") as f:
    f.write(content)
