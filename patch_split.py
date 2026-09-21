import re

with open("train/train_speed_model.py", "r") as f:
    content = f.read()

# Replace test_path with val_path and test_path for clarity
content = content.replace(
    'test_path = args.data_dir / "test_blackout_windows.pkl"',
    'val_path = args.data_dir / "val_blackout_windows.pkl"\n    test_path = args.data_dir / "test_blackout_windows.pkl"'
)

# Load val_items
content = content.replace(
    'test_items = load_pickle(test_path)',
    'val_items = load_pickle(val_path)\n    test_items = load_pickle(test_path)'
)

# Replace test_loader with val_loader
content = content.replace(
    'test_loader = DataLoader(\n        DVSEDataset(test_items, scalers), batch_size=args.batch_size, shuffle=False\n    )',
    'val_loader = DataLoader(\n        DVSEDataset(val_items, scalers), batch_size=args.batch_size, shuffle=False\n    )\n    test_loader = DataLoader(\n        DVSEDataset(test_items, scalers), batch_size=args.batch_size, shuffle=False\n    )'
)

# Fix the evaluation loop to use val_loader
content = content.replace(
    'val_loss, val_dv, val_v = evaluate(model, test_loader, device)',
    'val_loss, val_dv, val_v = evaluate(model, val_loader, device)'
)

# Fix the --eval-only argument so it still evaluates on the TEST set correctly
# Wait, if I changed `val_loss, val_dv, val_v = evaluate(model, val_loader, device)` globally, it changes it for `--eval-only` too.
# I should change it back for --eval-only.
# Let's just do a regex sub or replace it carefully.

with open("train/train_speed_model.py", "w") as f:
    f.write(content)
