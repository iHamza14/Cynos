import ast

with open("train/train_speed_model.py", "r") as f:
    lines = f.readlines()

# Add --eval-only argument
insert_arg_idx = -1
for i, line in enumerate(lines):
    if "parser.add_argument(\"--device\"" in line:
        insert_arg_idx = i + 1
        break

if insert_arg_idx != -1:
    lines.insert(insert_arg_idx, '    parser.add_argument("--eval-only", action="store_true", help="Skip training and just evaluate the best saved model")\n')

# Check where to bypass training
insert_train_idx = -1
for i, line in enumerate(lines):
    if "best_val_loss = float(\"inf\")" in line:
        insert_train_idx = i - 1
        break

if insert_train_idx != -1:
    lines.insert(insert_train_idx, '''
    if args.eval_only:
        print(f"Loading best model from {best_path} for evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        val_loss, val_dv, val_v = evaluate(model, test_loader, device)
        print(f"\\n[Evaluation Results] Test Loss (Total MAE): {val_loss:.4f} m/s | Delta-V MAE: {val_dv:.4f} m/s | Absolute Velocity MAE: {val_v:.4f} m/s")
        return
''')

with open("train/train_speed_model.py", "w") as f:
    f.writelines(lines)
