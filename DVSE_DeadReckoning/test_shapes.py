import torch
vr_seed = torch.randn(8, 1)
delta_v_pred = torch.randn(8, 60, 1)

v_cum = vr_seed + torch.cumsum(delta_v_pred.squeeze(-1), dim=1)
print("Correct v_cum shape:", v_cum.shape)

v_cum_bad = vr_seed.unsqueeze(1) + torch.cumsum(delta_v_pred.squeeze(-1), dim=1)
print("Bad v_cum shape:", v_cum_bad.shape)
