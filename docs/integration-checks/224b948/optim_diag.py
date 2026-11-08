"""Controlled optimizer diagnostic using the repo's own train_step, data and model.

Only the optimizer object differs between arms; everything else is the committed
code path. Prints, every `every` steps, the mean training reconstruction loss over
that window, the same loss for the zero-velocity prediction on the same batches
(a fixed yardstick), the mean pre-clip generator gradient norm, critic accuracies,
and the S11 Euler validation error (mean of 80/160/320/400 ms).

usage: python optim_diag.py <data root> <variant> <adam|sgd> <lr> <steps> [every]
"""
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from aged import data as D
from aged.config import Config
from aged.evaluate import mean_angle_error
from aged.losses import GeodesicLoss
from aged.runtime import build_model, seed_everything
from aged.train import critic_parameters, reconstruction, train_step

root, variant, opt_name, lr, steps = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
every = int(sys.argv[6]) if len(sys.argv) > 6 else 250
config = Config(variant=variant, learning_rate=lr, steps_per_epoch=steps, val_subject="S11")
seed_everything(0)
device = torch.device("cuda")
subjects = [s for s in D.TRAIN_SUBJECTS if s != "S11"]
norm = D.fit_normalisation(root, subjects)
train = D.WindowDataset(root, norm, subjects, length=steps * config.batch_size, seed=0)
val = D.WindowDataset(root, norm, ["S11"], length=256, seed=999)
model = build_model(config, norm).to(device)
geometry = GeodesicLoss(norm, config.geodesic_reduction).to(device)


def make(params):
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise SystemExit(f"unknown optimizer {opt_name}")


g_opt = make(model.predictor.parameters())
d_params = critic_parameters(model)
d_opt = make(d_params) if d_params else None


def validate():
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for o, t, a in DataLoader(val, batch_size=64):
            preds.append(model.predict(o.to(device), a.to(device)).cpu().numpy())
            targets.append(t.numpy())
    model.train()
    return mean_angle_error(np.concatenate(preds), np.concatenate(targets), norm)


obs = np.stack([val[i][0].numpy() for i in range(len(val))])
tgt = np.stack([val[i][1].numpy() for i in range(len(val))])
zv_err = mean_angle_error(np.repeat(obs[:, -1:], tgt.shape[1], 1), tgt, norm)
print(f"S11 zero-velocity euler: {zv_err}  mean {np.mean(list(zv_err.values())):.3f}", flush=True)
print(f"variant={variant} optimizer={opt_name} lr={lr} steps={steps} batch={config.batch_size} "
      f"clip={config.gradient_clip} lambda_adv={config.lambda_adv}", flush=True)
model.train()
acc = {"rec": 0.0, "zv": 0.0, "gn": 0.0, "fa": 0.0, "ca": 0.0, "n": 0}
started = time.time()
for step, (o, t, a) in enumerate(DataLoader(train, batch_size=config.batch_size), 1):
    o, t, a = o.to(device), t.to(device), a.to(device)
    with torch.no_grad():
        zv = reconstruction(config, geometry, o[:, -1:].expand_as(t).contiguous(), t)
    diag = train_step(model, geometry, (o, t, a), g_opt, d_opt, config)
    acc["rec"] += diag["reconstruction"]; acc["zv"] += float(zv); acc["gn"] += diag["generator_grad_norm"]
    acc["fa"] += diag.get("fidelity_accuracy", float("nan")); acc["ca"] += diag.get("continuity_accuracy", float("nan"))
    acc["n"] += 1
    if step % every == 0:
        n = acc["n"]
        err = validate()
        print(f"step {step:5d}  rec {acc['rec']/n:10.3f}  zero-vel {acc['zv']/n:8.3f}  "
              f"gnorm {acc['gn']/n:12.1f}  fid_acc {acc['fa']/n:.3f}  con_acc {acc['ca']/n:.3f}  "
              f"S11 euler {np.mean(list(err.values())):.3f}  ({time.time()-started:.0f}s)", flush=True)
        acc = {k: 0.0 for k in acc}; acc["n"] = 0
print(f"final S11 euler per horizon: {err}", flush=True)
