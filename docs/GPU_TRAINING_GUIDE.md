# GPU training: hyperparameters, experiment priorities and quick validation

Applies to integration commit `51d23d1` and later versions with the same interfaces. Run commands from the repository root in your existing GPU Python environment. No PyTorch reinstall is required. Training interfaces, short/long YAML files and integration v1 checkpoints remain compatible. Defaults also stay unchanged: **updating the code does not automatically switch Adam to SGD**.

## Evidence and current findings

The team-supplied `integration-checks.zip` records commit `224b948` on an RTX 4070 with a real Human3.6M text export. All 22 pipeline checks succeeded, including 14 automated tests. This is a different count from the 22 passing automated CPU tests at the initial reconciliation stage.

- Training, resume, validation, TensorBoard and S5 evaluation ran successfully on the earlier GPU version.
- Short-term zero-velocity values and four long-term action rows match the reference. The short-term table uses the documented inferred correction of 0.63 for sitting at 160 ms.
- At LR 0.005, Adam training deteriorated while SGD was much more stable. After 2000 updates, mean S11 errors were plain/Adam 10.155, plain/SGD 0.808, full/Adam 10.472 and full/SGD 0.718. Zero velocity on the same S11 windows scored 0.704.
- This supports checking the optimizer/LR combination first. It does not establish the original authors' optimizer or successful reproduction with SGD.
- The earlier record contains no step-0 evaluation, so it cannot show whether training performed worse than random initialization.

Original reports, logs, configurations, results and the data fingerprint are archived unchanged under [integration-checks/224b948](integration-checks/README.md). The uploaded ZIP remains in the workspace; the repository stores its extracted contents without a duplicate ZIP. No dataset or model weights are included. Local CPU checks do not constitute a rerun of those GPU experiments.

## Fixed settings and choices to validate

Fix data, seeds and update budgets before changing unspecified implementation details. Select candidates on S11 and reserve S5 for final reporting.

| Parameter | Current value | Paper evidence / validation policy |
| --- | --- | --- |
| Predictor hidden size, batch size, gradient clip | 1024, 16, 5 | Explicit paper settings; retain initially. Also fix critic hidden size at 1024, which is an implementation choice |
| Learning rate, lambda, formal epochs | 0.005, 0.6, 50 | Explicit paper settings; test SGD at 0.005 first. A lower Adam LR is a separate candidate |
| Observed / predicted frames | 50 / 10; long-term 50 / 25 | Explicit paper settings; validate short-term first, then train a separate long-term model |
| Geodesic loss | Sum over time/joints, mean over batch | Eq. 4 does not square the distance. Do not replace it with a square or mean merely to reduce its magnitude |
| Adversarial objective | `minimax` | Paper objective; report `non_saturating` separately as an engineering variant |
| `optimizer` | `adam` | Not named in the paper; **P0: test `sgd` first** |
| `learning_rate_decay` / `every` | 1.0 / 10000 | Unspecified; try 0.95 / 10000 after establishing stability |
| `critic_optimizer` / `critic_learning_rate` | Follow G | Independently configurable; retain initially, adjust D if training is imbalanced |
| Critic architecture / optimizers | `projected` / joint | Compare team `direct` / separate optimizers one change at a time |
| Sampling / earliest start | `windows` / 0 | Not fully specified; separately test `recordings` and 16 |
| `steps_per_epoch` | 2000 | Engineering choice; compare total updates, not epoch count alone |
| S11 / validation windows | Optional / 256, fixed seed 999 | Enable during development; exclude S11 from training and normalization fitting |

The learning rate applies to both G and D unless `--critic-learning-rate` is set. An Adam candidate can start at 0.0005, then try 0.0001 if needed. These are proposed experiments, not paper-specified values.

## Experiment priorities and combinations

Except for explicit changes below, fix all 15 actions, hidden size 1024, batch 16, clip 5, lambda 0.6, seed 0, S11 validation, minimax, geodesic sum and no LR decay. The diagnostic budget is **2 epochs x 1000 steps = 2000 updates**, with validation and a snapshot every 1000 updates.

| Priority / ID | Combination | Change relative to its reference | When to run |
| --- | --- | --- | --- |
| P0 / A | full + SGD, LR 0.005 | Change the optimizer from compatibility defaults | First check of the full model's stability |
| P0 / B | plain + SGD, LR 0.005 | Remove critics and use MSE relative to A | Check basic predictor learning; this is a reference, not a single-factor loss ablation |
| P1 / C | geodesic + SGD | Remove both critics relative to A | If A is unstable or to isolate the adversarial contribution |
| P1 / D | full + Adam, LR 0.0005 | Reduce LR from the earlier Adam 0.005 run | If SGD is unsatisfactory or an optimizer comparison is needed; no need to repeat known-poor Adam 0.005 |
| P1 / E | A + critic LR 0.001 | Reduce only D LR; keep G at 0.005 | If critics approach perfect accuracy while G validation stalls |
| P2 / F | A + `critic_architecture=direct` | Change only the critic architecture | After establishing a stable reference |
| P2 / G | A + separate critic optimizers | Add only `separate_critic_optimizers` | Same stage; this also changes per-critic clipping |
| P2 / H | A + `sampling_strategy=recordings` | Change only the recording distribution | Then separately test `minimum_start=16` |
| P2 / I | Stable candidate + decay 0.95/10000 | Change only the schedule | Run beyond 10000 updates to observe decay |
| P3 / J | A + `non_saturating` | Change only the generator objective | Stability extension; report the departure from paper minimax |

Avoid a full grid search. Start with A, then B/C as needed, and select later experiments based on the observed issue. F/G/H/J each compare with A; do not accumulate changes from preceding rows. `column` only changes the critic output layout and need not be an accuracy-search variable here.

The extra LR settings in D/E and the objective in J are engineering comparisons, not runs using every explicit paper hyperparameter. Report choices for unspecified schedules, architecture details and sampling too.

`configs/team-reference.yaml` simultaneously changes critics, optimizer grouping, sampling, schedule, objective and early stopping. Test individual changes first, then evaluate it as a labeled team combination. It is not a fully determined original-paper configuration.

## Fastest validation workflow

The single-line commands below work in Bash, Git Bash or PowerShell. Use the same dataset root and a new `--out` per run. An existing working CUDA environment needs no installer rerun.

### Step 1: environment, version and baseline

```bash
git rev-parse HEAD
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q
python -m aged.evaluate --root data/h36m --baseline --check-baseline --output experiments/current-baseline.json
```

These check deployment and evaluation, not learning quality. Baseline evaluation is primarily CPU computation. Once this installation and export pass, do not repeat the checks before every candidate. Conditional CUDA tests require usable CUDA; consult actual counts and skip reasons in the log.

The baseline gate requires all 80/160/320/400-ms entries for the selected actions. A horizon too short to provide them cannot pass.

### Step 2: run A only, for about 2000 updates

```bash
python -m aged.train --root data/h36m --config configs/short-term.yaml --variant full --optimizer sgd --val-subject S11 --epochs 2 --steps-per-epoch 1000 --save-every 1 --seed 0 --device cuda --out experiments/p0-A-full-sgd
```

Earlier RTX 4070 short-term full runs took about 40-45 ms/update, or 80-90 seconds of training for 2000 updates, plus loading, validation and saving. This is an estimate from older code, not a performance guarantee for other hardware or new combinations. With a tight budget, run A first. Tens or 200 updates can expose runtime errors but are insufficient to assess convergence.

2x1000 is a diagnostic budget, not the paper's epoch definition. Changing steps per epoch also changes epoch-dependent sampling, so do not expect exact pointwise equality with the earlier 1x2000 diagnostic. Use 2x1000 consistently for new candidates.

### Step 3: run B when a predictor reference is needed

```bash
python -m aged.train --root data/h36m --config configs/short-term.yaml --variant plain --optimizer sgd --val-subject S11 --epochs 2 --steps-per-epoch 1000 --save-every 1 --seed 0 --device cuda --out experiments/p0-B-plain-sgd
```

Compare the two epochs' `validation_angle_error`, `reconstruction` and gradients in `results.json`. Do not directly compare plain MSE magnitudes with full geodesic-sum magnitudes.

### Exact changes for later candidates

Start from A's full command, change `--out`, and apply the relevant single change. These fragments are not standalone commands.

| ID | Command change |
| --- | --- |
| C | Replace `--variant full` with `--variant geodesic` |
| D | Replace `--optimizer sgd` with `--optimizer adam`; add `--learning-rate 0.0005` |
| E | Add `--critic-learning-rate 0.001` |
| F | Add `--critic-architecture direct` |
| G | Add `--separate-critic-optimizers` |
| H | Add `--sampling-strategy recordings`; compare a separate run with `--minimum-start 16` afterward |
| I | Add `--learning-rate-decay 0.95 --learning-rate-decay-every 10000`; run both decay/no-decay arms for, for example, 12x1000 updates |
| J | Add `--adversarial-objective non_saturating` |

## Read results quickly

Use S11 development validation to select hyperparameters. Repeated S5 evaluations must not drive that selection.

| Observation | Continue / investigate criteria |
| --- | --- |
| Numerical health | Finite losses/validation and completed updates are minimum requirements |
| Learning trend | Compare validation at 1000/2000 updates. Small fluctuations are normal; jumps toward the earlier error scale of about 10 warrant investigation |
| Zero-velocity comparison | Use the same current S11 windows. The historical 0.704 only applies to matching data/protocol. Being worse does not alone prove a code bug or successful reproduction |
| Adversarial state | Near-perfect D accuracy, a minimax term near zero and stalled G validation jointly suggest imbalance. D accuracy near 50% is not sufficient evidence of success |
| Gradients | Logs show pre-clip norms. Large values warrant investigation but do not prove clipping failed |

This optional snippet reads A's saved configuration, reconstructs the **same S11 validation windows**, computes zero velocity and summarizes both epochs. Paste in Bash/Git Bash. PowerShell users can save the Python body to a temporary `.py` and run it in the existing environment. To inspect another candidate, change `run`.

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np
from aged.config import Config
from aged import data as D
from aged.evaluate import mean_angle_error

run = Path("experiments/p0-A-full-sgd")
cfg = Config(**json.loads((run / "config.json").read_text()))
assert cfg.val_subject is not None, "This comparison requires a validation run"
norm = D.Normalisation.from_dict(json.loads((run / "normalisation.json").read_text()))
val = D.WindowDataset("data/h36m", norm, [cfg.val_subject], cfg.actions,
                      cfg.observed_length, cfg.prediction_length,
                      cfg.validation_windows, 999,
                      cfg.sampling_strategy, cfg.minimum_start)
windows = [val[i] for i in range(len(val))]
observed = np.stack([w[0].numpy() for w in windows])
target = np.stack([w[1].numpy() for w in windows])
baseline = mean_angle_error(np.repeat(observed[:, -1:], cfg.prediction_length, axis=1), target, norm)
baseline_mean = float(np.mean(list(baseline.values())))
rows = []
for row in json.loads((run / "results.json").read_text())["history"]:
    errors = row["validation_angle_error"]
    average = float(np.mean(list(errors.values())))
    rows.append({"epoch": row["epoch"], "step": row["step"],
                 "per_horizon": errors, "mean": average,
                 "difference_from_zero_velocity": average - baseline_mean})
report = {"zero_velocity_s11": baseline, "zero_velocity_mean": baseline_mean, "model": rows}
(run / "validation-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
```

A negative difference means lower model error than zero velocity on the same windows. S11 uses random development windows; S5 uses fixed eight-window evaluation and raw targets. Do not directly compare these protocols or different sampling settings. This snippet does not evaluate random initialization; a claim about improvement over initialization needs a separate step-0 evaluation on the same validation set.

Use 2000 updates to reject clearly unstable candidates. A stable candidate still slightly worse than zero velocity can continue to 5000-10000 updates. Two slightly different validation points do not establish a robust conclusion. Formal comparison also needs per-action results and motion visualizations.

## Checkpoints, resume and formal experiments

- `last.pt` is overwritten after every completed epoch. S11 validation improvements update `best.pt`, selected by Euler error by default.
- `--save-every 1` additionally retains `epoch_001.pt`, etc. There is no automatic step-based checkpointing.
- Extend the same experiment using its saved configuration without `--config`:

```bash
python -m aged.train --root data/h36m --out experiments/p0-A-full-sgd --resume experiments/p0-A-full-sgd/last.pt --epochs 10 --device cuda
```

This retains 1000 updates/epoch and extends the total to 10000 updates, with S11 validation and per-epoch snapshots. Changing optimizer, LR, objective, sampling, architecture or epoch size requires a new run directory; `--resume` rejects those changes. `--init-from` also requires the same G optimizer type and does not convert Adam state to SGD.

After choosing stable settings:

1. Check seeds 0/1/2 on S11 and report means and variability, not only the best run.
2. Freeze settings and train from scratch on all six training subjects. Remove `--val-subject S11` with the short-term YAML; add `--final` when using a team-reference YAML that holds out S11. Refit normalization; do not change the split by resuming a development checkpoint.
3. A formal budget can use 50x2000 updates, explicitly identifying 2000 as an engineering epoch definition. Without S11, retain last/periodic checkpoints; there is no validation best. Disable early stopping when reporting a full 50-epoch paper run.
4. Train the separate 25-frame model with `configs/long-term.yaml` and verify its stability. `--horizon 100` enables rollout but does not establish long-term quality.
5. After settings are frozen, evaluate S5, analyze each action and visualize predictions. Keep development scores and final scores with refitted normalization separate.

Retain the code SHA, effective `config.json`, commands, `results.json`, normalization, checkpoints and data fingerprint. Distinguish evidence of pipeline operation, stable learning and reproduction of paper metrics.

Related: [paper settings and assumptions](PAPER_ALIGNMENT.md), [team contributions and compatibility](TEAM_INTEGRATION.md), [validation scope](VALIDATION.md).
