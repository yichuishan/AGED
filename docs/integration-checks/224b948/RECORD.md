# GPU integration check: `main` @ 224b948

**Date:** 2026-11-08, 14:51–14:59 (+08:00), 7.5 min wall time
**Purpose:** a short GPU run to check that the integrated pipeline works end to end on the full dataset: data loading, training, checkpoints, logs, resume and evaluation. These are **not benchmark results**, and the accuracy numbers below should not be quoted.
**Status:** all 22 pipeline checks passed. All 4 optimizer-diagnostic runs completed.

The integration branch does not yet bring in every feature branch. **The discriminator architecture, optimizer, LR schedule and sampling strategy still need reconciling.** This record documents what the code does today in each of those areas, so that the full run after reconciliation has a fixed reference point.

## Code under test

| | |
| --- | --- |
| Commit | `224b9489b74d3e20f3a6c93d4ac2e4269a749274` (`main`, equal to `origin/main`, "Update README.md", 2026-11-08 13:19 +08:00) |
| Working tree | clean (no tracked changes); every run's `results.json` records this commit with no `-dirty` suffix |
| Integration branch | `origin/integration/full-aged` @ `06490d7`; `src/` and `configs/` are byte-identical to `main`. `main` differs only in `README.md`, `data/README.md`, `docs/PROVENANCE.md`, `docs/VALIDATION.md` and `scripts/setup.sh` |

Get the SHA with `git rev-parse HEAD`.

## Environment and data

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070, 12 GB, driver 610.88 |
| Software | Python 3.12.1, torch 2.15.0+cu132 (CUDA 13.2, cuDNN 9.26), numpy 2.5.4, Windows 11 |
| Dataset | `data/h36m`: 210 files (7 subjects × 15 actions × 2 takes), 561,444,622 bytes |
| Dataset fingerprint | sha256 over sorted paths and contents: `cd922e8d3b43accb8d641d468817e9473f949593de51928e8efd70c7f70c8d2b` |
| Split | train on S1, S6, S7, S8, S9 and S11 (180 files, all read for normalisation and training windows); test on S5 (30 files, read by every evaluation); the dev run holds out S11 for validation |

## Configuration

The configs are `configs/short-term.yaml` and `configs/long-term.yaml`, used unchanged except for the step and epoch counts:

```
observed_length 50   prediction_length 10 (long-term: 25)   hidden_dim 1024   critic_hidden_dim 1024
batch_size 16   learning_rate 0.005   lambda_adv 0.6   gradient_clip 5.0   seed 0
condition_decoder true   share_recurrent_weights false   num_workers 0   threads 1
adversarial_objective minimax   geodesic_reduction sum   actions: all 15

CHECK OVERRIDES (the YAML default is 50 epochs x 2000 steps):
  6 variants + long-term:  --epochs 1 --steps-per-epoch 200
  resume:                  --resume full-short/last.pt --epochs 2
  dev:                     --val-subject S11 --track tensorboard --epochs 2 --steps-per-epoch 100
  device:                  --device cuda
```

### The four areas still to be reconciled, as they are at 224b948

The YAML configs **do not cover** the optimizer, schedule or sampling. These are hard-coded:

| Area | Behaviour at this commit | Where |
| --- | --- | --- |
| Optimizer | `torch.optim.Adam` (default betas and eps, no weight decay) for the predictor, and a second Adam for both critics together; lr 0.005 | `src/aged/train.py:136,138` |
| LR schedule | none: constant learning rate | `src/aged/train.py` |
| Discriminator | one class used for both critics: Linear(K→1024) → 1-layer GRU(1024) → Linear(1024→1) on the final hidden state. The fidelity critic sees the predicted frames; the continuity critic sees observed + predicted frames. Each batch gets one joint critic step, then one predictor step. Critics are trained with BCE on logits; the predictor uses minimax log(1−D) | `src/aged/critics.py`, `src/aged/train.py:43-85` |
| Sampling | random windows: each recording is chosen in proportion to its number of valid windows, with a uniform start inside it. The RNG is keyed by (seed, epoch, index), and each epoch draws steps_per_epoch × batch_size windows | `src/aged/data.py:96-130` |

Other details:
- Geodesic loss: the sum over future frames and joints of d_G (**not squared**), averaged over the batch (`src/aged/losses.py:61-69`).
- K = 48 features: the varying body-rotation channels; root rotation and translation are excluded.
- Gradients are clipped to norm 5.0, and the run fails if any are non-finite.
- Evaluation: S5, 8 fixed windows per action (RandomState 1234567890), Euler-angle L2 on raw targets.

## Checks run

| # | Check | Result |
| --- | --- | --- |
| 1 | `pytest -q` (tests/system, including the CUDA train/evaluate case) | 14 passed |
| 2–3 | Zero-velocity baseline, 10- and 25-frame horizons | matches published values (below) |
| 4–15 | Train and evaluate each of `full`, `geodesic`, `euclidean`, `plain`, `fidelity`, `continuity` (200 steps on CUDA, full data) | all write `last.pt`, `results.json`, `config.json`, `normalisation.json`, `commands.txt`; all S5 evaluations finite |
| 16–17 | Resume `full` from `last.pt` (optimizer and RNG state) to epoch 2, then evaluate | resumes at epoch 2; history has 2 finite epochs |
| 18–20 | Long-term `full` (25-frame training), evaluated at 25 and 100 frames | ok, finite |
| 21–22 | Dev run with S11 validation and TensorBoard, then evaluate `best.pt` | per-epoch validation recorded, `best.pt` written (epoch 1), 1 TensorBoard event file |

### Evaluation protocol: the zero-velocity baseline matches the paper

The baseline matches the published zero-velocity values (residual sup. [31], as reproduced in the AGED paper):
- **all 15 actions at 80/160/320/400 ms**, to two decimals; the 15-action average is **0.40 / 0.71 / 1.07 / 1.21**
- **walking, eating, smoking and discussion at 560/1000 ms**: 1.35/1.32, 1.04/1.38, 1.02/1.69 and 1.41/1.96

So the test-set sampling, action averaging and Euler metric are correct. They are not a source of discrepancy. Per-action values are in `results/zero-velocity.json` and `results/zero-velocity-long.json`.

### Per-run numbers (200 steps; for pipeline health only)

S5 error is the 15-action average Euler error at 80 / 160 / 320 / 400 ms. Zero-velocity gives 0.40 / 0.71 / 1.07 / 1.21.

| Run | ms/step | Recon. loss (last epoch) | Pre-clip predictor grad norm | Critic acc. fid / con | S5 error, model |
| --- | ---: | ---: | ---: | --- | --- |
| plain | 19.1 | 0.98 (MSE) | 2.7 | n/a | 2.71 / 2.54 / 2.40 / 2.40 |
| geodesic | 21.8 | 285.9 | 874 | n/a | 9.82 / 9.77 / 10.14 / 9.99 |
| fidelity | 27.6 | 274.4 | 521 | 0.98 / n/a | 8.15 / 8.84 / 9.94 / 10.11 |
| continuity | 37.7 | 274.9 | 578 | n/a / 0.99 | 8.58 / 8.84 / 9.99 / 10.46 |
| euclidean | 40.3 | 87.8 (MSE) | 16,098 | 0.97 / 0.97 | 11.17 / 11.07 / 10.58 / 10.51 |
| full, epoch 1 | 42.2 | 276.7 | 696 | 0.98 / 0.98 | 9.07 / 8.93 / 9.53 / 10.09 |
| full, resumed epoch 2 | 43.4 | 316.9 | 15,524 | 1.00 / 1.00 | 10.66 / 10.28 / 10.71 / 10.51 |
| full, long-term (H=25) | 64.1 | 824.3 | 1,455 | 0.99 / 0.99 | 10.28 / 10.26 / 10.39 / 10.43; 560/1000 ms: 10.41 / 10.39 |
| dev full (S11 validation) | 41.9 | 322.4 | 1,163 | 1.00 / 1.00 | S11 validation, epoch 1: 9.01 / 9.06 / 9.47 / 9.48; epoch 2: 10.00 / 9.98 / 10.25 / 10.20. S5 for best.pt (epoch 1): 9.02 / 8.72 / 9.30 / 9.91 |

Timing includes data loading (`num_workers 0`).

Projected time for a full 50 × 2000-step run on this GPU, not counting evaluation:
- full short-term: about **72 min**
- full long-term: about **107 min**

Checkpoints include optimizer state:
- 292 MB with both critics
- 219 MB with one critic
- 146 MB with no critic

`last.pt` is overwritten every epoch.

Not covered by this check: peak GPU memory, `num_workers > 0`, a full 2000-step epoch, the `non_saturating` objective, `geodesic_reduction mean`, and non-Windows or cluster environments.

## Optimizer diagnostic (separate from the pipeline checks)

This uses the repo's own `train_step`, model, data and loss, and changes **only the optimizer object**, in `optim_diag.py`. Each arm runs 2,000 steps at lr 0.005, batch 16, clip 5, with S11 held out. Validation uses 256 random S11 windows (seed 999); zero-velocity scores **0.704** on them (mean of 80/160/320/400 ms).

| Variant | Optimizer | S11 Euler error at 250 / 500 / 1000 / 2000 steps | Final per-horizon (80/160/320/400) | Pre-clip grad norm |
| --- | --- | --- | --- | --- |
| plain | Adam | 2.51 / 10.09 / 10.59 / 10.16 | 10.26 / 10.10 / 10.23 / 10.03 | 2 → 6×10⁴ by step 500 → 10⁷–10⁸ |
| plain | SGD | 0.94 / 0.88 / 0.84 / 0.81 | 0.43 / 0.66 / 1.01 / 1.14 | about 1.7–2.3 throughout |
| full | Adam | 10.14 / 10.15 / 10.31 / 10.47 | 10.42 / 10.57 / 10.45 / 10.46 | 1.2×10³ → 1–4×10⁴; critic accuracy 1.00 |
| full | SGD | 0.84 / 0.80 / 0.77 / 0.72 | 0.32 / 0.56 / 0.91 / 1.08 | about 250–265; critic accuracy about 0.52 |

Logs: `logs/diag-{plain,full}-{adam,sgd}.log`.

## Findings

1. **Training is numerically unhealthy at this commit, and the optimizer is the cause.**
   - **In the pipeline runs:** within 200 steps, every variant is 2–28× worse than zero-velocity. Pre-clip gradient norms for the geodesic and adversarial variants reach 5×10² to 1.6×10⁴. A second epoch makes `full` worse, not better.
   - **In the diagnostic:** the only thing that changes between arms is the optimizer. Adam at lr 0.005 diverges for both `plain` and `full`, reaching an S11 error of about 10 rad. Plain SGD at the same lr trains stably and improves steadily: `full` reaches 0.72 after 2,000 steps, against 0.70 for zero-velocity.
   - lr 0.005, batch 16 and clip 5 are the SGD settings of residual sup. [31]. **Reconcile the optimizer (and any LR schedule) first.** Accuracy numbers from Adam at lr 0.005 are not meaningful.
2. **The adversarial signal collapses under Adam.** Critic accuracy goes to 0.97–1.00 and the minimax generator term goes to about 0 (for example −0.0005 for `full`, epoch 2), so the predictor receives almost no adversarial gradient. Under SGD the critics stay near 0.52 accuracy. Look at this when reconciling the discriminator architecture and update schedule.
3. **The geodesic loss sums d_G without squaring it.** When the losses are reconciled, check whether Eq. 4 of the AGED paper squares the distance. Squaring would change the loss scale and the gradient magnitude.
4. **The evaluation protocol is verified** against the published zero-velocity values, so S5 numbers from a reconciled run will be comparable to the paper.

## Files

This folder holds the record, about 0.5 MB, uncommitted:
- `RECORD.md`: this file
- `manifest.json`: machine-readable record: commit, environment, dataset fingerprint, both YAML configs verbatim, overrides, implementation choices, every command with return code and timing, and a per-run summary
- `integration_check.py`: the driver that ran everything
- `optim_diag.py`: the optimizer diagnostic
- `logs/`: one log per step (pytest, baselines, training, evaluation, diagnostics)
- `results/<run>/`: `config.json`, `normalisation.json`, `results.json` (per-epoch history), `test-results*.json`, `rollout-100.json`, `commands.txt`. `full-short/results-epoch1.json` is the history before resume.
