# AGED — complete training and evaluation

A paper-based PyTorch reimplementation of **Adversarial Geometry-Aware Human
Motion Prediction** (Gui et al., ECCV 2018,
[DOI](https://doi.org/10.1007/978-3-030-01225-0_48)), integrated from the supplied
team source archive and the original predictor module.

The package implements the residual, action-conditioned GRU predictor,
SO(3) geodesic reconstruction, fidelity and continuity discriminators,
alternating adversarial training, ablations, resumable checkpoints and S5
evaluation with a zero-velocity baseline.

**Local CPU whole-pipeline validation passes. Team-supplied GPU records confirm
an earlier pipeline runs on real data, but Adam at LR 0.005 trained poorly.
The latest optional settings still need GPU validation; published accuracy has
not been reproduced.** Start with the
[GPU training and hyperparameter validation guide](docs/GPU_TRAINING_GUIDE.md)
(priorities, controlled combinations, fastest checks and checkpoint rules). See
[paper alignment](docs/PAPER_ALIGNMENT.md),
[source audit](docs/PROVENANCE.md) and [validation scope](docs/VALIDATION.md).

The remaining team branches are now reconciled functionally. See
[team contributions and compatibility](docs/TEAM_INTEGRATION.md) for the source
commits, preserved capabilities, and compatibility boundaries. Existing commands,
short/long YAML files, tensor layouts and v1 integration checkpoints still work;
the new team options do not silently change an experiment already in progress.

## Install for GPU training

Run commands from the repository root, using Python 3.10 or later on a machine
with an NVIDIA GPU and compatible driver. Check the machine first:

```bash
nvidia-smi
```

Select the supported CUDA wheel at the
[official PyTorch selector](https://pytorch.org/get-started/locally/).
Use its `--index-url` value as `TORCH_INDEX_URL`. For example, **only if that
selector recommends CUDA 12.8 for your machine**:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup.sh --cuda
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q
```

`setup.sh` defaults to CUDA mode. A fresh CUDA installation requires an explicit
official CUDA wheel index; an already-installed usable CUDA runtime can be
reused. A CPU wheel is replaced when a CUDA index is supplied, even if the CPU
package has a newer version. Setup fails if CUDA is still unavailable: it does
not silently declare a CPU installation ready for GPU training. A local CUDA
toolkit alone is not evidence that PyTorch can access a GPU.

### CPU validation only

For laptops or this CPU-only cloud environment, explicitly request CPU:

```bash
bash scripts/setup.sh --cpu
source .venv/bin/activate
python -m pytest -q
```

This validates the pipeline but does not perform the full GPU experiment.
Windows users can install the selector's wheel manually in a virtual environment
and then run `python -m pip install -r requirements.txt`; Git Bash activation is
`source .venv/Scripts/activate`. TensorBoard is optional:
`python -m pip install -e '.[tracking]'`.

In the prepared cloud machine, activate `/workspace/AGED/.venv/bin/activate`.
Each cloud task is already isolated; use the existing checkout and do not
create a Git worktree unless requested.

## Data

Use the licensed **99-column, 50 Hz exponential-map text export** described in
[data/README.md](data/README.md). The loader downsamples once to 25 Hz and derives
the motion feature count from training statistics, explicitly excluding global
translation and root rotation. It does not read BVH or CDF directly.

```text
data/h36m/S1/walking_1.txt
data/h36m/S1/walking_2.txt
...
data/h36m/S5/...        # test only
data/h36m/S6/...       # also S7, S8, S9, S11
```

Each subject has two takes of all 15 actions. No dataset download, redistribution,
or benchmark result is implied by cloning this repository.

## Train

### Local training dashboard

Run in the existing GPU Python environment, from the repository root:

```bash
python -m aged.web --open
```

Open **http://127.0.0.1:8765**. Enter or browse the local dataset/output paths,
edit every Config parameter, start a real training process, and inspect live
loss/validation curves, stdout/stderr and full error traces. The page supports
checkpoint configuration loading, resume, warm start and stopping after an
epoch is saved. It uses the same trainer and v1 checkpoints as the CLI, with
no new mandatory dependency. The UI starts from an explicit P0 SGD development
preset; CLI defaults stay unchanged. See [dashboard usage](docs/LOCAL_TRAINING_UI.md).

### Command line

Short term: 50 observed frames, 10 predicted frames (400 ms). Defaults use all
six training subjects, hidden size 1024, batch 16, LR 0.005, clip norm 5, 50 epochs,
lambda 0.6 and both recurrent discriminators.

Existing short/long configs retain **Adam at LR 0.005** for compatibility. The
team's earlier GPU diagnostic found this combination unstable. Before launching
50 epochs, follow the [short SGD check](docs/GPU_TRAINING_GUIDE.md#最快验证流程)
on S11; keeping the command interface compatible does not establish convergence.

```bash
python -m aged.train --root data/h36m \
  --config configs/short-term.yaml --out experiments/full-short --device cuda
```

Long term: train a separate 25-frame (1-second) model.

```bash
python -m aged.train --root data/h36m \
  --config configs/long-term.yaml --out experiments/full-long --device cuda
```

Use a new output directory for each experiment. Optional development validation
holds out S11 from both training and normalization:

```bash
python -m aged.train --root data/h36m --val-subject S11 \
  --out experiments/dev-full --track tensorboard --device cuda
```

S5 is never used for fitting statistics or selecting checkpoints.
For final six-subject runs, omit `--val-subject`. `last.pt` is always saved;
`best.pt` exists only for validation runs.

Available variants:

| Variant | Reconstruction | Discriminators |
| --- | --- | --- |
| `full` | Geodesic | Fidelity + continuity |
| `geodesic` | Geodesic | None |
| `euclidean` | MSE | Fidelity + continuity |
| `plain` | MSE | None |
| `fidelity` | Geodesic | Fidelity |
| `continuity` | Geodesic | Continuity |

Pass `--variant plain`, for example, with its own `--out`.
The default adversarial objective is the minimax form written in the paper.
`--adversarial-objective non_saturating` and
`--geodesic-reduction mean` are explicit experimental deviations.
Optimizer, update schedule, epoch size and ambiguous predictor details are
documented in [PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md).

Resume from a completed epoch using the saved model, critics, optimizers,
normalization, configuration and RNG states:

```bash
python -m aged.train --root data/h36m --out experiments/full-short \
  --resume experiments/full-short/last.pt --epochs 50 --device cuda
```

The total epoch count must exceed the saved epoch. Changed training objectives
or architecture are rejected on resume; use the same underlying dataset.
Results, environment/code version, effective configuration, normalization
metadata, command history and checkpoints are written into the run directory.

### Additional team training capabilities (optional)

Existing training commands need no extra flags. The following options bring in
the training/evaluation owner's capabilities without changing existing defaults:

| Options | Capability |
| --- | --- |
| `--optimizer sgd`, `--critic-optimizer adam`, `--critic-learning-rate` | Separate G/D optimizer choices and rates |
| `--learning-rate-decay 0.95 --learning-rate-decay-every 10000` | Step-based learning-rate schedule, restored on resume |
| `--sampling-strategy recordings --minimum-start 16` | Uniform recording sampling and configurable minimum frame |
| `--critic-architecture direct --critic-output-layout column` | Team direct-GRU critic and `[B,1]` scores |
| `--separate-critic-optimizers` | Independent optimizers and gradient clipping for each critic |
| `--patience 8 --select-on angle_error` | Validation-only early stopping; `val_loss` is also available |
| `--save-every 5` | Retain `epoch_005.pt`, etc., in addition to last/best |
| `--track wandb --project aged` | Optional wandb tracking after `pip install -e '.[wandb]'` |

The team aliases `--lr`, `--lr-critic`, `--lr-decay`, `--lr-decay-every`,
`--hidden`, `--adv-form non-saturating`, `--variant adv-euc`, and `--final`
are accepted. Canonical config values and original command flags remain valid.
The optional contributor notebook can be opened in Jupyter/VS Code; install
`python -m pip install -e '.[analysis]'` if its plotting dependencies are needed.
This is not required for the production training/evaluation commands.

To explicitly select the reconciled team reference settings for a **new** run:

```bash
python -m aged.train --root data/h36m --config configs/team-reference.yaml \
  --out experiments/team-reference --device cuda
```

That profile chooses direct/column critics, separate optimizers, SGD with decay,
uniform recording sampling and non-saturating GAN loss. These are reported
implementation choices, not additional facts stated by AGED. It holds out S11;
append `--final` for six-subject training. Do not switch an existing run to this
different architecture/objective using `--resume`.

Warm-start a **new** run using a checkpoint's predictor and optimizer while
initializing new critics. Epoch count and the LR schedule continue from source;
the dataset/normalization and predictor optimizer type must match:

```bash
python -m aged.train --root data/h36m --out experiments/warm-full \
  --init-from experiments/plain/last.pt --variant full --epochs 50 --device cuda
```

Without an explicit YAML, warm start inherits the source configuration before
CLI overrides. Changing the architecture requires a new run; checkpoint loading
does not silently reshape weights. Update a running experiment only after it
has stopped at a saved checkpoint, then use the same existing resume command.

## Evaluate

```bash
# Fixed eight-window S5 evaluation; always includes zero-velocity.
python -m aged.evaluate --root data/h36m \
  --checkpoint experiments/full-short/last.pt \
  --output experiments/full-short/test-results.json --device cuda

# Standalone zero-velocity table.
python -m aged.evaluate --root data/h36m --baseline \
  --output experiments/zero-velocity.json

# Long-term model, 1-second quantitative horizons.
python -m aged.evaluate --root data/h36m \
  --checkpoint experiments/full-long/last.pt --horizon 25 \
  --output experiments/full-long/test-results.json --device cuda

# Optional 4-second rollout, using the separately trained long-term model.
python -m aged.evaluate --root data/h36m \
  --checkpoint experiments/full-long/last.pt --horizon 100 \
  --output experiments/full-long/rollout-100.json --device cuda
```

Evaluation uses actual raw S5 targets, restores model predictions using saved
training statistics, converts rotations to Euler angles, excludes global motion
and reports mean vector error in radians at 80/160/320/400 ms (plus 560/1000 ms
when available). Each action row and the overall average include zero-velocity.
The eight-window sampling and Euler convention follow the protocol documented
in the team archive. The supplied GPU record for the earlier `224b948` version
matches the documented short-term zero-velocity reference (including its noted
Sitting/160 ms correction) and four long-term action rows. Repeat the baseline
check on the current installation and export. Longer rollouts do not establish
long-term accuracy.

The training/evaluation owner's published zero-velocity regression check is
available explicitly, without changing the normal evaluation command:

```bash
python -m aged.evaluate --root data/h36m --baseline --check-baseline \
  --output experiments/published-baseline-check.json
```

This writes actual/reference differences into JSON and exits nonzero on a
mismatch. It checks short-term entries only and documents the team's inferred
Sitting/160 ms table correction. Synthetic data must fail this benchmark check.

## Whole-pipeline tests

```bash
python -m pytest -q
```

Tests generate temporary synthetic export files, run all training variants,
check gradients and exact CPU resume, evaluate raw targets and baselines, test
both training horizons and 100-frame rollout, and invoke the actual command-line
entry points. Default-size predictor and critics are exercised together. A
complete CUDA train/evaluate case runs only when CUDA is available.
Synthetic tests are engineering validation, not the final research experiment.

## Layout

```text
configs/                 short/long settings plus optional team reference profile
data/README.md           data contract (recordings are ignored)
docs/                    architecture, paper alignment, audit and validation
docs/experiments/        original team experiment records, clearly labeled historical
notebooks/              adapted optional continuity experiment (outputs cleared)
scripts/setup.sh         environment helper
src/aged/
  models/                single encoder/decoder/predictor implementation
  config.py, data.py     experiment schema and preprocessing
  critics.py, losses.py  complete adversarial and geometry components
  train.py, evaluate.py  the two end-to-end entry points
  runtime.py, tracking.py
  web.py, web_worker.py   local HTTP dashboard and isolated training worker
  web_assets/            offline HTML/CSS/JS dashboard
tests/system/            whole-pipeline tests only
experiments/             ignored run outputs
```

There is no second top-level `models/` implementation, component-only test suite
or `smoke_train_predictor.py`. The production package is installed from `src/`.
Legacy standalone checkpoints have a different format and are intentionally not
silently loaded as complete AGED runs.
