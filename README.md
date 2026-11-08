# AGED — complete training and evaluation

A paper-based PyTorch reimplementation of **Adversarial Geometry-Aware Human
Motion Prediction** (Gui et al., ECCV 2018,
[DOI](https://doi.org/10.1007/978-3-030-01225-0_48)), integrated from the supplied
team source archive and the original predictor module.

The package implements the residual, action-conditioned GRU predictor,
SO(3) geodesic reconstruction, fidelity and continuity discriminators,
alternating adversarial training, ablations, resumable checkpoints and S5
evaluation with a zero-velocity baseline.

**CPU whole-pipeline validation passes. Published Human3.6M accuracy and GPU
execution have not been validated here.** See
[paper alignment](docs/PAPER_ALIGNMENT.md),
[source audit](docs/PROVENANCE.md) and [validation scope](docs/VALIDATION.md).

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

Short term: 50 observed frames, 10 predicted frames (400 ms). Defaults use all
six training subjects, hidden size 1024, batch 16, LR 0.005, clip norm 5, 50 epochs,
lambda 0.6 and both recurrent discriminators.

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
in the team archive; **agreement with published baseline tables still needs
real-data verification**. Longer rollouts do not establish long-term accuracy.

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
configs/                 short-term and long-term experiment settings
data/README.md           data contract (recordings are ignored)
docs/                    architecture, paper alignment, audit and validation
scripts/setup.sh         environment helper
src/aged/
  models/                single encoder/decoder/predictor implementation
  config.py, data.py     experiment schema and preprocessing
  critics.py, losses.py  complete adversarial and geometry components
  train.py, evaluate.py  the two end-to-end entry points
  runtime.py, tracking.py
tests/system/            whole-pipeline tests only
experiments/             ignored run outputs
```

There is no second top-level `models/` implementation, component-only test suite
or `smoke_train_predictor.py`. The production package is installed from `src/`.
Legacy standalone checkpoints have a different format and are intentionally not
silently loaded as complete AGED runs.
