# AGED — reimplementation

> **Repository scope:** This README preserves the team-project overview supplied
> during predictor integration. Team-wide paths such as `src/aged/` and
> `scripts/setup.sh` belong to the
> [team repository](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation).
> They are not included or validated in this standalone predictor checkout.
> Team documentation links below point to that repository. For the code available
> here, use [Standalone GRU predictor module](#standalone-gru-predictor-module).
> Aligning these shared files does not connect `models/` to `src/aged/train.py`.

AI6103 Deep Learning, group project. A from-scratch reimplementation of
**Adversarial Geometry-Aware Human Motion Prediction** (Gui, Wang, Liang and
Moura, ECCV 2018), with the ablations needed to test whether its two ideas
actually earn their place.

The task: given 2 seconds of observed skeletal motion, predict the next 400 ms of
joint angles.

## Why this paper

The course allows three project shapes. This is the first — *reimplement a 2015+
conference paper with no public codebase*. AGED qualifies on year (2018), venue
(ECCV, whitelisted) and topic (computer vision).

It also splits cleanly into four pieces with fixed interfaces, which is why it
was chosen over improving an existing codebase: everyone can build and test their
own part without waiting for anyone else.

**Checked 26 Sep 2026 — no public implementation found.** Searched the ECCV open
access page, Semantic Scholar, the authors' pages, the
[awesome-human-pose-prediction](https://github.com/karttikeya/awesome-human-pose-prediction)
list (which gives AGED a paper link and no code link — as it does for every 2018
entry) and general web search. The nearest repositories,
[garroud](https://github.com/garroud/human-motion-prediction-pytorch) and
[ornithos](https://github.com/ornithos/human-motion-prediction-pytorch), are
PyTorch ports of **Martinez et al. CVPR 2017**, not of AGED: residual supervision
without the geodesic loss or the two critics. They are useful references for the
predictor half and are not the paper we are reimplementing.

Caveat: this is the result of web search, not an authenticated GitHub code
search. If someone has a GitHub account handy, search the code index for the
geodesic loss together with two discriminators and record the result here.

## The method

A residual GRU sequence-to-sequence predictor, trained against two critics and a
geodesic loss.

```
observed (50 frames, 2 s)
      |
   Encoder  GRU, hidden 1024
      |
   Decoder  GRU, hidden 1024, residual: predicts velocities
      |
predicted (10 frames, 400 ms)
      |
      +--> Geodesic loss ........ distance on SO(3) against ground truth
      +--> Fidelity critic ...... sees the prediction alone
      +--> Continuity critic .... sees observed + prediction joined
```

Only the encoder and decoder survive to inference. The three signals on the right
exist during training and are then thrown away.

Each component answers one named failure of prior work: the continuity critic
kills the jump at the seam, the fidelity critic stops the collapse to an average
pose, and the geodesic loss stops treating rotations as ordinary numbers. See
[`docs/paper-notes.md`](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/docs/paper-notes.md).

![One training example](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/docs/media/motion_walking.gif?raw=true)

Black: the 50 observed frames. Blue: the 400 ms the model has to predict. The
grey ghost is the last observed pose. Dataset structure is documented in
[`docs/dataset.md`](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/docs/dataset.md).

## Getting started

```bash
git clone https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation.git
cd ai6103-team-project-aged-reimplementation

source scripts/setup.sh
```

That creates `.venv`, installs the dependencies, reports your torch version and
whether a GPU backend is visible, and leaves you inside the environment.

Note the `source`. Activating a virtualenv only changes the shell it runs in, so
a script run as a child process cannot activate anything in yours — `bash
scripts/setup.sh` works but then tells you to activate manually.

**Use the virtualenv.** Everyone needs the same torch and numpy for results to be
comparable, and `pip install` into the system python on macOS fights with
Homebrew. `.venv/` is gitignored. Check it took:

```bash
python -c "import sys, torch; print(sys.prefix); print(torch.__version__)"
```

The first line should end in `.venv`. On Apple Silicon, also check the GPU is
visible:

```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

Then get the data. **Register first** at
<https://vision.imar.ro/human3.6m/register.php> (academic address required) — the
licence forbids passing copies between team members, so everyone downloads their
own. The link printed in every other repo is **dead**; see
[`data/README.md`](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/data/README.md).

```bash
scripts/get_data.sh --mirror                  # downloads, unpacks, verifies
scripts/get_data.sh /path/to/download.zip     # if you downloaded it yourself
```

That unpacks, checks all seven subjects have 30 files, checks the 99 columns and
the size, then runs the loader's own verification. `--mirror` shows the licence
position and asks before downloading; it needs `pip install gdown`.

Expected output: 99 columns, 54 non-constant, 45 constant, about 2.9 hours of
motion at 25 Hz.

## Looking at the data

```bash
python tools/visualize_h36m.py --root data/h36m --out figures/
python tools/make_animation.py --root data/h36m --action walking --out figures/walking.gif
```

## Training and the ablations

```bash
python -m aged.train --root data/h36m --variant full
python -m aged.train --root data/h36m --variant euclidean   # geodesic loss -> MSE
python -m aged.train --root data/h36m --variant fidelity    # one critic
python -m aged.train --root data/h36m --variant continuity  # one critic
python -m aged.train --root data/h36m --variant plain       # no critics
```

Published settings, used as defaults: lambda 0.6, learning rate 0.005, batch size
16, 50 epochs.

## Evaluation

Train on six subjects, test on S5. Eight sequences per action, fixed seed. Mean
angle error in the original units at 80, 160, 320 and 400 ms — frames 1, 3, 7
and 9 at 25 Hz.

**Two known defects in the evaluation path, both measured, both blocking any
number that goes beside the published table**: windows are currently sampled at
random rather than using the canonical seeded set, and the metric is computed in
exponential-map space where the literature uses Euler angles. See
[`docs/preprocessing.md`](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/docs/preprocessing.md), findings 3 and 4.

**Zero-velocity — repeating the last observed frame — must appear in every
results table.** It is trivially simple and beats a surprising number of
published models at short horizons.

### What success looks like

Published AGED on walking is 0.22 / 0.36 / 0.55 / 0.67 at 80 / 160 / 320 /
400 ms, against 0.28 / 0.49 / 0.72 / 0.81 for residual supervision. Matching
those decimals in a different framework with different seeds is not the target.
Success is:

- the full model beats the Euclidean-loss ablation
- both beat zero-velocity at 320 ms and beyond
- the ordering of the published table is reproduced

A number that does not replicate is a finding to report, not a failure to hide.

## Layout

```
src/aged/data.py       loading, downsampling, normalisation   [complete, verified]
src/aged/model.py      the residual GRU predictor
src/aged/critics.py    one Critic class, used twice
src/aged/losses.py     geodesic loss on SO(3)
src/aged/train.py      adversarial training loop and ablations
src/aged/evaluate.py   protocol, metric, zero-velocity baseline
src/aged/tracking.py   TensorBoard / wandb logging and run metadata
tools/                 visualisation, result collection, preprocessing audit
tests/                 contract tests for the predictor
docs/                  orientation, the tensor contract, paper and data notes
experiments/           one folder per run, gitignored
```

`data.py` is complete and verified against the real dataset because it is shared
infrastructure and everything else depends on it; `model.py` is implemented and
passes its contract tests. **`critics.py` and `losses.py` are still specified
stubs** — they are the paper's two actual ideas, and implementing them is the
assignment. `train.py` runs the `plain` variant today and has a marked hole
where the critics plug in.

New to the repo? Read [`docs/orientation.md`](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/docs/orientation.md) — where
everything lives, what is done, what is yours, and how the pieces connect. Then
[`docs/interfaces.md`](https://github.com/akkvjoe/ai6103-team-project-aged-reimplementation/blob/HEAD/docs/interfaces.md), which is short and is the only thing
you have to obey. Every tensor is `(batch, time, D)`, float32, normalised.

## Data and licence

Human3.6M, exponential-map export. Covered by the
[Human3.6M EULA](http://vision.imar.ro/human3.6m/eula.php): academic use only.
Cite Ionescu et al. in the report, not the mirror the file came from. The dataset
is gitignored and must not be committed.

---

## Standalone GRU predictor module

AI6103 group project: a from-scratch PyTorch implementation of the GRU predictor
in **Adversarial Geometry-Aware Human Motion Prediction** (Gui et al., ECCV 2018,
[DOI](https://doi.org/10.1007/978-3-030-01225-0_48)). The user-supplied paper is
the algorithmic source; no external motion-prediction implementation was used.

The standalone module in `models/` implements the encoder, autoregressive
residual decoder, predictor, optional-component AGED wrapper, configuration,
tests and synthetic training smoke test. Connecting it to the team's `src/aged`
package, preprocessing, losses, critics and training/evaluation is separate
integration work. The predictor does not import or replace those team modules.

### Setup and checks

Run from the repository root. CPU Python 3.12 is validated. In the prepared cloud
environment, use `source /workspace/aged-venv/bin/activate`. Each cloud task is
already isolated; use the existing checkout without creating a Git worktree
unless explicitly requested.

For a fresh CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.smoke_train_predictor
```

For CUDA, install the appropriate official PyTorch CUDA wheel for the machine's
driver, then install the remaining requirements. Select CUDA explicitly in caller
code or pass `--device cuda` to smoke training. No CUDA hardware is present in
the prepared environment; its conditional test is skipped there.

Smoke training uses two synthetic constant-velocity sequences and **MSE solely
for debugging**. It checks finite gradients, at least 90% loss reduction,
optimizer compatibility and exact checkpoint output restoration. Defaults
override hidden dimension to 32 and K to 6 to keep the engineering check small.
It is not the geodesic objective, an adversarial trainer or an H3.6M result.

```bash
# Long-horizon debugging baseline; outputs are ignored by Git.
python -m scripts.smoke_train_predictor --pred-len 25 --checkpoint outputs/smoke_long.pt
```

### Predictor usage

```python
import torch
from models import AGEDModel, MotionPredictor
from models.config import ModelConfig

K = 54  # Synthetic example only; replace with your preprocessing feature count.
config = ModelConfig.from_yaml("configs/model.yaml", motion_dim=K)
predictor = config.build()  # H=1024, one GRU layer, 15 actions, residual outputs.
model = AGEDModel(predictor=predictor)

observed = torch.randn(2, config.observed_length, K)  # [B,50,K]
actions = torch.tensor([0, 14])                     # [B], integer IDs
model.eval()
with torch.no_grad():
    predicted = model.predict(observed, actions, pred_len=10)
    long_prediction = model.predict(observed, actions, pred_len=25)
assert predicted.shape == (2, 10, K)
assert long_prediction.shape == (2, 25, K)

# Equivalent direct API:
predictor = MotionPredictor(motion_dim=K, hidden_dim=1024, num_actions=15)
```

The YAML intentionally leaves `motion_dim: null` until preprocessing supplies K.
Do not label 54 as the paper's fixed feature count. Prediction horizons can be
10, 25, 100 or any other positive number of frames, with no architecture changes.
`num_actions=None` disables conditioning as a development variant.

```text
Observed motion [B,T,K] + integer action IDs [B]
        ↓ append per-frame one-hot
Linear input map → GRU encoder (1 layer, hidden 1024)
        ↓ final hidden [1,B,1024] + last observed seed [B,K]
GRU decoder: project input → recurrent step → output delta
        ↓ add previous pose; feed predicted pose back
Predicted future motion [B,H,K] (seed excluded)
```

Encoder action concatenation, GRU size, residual prediction and feedback are
stated in the paper. Parameter tying, decoder action concatenation and exact
seed slicing are ambiguous; selected defaults and switches are disclosed in
[IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md). Read
[INTEGRATION.md](docs/INTEGRATION.md) before attaching losses or discriminators.
No future ground truth is accepted by the predictor in either train or eval mode.

### Files

```text
models/    encoder.py, decoder.py, predictor.py, aged.py
           config.py, contracts.py, _inputs.py
configs/   model.yaml
scripts/   smoke_train_predictor.py
tests/     test_encoder.py, test_decoder.py, test_predictor.py
docs/      IMPLEMENTATION_NOTES.md, INTEGRATION.md
```

Tests cover 50-frame history, K=54 shapes, horizons 10/25/100, action
concatenation, seed/state handoff, residual and absolute variants, feedback,
gradient propagation, deterministic eval, tied/untied checkpoint restore,
input validation, predictor-only integration and tiny-batch overfit. One test
also executes the full 1024-hidden default. CUDA testing is conditional.
