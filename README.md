# AGED — predictor reimplementation

AI6103 group project: a from-scratch PyTorch implementation of the GRU predictor
in **Adversarial Geometry-Aware Human Motion Prediction** (Gui et al., ECCV 2018,
[DOI](https://doi.org/10.1007/978-3-030-01225-0_48)). The user-supplied paper is
the algorithmic source; no external motion-prediction implementation was used.

This repository implements the encoder, autoregressive residual decoder,
predictor, optional-component AGED wrapper, configuration, tests and synthetic
training smoke test. Geometry loss, discriminators, H3.6M preprocessing and
full benchmark training/evaluation remain team integration work.

## Setup and checks

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

## Predictor usage

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

## Files

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
