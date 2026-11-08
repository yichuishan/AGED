# Validation scope

Validated environments: Python 3.12 with PyTorch 2.14.1+cpu and a clean
installation with PyTorch 2.15.0+cpu. No usable CUDA GPU or
licensed Human3.6M recordings are present in this environment.

Run from the repository root after installation:

```bash
python -m pip check
python -m pytest -q
```

Tests under `tests/system/` create a temporary 99-column, 50 Hz text export
with the real subject/action layout. It has eight varying body features that
do **not** form rotation triplets; this makes invalid reduced-vector reshaping
detectable. They exercise the production loader, model, losses, trainer,
checkpoint restorer and evaluator, not dummy versions.

Coverage:

- All six training variants perform actual predictor updates; every active
  discriminator also changes its weights with finite gradients.
- Train-only normalization, S5 leakage rejection, explicit root exclusion,
  action labels, complete joint restoration and epoch-varying windows.
- Correct zero-angle rotation derivative, known angular distances, finite
  gradients, raw-target retention and root-independent Euler evaluation.
- Exact CPU model and optimizer equality after resume versus uninterrupted
  training. Different objective settings are rejected on resume.
- Both 10-frame and 25-frame training, with evaluation rollout to 100 frames.
- Default 1024-hidden predictor and both default-size critics execute together.
- Real subprocess invocations of `python -m aged.train`, resumed training and
  `python -m aged.evaluate`, including persisted JSON results.
- Conditional full CUDA training/evaluation test; skipped on this machine.

The whole-pipeline suite completed with **13 passed, 1 CUDA skipped**, including
the clean CPU environment prepared with `bash scripts/setup.sh --cpu`. In the environment
with optional TensorBoard installed, the actual CLI train/resume/evaluate test
also passed and verified that event files were written. Both installed console
entry points were checked from outside the repository. Synthetic scores are
deliberately not listed as Human3.6M benchmark numbers.

The installer now defaults to CUDA mode for formal GPU training. CPU mode,
invalid option rejection and rejection of an existing CPU wheel in CUDA mode
without a CUDA installation index were exercised. The official cu128 index was
queried for available wheels, but CUDA wheel installation, driver compatibility
and real GPU execution remain unverified on this CPU-only machine.

Before reporting a reproduction result, run on the authorized real export,
verify zero-velocity against published tables, check action averaging and the
fixed-window/Euler convention, train the requested horizon for 50 epochs, and
evaluate on S5 using checkpoint statistics. Preserve configuration, code commit,
checkpoints and JSON results for every reported experiment.
