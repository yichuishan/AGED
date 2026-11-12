# Diagnostic fixes and explicit V2 recipes

Prepared on 2026-11-11 against the supplied team archive at
`84e5ba24ca599cb3f4e1b1a978980002343ca011`.

## What this update changes

| Files | Change |
|---|---|
| `docs/integration-checks/5a48f7d/grad_balance.py` | Follow the checkpoint's minimax/non-saturating setting; add per-critic gradients and cosines; support CPU/CUDA selection and critic-free checkpoints. |
| `tests/system/test_gradient_diagnostics.py` | Check both objectives against explicit logit formulas, read-only behavior, full/single/no-critic variants, and CLI checkpoint compatibility without S5. |
| `tests/system/test_team_integration.py` | Replay recorded CLI commands using native Windows quoting instead of POSIX parsing; include paths with spaces. |
| `configs/v2-short-full.yaml` | Explicit final short-term full recipe: Adam 5e-4, 8k updates. |
| `configs/v2-short-geodesic.yaml` | Matched short-term geodesic-only recipe. |
| `configs/v2-long-full.yaml` | Explicit retained long-term recipe: SGD 0.005, 100k updates. |
| `src/aged/web.py`, `src/aged/web_assets/app.js` | Add the three named recipes to the dashboard and explain their intended use. Existing compatibility presets remain available. |
| `tests/system/test_web_pipeline.py` | Verify that dashboard recipe settings equal the corresponding YAML settings. |
| This document | Installation, validation, limitations and reporting corrections. |

The predictor, geometry implementation, critic architecture, main training step,
evaluator, package dependencies and checkpoint format are unchanged. Existing
checkpoints and run directories remain usable. Existing benchmark scores do not
need to be recomputed because of this patch.

## Apply the download

The ZIP contains only the ten updated/new files, stored at their project-relative
paths. It is an overlay, not a complete project. It does not contain `.git`,
datasets, checkpoints, old logs or a virtual environment.

1. In the local team repository, inspect `git status`. Commit or preserve any
   local edits to the affected files before replacing them.
2. Optionally create a review branch:

   ```text
   git switch -c fix/diagnostics-v2-presets
   ```

3. Extract the ZIP **into the repository root**, where `pyproject.toml`, `src`,
   `configs`, `docs` and `tests` already exist. Merge the folders and replace only
   the matching files. Do not replace/delete the whole existing folders.
4. Use the existing Python environment. No new dependencies are needed. If this
   package was installed non-editably, reinstall the changed source with
   `python -m pip install --no-deps -e .`. Restart any running dashboard and refresh
   the browser after updating the files.
5. Review and test:

   ```text
   git diff --stat
   git diff --check
   python -m pytest -q
   ```

   Run pytest in an environment with the project's existing dev dependencies.

## Run the corrected gradient diagnostic

Run from the repository root, with the same environment that runs training.
Replace both paths with actual local paths. Paths containing spaces must be quoted.

```text
python docs/integration-checks/5a48f7d/grad_balance.py "experiments/full-short-seed0/epoch_008.pt" --root "D:/datasets/h36m" --device cuda --batches 10
```

For a CPU check, change `--device cuda` to `--device cpu`; fewer batches can be used.
The default device is `auto`. Omitting `--init` measures the saved trained weights.
`--init` instead measures a newly initialized model using that checkpoint's config.

For comparison, run the same command on each retained final seed checkpoint.
There is no requirement to retrain. The old screen checkpoints may have been
deleted: exact diagnostics for those models require another copy or retraining.
Do not relabel a newly trained model's diagnostics as measurements of an old run.

The script prints a JSON line including:

- `adversarial_objective`: objective loaded from the checkpoint configuration.
- `diagnostic_version: 2`: distinguishes corrected output from historical records.
- `mean.g_rec`, `mean.g_adv`, `mean.ratio`, `mean.cosine`: the original aggregate keys.
- `mean.fidelity_g_adv`, `mean.fidelity_ratio`, `mean.fidelity_cosine`, and the
  corresponding continuity fields when those critics exist.
- `mean.critics_cosine`: alignment between the two weighted critic gradients.

Norm ratios are **pre-clip gradient statistics, not fractions of an actual optimizer
update**. Each per-critic gradient includes lambda. Batchwise ratios and cosines are
averaged; the aggregate ratio is not a ratio of aggregated norms. Zero-norm cases
use a denominator floor of 1e-12, and zero-gradient cosines are reported as zero.
No optimizer-state counterfactual is calculated by this script. That analysis
would also require the saved optimizer state and matching clipping/moment updates.

The diagnostic does not call an optimizer, change parameter values or `.grad`
buffers, write checkpoints, or read S5. It restores critic gradient flags after
each batch. New fields are absent when the corresponding critic does not exist.

## Explicit recipes for future runs

These configs package the already selected recipes; they are not new accuracy
improvements. They do not modify the legacy defaults. They use all six training
subjects, all 15 actions, no held-out validation subject, and no early stopping.

| Config | Model | Optimizer / LR | Updates | Final evaluation checkpoint |
|---|---|---|---:|---|
| `configs/v2-short-full.yaml` | 10-frame full | Adam / 0.0005 | 8 x 1000 | `epoch_008.pt` |
| `configs/v2-short-geodesic.yaml` | 10-frame geodesic only | Adam / 0.0005 | 8 x 1000 | `epoch_008.pt` |
| `configs/v2-long-full.yaml` | 25-frame full | SGD / 0.005 | 50 x 2000 | `epoch_050.pt` |

Each config saves snapshots each epoch. Long-term snapshots consume substantial
space; the existing `--save-every` option can change retention without changing
the model update recipe. Keep any checkpoints needed for planned diagnostics.

Examples for a **new** run, using a new output directory:

```text
python -m aged.train --config configs/v2-short-full.yaml --root "D:/datasets/h36m" --out experiments/v2-full-short-seed0 --device cuda --seed 0
python -m aged.train --config configs/v2-short-geodesic.yaml --root "D:/datasets/h36m" --out experiments/v2-geo-short-seed0 --device cuda --seed 0
python -m aged.train --config configs/v2-long-full.yaml --root "D:/datasets/h36m" --out experiments/v2-full-long-seed0 --device cuda --seed 0
```

Use seeds 1 and 2 for additional formal runs, each with a different output directory.
Do not use these commands to overwrite existing experiments. Resume still loads
the original checkpoint's configuration; a new preset must not silently alter it.

Start the existing dashboard as before with `python -m aged.web`. New choices are
`V2 final · Short-term full / Adam 8k`,
`V2 final · Short-term geodesic / Adam 8k`, and
`V2 retained · Long-term full / SGD 100k`.
The initial legacy P0 smoke-check choice is unchanged in purpose.

For new settings or architecture experiments, hold out S11, use a separate output
directory, and keep S5 for final testing. These formal presets do not recreate the
separate action-balanced S11 selection scripts automatically.

## Interpretation corrections for the historical reports

Historical results/logs are preserved. Read them with these qualifications:

- The old gradient probe always used minimax, including for non-saturating runs.
  That affects its gradient interpretation, not the main training objective or
  the reported evaluation scores. Minimax runs do not have this dispatch bug.
- The approximately 21% raw-gradient ratio from a high-lambda diagnostic is not a
  measurement of the final Adam model's actual update contribution.
- Both Adam and SGD predictor gradients are clipped by the production trainer.
- At 400 ms, equal averaging of the paper's printed 15 action entries gives
  1.1793 for Residual supervised, 1.0913 for geodesic-only, and 0.9653 for full AGED;
  their printed Average values are 1.15, 1.07 and 0.97. Report both definitions
  rather than silently replacing one with the other.
- The short-term final results have three seeds; the retained SGD long-term
  result has one. The phoning-excluded score is a secondary diagnostic; the
  official 15-action score remains primary.
- A correctly wired critic and a nonzero gradient do not establish an accuracy
  gain. Root context, sequence-boundary alternatives and long-term paired
  ablations remain experiments, not fixes included in this update.

## Validation performed for this package

- Complete production/system suite on Linux with CPU PyTorch:
  **39 passed, 2 CUDA tests skipped**, 42.40 seconds.
- Regression coverage includes both loss objectives, full/fidelity/continuity/
  no-critic models, unchanged weights and gradient buffers, and CLI diagnostics
  on existing-format checkpoints without S5 or held-out S11 data.
- `node --check src/aged/web_assets/app.js` passed.
- Chromium/Playwright checked that all three dashboard presets populate and
  export the expected optimizer, LR, horizon and update budget; no page errors.
- No real Human3.6M GPU training or evaluation was rerun. This environment has no
  CUDA runtime available for those tests. The Windows quoting repair was reviewed
  and the portable test passed on Linux; native Windows execution should be
  confirmed by running the same pytest suite locally.

## Commit and push

Run in the local team repository after reviewing the changes. The following
explicit path list avoids staging unrelated local work:

```text
git add -- docs/integration-checks/5a48f7d/grad_balance.py tests/system/test_gradient_diagnostics.py tests/system/test_team_integration.py tests/system/test_web_pipeline.py configs/v2-short-full.yaml configs/v2-short-geodesic.yaml configs/v2-long-full.yaml src/aged/web.py src/aged/web_assets/app.js docs/DIAGNOSTIC_FIXES_20261111.md
git diff --cached --stat
git commit -m "Fix gradient diagnostics and add explicit V2 training presets"
git push -u origin HEAD
```

`origin` must point to the intended team repository (`git remote -v`).
The final command pushes the current branch under its current name. If you
created `fix/diagnostics-v2-presets`, it pushes that branch; open a pull request
to the team main branch afterward. No force push is needed.
