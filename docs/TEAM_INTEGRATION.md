# Team contributions, stable interfaces and compatibility

This integration retains functionality from each supplied branch while leaving
the existing full-pipeline workflow usable. Contributors are identified by the
provided branch snapshots, not guessed author names. Keep the original team PRs
as the authoritative contribution history; archives do not contain full Git
author objects.

## Contribution map

| Source | Preserved capability | Current implementation / evidence |
| --- | --- | --- |
| Predictor/main | Encoder, residual autoregressive decoder, action conditioning, full geometry and data contract | `models/`, `losses.py`, `data.py`; original whole-pipeline tests remain |
| continuity `8d29ffa` | Recurrent history+future classifier, one seed, loss helpers, seam sensitivity | `critics.continuity_input`, common `Critic`; pipeline seam and gradient tests |
| continuity `8d29ffa` | Real/fake experiment using frozen and swapped futures | Adapted `notebooks/continuity_sanity.ipynb`; original results/figures in `docs/experiments/continuity/` |
| fidelity `52cadee` | Future-only classifier, final-state readout, input validation and independent parameters | `critics.Critic`, `fidelity_input`; input checks and parameter/gradient assertions |
| fidelity reconciliation | Explicit minimax/non-saturating helpers, stable logits, detached D input and frozen D feedback | `critic_loss`, `generator_loss`, `generator_loss_non_saturating`; objective/derivative/extreme-logit tests |
| fidelity reconciliation | Direct-GRU and column output contract, existing component checkpoint naming | Selectable `direct` / `column`; constructor aliases and explicit strict checkpoint import |
| training/eval `a042021` | SGD/Adam, separate D rates/optimizers, step-based LR decay | `train.py`, `Config`; full updates and exact scheduled resume tests |
| training/eval `a042021` | Uniform recording windows, configurable first start, epoch-dependent draws/action IDs | `WindowDataset`; real-file sampling-distribution and frame-marker tests |
| training/eval `a042021` | Warm start, early stopping, metric selection and periodic snapshots | `--init-from`, `--patience`, `--select-on`, `--save-every`; end-to-end checks |
| training/eval `a042021` | Fixed test-window protocol, Euler body error, numerical baseline comparison | Existing evaluator plus `--check-baseline`; raw target handling retained |
| training/eval/main | JSON, TensorBoard and optional wandb tracking | Existing JSON schema, optional tracker backends; no new mandatory dependency |
| training/eval | Replayable module command history | `commands.txt` uses `python -m aged.train`; effective-config snapshots are retained per invocation |

## What existing users do not need to change

- `python -m aged.train` / `python -m aged.evaluate`, installed console commands,
  every existing flag and the short/long YAML files retain their meanings.
- Predictor input `[B,T,K]`, integer action IDs `[B]`, future-only output
  `[B,H,K]`, dataset triples and raw 99-channel export layout remain unchanged.
- Default Adam, constant LR, weighted windows starting at zero, minimax loss,
  root exclusion, projected/flat critics and joint D optimizer are retained.
- Existing integration **format-v1** checkpoints load without conversion;
  omitted new fields receive their old default meanings. The same resume/eval
  commands work and use saved normalization. Default model parameter names
  are unchanged. Do not reinstall CUDA or create a new environment just for
  this source update if the package was installed editable as in the README.
- Existing evaluation JSON keys remain; extra fields appear only for added
  features. No S5 metric is used for training checkpoint selection.

Stop an active job at a checkpoint before updating its source. If deployed as a
non-editable wheel rather than the documented editable install, update the
installed package too. Compatibility means new code reading old integration
files, not old code reading every new option or silently converting every
unrelated checkpoint family.

## Opt-in settings and team defaults

`configs/team-reference.yaml` is an explicit **new-run** profile reproducing the
team's selected policy: direct/column critics, separate optimizers, SGD with
0.95 decay every 10000 updates, uniform recording sampling, minimum start 16,
non-saturating adversarial loss and S11 validation. It does not silently replace
an existing run's settings. The short/long files are unchanged.

The paper does not settle every optimizer, schedule, sampling or projection
detail. These choices remain documented implementation decisions. Geodesic
restoration/root exclusion and the current Euler convention stay unified; older
stub loss files or verbatim external utility ports are not copied back in.

## Public critic compatibility

```python
from aged.critics import Critic

# Existing call remains projected and returns [B].
legacy = Critic(motion_dim=K, hidden_dim=H)

# Equivalent team classifier functionality with the selected team contract.
team = Critic(n_dims=K, hidden=H, architecture="direct", output_layout="column")
```

Helpers accept either output layout and preserve batch size one. `fc` and
`classifier` are read-only aliases to the head; they do not register duplicate
state keys. In direct mode, `load_team_state_dict(state)` explicitly maps one
component's `fc.*` or `classifier.*` to `output_projection.*` and loads strictly.
It rejects projected mode: their recurrent input matrices have different shapes.
This is not automatic conversion of legacy team **full-run** files or a changed
54/48-channel mask. The supported no-conversion resume target is the existing
integration v1 format, with matching normalization and architecture.

The loss helper names follow the fidelity reconciliation: `generator_loss` is
minimax; continuity's older non-saturating meaning is now explicitly named
`generator_loss_non_saturating`. This avoids a silent objective swap. Accuracy
uses the team's strict positive/negative thresholds; exact-zero logits are not
counted as correct classifications. This only affects that diagnostic tie case.

## Resume versus warm start

- Resume restores G, D, optimizers, normalization, step count and RNG. It rejects
  changed objectives, architecture, sampling and optimizer policies. Epoch limit,
  runtime/logging settings, patience and snapshot frequency may change.
- Warm start opens a new run, copies G and its optimizer, verifies the same
  normalization and keeps new critics fresh. Epoch and LR-step progress continue;
  new history/selection starts fresh. It does not need the source file again
  when subsequently resuming the new run.

Zero G/D learning rates are accepted for the team's bounded initialization or
freeze diagnostics. They are not a useful default for a learning experiment.

For old v1 files without a step field, completed steps are inferred as saved
epoch times steps-per-epoch. This is valid for that format's epoch-boundary
checkpoints. Old files without an early-stop counter start that new feature's
counter at zero. Cross-device bit-exact trajectories are not promised.

## Verification and limits

New integration tests execute actual optimizers, geometry, data sampling,
checkpoint loads and CLI entry points. They retain important component-level
assertions inside the whole-pipeline suite, including extreme-score derivatives,
gradient isolation, seam sensitivity, direct-weight imports and independence.

A real checkpoint generated by the unmodified `be674c6` code was retained before
editing. On CPU PyTorch 2.15, the new code produced identical inference and
bit-exact resumed model and optimizer tensors against an actual old-code
continuation. Loss/backward ordering is deliberately preserved in default mode.
Automated tests also exercise v1 files without the new optional fields.

The optional notebook passed bounded synthetic runs for all four experiments,
plus sampling through the actual current dataset class. Historical results are
not rerun claims. Real Human3.6M accuracy, CUDA/MPS execution and remote wandb
connectivity remain outside current hardware/data validation.
