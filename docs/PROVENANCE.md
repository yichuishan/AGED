# Source provenance and integration audit

Integration inputs:

- Personal repository `yichuishan/AGED`, base commit
  `972292670a10cfacda8d6470712376f8ba40b2a6`.
- User-uploaded `ai6103-team-project-aged-reimplementation-main.zip`, SHA-256
  `8b2f890e7eb1185afd02a339fead0eb14452f61a0e49354bce832b6de2eded27`.
  Its archive comment records revision `f4a4ba458e5945ea28e4d086571cb5320151f0f2`.
- User-uploaded primary paper PDF, SHA-256
  `d72b6e3877f6ff8043e38a9e2766163a92eb5195cfb37295f6b92ba85442a2d1`.

The private team Git remote was not readable from this environment. The ZIP,
not a guessed live repository version, is the reviewed source snapshot.

## Findings in the supplied snapshot

| Finding | Resolution |
| --- | --- |
| `src/aged/critics.py` still raised `NotImplementedError` | Implement one recurrent classifier, instantiate independent fidelity/continuity critics |
| Adversarial variants in training raised `NotImplementedError` | Complete alternating updates with detached discriminator inputs, frozen discriminator weights during predictor update |
| Two competing predictors existed, one lacking action conditioning | Move personal encoder/decoder into the sole `aged.models` package; remove duplicate predictor entry points |
| Geodesic loss received normalized, reduced coordinates | Restore full raw triplets differentiably before geometry; exclude root coordinates |
| Original Rodrigues coefficient clamp lost the first derivative at zero | Use sinc coefficients; whole-pipeline validation includes the zero-angle Jacobian |
| Reduced features included translation and root rotation | Select varying coordinates in channels 6:99 explicitly |
| Fixed evaluation selection was a stub; metric used exponential-map error | Implement documented fixed windows and Euler-angle evaluation from mathematical formulas |
| Evaluation targets could be reconstructed from train means | Keep the actual raw S5 targets; baseline repeats actual raw seed |
| Checkpoints lacked normalization and resumable optimizer/RNG states; final training did not save a checkpoint | Save full portable `last.pt` every epoch, optional validation `best.pt`, and test exact CPU resume |
| Evaluation used `os` without a module-level import | Replace legacy entry point with tested train/resume/evaluate commands |
| Repeated dataset windows every epoch | Deterministic epoch-indexed sampling |
| Stale notebook, standalone tests and smoke script described incomplete modules | Keep one train/evaluate workflow and whole-pipeline tests; document actual remaining limits |

The team's action vocabulary, subject split, export layout, normalization
structure, geometric `atan2` approach and experiment-tracking pattern informed
the integration. The existing personal predictor was retained and relocated.
The discriminator, trainer and checkpoint lifecycle were completed here.

The archive also contained audit scripts labeled as verbatim ports of another
implementation. They were inspected for documented protocol constants but were
not copied into this package or run. Rotation conversion and sampling were
written independently from mathematical formulas and the team's stated protocol.
No unofficial AGED implementation was searched for or used.

Historic data-audit claims, media, notebook commands and obsolete component-only
examples are not presented as newly verified facts. Original personal files are
recoverable in Git history; the provided team ZIP remains the source record.
