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
The initial integration completed a discriminator, trainer and checkpoint
lifecycle from that main snapshot. It did not include the then-unmerged work
described below; those contributions have now been reconciled explicitly.

## Subsequently provided feature branches

| Branch archive | Recorded revision | Archive SHA-256 |
| --- | --- | --- |
| continuity | `8d29ffae5baab34585fa5c349dd3ab2065ecc03a` | `746b1cea8f0fda5c2d96d7a1e5997cc32196b70415ef9c45d6dd8eda27b44976` |
| fidelity | `52cadeefaf80c71f93336d3ae31f0af6df96a430` | `211edddef27453408edae8b155e626df76b68ce9fe0633257147b098c23ae186` |
| training/eval | `a0420214e112d3343a4e67cc7da47873e22f0087` | `50c3994ee6c7a989b08183bf12a83282b6df192c121b624b18d23098b5fa536c` |

The common critic, stable loss helpers, continuity experiment, optimization and
sampling policies, warm start, early stopping, periodic snapshots and published
baseline check are team contributions. They are retained or adapted into the
current API, not claimed as newly invented work by the integrator. See
`TEAM_INTEGRATION.md` for the function/test map. Original continuity assets are
copied byte-for-byte; the notebook is adapted with outputs cleared and long
experiments disabled by default.

ZIP snapshots lack full author/parent Git objects. Recording source revisions
does not recreate the members' original Git histories; preserve the original
branches/PRs for authorship records. Common older stub files in those branches
were not used to roll back completed geometry or preprocessing fixes.

The archive also contained audit scripts labeled as verbatim ports of another
implementation. They were inspected for documented protocol constants but were
not copied into this package or run. Rotation conversion and sampling were
written independently from mathematical formulas and the team's stated protocol.
No unofficial AGED implementation was searched for or used.

Historic data-audit claims, media, notebook commands and obsolete component-only
examples are not presented as newly verified facts. Original personal files are
recoverable in Git history; the provided team ZIP remains the source record.
