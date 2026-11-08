# Original team GPU integration records

The files in [224b948/](224b948/) are restored byte-for-byte from the supplied
`integration-checks.zip`, retaining the original `docs/integration-checks/224b948/`
layout. All 75 original files are included. The ZIP itself remains in the upload
workspace; this directory stores its extracted contents without a duplicate ZIP.

| Record | Contents |
| --- | --- |
| [RECORD.md](224b948/RECORD.md) | Original report, environment, findings and limitations |
| [manifest.json](224b948/manifest.json) | Code revision, data fingerprint, commands, configuration and run summaries |
| [logs/](224b948/logs/) | 26 original pipeline and optimizer-diagnostic logs |
| [results/](224b948/results/) | 45 original configuration, normalization, metric and command-history files |
| [integration_check.py](224b948/integration_check.py) | Original pipeline driver |
| [optim_diag.py](224b948/optim_diag.py) | Original optimizer comparison driver |

These are historical runs of commit `224b948` on the team's RTX 4070, not fresh
results from the current code. The archive contains no dataset or model weights.
Original report wording is preserved, including questions raised at the time.
The question about squaring geodesic distance has since been checked: paper
Eq. 4 sums the unsquared distance.

Current interpretation and next experiments are documented in
[VALIDATION.md](../VALIDATION.md) and the
[GPU training guide](../GPU_TRAINING_GUIDE.md).
