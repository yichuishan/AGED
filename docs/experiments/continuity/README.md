# Preserved continuity-owner experiment

These files are copied unchanged from the supplied continuity branch
`8d29ffae5baab34585fa5c349dd3ab2065ecc03a`, original directory
`experiments/critic-sanity-20261102-232155/`:

- `results.json`: data/settings and reported fresh-sample classification accuracy.
- `table.md`: the author's summary table.
- `critic_sanity.png`: original figure. The duplicate PDF export was removed
  during cleanup; the notebook can still export a new PDF for a rerun.

They report a direct-GRU critic trained on MPS with 54 retained features, hidden
128, batch 64, LR 0.001 and 4000 updates. Swapped real futures were detected by
the history-conditioned critic (1.0), whereas the future-only control was at
chance (0.5). These are the contributor's provided records, not new results
recomputed by this integration, and not AGED prediction-angle errors.

`notebooks/continuity_sanity.ipynb` preserves the experiment functionality with
current APIs, three-item dataset batches and direct critics. Long experiments
are opt-in and notebook outputs are cleared. New real-data reruns use current
body-only normalization; they must record their own feature count and cannot
claim identical settings to these historical 54-channel results.

Shape/gradient checks and test execution are centralized under `tests/system/`;
the notebook keeps only experiment setup, sampling, training and reporting.
