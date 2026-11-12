# AGED diagnostic fixes download

This directory delivers the diagnostic-fix overlay prepared on 2026-11-11.

- **Download:** [aged-fixes-20261111.zip](aged-fixes-20261111.zip). Open the file on GitHub and choose **Download raw file**.
- **Instructions and commands:** [INSTALL.md](INSTALL.md).
- **Reviewed team archive:** `84e5ba24ca599cb3f4e1b1a978980002343ca011`.

The ZIP contains 10 updated/new files with project-relative paths. Extract it into
the reviewed team project's root, merging the `configs`, `docs`, `src` and `tests`
folders and replacing matching files. Preserve any local edits first. Existing
datasets, checkpoints and historical evaluation results remain in their locations.

It fixes the gradient diagnostic's loss dispatch and Windows test command replay,
adds explicit V2 training configurations and dashboard presets, and includes
regression tests and documentation. Core training/evaluation algorithms and the
checkpoint format are unchanged. Existing benchmark scores remain valid.

Validation: 39 CPU tests passed; 2 CUDA tests were skipped. Three dashboard presets
were checked in Chromium. No new Human3.6M GPU run or native Windows test run was
performed.

SHA-256 of the ZIP:

```text
64de5fe63ae2c8fad0328391e256098cda24f5006adfa68399aeb527a3de8e32
```
