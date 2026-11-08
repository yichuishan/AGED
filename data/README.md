# Required data

This repository contains no Human3.6M recordings. Obtain the dataset through
your authorized academic access and prepare the 99-column exponential-map text
export. Register at <https://vision.imar.ro/human3.6m/register.php> and follow
the dataset's terms. The loader does not parse BVH or the original CDF archive.

```text
data/h36m/
├── S1/
│   ├── directions_1.txt
│   ├── directions_2.txt
│   └── ...
├── S5/
├── S6/
├── S7/
├── S8/
├── S9/
└── S11/
```

Each subject has two takes for each of the 15 action names listed in
`aged.data.ACTIONS`. Each comma-separated file is `[frames,99]` at **50 Hz**:
global translation in columns 0:3, then 32 axis-angle joint rotations in 3:99.
The loader takes every second frame, producing 25 Hz. Do not pre-downsample.

Model features are the varying body-rotation coordinates in columns 6:99;
feature count is inferred from the selected training subjects, not fixed to 54.
Root rotation and translation are excluded even if they vary in the export.
The full training mean/scale and coordinate mask are retained in checkpoints.
Train subjects are S1, S6, S7, S8, S9, S11; S5 is test-only. An optional S11
development holdout is excluded from fitting both the predictor and statistics.

Whole-pipeline tests generate tiny synthetic files with this layout under
pytest's temporary directory. They are not Human3.6M data and their scores
must not be used as research results.
