from pathlib import Path

import numpy as np
import pytest

from aged.data import TRAIN_SUBJECTS, TEST_SUBJECT


@pytest.fixture(scope="session")
def motion_export(tmp_path_factory):
    """Tiny synthetic export, not Human3.6M or a benchmark result."""
    root = tmp_path_factory.mktemp("motion-export")
    moving = [6, 7, 10, 15, 17, 23, 42, 98]  # Eight reduced features: not joint triplets.
    time = np.arange(420)[:, None]
    for subject_index, subject in enumerate([*TRAIN_SUBJECTS, TEST_SUBJECT]):
        (root / subject).mkdir()
        for action_index, action in enumerate(("walking", "eating")):
            for take in (1, 2):
                frames = np.broadcast_to(np.linspace(0.001, 0.03, 99), (420, 99)).copy()
                phase = subject_index * 0.3 + action_index * 0.5 + take * 0.1
                frames[:, moving] += 0.04 * np.sin(time * np.linspace(0.005, 0.02, len(moving)) + phase)
                frames[:, :6] += time * np.linspace(0.0001, 0.002, 6)
                np.savetxt(root / subject / f"{action}_{take}.txt", frames, delimiter=",", fmt="%.9f")
    return Path(root)
