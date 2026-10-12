import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible_cpu_tests():
    torch.manual_seed(7)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)
