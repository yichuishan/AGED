import pytest
import torch
from torch.nn import functional as F

from models import MotionEncoder


def test_encoder_one_hot_linear_and_final_hidden():
    encoder = MotionEncoder(7, hidden_dim=13, num_actions=3)
    observed = torch.randn(2, 50, 7)
    actions = torch.tensor([0, 2])
    captured = []
    handle = encoder.input_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    hidden = encoder(observed, actions)
    handle.remove()
    assert hidden.shape == (1, 2, 13)
    assert captured[0].shape == (2, 50, 10)
    torch.testing.assert_close(captured[0][:, :, :7], observed)
    expected_labels = F.one_hot(actions, 3).float()[:, None, :].expand(-1, 50, -1)
    torch.testing.assert_close(captured[0][:, :, 7:], expected_labels)
    _, expected_hidden = encoder.gru(encoder.input_projection(captured[0]), torch.zeros(1, 2, 13))
    torch.testing.assert_close(hidden, expected_hidden)


def test_encoder_resets_hidden_between_calls_and_preserves_batch_independence():
    encoder = MotionEncoder(5, hidden_dim=9).eval()
    observed = torch.randn(2, 4, 5)
    first = encoder(observed)
    encoder(torch.randn(2, 12, 5))
    torch.testing.assert_close(encoder(observed), first, rtol=0, atol=0)
    torch.testing.assert_close(encoder(observed.flip(0)), first.flip(1))


@pytest.mark.parametrize("motion", [torch.randn(2, 7), torch.randn(2, 4, 8), torch.empty(2, 0, 7)])
def test_encoder_rejects_invalid_motion(motion):
    with pytest.raises(ValueError):
        MotionEncoder(7, hidden_dim=8)(motion)


@pytest.mark.parametrize(
    "actions, exception",
    [(None, ValueError), (torch.tensor([0]), ValueError),
     (torch.tensor([0.0, 1.0]), TypeError), (torch.tensor([0, 3]), ValueError),
     (torch.tensor([-1, 0]), ValueError)],
)
def test_encoder_rejects_invalid_action(actions, exception):
    with pytest.raises(exception):
        MotionEncoder(7, hidden_dim=8, num_actions=3)(torch.randn(2, 5, 7), actions)


def test_unconditioned_encoder_rejects_supplied_action():
    with pytest.raises(ValueError, match="unconditioned"):
        MotionEncoder(7, hidden_dim=8)(torch.randn(2, 5, 7), torch.tensor([0, 1]))
