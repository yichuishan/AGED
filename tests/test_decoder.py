import pytest
import torch

from models import MotionDecoder


def test_decoder_uses_seed_then_its_own_predicted_frames_and_fixed_actions():
    decoder = MotionDecoder(5, hidden_dim=9, num_actions=3)
    seed = torch.randn(2, 5)
    hidden = torch.randn(1, 2, 9)
    original_hidden = hidden.clone()
    captured = []
    handle = decoder.input_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    prediction = decoder(seed, hidden, torch.tensor([0, 2]), pred_len=4)
    handle.remove()
    torch.testing.assert_close(captured[0][:, :5], seed)
    for step in range(1, 4):
        torch.testing.assert_close(captured[step][:, :5], prediction[:, step - 1, :])
        torch.testing.assert_close(captured[step][:, 5:], captured[0][:, 5:])
    torch.testing.assert_close(hidden, original_hidden)  # Caller state is not modified.


@pytest.mark.parametrize("residual", [True, False])
def test_residual_accumulates_and_absolute_mode_does_not(residual):
    decoder = MotionDecoder(3, hidden_dim=7, residual_prediction=residual)
    delta = torch.tensor([0.2, -0.1, 0.05])
    with torch.no_grad():
        decoder.output_projection.weight.zero_()
        decoder.output_projection.bias.copy_(delta)
    seed = torch.randn(2, 3)
    prediction = decoder(seed, torch.zeros(1, 2, 7), pred_len=5)
    if residual:
        expected = seed[:, None, :] + torch.arange(1, 6)[None, :, None] * delta
    else:
        expected = delta.expand(2, 5, 3)
    torch.testing.assert_close(prediction, expected)


def test_late_loss_backpropagates_through_earlier_autoregressive_outputs():
    decoder = MotionDecoder(4, hidden_dim=8)
    recurrent_outputs = []

    def retain_output(module, args, output):
        output[0].retain_grad()
        recurrent_outputs.append(output[0])

    handle = decoder.gru.register_forward_hook(retain_output)
    seed = torch.randn(2, 4, requires_grad=True)
    hidden = torch.randn(1, 2, 8, requires_grad=True)
    decoder(seed, hidden, pred_len=5)[:, -1].square().mean().backward()
    handle.remove()
    assert seed.grad is not None and seed.grad.abs().sum() > 0
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    for output in recurrent_outputs:
        assert output.grad is not None and output.grad.abs().sum() > 0


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True])
def test_decoder_rejects_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="pred_len"):
        MotionDecoder(4, hidden_dim=8)(torch.randn(2, 4), torch.zeros(1, 2, 8), pred_len=horizon)


def test_decoder_rejects_invalid_state():
    decoder = MotionDecoder(4, hidden_dim=8)
    with pytest.raises(ValueError, match="hidden must have shape"):
        decoder(torch.randn(2, 4), torch.zeros(2, 8))
    with pytest.raises(ValueError, match="same device and dtype"):
        decoder(torch.randn(2, 4), torch.zeros(1, 2, 8, dtype=torch.float64))
