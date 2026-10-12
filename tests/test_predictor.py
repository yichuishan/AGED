import io

import pytest
import torch

from models import AGEDModel, MotionPredictor
from models.config import ModelConfig
from scripts.smoke_train_predictor import run_smoke


@pytest.mark.parametrize("horizon", [10, 25, 100])
def test_shapes_and_prediction_lengths(horizon):
    config = ModelConfig.from_yaml("configs/model.yaml", motion_dim=54, hidden_dim=32)
    model = config.build()
    observed = torch.randn(2, config.observed_length, config.motion_dim)
    prediction = model(observed, torch.tensor([0, 14]), pred_len=horizon)
    assert prediction.shape == (2, horizon, 54)
    assert torch.isfinite(prediction).all()


def test_paper_hidden_size_forward_and_default_config():
    config = ModelConfig.from_yaml("configs/model.yaml", motion_dim=54)
    assert (config.hidden_dim, config.num_layers, config.num_actions) == (1024, 1, 15)
    assert (config.observed_length, config.prediction_length) == (50, 10)
    assert config.residual_prediction
    model = config.build().eval()
    with torch.no_grad():
        prediction = model(torch.randn(2, 50, 54), torch.tensor([1, 2]))
    assert prediction.shape == (2, 10, 54)
    assert torch.isfinite(prediction).all()


def test_same_model_has_horizon_independent_prefix_and_no_persistent_state():
    model = MotionPredictor(7, hidden_dim=16, num_actions=3).eval()
    observed, actions = torch.randn(2, 50, 7), torch.tensor([0, 2])
    with torch.no_grad():
        short = model(observed, actions, pred_len=10)
        long = model(observed, actions, pred_len=25)
        model(torch.randn(2, 50, 7), actions, pred_len=100)
        repeat = model(observed, actions, pred_len=10)
    torch.testing.assert_close(long[:, :10], short, rtol=0, atol=0)
    torch.testing.assert_close(repeat, short, rtol=0, atol=0)


def test_all_predictor_parameters_receive_finite_nonzero_gradients():
    model = MotionPredictor(7, hidden_dim=16, num_actions=3)
    observed = torch.randn(2, 50, 7, requires_grad=True)
    prediction = model(observed, torch.tensor([0, 2]), pred_len=10)
    prediction.pow(2).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert observed.grad is not None
    assert observed.grad[:, 0, :].abs().sum() > 0  # History affects future through encoder.


def test_predictor_transfers_last_seed_and_encoder_state_without_detaching():
    model = MotionPredictor(4, hidden_dim=8)
    captured_hidden = []
    captured_decoder_args = []
    encoder_handle = model.encoder.register_forward_hook(
        lambda module, args, output: captured_hidden.append(output)
    )
    decoder_handle = model.decoder.register_forward_pre_hook(
        lambda module, args: captured_decoder_args.append(args)
    )
    observed = torch.randn(2, 50, 4)
    model(observed, pred_len=3)
    encoder_handle.remove()
    decoder_handle.remove()
    seed, hidden, _, _ = captured_decoder_args[0]
    torch.testing.assert_close(seed, observed[:, -1, :])
    assert hidden is captured_hidden[0]
    assert hidden.grad_fn is not None


def test_eval_is_deterministic_even_with_multilayer_dropout():
    model = MotionPredictor(6, hidden_dim=12, num_layers=2, dropout=0.3).eval()
    observed = torch.randn(2, 50, 6)
    with torch.no_grad():
        first = model(observed)
        second = model(observed)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("shared", [False, True])
def test_checkpoint_round_trip_and_parameter_sharing(shared):
    kwargs = dict(motion_dim=5, hidden_dim=11, num_actions=3, share_recurrent_weights=shared)
    model = MotionPredictor(**kwargs).eval()
    assert (model.encoder.gru is model.decoder.gru) is shared
    assert (model.encoder.input_projection is model.decoder.input_projection) is shared
    observed, actions = torch.randn(2, 50, 5), torch.tensor([0, 2])
    with torch.no_grad():
        expected = model(observed, actions, pred_len=25)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = MotionPredictor(**kwargs).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(observed, actions, pred_len=25)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (restored.encoder.gru is restored.decoder.gru) is shared


def test_optional_components_and_no_future_ground_truth_input():
    predictor = MotionPredictor(6, hidden_dim=12)
    model = AGEDModel(predictor)
    observed = torch.randn(2, 50, 6)
    assert model.fidelity_discriminator is None
    assert model.continuity_discriminator is None
    torch.testing.assert_close(model.predict(observed, pred_len=10), predictor(observed))
    torch.testing.assert_close(model(observed), predictor(observed))
    with pytest.raises(TypeError):
        predictor(observed, future_gt=torch.randn(2, 10, 6))
    # Arbitrary modules used solely as registration sentinels; never fake algorithms.
    fidelity, continuity = torch.nn.Linear(6, 1), torch.nn.Linear(6, 1)
    attached = AGEDModel(predictor, fidelity, continuity)
    assert attached.fidelity_discriminator is fidelity
    assert attached.continuity_discriminator is continuity
    assert "fidelity_discriminator.weight" in attached.state_dict()
    torch.testing.assert_close(attached.predict(observed), predictor(observed))


def test_encoder_only_conditioning_variant():
    model = MotionPredictor(6, hidden_dim=12, num_actions=3, condition_decoder=False)
    assert model.encoder.input_projection.in_features == 9
    assert model.decoder.input_projection.in_features == 6
    assert model(torch.randn(2, 50, 6), torch.tensor([0, 2])).shape == (2, 10, 6)
    with pytest.raises(ValueError, match="matching"):
        MotionPredictor(6, num_actions=3, condition_decoder=False, share_recurrent_weights=True)


def test_tiny_overfit_with_optimizer_and_checkpoint(tmp_path):
    result = run_smoke(
        ModelConfig(motion_dim=6, hidden_dim=24), steps=150, seed=7,
        checkpoint=tmp_path / "predictor.pt",
    )
    assert result["final_mse"] < 0.1 * result["initial_mse"]
    assert result["checkpoint_max_error"] == 0


def test_arbitrary_observation_length_and_dtype():
    model = MotionPredictor(7, hidden_dim=9, num_actions=3).double()
    output = model(torch.randn(3, 8, 7, dtype=torch.float64), torch.tensor([0, 1, 2]), pred_len=4)
    assert output.shape == (3, 4, 7)
    assert output.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime unavailable")
def test_cuda_forward_backward():
    model = MotionPredictor(7, hidden_dim=16, num_actions=3).cuda()
    observed = torch.randn(2, 50, 7, device="cuda")
    output = model(observed, torch.tensor([0, 2]), pred_len=25)  # Labels may be CPU IDs.
    assert output.device.type == "cuda"
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_config_requires_preprocessing_dimension_and_rejects_invalid_values():
    with pytest.raises(ValueError, match="motion_dim"):
        ModelConfig.from_yaml("configs/model.yaml")
    with pytest.raises(ValueError, match="positive integer"):
        ModelConfig(motion_dim=0)
    with pytest.raises(ValueError, match="between layers"):
        ModelConfig(motion_dim=6, dropout=0.3)
    with pytest.raises(ValueError, match="boolean"):
        ModelConfig(motion_dim=6, residual_prediction="false")
