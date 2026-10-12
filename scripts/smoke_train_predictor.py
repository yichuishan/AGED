"""Synthetic MSE overfit check. This does NOT implement the AGED training objective."""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch

from models.config import ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synthetic_batch(config: ModelConfig, batch_size: int, device: torch.device):
    """Constant-velocity tensors; toy coordinates, not valid H3.6M rotations."""
    start = 0.1 * torch.randn(batch_size, 1, config.motion_dim, device=device)
    velocity = 0.002 * torch.randn(batch_size, 1, config.motion_dim, device=device)
    times = torch.arange(
        config.observed_length + config.prediction_length, device=device, dtype=start.dtype,
    )[None, :, None]
    complete = start + times * velocity
    observed = complete[:, :config.observed_length, :]
    future = complete[:, config.observed_length:, :]
    action = None
    if config.num_actions is not None:
        action = torch.arange(batch_size, device=device) % config.num_actions
    return observed, future, action


def run_smoke(
    config: ModelConfig,
    steps: int = 250,
    batch_size: int = 2,
    seed: int = 7,
    device: str = "cpu",
    checkpoint: str | Path | None = None,
) -> dict[str, float]:
    """Train a tiny batch, check gradients and optionally verify checkpoint restore."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    seed_everything(seed)
    target_device = torch.device(device)
    model = config.build().to(target_device)
    observed, future, action = synthetic_batch(config, batch_size, target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Debugging choice.
    criterion = torch.nn.MSELoss()  # Temporary regression baseline, NOT geodesic loss.
    model.train()
    with torch.no_grad():
        initial = criterion(model(observed, action), future).item()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(observed, action), future)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite smoke loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("missing or nonfinite predictor gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(observed, action)
        final = criterion(prediction, future).item()
    if not final < 0.1 * initial:
        raise RuntimeError(f"tiny overfit failed: initial={initial:.6g}, final={final:.6g}")
    result = {"initial_mse": initial, "final_mse": final, "loss_ratio": final / initial}
    if checkpoint is not None:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_config": asdict(config), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "steps": steps,
        }, path)
        saved = torch.load(path, map_location=target_device, weights_only=True)
        restored = ModelConfig(**saved["model_config"]).build().to(target_device)
        restored.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.005)
        restored_optimizer.load_state_dict(saved["optimizer"])
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(observed, action)
        torch.testing.assert_close(prediction, restored_prediction, rtol=0, atol=0)
        result["checkpoint_max_error"] = (prediction - restored_prediction).abs().max().item()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--motion-dim", type=int, default=6, help="Synthetic feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Small debugging override")
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint", default="outputs/smoke_predictor.pt")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = ModelConfig.from_yaml(
        args.config, motion_dim=args.motion_dim, hidden_dim=args.hidden_dim,
        prediction_length=args.pred_len,
    )
    print("DEVELOPMENT BASELINE: synthetic motion + MSE, not the AGED objective")
    decoder_kind = "residual" if config.residual_prediction else "absolute"
    decoder_condition = "with action" if config.condition_decoder else "without action"
    print(
        f"Observed [B,{config.observed_length},{config.motion_dim}] -> one-hot conditioning "
        f"-> Linear -> GRU encoder (H={config.hidden_dim}, L={config.num_layers}) "
        f"-> hidden state + last observed seed -> autoregressive {decoder_kind} GRU decoder "
        f"({decoder_condition}) "
        f"-> predicted future [B,{config.prediction_length},{config.motion_dim}]"
    )
    for name, value in run_smoke(
        config, args.steps, args.batch_size, args.seed, args.device, args.checkpoint,
    ).items():
        print(f"{name}: {value:.8g}")


if __name__ == "__main__":
    main()
