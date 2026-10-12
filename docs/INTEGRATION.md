# Contracts for the team

All tensors use batch-first layout. Predictions contain **future frames only**:

| Name | Shape | Meaning |
| --- | --- | --- |
| `observed` | `[B,T,K]` | `x_1 ... x_T`; final entry is decoder seed. |
| `future_gt` | `[B,H,K]` | `x_(T+1) ... x_(T+H)`; never predictor input. |
| `predicted` | `[B,H,K]` | `x_hat_(T+1) ... x_hat_(T+H)`; includes no duplicate seed. |
| `action` | `[B]`, integer | One class per sequence, IDs in `[0,A)`; one-hot made by model. |
| `hidden` | `[L,B,D]` | PyTorch GRU state; default L=1, D=1024. Internal interface only. |

Use floating-point motion tensors on the model's device and with its dtype.
Action IDs may be CPU tensors; they are copied to the motion device internally.
Output preserves motion layout, dtype and device. To select CUDA, move model and
motion together with `.to(device)`; no model code selects a global device.

## Data preprocessing owner

Supply standardized exponential-map features of relative joint rotations, plus
future targets in the **same** space. The paper specifies downsampling by two,
50 observed frames, training on six subjects and testing on subject five.
At the resulting 25 Hz, horizons are 10 / 25 / 100 frames (400 ms / 1 s / 4 s).
Configure K from your retained-channel metadata; no fixed feature count is assumed.

Keep action ID mapping, retained channel order, full joint-triplet restoration,
standardization means/scales, root-handling and frame rate as shared metadata.
Use training-only statistics to prevent leakage. The paper excludes whole-body
translation/rotation in evaluation. Exact masking and normalization details are
not supplied by the paper and are not fabricated here.

To calculate rotation geometry, undo standardization and restore any omitted
channels **differentiably**, before interpreting triplets as rotations. If reduced
K is not divisible by 3 or no longer groups complete joints, the loss adapter must
reconstruct the full joint layout. Never reshape a reduced vector blindly into
rotations. Avoid NumPy conversions or `.detach()` in the generator loss path.

## Predictor owner

```python
from models import AGEDModel, MotionPredictor

predictor = MotionPredictor(motion_dim=K, hidden_dim=1024, num_actions=15)
model = AGEDModel(predictor=predictor)  # Both discriminators optional.
predicted = model.predict(observed, action, pred_len=H)
```

`model(...)` also delegates to prediction. Repeated calls start fresh encoder
states. No loss, optimizer, DataLoader, hidden state persistence, target frame,
or discriminator call is part of prediction. Calling `.predict()` does not
implicitly switch to eval or disable gradients; the trainer controls these modes.

## Geodesic loss owner

Implement this contract (also defined as a typing Protocol in `models/contracts.py`):

```python
loss_geo = geodesic_loss(predicted, future_gt)  # Scalar tensor.
```

The loss module must hold or receive preprocessing metadata to restore raw
rotation coordinates. It compares matching future frames only. Eq. 4 sums
geodesic distances over future frames and joints. Batch reduction is not specified
by the paper: document the agreed reduction and account for its impact on the
adversarial/regression scale. Ensure stable, valid rotation conversion and a
well-defined gradient at singular/identity cases; loss mathematics is not
implemented by this scaffold.

The smoke script's MSE is a temporary debugging criterion, **not** Eq. 4 and
not an approximation endorsed for reporting AGED results.

## Fidelity discriminator owner

Sec. 3.2 / Eq. 6: unconditional, examines **future sequence only**.

```python
score_fake = fidelity_discriminator(predicted)  # [B,1], P(real)
score_real = fidelity_discriminator(future_gt)  # [B,1], P(real)
```

Do not append observed frames, action labels, or random noise for this contract.
The paper uses a single GRU, final hidden representation, fully connected layer
and sigmoid. `[B,1]` is our explicit output-shape agreement; the paper specifies
one real probability per sequence but not its tensor layout. If the team chooses
logits for numerical stability, agree to change the contract and use a logits
loss; do not feed sigmoid probabilities to `BCEWithLogitsLoss`.

## Continuity discriminator owner

Sec. 3.2 / Eq. 7: conditional on history by examining the concatenation of
history and future, **along the time axis**.

```python
score_fake = continuity_discriminator(observed, predicted)  # [B,1], P(real)
score_real = continuity_discriminator(observed, future_gt)  # [B,1], P(real)
```

Implement the two-argument API with internal composition:

```python
complete = torch.cat((observed, future), dim=1)  # [B,T+H,K]
```

The sequence has exactly one copy of seed `x_T`, as the last observed entry.
The discriminator sees the `x_T -> x_hat_(T+1)` boundary. Concatenating along
features, evaluating future alone, or inserting the seed a second time would
change the paper's discriminator input. An existing one-input discriminator can
be adapted to this two-argument API by its owner. No adapter or fake classifier
is implemented here.

## Training / evaluation owner

Attach actual `nn.Module` implementations when ready:

```python
model = AGEDModel(predictor, fidelity_discriminator, continuity_discriminator)
predicted = model.predict(observed, action, H)
loss_geo = geodesic_loss(predicted, future_gt)
score_fidelity = model.fidelity_discriminator(predicted)
score_continuity = model.continuity_discriminator(observed, predicted)
```

The full model should combine Eq. 4 with Eqs. 6–8 (lambda 0.6, p. 8).
Document any alternative adversarial objective rather than silently claiming it
is the paper's minimax objective. For discriminator updates, detach predicted
motion to avoid updating the predictor; for predictor updates, preserve its graph
and freeze discriminator parameters rather than detaching their inputs. Use
separate optimizer parameter sets: `model.parameters()` includes every attached
module and is unsuitable as a predictor-only optimizer after integration.

Reported training settings: LR 0.005, batch 16, clipping L2 norm 5, 50 epochs.
The original optimizer and update schedule are unspecified. Short-term training
uses a 10-frame target; long-term training uses a 25-frame target and can roll out
to 100 frames. Arbitrary horizons do not imply identical quality at every horizon.

Checkpoint the configuration, state dict, optimizer state and preprocessing
metadata together. Build the same tying/conditioning architecture before loading;
these choices are not inferred from a state dict. `AGEDModel.state_dict()` includes
attached discriminators automatically.

Evaluation should call `.eval()` and use `torch.no_grad()`. Own the paper's angle
error conversion, root exclusion, H3.6M sampling and horizons. The synthetic MSE
results from this project are engineering checks, not benchmark scores.
