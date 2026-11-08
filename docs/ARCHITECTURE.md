# Complete pipeline and tensor contracts

```text
99-channel export (50 Hz)
  -> load and downsample once (25 Hz)
  -> fit training-only statistics, select varying body coordinates
  -> observed [B,50,K], target [B,H,K], integer action [B]
  -> one-hot-conditioned GRU encoder -> hidden + last observed seed
  -> residual autoregressive GRU decoder -> prediction [B,H,K]
       |-> restore full raw joint triplets -> SO(3) geodesic regression
       |-> fidelity GRU -> P(real) for future only
       `-> continuity GRU -> P(real) for observed + future along time
```

`K` is measured from normalization metadata and can be indivisible by three.
Only **restored** 3-vector joint coordinates may be interpreted as rotations.
The predictor returns future frames only; its seed is not included in output.
No future ground-truth frame is ever passed into the predictor. GRU state is
reset for each input batch. Predictions feed back without `.detach()`.

## Package layout

| File | Responsibility |
| --- | --- |
| `data.py` | Text export, training moments, root exclusion, action IDs, reproducible windows |
| `models/encoder.py` | Per-frame action concatenation, affine projection, GRU encoding |
| `models/decoder.py` | Seed/state initialization, affine projection, GRU, residual feedback |
| `models/predictor.py` | Predictor API and parameter-sharing switch |
| `models/aged.py` | Registration of predictor and optional discriminators |
| `critics.py` | One recurrent classifier architecture, independently instantiated twice |
| `losses.py` | Complete pose restoration and stable Rodrigues / shortest-angle geometry |
| `train.py` | Alternating optimizers, clipping, validation, epoch/checkpoint lifecycle |
| `evaluate.py` | Fixed S5 windows, Euler angle L2 metric, raw-pose zero-velocity baseline |
| `runtime.py` | Device selection, assembly, checkpoint format and restoration |
| `config.py` | One serializable configuration schema |
| `tracking.py` | Optional TensorBoard, runtime and code-version metadata |

## Adversarial updates

For each batch, the predictor generates a future sequence from history and
action IDs. Discriminators minimize the negative real/fake log-likelihood using
**detached** generated futures. Their optimizer has no predictor parameters.

Then discriminator weights are frozen. Fresh discriminator forwards on the
attached prediction preserve the gradient into the predictor. The generator
minimizes geodesic reconstruction plus lambda times the selected adversarial
objective. Both optimizers have finite-gradient checks and norm clipping.
One discriminator update precedes one predictor update per batch.

`Critic.forward(sequence)` returns probabilities `[B]`. Training uses its
`logits(sequence)` to evaluate log-probabilities stably. Fidelity consumes
`[B,H,K]`. Continuity consumes `torch.cat((observed,future), dim=1)` of shape
`[B,50+H,K]`; the last observed seed appears exactly once. Neither discriminator
receives action IDs or noise.

## Normalization and geometry

Statistics are computed over downsampled training recordings. Coordinates with
standard deviation below 1e-4 are omitted; six global-motion channels are always
omitted. A model prediction is scattered back into the full 99-channel training
mean, undoing normalization differentiably before loss computation. Body joint
vectors in 6:99 become 31 rotation matrices. Nonselected coordinates are fixed
at their training means.

Rodrigues coefficients use `sinc(theta/pi)` and
`0.5*sinc(theta/(2*pi))**2`. This preserves the correct first derivative at
zero; a clamped denominator alone did not. Geodesic distance uses
`atan2(norm(skew_vector), (trace-1)/2)` on the relative rotation, selecting
the angle in [0,pi]. The norm has a zero subgradient at identity; the shortest
distance has no unique derivative at the pi cut locus.

S5 evaluation retains the complete raw target and raw observed seed, including
channels absent from the training mask. Predictions restore omitted coordinates
from training means; the zero-velocity baseline repeats the actual raw seed.
Global motion is then excluded from both. Validation uses reconstructed reduced
targets and serves checkpoint selection, not a published test result.

## Checkpoints and portability

`last.pt` contains the entire model (including critics), both optimizer states,
normalization statistics, action vocabulary, configuration, epoch/history,
best validation score and Python/PyTorch RNG states. `best.pt` is written only
when a development validation score improves. Full six-subject training always
writes `last.pt`; it never selects a model using S5.

Epoch-based window sampling is stateless in `(seed,epoch,index)` and changes
between epochs. CPU resume is tested against uninterrupted training, including
optimizer tensors. Resume must preserve data contents and training settings;
only total epochs, device, thread/worker counts and logging can be changed.
Evaluation reuses checkpoint statistics and never refits them on S5.

All model tensors follow the caller's device. Select CUDA or MPS explicitly or
use `auto`; explicit requests fail if unavailable. CUDA/MPS numerical behavior
and cross-device exact resume have not been validated on this CPU-only machine.
