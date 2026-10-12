# Predictor specification and implementation decisions

Primary source: Liang-Yan Gui, Yu-Xiong Wang, Xiaodan Liang, José M. F. Moura,
**Adversarial Geometry-Aware Human Motion Prediction**, ECCV 2018,
DOI: [10.1007/978-3-030-01225-0_48](https://doi.org/10.1007/978-3-030-01225-0_48).
The user-supplied paper PDF (18 pages) was read directly. No external motion
prediction implementation was searched for, copied, or adapted. Page numbers below
refer to the paper's printed page numbers.

## Facts stated in the paper

| Question | Paper specification | Implementation |
| --- | --- | --- |
| One motion frame | `x_t` is a vector in `R^K`, consisting of joint angles in exponential-map representation; each joint has three components (p. 5, Problem Formulation; pp. 6–7, Eq. 4). | Caller supplies `[B,T,K]`; no fixed `K` in model. |
| Preprocessing | Standardized inputs and relative joint rotations (p. 5); downsample H3.6M by two, train on six subjects and test on subject five (p. 9). | Preprocessing remains a teammate's responsibility. |
| Observations | 50 frames = 2 seconds, for both short and long prediction (p. 9). | YAML `observed_length: 50`; GRU also accepts other nonempty lengths. |
| Short prediction | 10 frames = 400 ms; short-term loss trained over 400 ms (pp. 9–10). | Default prediction length 10. |
| Long prediction | 25 frames = 1 second, or 4 seconds (p. 9); long-term network trained over 1 second, visualizations extend to 4 seconds (pp. 13–14). | `pred_len=25` or `100` (100 derived from the stated 25 Hz); architecture is unchanged. This does not claim one checkpoint reproduces both training regimes. |
| Encoder / decoder | Each consists of a single GRU cell with hidden size 1,024 (p. 8, Implementation Details). | One unidirectional GRU layer each, hidden dimension 1024 by default. Multiple recurrent timesteps are not multiple GRU layers. |
| Linear mappings | Linear mappings between K-dimensional input/output and 1,024-dimensional hidden state; Fig. 2b depicts an input Linear before GRU and an output Linear after decoder GRU (pp. 5, 8). | Encoder and decoder each have `input_projection`; decoder has `output_projection`. |
| Action conditioning | Each encoder input frame concatenated with a one-hot action-class vector (p. 6, Sec. 3.1); H3.6M has 15 activities (p. 9). | Public integer IDs `[B]`, expanded internally to `[B,T,15]`; YAML uses 15 actions. |
| Decoder initialization | Encoder hidden representation and a seed motion frame fed into the decoder (p. 6); Fig. 1 shows a seed preceding prediction (p. 2). | Encoder state passed directly to decoder; seed indexing decision below. |
| Residual output | A residual connection models motion velocities rather than absolute angles (p. 6). | Predicted delta added to preceding pose in decoder loop. |
| Feedback | Decoder takes its own output as next timestep input (p. 6). | Prediction used immediately at next timestep, with the full gradient path retained. |
| Future ground truth | Ground-truth sequence supplies regression and discriminator targets (pp. 5–8). Decoder is described as consuming its own output. | No future GT argument and no teacher forcing in either training or inference. |
| Noise | Predictor has no noise inputs (p. 8). | No random latent vector or input noise. |
| Full objective | Geodesic regression + fidelity/continuity adversarial losses, weight lambda 0.6 (p. 8, Eq. 8). | Outside the predictor; not implemented here. Smoke MSE is not this objective. |
| Training settings | LR 0.005, batch 16, maximum gradient L2 norm 5, 50 epochs (p. 9). | Documented for training teammate; smoke uses LR 0.005 / norm 5 but batch 2 and Adam for debugging only. |

The exponential map is a 3-vector encoding a rotation, not an Euler-angle vector.
Do not interpret normalized output coordinates as radians without undoing
standardization. In Eq. 4, `K/3` assumes complete joint triplets; a reduced feature
dimension such as 54 is an arbitrary engineering example here, not a verified
H3.6M preprocessing convention.

## Decisions needed because the paper is ambiguous

| Missing detail | Chosen interpretation | Alternatives / limits |
| --- | --- | --- |
| Encoder/decoder parameter tying | Default **untied**, with separate linear input maps and GRUs. The paper says “respectively” but never explicitly says tied/shared. | `share_recurrent_weights=True` aliases both input mapping and GRU; output projection stays decoder-only. Do not claim either tying policy is proven by this paper's reference to [31]. |
| Action in decoder | Default concatenate the same one-hot action at every decoder step. Explicitly stated for encoder only; extending it to decoder keeps action information fixed during feedback. | `condition_decoder=False` implements the literal encoder-only reading. Sharing is rejected when input dimensions differ. |
| Exact seed index | Seed is the last observed pose `x_T`; encoder consumes **all T observed frames**, then decoder processes `x_T` again to predict `x_(T+1)`. | Figures distinguish conditioning/seed but do not specify slicing or whether to exclude seed from encoder. This choice keeps the public 50-frame window inclusive of seed. |
| Initial hidden state | Zero encoder initial state; final encoder state initializes decoder directly, without bridge or detached copy. | No initialization distribution or learned bridge is described. |
| Exact residual equation | `delta_(t+1) = Linear(h_(t+1))`; `x_hat_(t+1) = previous_pose + delta_(t+1)`, where `previous_pose=x_T` at step one and the preceding prediction thereafter. | Paper states velocity residuals, but does not provide the recurrence equation. We add in standardized feature space and unnormalize externally. No addition to action one-hot channels. |
| Input mapping dimensions / activation | Explicit affine `Linear(K+A,H)` followed by `GRU(H,H)`; encoder-only conditioning changes decoder mapping to `Linear(K,H)`. No extra activation. | Fig. 2b and linear-mapping text motivate explicit input maps; the paper does not specify whether GRU's internal input affine suffices or where one-hot is embedded. No hidden modern embedding MLP is added. |
| Projection sharing and bias | Untied input projections by default; all Linear/GRU modules use PyTorch's standard biases and initialization. | Projection tying and initialization are unspecified. Sharing mode ties both input maps and recurrent weights, and is an exploratory option. |
| Dropout | Zero dropout; deterministic default training forward pass. | Paper does not specify predictor dropout. Nonzero configurable dropout is GRU inter-layer dropout and requires more than one layer, thus an experimental architecture variant. |
| Optimizer | Smoke uses Adam. | Paper gives LR / clipping but does not name the optimizer. Adam is a debug choice, not a confirmed original setting. |
| Action ID order / normalization / retained channels | No default action-name order, channel mask or normalization statistics are invented. | These must come from shared preprocessing metadata and remain fixed across train, validation and checkpoint restore. |

`MotionPredictor(..., num_actions=None)` is an optional unconditioned variant for
standalone tests. The paper-oriented YAML instead selects 15 actions, one layer,
1024 hidden units and residual prediction. Nonresidual, multi-layer, different hidden
size, and unconditioned modes are development variants, not claimed reproductions.

## Recurrence and sequence boundary

```text
X       = [x_1, ..., x_T]                 observed, T=50 in paper
X_gt    = [x_(T+1), ..., x_(T+H)]         target only; never predictor input
X_hat   = [x_hat_(T+1), ..., x_hat_(T+H)]  returned future; excludes seed

h_enc_0 = 0
h_enc_T = Encoder(Linear([X, action_one_hot]), h_enc_0)
h_dec_0 = h_enc_T
previous = x_T
for each of H future steps:
    h_dec = GRU(Linear([previous, action_one_hot]), h_dec)
    delta = output_Linear(h_dec)
    predicted = previous + delta
    previous = predicted
```

This schematic includes the documented decoder-conditioning assumption. No GT
frame appears in this recurrence, even when the model is in `.train()` mode.
Hidden states are local to each call: nothing is carried across minibatches.
Return shape is `[B,H,K]` for any positive horizon. Prediction length is a number
of output frames, not the GRU hidden dimension.

## Mapping to files

- `models/encoder.py`: Fig. 2b linear input map, Sec. 3.1 one-hot conditioning and recurrent encoding, p. 8 hidden size / layer count.
- `models/decoder.py`: Sec. 3.1 residual prediction and explicit autoregressive feedback; assumption-sensitive seed/state passed by caller.
- `models/predictor.py`: combines encoder final hidden and last observed seed; optional parameter aliases.
- `models/config.py`, `configs/model.yaml`: central defaults, preprocessing-supplied K, clearly labeled assumption switches.
- `models/aged.py`: component registration and prediction-only delegation; no adversarial training hidden inside it.
- `models/contracts.py`, `docs/INTEGRATION.md`: Sec. 3.2 and Eqs. 4, 6–8 interfaces, without fake implementations.
- `scripts/smoke_train_predictor.py`: temporary synthetic MSE baseline with optimizer and checkpoint restore.
- `tests/`: conditioning, recurrence boundary, residual accumulation, gradient flow, horizons, defaults, checkpoint and overfit checks.

## Fidelity review and scope

Rechecked against Sec. 3.1, Fig. 2b, Implementation Details and experiment sections
4, 4.1, 4.3. This implements the predictor under the disclosed interpretations;
it is not a complete reproduction of AGED results. Geodesic loss, both recurrent
discriminators, real H3.6M preprocessing, adversarial training and official benchmark
evaluation remain teammates' modules. There is no dataset in this checkout.

CPU execution with PyTorch 2.14.1+cpu is the validated runtime. CUDA code paths are
device agnostic and have a conditional test; CPU checks do not establish CUDA
readiness or numerical equivalence to the 2018 implementation. No pretrained
weights or external implementation are used. Exact original parameter sharing,
seed slicing and decoder action handling remain unconfirmed by the supplied paper.

## Recorded validation

Executed from the repository root using `/workspace/aged-venv/bin/python`:

| Check | Observed outcome |
| --- | --- |
| `python -m pytest -q` | 35 passed, 1 skipped; the skipped test is CUDA execution. |
| `python -m scripts.smoke_train_predictor` | 250 Adam steps, H=32, K=6, B=2, T=50, horizon=10: MSE 0.79337829 -> 0.0000064077258; exact checkpoint restoration. |
| `python -m scripts.smoke_train_predictor --pred-len 25 --checkpoint outputs/smoke_long.pt` | 250 steps: MSE 7.2575917 -> 0.000027468546; exact checkpoint restoration. |
| Paper hidden-size forward | Test executes B=2, T=50, K=54, hidden=1024, actions=15, horizon=10 successfully. |
| Reusable installation script | Rerun successfully; dependency check, complete pytest run and 10-frame smoke all pass. |
| Real H3.6M / final AGED objective | Not executed: dataset and teammates' modules are outside this deliverable. |

These are engineering checks on synthetic coordinates, not research benchmark scores.
