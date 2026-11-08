# Paper alignment, assumptions and evaluation limits

Primary source: Gui, Wang, Liang and Moura, **Adversarial Geometry-Aware Human
Motion Prediction**, ECCV 2018,
[DOI 10.1007/978-3-030-01225-0_48](https://doi.org/10.1007/978-3-030-01225-0_48).
Page numbers refer to the user-supplied 18-page PDF. The paper remains the
algorithmic specification; no external motion-prediction implementation was
retrieved during integration.

## Explicit paper facts

| Specification | Source | Implementation |
| --- | --- | --- |
| Standardized relative joint rotations in exponential-map form | p. 5, problem formulation | Training-only statistics; full raw triplets restored for geometry |
| Single-layer GRU, hidden 1024, encoder and decoder | p. 8, implementation details | `models/`, default configuration |
| Linear mappings between motion and GRU hidden space | Fig. 2b, p. 8 | Explicit input affine maps; output affine predicts K deltas |
| One-hot action concatenated to each encoder frame | p. 6, Sec. 3.1 | Public integer IDs, internal one-hot expansion |
| Residual connection models velocity; decoder feeds back own output | p. 6 | Delta added to previous pose; no teacher forcing or detached feedback |
| Fidelity judges future; continuity judges concatenated history/future | pp. 7–8, Eqs. 6–7 | Independent recurrent critics, no action or noise input |
| GRU followed by fully connected layer and sigmoid in critics | p. 8 | `Critic.forward`, stable logits path for optimization |
| Geodesic regression plus two adversarial losses, lambda 0.6 | pp. 7–8, Eqs. 4–8 | Sum over future time/joints, mean over minibatch; minimax default |
| LR 0.005, batch 16, clip norm 5, 50 epochs | p. 9 | Configuration defaults |
| Downsample by two, six training subjects, test S5 | p. 9 | Loader split and fit-time S5 rejection |
| 50 observed frames; 10/25 predicted; 4-second rollout | p. 9, Secs. 4.1/4.3 | Short/long configs; evaluation can roll out 100 frames |
| Exclude whole-body translation and rotation in evaluation | p. 9 | Global channels 0:6 excluded, independently of observed variance |

## Implementation choices the paper does not determine

- **Parameter sharing:** untied encoder/decoder GRUs and input maps by default;
  `share_recurrent_weights` can alias them. The paper does not explicitly say tied.
- **Decoder action conditioning:** action one-hot is also concatenated at every
  decoder step by default. This extends the explicit encoder description and is
  switchable with `condition_decoder` in YAML.
- **Seed boundary:** encode all 50 observed frames; feed the last frame again as
  decoder seed. Pass the encoder final state directly. Initial encoder state is
  zero. Exact slicing and initialization are not fully specified in the paper.
- **Affine details:** input dimension is K+15, mapped to H; no extra activation.
  Standard PyTorch initialization and biases are used. Critic hidden size 1024 is
  chosen consistently with the predictor; a separate critic size is configurable.
- **Optimization:** Adam, one joint critic update then one predictor update per
  batch; the original optimizer and schedule are not named. The generator's
  default is the paper's saturating minimax form `log(1-D(fake))`. Optional
  `non_saturating` uses `-log D(fake)` and is explicitly a deviation.
- **Loss scale:** Eq. 4 sums time/joints; minibatches are averaged here. Optional
  mean-over-time/joints changes relative adversarial strength and is an ablation.
  Discriminators minimize negative log-likelihood without lambda scaling; positive
  lambda does not change their optimum. Regression/adversarial scales must be
  reported alongside results, not silently altered.
- **Input export:** comma-separated 99-channel data at 50 Hz is the supplied
  team's export contract, not a file format specified by AGED. No BVH parser is
  included. Normalization at 25 Hz, threshold 1e-4, streaming population moments,
  length-weighted window sampling and 2000 steps/epoch are engineering choices.
- **Validation:** default training uses all six training subjects. An optional
  S11 development holdout fits statistics on the other five and selects `best.pt`
  by mean Euler error. S5 is never used for checkpoint selection.

## Evaluation protocol and remaining reproducibility limits

The supplied team archive documented the fixed eight-seed evaluation parameters
but left the executable selection function unfinished. This package independently
implements those documented constants: NumPy `RandomState(1234567890)`, four
alternating draws from each take, prefix 50, suffix at least 100, random start
in `[16,T-50-suffix)`, future boundary `start+50` at 25 Hz. These details are
benchmark conventions documented by the team, **not all stated by AGED**.

Raw rotation vectors are converted through SO(3) to Euler angles using the
convention `R=Rx(-a)Ry(-b)Rz(-c)`. Global translation/rotation are excluded;
for each sequence, target Euler coordinates with standard deviation >1e-4 enter
the L2 norm. Errors are averaged over sequences at future indices 1,3,7,9
(80,160,320,400 ms), and 13,24 for long prediction (560,1000 ms).
This implements the team's documented benchmark convention; the AGED text calls
the metric MSE while conventional tables use mean Euler vector error. This
distinction and Euler convention must be checked against real baseline results.

The legacy evaluation instead sampled arbitrary windows and compared reduced
exponential-map vectors. Those numbers are not interchangeable with these scores.
All evaluations include a raw-pose zero-velocity baseline for comparison.

**Real Human3.6M data, a matching baseline table, GPU execution and a full
50-epoch run have not been validated in this environment. No published accuracy
claim follows from passing synthetic tests.** The short-term model must be
trained on 10-frame loss, and the long-term model on 25-frame loss, before
reporting the corresponding results; rolling a checkpoint farther is only an
inference capability, not evidence of long-term quality.
