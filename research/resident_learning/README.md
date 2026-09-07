# Current CNS resident learning

This is the Torch/ROCm training boundary for the native v10 resident. Its only
controller inputs are authenticated masked full-CNS `Z512`, the previous
delivered action receipt, reset, and private recurrent state. Physical rewards
and delivered actions are teacher signals. Raw images, physiology, positions,
material kinds, identities, and geometry cannot be stored as model inputs.

The joint objective updates the exact canonical native tensors: recurrent
GRU256, goal/key encoder, four local action proposals, three latent transition
predictors, and the sequence selector/value heads. Experienced action suffixes
are presented in the same 234-value candidate layout used by Rust. Temporal
contrastive fitting shapes the 128-value private goal reservoir key, while
multi-step prediction and action coverage fit motor proposals and forecasts.
There is no parallel actor or inference-only research head.

`ClosedLoopCollector` accepts one initial Z512 observation and then alternating
delivered-action/reward/next-Z transitions. A privileged physical teacher can
calculate commands and scalar rewards outside the collector. Its geometry,
object labels, physiology, and raw sensor arrays have no field in the episode
format. `pack-joined` seals equivalent arrays exported by the browser or native
world and requires their CNS service identity to match the resident artifact.

The current corpus is six independent training worlds plus two held-out worlds.
Every episode is divided into gapless approach, heading-correction, stop,
withdraw, and contact-recovery bouts.  The trainer samples those skills
uniformly instead of sampling ticks uniformly, so a long turning phase cannot
dominate stopping or recovery.  World seed, variation seed, and layout identity
must all be unique, and held-out identities cannot occur in training. Bout
labels are privileged sampling/target metadata; they never enter a tensor given
to the resident. Each optimized 32-tick window first reconstructs up to 32
actual preceding CNS/action ticks without gradients, then detaches that private
state. This matches live recurrent context over the recent 1.6 seconds without
claiming retention beyond the explicit burn-in horizon.

An achieved future CNS key conditions each inverse motor fit. Deployment can
obtain that kind of goal only through the resident's own achieved-history
reservoir. The native predictor ensemble fits one-to-eight-step CNS outcomes;
the existing selector, value estimate, and acquired-suffix termination head use
the same physical sequence and scalar return. Held-out teacher variations and
world starts can be supplied as validation episodes, with before/after losses
recorded both overall and per skill in the final receipt. Predictor loss covers
every horizon from one through eight and receives only the current experienced
CNS key; the future achieved key is reserved for hindsight inverse-action
supervision and never enters a dynamics forecast context. Selector/value gradients stop at the
canonical head inputs so they cannot distort the CNS transition model to create
an easy classification shortcut. Termination learns from both the suffix that
was useful in the current context and a mismatched experienced suffix.

`scripts/train_cns_resident.py pack-corpus` seals the eight exact joined source
archives and writes their authenticated split manifest. It also authenticates
the complete collection receipt and its stable curriculum-contract identity;
placement-only source amendments remain explicit per episode without relabeling
already collected experience. `train` consumes that
single manifest and publishes a new immutable resident and sequence-control
artifact. Durable checkpoints move every model and optimizer tensor to CPU and
include lineage, RNG, validation baseline, and history, so another ROCm or CPU
host can resume them. `compare-native` checks the exported GRU state, goal key,
and four local proposals against the native implementation at a reset boundary.
