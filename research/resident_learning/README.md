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

An achieved future CNS key conditions each inverse motor fit. Deployment can
obtain that kind of goal only through the resident's own achieved-history
reservoir. The native predictor ensemble fits one-to-eight-step CNS outcomes;
the existing selector, value estimate, and acquired-suffix termination head use
the same physical sequence and scalar return. Held-out teacher variations and
world starts can be supplied as validation episodes, with before/after losses
recorded in the final receipt.

`scripts/train_cns_resident.py convert-recording` can remove physical geometry
from an actual joined assay after deriving scalar displacement reward. `train`
publishes a new immutable resident and sequence-control artifact. `compare-native`
checks the exported GRU state, goal key, and four local proposals against the
native implementation at a reset boundary.
