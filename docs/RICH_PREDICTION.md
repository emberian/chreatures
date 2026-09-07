# Recurrent rich consequence organ

The current predictor is a three-member recurrent ensemble trained from authenticated `chreatures-sensorimotor-play-rich-v3` trajectories. It forecasts body-local achieved-sensory codes and all twelve physiology channels from delivered canonical actions. It does not receive coordinates, object labels, world identity, rewards, or unobserved state.

At decision tick `t`, the 1,560-value context is four frozen 256-value frame codes ending at `t`, the exact native effective worker context (`state + recurrent_adapter`) used to choose action `t`, 384 canonical neural readouts, twelve raw physiology values, and the previous delivered action12. Candidate input is `[B,K,H,12]`, with `1 <= H <= 8` and a 50 ms interval. Each independent member projects the context to 256 values, advances a 256-state `GRUCell` once per candidate action, and emits a frame-code proposal256 plus physiology proposal12 per step.

Physiology proposals use a smooth bounded link during both fitting and inference. For current state `p`, upper headroom `u`, lower headroom `d`, proposal `q`, and epsilon `1e-4`, stable positive and negative parts of `q` are computed from `sqrt(q² + epsilon²)`. The decoded delta is `u*tanh(qplus/max(u,epsilon)) - d*tanh(qminus/max(d,epsilon))`. The loss is applied to these decoded physical deltas and their cumulative trajectories. No post-hoc clipping invents valid physiology.

The native hot path reuses cohort-owned buffers and returns member deltas, ensemble means, uncalibrated RMS disagreement, cumulative absolute code and physiology, and prefix-validity flags. Runtime v6 evaluates a declared four-tick constant-action counterfactual; only its first action is delivered before replanning. This is a model-based score, not evidence that the counterfactual action caused the observational training outcome.

Training requires explicit whole-world partitions. For the first corpus, worlds 0–3 fit all moments and weights, world 4 supplies validation and per-horizon goal-space error, and world 5 is opened once for the final report. The frozen resident checkpoint is replayed only to encode frames. Effective recurrent context is recorded directly from native execution because candidate `recurrent_gain` creates private state that shared Torch GRU replay cannot reconstruct exactly.

The first collection completed 393,216 transitions in 1,716.603 wall seconds
(229.06 resident transitions/s including compression and checkpoints). Six
eight-resident worlds each supplied two 4,096-tick episodes. Disk-backed arrays
keep collection memory bounded; packet hashes and the authenticated relay to
persvati are retained in the [collection receipt](../data/training/rich-sensorimotor-play-v3.receipt.json).
Its frozen v5 source and update-160 policy precede the reciprocal-v6 chemistry
and memory changes. Fits on this corpus are inherited priors for the new world;
held-out source-world error does not validate the new ecological dynamics.

```bash
/home/ember/kaxsim/.venv7/bin/python scripts/train_rich_prediction.py \
  /home/ember/chreatures-data/sensorimotor-play/rich-v3-predictor-seed20260918 \
  --representation-checkpoint /home/ember/chreatures-data/models/population-v5-update160/development.pt \
  --validation-worlds 4 --heldout-worlds 5 \
  --trusted-checkpoint --device cuda \
  --output /home/ember/chreatures/runs/prediction/rich-recurrent-v3-seed20260918
```

The artifact pins the trajectory manifest and packet hashes, complete action and physiology orders, exact resident checkpoint and representation tensor identities, normalization floors, temporal alignment, bounded decoder, tensor receipts, and H1–H8 validation and final-heldout goal errors. Existing residents retain their older immutable predictor artifacts.

## First executed fit

The first fixed three-member fit ran on an AMD Radeon 890M with Torch 2.10.0
and ROCm 7.0 for 399.791 seconds. Its corpus contains 393,216 transitions;
fitting used only world slots 0–3. World 4 remained validation-only, and world 5 was opened once
for the final held-out report.

At the runtime scoring horizon H4 (0.20 seconds), held-out goal-code RMS was
0.234636, compared with 0.550485 for a persistence forecast. Held-out H4 RMSE
was 0.152928 for the visual code, 0.254890 for the body code, and 0.070819 for
the twelve bounded physiology channels. These are learned representation and
normalized physiology units, not physical calibration or evidence of improved
control. Ensemble disagreement is residual member spread and is not calibrated
epistemic uncertainty.

The fit used trajectories from the instrumented frozen-v5 update-160 controller
and source-world dynamics. Reciprocal-v6 uses this predictor as an inherited
bounded prior. The held-out result does not validate transfer to v6 chemistry,
new ecology, or a fresh resident's action choices.

The compact [fit and export receipt](../data/training/rich-recurrent-v3/fit-export-receipt.json)
pins the corpus, split, representation checkpoint, source files, predictor and
native-v6 artifact identities, embedded consequence laws, held-out metrics, and
hardware run. [Native parity](../data/training/rich-recurrent-v3/native-parity.json)
used the fitted weights with synthetic B3 x K4 x H4 numerical inputs; maximum absolute error was `1.76e-6` for cumulative
codes and at most `9.54e-7` for per-member deltas. The check is reproducible
with `scripts/validate_rich_prediction_native.py` against the immutable
predictor and compiled `_cognitive_core` extension.
