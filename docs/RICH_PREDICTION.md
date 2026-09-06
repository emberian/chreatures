# Rich sensory consequence prediction

The rich consequence organ forecasts one 50 ms sensory/body transition under
a candidate motor command. It is a separate learned component for fresh
research residents. It does not alter the frozen rich visual representation,
the resident's achieved-history memory, or an existing life.

## Contract

For a transition ending at index `t`, the 1,426-column input is ordered as:

1. frozen 256-column frame codes for `t-3`, `t-2`, `t-1`, and `t`;
2. the actual 384 MaleCNS readouts at `t`;
3. the actual eight-axis executed action plus oral command from `t-1`;
4. the candidate eight-axis action plus oral command for `t`.

The target has 262 columns: the frozen frame-code delta from `t` to `t+1`,
followed by the six raw physiology deltas over the same transition. A row is
excluded if a reset occurs anywhere from `t-3` through `t+1`.

All input and target moments come only from training worlds 0, 1, and 2. Input
standard deviations have a 0.02 floor. Runtime inference clamps standardized
inputs to `[-8,8]`; a candidate with any clipped coordinate receives zero
selection tilt. Code-delta output scales have a `1e-3` floor and physiology
delta scales have a `1e-4` floor. Raw next physiology is the current raw value
plus the predicted delta. It is not clamped; the caller marks it invalid when
it is nonfinite or outside the six physical bounds recorded in the artifact.

There are three separately parameterized models. Each is
`Linear(1426,256) -> tanh -> Linear(256,256) -> tanh -> Linear(256,262)`.
The forecast is their arithmetic mean after target denormalization. Per-output
disagreement is the population RMS member deviation in raw target units. It is
an uncalibrated diagnostic, not a probability interval.

The portable NPZ includes every member tensor, the observation/input/output
normalizers, empirical train residual scales, and exact copies of the visual,
body, and goal encoders. Every array has a dtype, shape, and SHA-256 receipt in
the embedded metadata. Runtime inference needs NumPy or the native Rust mirror;
it does not require PyTorch.

## Actual training run

The source data were the two authenticated packets in
`/home/ember/chreatures-data/sensorimotor-play/rich-v2-seed20260912` on
persvati. Their manifest content identity is
`14bb5f01f6fa2a69323180ed9c7d683e9ae1127f4bc22f11a3d0609c8fc855fe`.
Each packet records 4,096 transitions for 24 residents. The frozen bootstrap
was
`/home/ember/chreatures/runs/sensorimotor-rich/bootstrap-v1-seed20260912/rich-worker.pt`,
SHA-256
`cadebcece6034af1346c71c68d9962ea0f598f3f4e81263a5a4beee1db50543f`.

The executed command was:

```bash
/home/ember/kaxsim/.venv7/bin/python scripts/train_rich_prediction.py \
  /home/ember/chreatures-data/sensorimotor-play/rich-v2-seed20260912 \
  --bootstrap /home/ember/chreatures/runs/sensorimotor-rich/bootstrap-v1-seed20260912/rich-worker.pt \
  --output /home/ember/chreatures/runs/sensorimotor-rich/prediction-v1-seed20260912 \
  --epochs 10 --batch-size 1024 --seed 20260912 --device cuda
```

PyTorch 2.10.0 with ROCm 7.0 trained the ensemble on the AMD Radeon 890M in
37.0 seconds. Training used 147,312 rows from worlds 0–2. World 3 supplied
49,104 descriptive validation rows. The frozen representation bootstrap had
already seen every world, including world 3, so this is predictor validation
rather than an end-to-end held-out generalization result.

On world 3, all-output RMSE measured in training-target scales was 0.7412. The
zero-delta forecast scored 0.9752 and permuting candidate actions among the
same rows scored 1.3595. The model scored 0.6221 for 29,592 near-unchanged
action rows, 0.8731 for 18,858 moderately changed rows, and 1.3266 for 654 rows
whose mean absolute action change exceeded 0.25. No training, validation, or
permuted validation row crossed the eight-sigma input clamp.

For goal-space calibration, the predicted next code replaced the final member
of `[e[t-2], e[t-1], e[t], e[t+1]]` before applying the frozen goal encoder.
The predicted window scored 0.05449 overall RMS against the actual next goal
code, compared with 0.09880 for a zero-delta next frame. Runtime uses 0.05449
as an empirical error scale with a `1e-4` floor. It is not a confidence level.

The final artifact is
`data/genomes/rich-consequence-ensemble-v1.npz`, 7,545,779 bytes, file SHA-256
`7192e01191282a736299013159b7852b291c581ad06faff3b33b37e157d4352f`,
and logical tensor identity
`bc18bf8c25ab2d325c86ca5c70f65c1319f29d5cf0720e837dc7663e7aa1d4bf`.
The full fit receipt is
`data/training/rich-consequence-v1.receipt.json`. The copied encoder bundle
identity is
`bd3d70f5803c5b06f76143aa2f79c51c824736b9142a54c4dc2b3a26211f0ff0`.
The 62,008-byte native parity fixture at
`data/training/rich-consequence-v1.native-reference.npz` contains three actual
world-3 rows and their normalized inputs, three member outputs, ensemble
summary, and actual/predicted goal codes. It contains no resident identities,
coordinates, or object labels. Its SHA-256 is
`4910fed8c964089bcbda6f6470d82e1961b7ec3098e36c9cf4132306451229b1`.
A direct NumPy float32 forward pass over those rows matched the recorded AMD
PyTorch outputs within `1.65e-6` for normalized member outputs, `4.19e-7` for
raw member outputs, `1.42e-7` for the ensemble mean, and `2.24e-7` for RMS
disagreement. Input normalization matched exactly.

These are observational transitions from exploratory behavior. Better
prediction than the two declared comparisons does not establish causal action
effects, welfare improvement, or robust behavior in a new environment.

## Experimental action-suffix horizons

The H1 artifact above remains unchanged. The optional `--suffix-horizons 5 20`
mode fits a separate experimental format,
`chreatures-rich-action-suffix-consequence-ensemble-v1`. For horizon `H`, its
input is the same frozen four-frame code window, current 384 neural readouts
and previous action/oral vector, followed by the `H` actual executed
action/oral vectors from `t` through `t+H-1`. The input dimensions are 1,462
for H5 and 1,597 for H20.

Each model predicts four distinct future frame-code deltas relative to `e[t]`,
covering `e[t+H-3]` through `e[t+H]`, plus the six-coordinate physiology delta
at `t+H`. Its 1,030 outputs therefore preserve the temporal endpoint window
needed by the frozen goal encoder. They do not duplicate one forecast frame
four times. A row is accepted only when no reset occurs from `t-3` through
`t+H`.

The executed persvati command was:

```bash
/home/ember/kaxsim/.venv7/bin/python scripts/train_rich_prediction.py \
  /home/ember/chreatures-data/sensorimotor-play/rich-v2-seed20260912 \
  --bootstrap /home/ember/chreatures/runs/sensorimotor-rich/bootstrap-v1-seed20260912/rich-worker.pt \
  --output /home/ember/chreatures/runs/sensorimotor-rich/action-suffix-v1-seed20260912 \
  --epochs 10 --batch-size 1024 --seed 20260912 --device cuda \
  --suffix-horizons 5 20
```

The same AMD Radeon 890M, PyTorch 2.10.0 and ROCm 7.0 setup trained three
independent float32 members at each horizon. H5 used 147,168 train rows and
49,056 descriptive validation rows, completing in 53.2 seconds. H20 used
146,628 train rows and 48,876 validation rows, completing in 50.5 seconds. No
member failed and no train, validation or action-permuted row crossed the
eight-sigma input clamp.

| Horizon | Model scaled RMSE | Persistent-window baseline | Permuted suffix | Goal RMS | Baseline goal RMS | Permuted goal RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 ticks / 0.25 s | 0.6746 | 1.1133 | 1.0347 | 0.2369 | 0.5099 | 0.5380 |
| 20 ticks / 1.00 s | 0.6685 | 0.9842 | 0.9532 | 0.2918 | 0.5350 | 0.5660 |

The collected validation suffixes were all variable at exact float precision.
Only one H5 row, and no H20 row, kept every motor coordinate within 0.05 for
the whole suffix. Constant-zero and repeat-first candidate suffixes stayed
inside the fitted coordinatewise eight-sigma box, but that weak domain check
does not establish behavioral support. Their measured standardized distances
and the sparse stop counts are retained in each fit receipt.

The artifacts are
`data/genomes/rich-action-suffix-h005-v1.npz` (9,856,842 bytes, SHA-256
`d9fcb262bb25ad97e4c3f9139c73a5efc58a9999a2ce3530d07cd9421da86a33`,
logical identity
`ce4e9725fc9de990147405c00924a2eebc41fe36e1ba14aed56de3164129e8be`)
and `data/genomes/rich-action-suffix-h020-v1.npz` (10,244,494 bytes, SHA-256
`b2e23be61718f088ce9d98a21ee36abdd2e1a66a9ec231032b6dbb02652f73e1`,
logical identity
`247d5a436b16da439745d6af95a6abe6bbb5ca42169e9ff0db6338a93d9927a3`).
Their receipts and three-row actual validation references live under
`data/training/`; `rich-action-suffix-v1.recipe.json` pins the command, source
hashes, data, hardware, artifacts and measured NumPy float32 reload parity.

A possible later controller could compare constant five- or twenty-tick motor
plans, execute only the first action, then replan. This experiment implements
no planner, native runtime, resident integration or behavior claim.
