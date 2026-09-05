# GPU circuit sensory-history experiments

These runs exercise the real 6,789-neuron, 564,810-edge circuit in batches on
both AMD GPUs. They are circuit sensory-history experiments, not embodied
lives: the inputs are deterministic local sensory records, and no physical
world is simulated.

## Runtime

`TorchCircuit` mirrors the recurrent portion of `Brain.step` for independent
brains sharing one sparse connectome:

- two rate substeps with `dt=0.05`, `tau=0.16`, gain `0.92`, baseline `0.005`,
  rectified tanh target, support, and adaptation;
- the calibrated downstream decoder and context trace;
- odor eligibility, sound trace, nutrition modulator, values, sound memory,
  reinforcement, and slow extinction;
- float32 and float16 state, COO sparse recurrence, and an explicit recurrent
  edge-silencing switch.

The normal simulation remains NumPy/SciPy. PyTorch is imported only when this
optional batched runtime is used.

## Acquisition

Every circuit experienced the same repeating sequence of three bilateral odor
cues. Each 128-step cue block also contained measured-shaped RGB, looming,
sound, shade, and touch channels. Circuits were assigned history 0, 1, or 2;
only the assigned odor produced a `0.004` nutrition pulse. The 128-step blocks
separate cues by more than the four-second eligibility time constant, while
still allowing delayed reinforcement.

After 2,048 upbringing steps (102.4 simulated seconds), learning was frozen,
transient neural state was reset, and every circuit received the same 64-step
probe for each odor. Three conditions isolate the causal mechanisms:

| Condition | Plasticity | Recurrent edges |
| --- | --- | --- |
| `learned_recurrent` | enabled | enabled |
| `plasticity_disabled` | disabled | enabled |
| `recurrent_silenced` | enabled | silenced |

The exact executed commands, environment, seed, library versions, code hashes,
connectome hash, device memory, runtimes, and artifact hashes are recorded in
each `metadata.json`. From a checkout with ROCm PyTorch, SciPy, and NumPy:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
PYTHONPATH=. \
/home/hbox/h1-ghost/venv/bin/python scripts/gpu_nursery.py \
  --output-dir /tank/chreatures/runs/gpu/hbox-final-b256-s2048-seed20260905 \
  --batch-size 256 --steps 2048 --sample-every 16 \
  --parity-steps 256 --seed 20260905
```

```sh
PYTHONPATH=. \
/home/ember/kaxsim/.venv7/bin/python scripts/gpu_nursery.py \
  --output-dir /home/ember/chreatures-compute/runs/gpu/persvati-final-b64-s2048-seed20260905 \
  --batch-size 64 --steps 2048 --sample-every 16 \
  --parity-steps 256 --seed 20260905
```

On the staged hosts, SciPy was supplied from a project-only target directory:
`/tank/chreatures/envs/python-packages` on hbox and
`/home/ember/chreatures-compute/envs/python-packages` on persvati. Those paths
were prepended to `PYTHONPATH`; the existing ROCm environments were unchanged.

## Results

The finalized connectome SHA-256 is
`d2fb9bee3d591dc5af2be3a8eda0aba2a3c9f3b62b6c23d0a24cbdb4a72e3567`.
Both hosts used float32.

| Measurement | hbox RX 6750 XT | persvati Radeon 890M |
| --- | ---: | ---: |
| Batch size | 256 | 64 |
| Learned recurrent runtime | 17.85 s | 21.80 s |
| Plasticity-disabled runtime | 17.40 s | 20.59 s |
| Recurrent-silenced runtime | 1.97 s | 8.74 s |
| Peak allocated GPU memory | 105.6 MiB | 84.4 MiB |
| NumPy/Torch parity, maximum absolute error over 256 steps | 3.58e-7 | 4.62e-7 |
| Parity tolerance | 2.00e-4 | 2.00e-4 |

Persvati's existing PID 1656947 held `/dev/kfd` and the GPU reported 99-100%
busy before this replication. The run was deliberately smaller and its timing
is a contended measurement. That process was not interrupted.

The learned value vector and standardized probe predictions on hbox were:

| Reinforced odor history | Final values | Probe predictions for odors 0 / 1 / 2 | Assigned probe minus other probes |
| --- | --- | --- | ---: |
| 0 | 0.397 / 0.229 / 0.178 | 0.219 / 0.104 / 0.096 | +0.119 |
| 1 | 0.268 / 0.349 / 0.130 | 0.148 / 0.159 / 0.070 | +0.050 |
| 2 | 0.232 / 0.261 / 0.295 | 0.128 / 0.119 / 0.160 | +0.036 |

Against the plasticity-disabled baseline, prediction for each history's
assigned probe increased by 0.098, 0.059, and 0.095 respectively. With
plasticity disabled, values remained at 0.220 / 0.220 / 0.120. With recurrent
edges silenced, final mean activity fell from 0.1787 to 0.0213, probe decoding
changed by as much as 0.567, and history-specific probe separation disappeared
to numerical zero. The recurrence is therefore necessary for the calibrated
sensory response and the history effect requires plasticity.

Persvati replicated cohort-level final values, decoded probes, and prediction
probes within `5.96e-7` maximum absolute difference from hbox despite the
different GPU, ROCm release, batch size, and competing load.

## Artifacts

Each run contains compressed activity/support/decoder/value time series, final
learned state, standardized probe arrays, a CSV summary, and metadata:

- `runs/gpu/hbox-final-b256-s2048-seed20260905/`
- `runs/gpu/persvati-final-b64-s2048-seed20260905/`

The circuit-only result supports using hbox for larger batches and persvati for
capacity-sensitive replications. It does not establish behavior in the live
world; motor behavior must be tested through `Brain.step` and the world runtime.
