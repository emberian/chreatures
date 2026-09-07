# Physical-world CNS sensory probe

This bounded research program asks whether the synthetic-trained current V2 CNS
retains information about its actual sensory experience in eight physical worlds.
It does not train a resident policy or replace a published CNS artifact.

The input corpus is authenticated through the combined receipt, each episode
receipt binding, every raw afferent sidecar hash and the CNS service identity.
Episodes 00–05 supply training data; entire worlds 06–07 are held out. Latents
and afferents are aligned at each physical tick: Z[t] follows the CNS step on
afferent[t], then the teacher advances physics with its action. A 16-tick full
Torch replay verifies that alignment before any CNS fine tuning.

The compact probe receives only Z512. Its current head is linear and its future
head uses a GRU64 over Z. Targets are 24 optical features (mean RGB in four
sectors of each actual atlas eye) and all 43 body-local afferents. The sectors
are target compression, never numerical policy input. Body coordinates, object
labels, teacher actions, reward and bout labels do not enter either probe head
or the neural model. Raw afferents enter only the existing CNS adapters.

Target means and scales come exclusively from training worlds. Constant training
channels are excluded; per-channel scores and used-channel counts accompany
modality scores. Current prediction is compared with the training-mean baseline,
and four-tick future change is compared with persistence. A separate zero-edge
score supplies only the CNS readout bias to the same temporal probe; an actual
full-graph zero-edge run verifies that constant-output identity. This exposes
possible time-since-reset prediction in the head without pretending it is CNS
sensory information.

The first output is a frozen-CNS probe fit on the recorded real-world Z512. The
optional bounded fine tune uses the canonical `TrainableCNSAdapter`, all existing
trainable afferents/type parameters/readout, and the same probe. Chronological
TBPTT carries private neural and probe state across 32-tick chunks, with four
lookahead observations used only as offline targets. No world boundary is
crossed; held-out worlds never supply optimizer targets. Parameters are exported
only as a research candidate with a new identity. No service process or public
resident binding is changed.

Run from an isolated hbox source directory with the project-owned ROCm environment:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python \
  -m research.physical_cns_probe.fit \
  --corpus /tank/chreatures/runs/live-wave-v2/skill-curriculum-512x8/output/corpus-56616258c \
  --manifest-sha 75ed2d59169327490b073ecd6db3d856458badd90b91ebaea48f3871f962355b \
  --service /path/to/immutable/cns-service-v2-temporal-trained.bin \
  --atlas data/ports/optic-anatomy-audit-v1.npz \
  --output /new/tank/output/path
```

The output directory must be new. `receipt.json` is published after the frozen
probe and after each fine-tuning group, preserving useful measurements if later
work fails. Bulky raw data, neural arrays and candidate parameters stay on `/tank`.

## Executed eight-world result

The prespecified run completed in **521.26 seconds** on the hbox RX 6750 XT:
160 frozen-probe updates, then two chronological fine-tuning epochs with 90
optimizer steps. Each pass optimizes 480 target ticks per training world, with
four lookahead inputs through tick 483; the last 29 inputs do not supply optimizer
steps. Evaluation uses the complete 513-input chronology. Full Torch replay agreed with the original recorded Dawn
latents over the first 16 physical ticks to maximum absolute error `5.69e-5`.
The initial preflight corrected only the literal held-out split label from
`heldout-world` to the actual manifest's `heldout-worlds`, before any training.
The executed source hashes are preserved separately.

All values below are held-out-world skills; current targets use the training
mean baseline, and future targets use persistence. They use fixed training-only
scales and 24 optical, 35 varying body-current and 32 body-change channels.

| Target | Frozen CNS probe | Fine-tuned candidate | Candidate with zero graph edges |
| --- | ---: | ---: | ---: |
| Optical current | .08540 | .08499 | .00915 |
| Body current | -.01329 | .03252 | -.00090 |
| Optical four-tick change | -.000682 | -.000131 | -.000128 |
| Body four-tick change | -.000300 | .000271 | -.000163 |

The small body-current improvement does not establish broad useful physical
transfer. Optical reconstruction did not improve and future changes remain at
persistence. This candidate therefore does **not** justify replacing the
published CNS or starting a new resident lineage on the strength of these
results. The poor frozen result is retained as consequential evidence rather
than hidden behind the earlier procedural visual-prediction scores.

These are probe limits, not proof that Z contains no nonlinear body information.
The current head is linear, and the temporal head deliberately has no executed
command input, so this is not an action-conditioned forecasting test. Input
clipping does not explain the result: training body values had absolute maximum
1 and no entries outside the adapter's [-8,8] standardized range. Several body
channels vary much less than in the earlier procedural excitation.

The actual all-edge-zero verification produced a maximum deviation of **0.0**
from the constant CNS readout-bias latent. Thus its probe scores explicitly
expose time-since-reset or bias prediction without a sensory graph route.

The compact completed receipt is also preserved locally at
`/Users/ember/paperbin/chreatures/integration/physical-cns-probe-20260907/receipt.json`,
SHA-256 `1c9758c2eca5e76f2974858f456fab9d333821ea8f3a9d3d20432c6372658db5`.
Bulk output remains under
`/tank/chreatures/runs/live-wave-v2/physical-cns-probe-20260907/results-r2`.
The research-only `research-candidate-parameters.npz` is 42,257,349 bytes with
file SHA-256 `65116ed54d7299cd7b8003b4c73fde2d27736af5be99cd474d25c6c3ede78003`
and parameter identity
`ee3cb6c51664ff02e01ae48c99949daef5b333bb61bdc93bb4bd9014eb7a2f3b`.
Evaluation latents and probe weights stay beside it as research outputs. No
resident corpus, cold parent, new service or policy-training run was produced.
