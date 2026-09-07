# Full MaleCNS operating-point research

The production equation/array boundary is frozen in
[`CNS_DYNAMICS_V2.md`](../../docs/development/CNS_DYNAMICS_V2.md).
These isolated research programs authenticate a sealed v1 service as an
anatomical and afferent seed; they never reinterpret its private state as V2.

From the repository root:

```sh
.venv/bin/python -m research.dynamics_v2.characterize \
  --service /path/to/cns-service-fitted-v1.bin \
  --atlas data/ports/optic-anatomy-audit-v1.npz \
  --output /new/path/to/regime-audit

HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python \
  -m research.dynamics_v2.train \
  --service /path/to/cns-service-fitted-v1.bin \
  --atlas data/ports/optic-anatomy-audit-v1.npz \
  --output /new/path/to/fit --gain .9 --updates 64

HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python \
  -m research.dynamics_v2.export_fixture \
  --service /path/to/cns-service-fitted-v1.bin \
  --fit /path/to/fit/fitted-parameters.npz \
  --output /new/path/to/fixture
```

The training objective weights current feature MSE plus twice future-change MSE,
with a .01 mean-square prior on raw type gains/time constants. No activity
variance, entropy or animation score is optimized. Readout and type dynamics
learn from full-graph sparse gradients. Optical and body afferent weights are
frozen for this mechanistic calibration. All raw features remain offline target
values or afferent inputs. Fitted prediction heads are not controllers.

The fixture exporter supplies all five exact type arrays, `afferent.neutral_drive`
and the rank-64 factorized Z512 readout; service packing stays with root. It also
supplies eight B2 physical-dt Torch steps with canonical sensory, rate, adaptation,
support and latent arrays, suitable for the joined native/browser check. The
fixture is generated from the executed fitted parameters, not synthesized from
the desired target image.

Use unique output paths. Research programs refuse existing output directories.
A failed process does not resume private world state or overwrite old artifacts.

The current research trainer wraps the canonical
`research.sensorimotor_skills.cns_adapter.TrainableCNSAdapter`; it has no separate
Torch recurrence. The completed first and second fitting runs retain immutable
source tarballs beside their artifacts because the training head and frequency
protocol changed. Their results must not be relabeled as executions of a later
source revision.

`crossed_regimes.py` supplies the full 27-setting graph response design, an
explicit `--baseline-only` historical control and `--setting GAIN TAU ADAPT`
confirmation. `gam_fit.py` fits native GAMs with configuration holdouts;
`GAM_RESPONSE_SURFACE.md` records the actual fit and confirmation. `audit_fit.py`
replays the trained task through the canonical model, checks joint full-graph
training gradients, exports canonical V2 parameter metadata, and measures a
local all-edge ablation. None of these tools mutate a running resident.
