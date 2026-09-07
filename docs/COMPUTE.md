# Compute hosts

Measured on 2026-09-05. PyTorch uses the `cuda` device API for both ROCm
hosts. No driver changes are required. Use the project-isolated environments
below; the original hbox `h1-ghost` environment was retired.

## Recommended environments

| Host | GPU | Python | PyTorch / HIP | Required runtime setting |
| --- | --- | --- | --- | --- |
| `hbox` | XFX Radeon RX 6750 XT, Navi 22, 12 GiB VRAM (`1002:73df`, subsystem `1eae:6710`, native `gfx1031`) | `/tank/chreatures/envs/rocm-dev/bin/python` | 2.9.1+rocm6.3 / 6.3.42134 | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| `persvati` | Ryzen AI 9 HX PRO 370 integrated Radeon 890M (`1002:150e`, `gfx1150`) | `/home/ember/kaxsim/.venv7/bin/python` | 2.10.0+rocm7.0 / 7.0.51831 | none |

The hbox wheel can enumerate native `gfx1031`, but kernels fail with
`invalid device function` without the override. With the override it reports
`gfx1030` and all probes pass. Keep the override scoped to each process.

Persvati's older environments also execute a small GPU matrix multiply when
started with `HSA_OVERRIDE_GFX_VERSION=11.0.0`:

- `/home/ember/kaxsim/.venv/bin/python` (PyTorch 2.5.1+rocm6.2)
- `/home/ember/restrans-exp/.venv-exp/bin/python` (PyTorch 2.5.1+rocm6.2)
- `/home/ember/h1-distributed/venv/bin/python` (PyTorch 2.9.1+rocm6.3)

The ROCm 6.2 environments warn that hipBLASLt does not support the overridden
architecture and fall back to hipBLAS. Prefer the native ROCm 7 environment.

## Reproduce the probe

From the repository root:

```sh
scripts/remote_probe.sh hbox /tank/chreatures/envs/rocm-dev/bin/python
scripts/remote_probe.sh persvati /home/ember/kaxsim/.venv7/bin/python
```

The wrapper copies `scripts/compute_probe.py` to the host, sets cache paths,
and emits JSON. Override sizes or timing counts with normal probe options:

```sh
scripts/remote_probe.sh hbox /tank/chreatures/envs/rocm-dev/bin/python \
  --matmul-size 4096 --warmup 10 --repetitions 50
```

Remote project storage and environments are reserved at:

- hbox: `/tank/chreatures/{cache,envs,probes,data}`. Put all new hbox data,
  package caches, environments, and builds under `/tank/chreatures`; `/` is
  97% full.
- persvati: `/home/ember/chreatures-compute/{cache,probes}`.

## Probe results

Default workload: 2048x2048 dense matrix multiply; index-add from 262,144
rows of width 64 into 65,536 rows; and 4096x4096 COO sparse matrix multiply
with 131,072 requested nonzeros and a 64-column dense right-hand side. Each
timing is the mean of 20 repetitions after five warmups. Correctness checks
against CPU references passed for every kernel.

| Kernel | hbox RX 6750 XT | persvati Radeon 890M |
| --- | ---: | ---: |
| FP32 dense matmul | 11.87 TFLOP/s | 0.89 TFLOP/s |
| FP16 dense matmul | 20.97 TFLOP/s | 3.22 TFLOP/s |
| FP32 index-add | 576.6 M updates/s (36.90 G values/s) | 82.2 M updates/s (5.26 G values/s) |
| FP32 sparse matrix multiply | 153.7 GFLOP/s | 48.3 GFLOP/s |

Persvati was already at 100% reported GPU busy during measurement, with PID
1656947 (`python`) holding `/dev/kfd`; its throughput numbers are contended and
should be remeasured when that existing job finishes. Hbox reported 0% GPU busy
outside the probe.

Resource snapshot at 2026-09-05 16:12 EDT:

| Host | Available RAM | GPU memory visible to PyTorch | Free project disk | Other use |
| --- | ---: | ---: | ---: | --- |
| hbox | 12 GiB | 12.0 GiB (about 98 MB used at rest) | 829 GiB on `/tank` | no `/dev/kfd` holder |
| persvati | 73 GiB | 41.9 GiB unified aperture; 8 GiB VRAM reservation | 661 GiB on `/` | GPU 100% busy, PID 1656947 |

Use hbox first for dense or scatter-heavy batches that fit in 12 GiB VRAM; it
was about 7x faster for index-add in this snapshot. Use persvati for larger
resident state because the integrated GPU exposes a much larger unified-memory
aperture. Before long experiments, check current load with fields that do not
expose command lines:

```sh
ssh hbox 'free -h; df -h /tank; ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head'
ssh persvati 'free -h; df -h /; ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head'
```

Coordinate before starting persistent experiments. Do not interrupt existing
GPU users. For the planned batched brain update, both `index_add_` and COO
`torch.sparse.mm` are operational; benchmark the real state shape on both
hosts before selecting the representation.

## Training world transport

Current training uses one spawned process per physical world, with 1–32
residents per world declared by the regional profile.
`ProcessWorldPool` keeps fixed numeric arrays in one parent-owned shared-memory
block: direct `float32 [world,resident,4096]` rich retina, pooled
`float32 [world,resident,351]` neural channels, `float32 [world,resident,12]`
executed controls, twelve physiology channels, and fixed body and outcome rows.
The initial population campaign uses eight residents per world, with separate
arrays for actual release, secretion, and allocation flows. Each worker writes
only its world row. Hot pipe messages contain an operation and monotonically
increasing sequence number; the parent reads a cohort only after every row has
acknowledged that sequence. A worker error closes the entire pool and its shared
memory, so callers cannot consume a partly updated cohort. World construction,
snapshot, restore, terminal outcomes, and detailed ecosystem telemetry remain
rare structured pipe operations.

`ProcessWorldPool.timing_snapshot()` reports hot-call wall time and summed and
maximum per-world CPU time for observation and advance. Training receipts use
this measurement together with neural and optimizer timings; it is a whole-path
measurement rather than a GPU kernel throughput figure.

### Rich corpus relay

The current rich-v4 collector seals 512-tick packets and complete coupled
checkpoints every 1,024 ticks. It atomically updates `progress.json` only after
the referenced files are durable, then writes the completed `manifest.json`
after the world pool closes. Relay a running hbox collection directly to
persvati with:

```sh
python scripts/platform/relay_sensorimotor_corpus.py \
  /tank/chreatures/runs/collections/COLLECTION \
  --destination-host persvati \
  --destination-root /home/ember/chreatures-data/sensorimotor-play/COLLECTION \
  --receipt /tank/chreatures/runs/collections/COLLECTION.relay.json \
  --watch
```

The relay accepts only monotonic prefix-extending progress receipts bound to
one collection identity. It authenticates every declared file while streaming,
uses a temporary destination name, fsyncs the destination file and directory,
and publishes the destination manifest last. A repeated invocation with the
same source and destination is idempotent; divergent destination bytes fail
closed. The relay does not open tensors or choose training rows. The trainer's
pinned split selects training and validation worlds, and final holdout worlds
remain excluded from fitting and model selection.

## Population campaign jobs

Population jobs use `scripts/platform/campaign_job.py`; the launcher supervises
ordinary commands and does not implement population search. A sealed
`chreatures-campaign-job-v1` manifest pins the host, exact source, executable,
environment settings, graph/ports/controller compatibility group, immutable
artifacts, candidate IDs, resource request, command argv, and external bulk
paths. Commands are argv arrays and never pass through a shell. Environment
values in a manifest must not contain credentials.

Create and validate a sealed job before launch:

```sh
python scripts/platform/campaign_job.py seal job.unsealed.json job.json
python scripts/platform/campaign_job.py validate job.json
python scripts/platform/campaign_job.py launch job.json
python scripts/platform/campaign_job.py status job.json
```

The resource check happens before graph or weight hashes are read. It requires
the declared available host RAM and bulk disk, and checks available AMD VRAM
when Linux exposes it through sysfs. A source may be an exact clean Git checkout
or an extracted source tree with a hash-pinned archive receipt and byte hashes
for every scoped entry point. The executable has its own byte hash, and the
environment names one or more immutable receipt artifacts for its package/ABI
identity. Immutable files use byte hashes; large directories use a hash-pinned
identity file plus their logical content hash.

Each candidate gets a private external directory containing an identity marker.
Candidates in one `compatibility_group` may share immutable graph and base
weights, while controller adapters, neural state, body state, RNG, memory and
checkpoints stay private. The sealed paths also name a campaign-relative shared
cache for compiled kernels and read-only reusable caches. The generic launcher
never merges candidate state.

Launch is idempotent for a sealed identity. A pending, running, or completed job
is returned without starting another process. A failed job remains failed until
`resume` is requested. Resume is available only when the original sealed
manifest includes a distinct `resume_argv` and versioned `resume_mode`
description; the application command remains responsible for loading and
validating its coherent checkpoint.

Supervision is stored below the manifest's campaign-relative supervision path:

```text
job.json
state.json
attempts/0001/supervisor.pid
attempts/0001/child.pid
attempts/0001/run.log
attempts/0001/exit-status.json
```

The launcher uses a file lock, a detached process session, and atomic JSON/PID
writes. `state.json` reports `pending`, `running`, `completed`, or `failed` and
points to the immutable attempt receipt. Failed attempts and logs are retained.
Current external roots are `/tank/chreatures/campaigns/v1` on hbox,
`/home/ember/chreatures-campaigns/v1` on persvati, and `runs/campaigns/v1` on
the M2. The M2 root is for orchestration and compact receipts because its local
disk has much less free space than either AMD host.

### Heterogeneous population episodes

`scripts/evaluate_population.py` executes one hardware-sized compatibility
group in the current physical world, full connectome, and native developmental
controller. The assignment document contains `worlds`; every row provides an
exact `{split,index}` environment selector, an unsigned seed, and a nonempty
list of complete `chreatures-population-genome-v1` values. All worlds in one
job have the same resident count, which is checked against the pinned profile.

```sh
python scripts/evaluate_population.py \
  --profile /bulk/campaign/profile.json \
  --resident-artifact /bulk/models/developmental-resident-population-v4.npz \
  --graph /bulk/data/malecns/derived \
  --port-bundle /bulk/data/ports/retinal-v2-maps.npz \
  --neural-recipe data/ports/neural-variant-canonical-v1.json \
  --assignments /bulk/campaign/batches/000001.json \
  --output /bulk/campaign/runs/000001 \
  --steps 1200 --checkpoint-every 600 --telemetry-every 120 \
  --device cuda --brain-backend tiled --physical-backend fast
```

The evaluator compiles and binds each candidate's neural phenotype, applies its
native controller adapter, delivers all twelve current actions, and records
actual physiology, organ flows, physical outcomes and native trajectory
summaries. Checkpoints contain the complete world, neural, controller/private
RNG, trajectory, boundary observation and previous-action state. `--resume`
accepts only the same evaluation identity and restores the latest authenticated
whole-step checkpoint before sampling another action. A failure writes one
diagnostic that references the last completed checkpoint and every affected
candidate. Its content-addressed copy remains under `failures/` across explicit
resume attempts; the evaluator does not retry or reset a life on its own.
