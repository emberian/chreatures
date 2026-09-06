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

Current training uses one spawned process per three-resident physical world.
`ProcessWorldPool` keeps fixed numeric arrays in one parent-owned shared-memory
block: `float32 [world,3,351]` observations, `float32 [world,3,9]` motor and oral
commands, and fixed body, physiology, and outcome rows. Each worker writes only
its world row. Hot pipe messages contain an operation and monotonically
increasing sequence number; the parent reads a cohort only after every row has
acknowledged that sequence. A worker error closes the entire pool and its shared
memory, so callers cannot consume a partly updated cohort. World construction,
snapshot, restore, terminal outcomes, and detailed ecosystem telemetry remain
rare structured pipe operations.

`ProcessWorldPool.timing_snapshot()` reports hot-call wall time and summed and
maximum per-world CPU time for observation and advance. Training receipts use
this measurement together with neural and optimizer timings; it is a whole-path
measurement rather than a GPU kernel throughput figure.
