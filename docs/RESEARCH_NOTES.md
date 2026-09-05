# Sources that change implementation choices

Investigated 5 September 2026 against the actual Chreatures implementation.
The initial Kagi attempts used an obsolete request shape. After inspecting the
existing local client, POST with Bearer authorization and a JSON `query` worked.
The papers below were found with ordinary web search and read from primary
sites; the later sparse-kernel search used Kagi. Downloaded papers, search
results and SHA-256 receipts are stored in the local paperbin directory.

## Throughput: optimize the coupled system

[UniLab](https://arxiv.org/abs/2605.30313) separates parallel CPU simulation from
GPU learning using explicit movement and synchronization of data. Its reported
3–10× training improvements belong to its benchmark configurations, not ours.
The relevant design lesson is end-to-end throughput rather than moving every
subsystem onto the GPU. Its [current source](https://github.com/Motphys/UniLab)
has a dedicated ROCm dependency path and separate simulator/learner packages.

[MuJoCoUni](https://arxiv.org/abs/2605.24922) is especially pertinent: a native
C++/pybind11 persistent batch executor supplies short stepping, sparse resets
and batched sensor operations while retaining upstream CPU MuJoCo solver
semantics. This is a narrower adoption boundary than replacing the world with
an entirely different simulator. We have not installed or benchmarked it yet.

[MuJoCo's MJX documentation](https://mujoco.readthedocs.io/en/latest/mjx.html)
distinguishes MJX-JAX (including AMD support through XLA) from the NVIDIA-focused
Warp implementation. JAX favors large batches of similar scenes; its contact
scaling and unsupported features require checking against our actual garden.
[AMD's own example](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html)
runs MuJoCo/JAX on ROCm 7.2, tested on RX 7900 XTX. That is evidence of a route,
not verification of our RX 6750 XT or Radeon 890M environments.

**Our measured decision:** the current 48-resident learner achieves about 230
resident-steps/s, with GPU utilization around 90% and roughly two CPU cores
occupied. The full sparse neural step is the immediate bottleneck. First test
neuron-major contiguous state, fixed-cohort batching, fewer host synchronizations
and appropriate sparse-kernel layouts. A physics rewrite cannot remove this
bottleneck. Keep the current run's checkpoint before comparing implementations.

The [rocSPARSE changelog](https://github.com/ROCm/rocSPARSE/blob/develop_deprecated/CHANGELOG.md)
records layout-dependent SpMM improvements. It supports investigating the matrix
and dense-state layout; it does not establish a particular speedup on our graph.

## Memory: stable representations and rapid binding

[Universal Hopfield Networks](https://proceedings.mlr.press/v162/millidge22a/millidge22a.pdf)
factors associative recall into similarity, separation and projection, and
compares alternative similarity functions. That suggests a compact experiment
on our actual partial views: compare distance metrics, contextual disambiguation
and interference. A nearest-neighbor store need not acquire a grand biological
name to become a useful memory organ.

[TEM-t](https://arxiv.org/html/2112.04035) relates transformers with recurrent
position encodings to hippocampal models and spatial representations. The useful
architectural distinction is reusable relational structure versus rapid binding
of particular observations. In our world, movement/proprioception can update a
context estimate; absolute simulator coordinates must remain outside the organ.

[Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) carries learned
memory tokens across sequence segments. It is a candidate for a limited working
context, with the separate episodic store retaining selected personal events.
Its language and algorithmic results are not evidence of embodied competence in
this garden.

**Our implementation inference:** preserve raw perceptual features plus encoder
and projection versions. Online training or running normalization can change
latent coordinates; comparing old keys to new queries can then manufacture
forgetting or novelty. Reprojecting retained features allows explicit control of
that effect. The visual-memory lane is implementing this boundary.

## Current epistemic boundary

The native small VLM really processed a resident's rendered view on the Radeon
890M, but described part of the artificial garden as a tree. Its hypotheses are
kept out of the controller. The next experiment asks whether its dense visual
features contribute useful discrimination and contextual recall beyond pixels
or the compact retina; fluent labels alone do not justify integration.

Correction after inspecting the existing local client: Kagi v1 uses POST, Bearer authorization, and a JSON `query` field. That request succeeded. The sparse-kernel search is saved in `kagi-sparse-kernels.json` in paperbin; the earlier failure was an integration error, not a bad key.
