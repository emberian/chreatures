# Weekend handoff — September 8, 2026

The user requested quiescence at 8:45 PM EDT, superseding the earlier 10 PM
autonomous target. The short experiments already running finished and were
sealed. **No new training, browser-model export, or experiment wave is running.**
Resume development only on renewed user direction; no incomplete experimental
mutation needs retrying.

## Public state

The live garden uses the initialized V5 full MaleCNS model, two articulated
NeuroMechFly bodies and the unified Rust/MuJoCo Wasm host. Selected release:
`v5-garden-2c040fd`; archive SHA
`505456e9b214bb23865c8193c5af441d0409a36ca3634d0b5c0155ac958977e4`.
The startup Three.js module failure, material/texture color omissions, lighting,
and Follow camera are repaired. The imported textures currently contribute
per-material mean albedo, not UV-resolved patterns.

Native encounter programs schedule actual tones, screen patterns and toy forces
before sensation. CHLIVE6 saves retain the pending program, delivered screen,
neural state, private memory and physics. The final 131-advance joined run
restored the complete coupled state exactly. Prior lives keep their pinned engine.

`ecology.html` adds an observation-only growth atlas: 52 completed worlds,
1,716 sparse samples, four inherited branching parameters, actual committed
capsules, static habitat geometry and native GAM predictions. Moving flies and
resource packets are omitted from this recording. Playback never interpolates
unrecorded branches. Native Universal Weave serialized and reloaded its
72-node/123-edge evidence graph, including the preserved failed launch.

## Training and embodied comparison completed

Chronological training at `a876bc86adbf76ca9104c68ff205a5d71d0db239`
carried all nine V5 neural fields through complete world histories. The run
completed 512 CNS segments and 512 private-resident updates in 4,225 seconds.
Its 48-world corpus split is 32 training, eight validation and eight held-out
worlds. Bounded 16-tick gradient windows do not give credit across an entire life.

The exact production Rust resident then ran on hbox with the frozen physical
host `75fd8fc5304b8c76ada70bdb5251adecd787cd5f` and assessment source
`2704d2b2ddc2a38f7ba950f9918fa56fc1aae15b`. Every condition used the same
RX6750XT and two held-out B4 starting worlds; the separate persvati baseline was
not pooled into this comparison.

| Condition | Falls / 8 residents | Mean upright time | Horizontal endpoint motion | Mechanical work |
| --- | ---: | ---: | ---: | ---: |
| Initialized, zero context | 0 | 5.12 s | 0.08093 mm | 23.71 |
| Trained CNS, zero context | 0 | 5.12 s | 0.13832 mm | 54.53 |
| Trained CNS, private memory | 0 | 5.12 s | 0.13787 mm | 57.76 |

Work uses synthetic model units. Residents within a shared world are not
independent world replicates. All conditions remained supported for nearly the
entire trial. This is a substantial improvement over the earlier V4 children
that immediately toppled, but does not establish purposeful locomotion.
Private context reached the CNS; its additional physical effect was tiny and
work increased. Private checkpoints restored byte-exactly. Memory occupancy
is not evidence of a useful learned association. **No trained child was promoted.**

Authoritative bulk artifacts remain at:

```
hbox:/tank/chreatures/runs/fly-learning-v5/training/chronological-a876bc8-r4
hbox:/tank/chreatures/runs/fly-learning-v5/assessment/full-three-arm-2704d2b-1199030f
```

| Artifact | SHA-256 |
| --- | --- |
| Training result | `cb4dec9eae4729c75e95b46ffdaa8f9531ee33534df71b565b865effca64ee15` |
| Trained CNS service | `1199030f07cc74404a8a6fb8759cee542db66b136094f618808145e52fbcff5d` |
| Private resident file | `3f28e38a0b74d0bcd5c055377e1ff2ebbd319be03afefbc405f3719c7392f834` |
| Complete physical assessment | `36cd13b367d920d92830070bfffa3aa35f64ebf6a8aa3fb308e7cbbd233fa912` |
| Physical diagnosis | `31fe1bcd0120f01b5571dc1835b670e6a0d1ea65c026bdcff9305d795cf783fa` |

Compact local copies and exact deferred export commands are in
`~/paperbin/chreatures/integration/fly-v5/trained-1199030f/artifact-chain.json`.
The complete assessment relay and separate lane handoff are in
`~/paperbin/chreatures/integration/cns-v5-trained-assessment-2704d2b/`.
Trained Torch/WebGPU parity and a trained browser pack were **not started**.
Initialized-service parity cannot stand in for this unexecuted check.

## Native ecology and GAM completed

All 48 fixed genotype–layout worlds and four GAM-proposed worlds on an untouched
fourth layout completed 32 model seconds. This experiment used neutral MOTOR92
controls and did not run a CNS. Responses ranged from 16 to 52 actual physical
constructions, with 1.28–4.16 synthetic construction material units.

Five varying inputs remain in the fitted models: four inherited genes and
measured preconstruction permeability. Two constant environmental columns were
removed after the initial additive fit rejected their degenerate design; the
original fit is preserved. Joint models were selected for construction, material,
resource change and aperture change; the additive model was selected for final
absolute permeability. Selection used whole-genotype leave-out error.

Interpretation must retain these findings:

- Final aperture mostly persists from the initial physical arrangement. On the
  untouched layout, persistence RMSE was 0.000875 versus the conditional GAM's
  0.003904. The model did not beat this simple reference.
- Joint construction/material/resource genotype-cross-validation errors were
  slightly worse than the training-mean reference. Withheld-genotype errors
  improved modestly; those are different comparisons.
- Material cost is exactly 0.08 per construction here. Count and material are
  dependent response axes, not separate ecological achievements.
- The initially reported roughly −0.75 aperture change used an unsampled
  all-open tick-zero placeholder. Correct preconstruction measurement is tick
  100, before growth begins at 200; changes are roughly ±0.0003 in the fitting
  worlds. The correction is authenticated, not silently substituted.
- The first 48-world launch failed at topology recompilation because its
  temporary fixture lost the mesh asset directory. Every failure remains
  preserved separately; the corrected 52-world campaign had no failures.

The complete directory is
`~/paperbin/chreatures/integration/fly-ecology-atlas/committed-response-seed20260930/campaign-v3-r2/`.
It contains `plan.json`, `results/`, `fit-r2/`, `confirmation/`,
`analysis-correction.receipt.json`, `secondary-diagnostics.json`, `receipt.json`
and `artifact-manifest.json`. The seal is
`dcfc4c1c9099ba222858f1a411bda58603f278968694e8f23c68b5150619428e`.
The atlas exporter and native Weave artifacts are in
`~/paperbin/chreatures/integration/native-growth-atlas-final/`.

## Performance and process state

Random paging of the 403 MB final latent replay slowed private training to about
53 seconds/update. A read-only sequential warm restored 3.44 seconds/update
without changing training state. Commit `c907d02` prevents that file-access
pattern in future runs: a preflighted 512 MiB budget, one managed read-only copy
per latent replay, and no replay allocation for goal-free fits.

One real B2 complete-path profile under concurrent ecology load measured
159.7 ms/tick: physical advancement 90.3 ms, private resident 30.9 ms, full CNS
25.5 ms, observer 7.8 ms and senses 5.0 ms. A checked in-place force-clear
experiment showed no benefit and was reverted. Do not claim that allocation
removal established a speedup. A useful next profile would distinguish native
collision/constraint solving from aerodynamics, preserving the current physics.

Training, assessment, ecology workers and temporary render servers have exited.
The hbox training GPU was idle at handoff. Persvati became unreachable during
artifact relay; its already-sealed initialized baseline remains separate.
The protected frozen worlds, unrelated jobs and owner Telegram bot were not
stopped or rewritten. Telegram is a polled mailbox while an agent works, not an
autonomous model-turn service after quiescence.

## Next frontier after the user resumes

Start from these sealed results; do not repeat the completed training, six-arm
assessment, or 52-world campaign just to re-establish their status.

1. Export the trained model through the existing packer using the exact command
   in `artifact-chain.json`, then check its own WebGPU numerical and joined-body
   behavior before considering a public selection change.
2. Give the stable trained body learning opportunities with a physically
   observable objective and multiple consequences. Keep teacher targets and
   curriculum labels outside policy inputs; an organism cannot infer a hidden
   teacher agenda. Memory and interaction should progress with these opportunities.
3. The private context currently has little bodily authority in the short assay.
   Study its CNS-derived effect over useful horizons and learned histories,
   rather than amplifying it merely until motion looks exciting.
4. Broaden construction costs, morphology and environment interactions so
   ecological responses are less tightly tied to one branch counter. Preserve
   full material accounting and compare the existing GAM with persistence and
   mean references before using its proposals as evidence of adaptation.

These are subsequent work directions, not background jobs or completed claims.
