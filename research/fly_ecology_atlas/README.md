# Physical developmental response atlas

This research lane runs the current MuJoCo fly habitat and native ecological
growth at 100 Hz for eight simulated seconds per world. Two flies remain present
under explicitly diagnostic zero MOTOR92 commands. No CNS is evaluated or replaced
by this atlas. Every colony branch uses the production local proposal, measured
clearance, finite biomass/ATP allocation and atomic physical commit.

The frozen design has 24 deterministic maximin Latin-hypercube inherited-gene
settings and three physical illumination/clearance layouts. The last six entire
gene settings are held out. All four varied genes remain inside the native declared
range: branch angle, lateral probability, phototropism and contact avoidance.
Other chemistry, developmental costs and directional persistence stay fixed.
Layout changes move the actual screen or an existing physical obstacle; light and
clearance are measured by the host, never injected as invented sensor values.

`prepare.py` freezes host/Wasm bytes and scene variants. `campaign.py --pilot`
executes the first actual world once, retaining it as a campaign member. Use its
complete wall time to choose independent workers. The approved campaign starts
with four workers and has up to 120 minutes after the photon/metabolic coupling
and clearance scratch-model reuse are frozen. The campaign preserves failed runs and never retries an unknown
physical mutation. A budget stops new launches, while already-running bounded
worlds finish coherently.

`fit.py` requires all requested outcome receipts and fits the installed native
`gamfit` library. It compares a joint four-dimensional Duchon smooth (12 centers) and additive
models with mean baselines,
reports whole-setting and whole-layout leave-out errors, and preserves constant
or rejected targets. Native artifacts are serialized, reloaded, and their
predictions compared. The six heldout settings never tune model selection or
proposals. Selection uses training leave-setting-out evidence and asks for distinct
canopy/route-permeability outcomes, not one global winner.

Each diversity proposal runs again through `run.mjs` on the fourth, previously
unused physical layout. `confirm.py` records predicted and actual outcomes without
converting mere execution into a successful prediction or promoted genotype.

All generated worlds, raw receipts, native models and checkpoints belong under
`~/paperbin/chreatures/integration/fly-ecology-atlas/`, not large repository assets.
Run Python orchestration as modules from the repository root; native GAM fitting
uses `integrations/.venv/bin/python`.

The initial no-photon run under `seed20260910` is excluded from fitting and retained
only as a throughput record. Final worlds use a distinct plan and exact frozen host
and native module hashes. The historical host did not automatically sample route clearance. Its supplied
openness stayed at the initialization default 1.0. The reported route conductance
was derived from that default, declared route geometry and pool diffusivity; it
was not a physical obstruction measurement. See `scope-amendment.json`.
Captured photon energy and actual construction expenditure accompany the geometry.

The response surface predicts a gene setting averaged across the three training
layouts. Whole-layout leave-out tests environmental transfer; the unused fourth
layout is an actual physical confirmation, not an extra fitting row. The historical default
route inputs remained constant and received no fitted surface.

Reproduce in a new output directory:

```sh
integrations/.venv/bin/python -m research.fly_ecology_atlas.prepare --fixture native/browser-world/fixtures/fly-ecology/world.json --runtime native/browser-world/runtime.mjs --output /absolute/new-campaign
integrations/.venv/bin/python -m research.fly_ecology_atlas.campaign --plan /absolute/new-campaign/plan.json --results /absolute/new-campaign/results --workers 4 --budget-minutes 120
integrations/.venv/bin/python -m research.fly_ecology_atlas.fit --plan /absolute/new-campaign/plan.json --results /absolute/new-campaign/results --output /absolute/new-campaign/fit
```

For each generated `fit/confirmation-*.json`, invoke the frozen `run.mjs` with
`--proposal` pointing to that file, `--setting` matching its `setting_id`, and
`--layout confirmation-layout`. Write results as
`confirmation-results/<setting_id>--confirmation-layout.json`, then run
`research.fly_ecology_atlas.confirm` with `--fit`, `--results`, `--plan`, and an
exclusive `--output` receipt path. Confirmations use the same finite native engine;
the GAM only proposes inherited parameters.

The initial tensor analysis attempt was stopped before its first saved model
after observed multi-minute execution and roughly 2.1 GiB RSS. The final joint
model uses the same native library’s bounded multivariate Duchon basis. Both
analysis recipes and the stopped attempt remain in the campaign directory; the
physical design and heldout split were never changed. `seal.py` binds the final
physical campaign, native models, host receipt and confirmation into one compact
receipt, including mean-baseline errors for constant targets.

Executed receipt: `receipt.json` (SHA-256
`1b351c12a1811de8740f2905a3032f6ede7538cd45ae601dad0c8fb58de068db`).
The corrected campaign completed all 72 eight-second worlds in 1734.09 wall
seconds using four workers. Eight native joint/additive models were saved and
reloaded. Training leave-setting-out selection chose the joint height model and
additive span/branch/blocked-rate models. Selected height heldout RMSE was
0.621 mm versus the mean baseline’s 0.724 mm; selected span RMSE was 0.576 mm
versus 0.568 mm. The unsampled route-openness default stayed exactly 1.0 and received no GAM fit.

Three fresh settings physically completed on the unused layout, producing 7, 7
and 5 branches with heights 3.064, 5.238 and 3.749 mm. They paid construction ATP
costs 0.056, 0.056 and 0.040. The intended height ranking did not hold: height
errors were 0.898, 1.512 and 0.779 mm. These are viable distinct physical
structures with imperfect surrogate predictions, not promoted genotypes or
evidence of physically measured or controllable route permeability. All raw receipts, eight native
models and three final world checkpoints remain in the private campaign
directory stated in the artifact paths.
