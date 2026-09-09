# Committed ecology response experiment

This campaign asks which inherited colony programs produce measurably different
**committed** construction, resource use and physical route aperture in the
current native MuJoCo world. Clearance approval is an intermediate host decision;
it is never counted as a branch.

## Frozen design before execution

- Use 16 maximin Latin-hypercube genotypes across branch angle, lateral
  probability, phototropism and contact avoidance. Twelve are fitting groups and
  four are whole-genotype holdouts.
- Run every genotype in three fitting layouts for 32 simulated seconds at 100 Hz:
  48 independent worlds. A unit is one fresh native process with unique ecology
  and physics seeds. The two colonies inside that world share transport and
  resources, so they are subsamples and are aggregated into one response row.
- Reserve a fourth physical layout. It supplies no response to fitting or proposal
  selection. After fitting, execute at most four diversity proposals there.
- Run at most four CPU workers only after root pins the native binary and four
  authenticated fixtures. Current scheduling preference is local M2; do not use
  hbox while the V5 fit/evaluation lane owns it. Preserve every failed unit and
  never retry an unknown mutation into the same output path.

The expected cost is about 25–40 minutes for the 48 fitting worlds plus roughly
5–10 minutes for four confirmations, based on the earlier native 32-second world
distribution. Root must approve the sealed plan and allocation before launch.

## Authoritative observations

Each sparse research observation records the full current committed structure set
with `owner_id`, physical binding, position/orientation, dimensions and material.
The runner requires every binding to exist among current native entity IDs. It
also records measured route aperture vectors and the new postcommit host counters:
committed constructions/births, separate construction/birth material, capacity and
resource rejection, aborts, and the last verified commit. Cumulative construction
material must cover material retained in current physical structures.

The world responses are committed construction count, construction material,
colony resource change, final area-weighted route permeability and permeability
change. Canopy extent, photon capture, clearance outcomes and elemental residuals
remain diagnostics. A completed response is rejected if postcommit counters,
native ecology structures or physical entity bindings disagree.

The earlier 72-world aperture campaign remains valid evidence for its frozen
route observations, but it predates both authoritative committed-growth counters
and immediate postcommit route invalidation. Its rows are therefore not pooled
into this joint response fit, and its clearance counts are never retrofitted into
branch counts.

## Native GAM and the next experiments

The installed Rust `gamfit` engine fits joint seven-variable Duchon surfaces and
additive alternatives for each response. Model choice uses whole-genotype
leave-out error; whole-layout leave-out error measures environmental transfer.
The four genotype holdouts are reported but cannot tune models or proposals.

The fitter evaluates a fixed candidate set through all three measured layout
signatures and selects up to four points by maximin distance in both predicted
construction/resource/aperture response space and gene space. It has no scalar
winner and promotes nothing. The untouched-layout run tests whether the proposed
set actually spans distinct responses.

Layout leave-out error and measured aperture signatures also rank the next
environmental experiments. Those outputs direct the habitat composer toward new
obstruction/light layouts where the response law transfers poorly. A proposed
signature becomes an experiment only after the native host measures the new
layout's initial aperture; the GAM never invents route openness or airflow.
