# Native GAM response surface for CNS dynamics V2

`gam_fit.py` fits an analyst-only response surface after the actual full-graph
crossed sweep. It requires one flat record for every point in the frozen 3×3×3
design:

- `gain`: `0.7`, `1.05`, `1.4`
- `tau`: `0.04`, `0.08`, `0.16` seconds
- `adaptation_gain`: `0.05`, `0.15`, `0.35`

The JSON root is `{ "format": "...", "records": [...] }`. Every record contains
finite numeric `gain`, `tau`, `adaptation_gain`, `temporal_probe_mse`, `skill`,
`persistence_mse`, `recovery_memory`, and `saturation`; `run_id` is optional.
`skill` is the sweep's mean of the three channelwise
`1 - temporal_probe_mse / persistence_mse` values. It is not reconstructed from
the two aggregate MSE columns because a ratio of means need not equal a mean of
ratios. `recovery_memory` is residual neural RMS one second after neutral input,
normalized by the last driven RMS. `saturation` is the sampled fraction with
`abs(r-r0) > .19`.

Run the fit in the existing isolated native GAM environment:

```sh
integrations/.venv/bin/python research/dynamics_v2/gam_fit.py \
  --input /path/to/crossed_sweep.json \
  --output-dir /path/to/fresh/gam-response
```

The laptop project `.venv` does not contain GAM. The pinned installation is
`gamfit==0.1.259`, built from SauersML/gam commit
`7c7eca8ac4826de95c8e743a20294bee132a9bcc`, in `integrations/.venv`. If that
environment must be rebuilt, use the repository's
`integrations/gamfit-requirements.txt`; do not install into the project or system
environment.

The primary model for each varying response is a native three-axis tensor smooth,
`te(gain_scaled,tau_scaled,adaptation_gain_scaled,k=15)`. Scaling is fixed from
the declared sweep bounds rather than estimated per fold. This surface contains
joint response structure without pretending 27 settings can support three
separate large pairwise smooths. Skill also gets a smaller additive native GAM
benchmark. Every reported predictive score leaves one complete parameter
configuration out, and the report states whether the joint skill surface beats
that additive benchmark. A response that is exactly constant is recorded as
non-identifiable instead of adding fake jitter to force a fit.

The query searches a dense grid only inside the measured parameter box. It
maximizes predicted skill subject to predicted saturation at or below `0.01` by
default. Recovery memory is reported beside the selected point but is absent
from the objective: a persistent residual is not task success. As a general
rule, a newly selected interpolation remains proposed until the full graph runs
at that exact setting. These raw research outcomes never enter a resident
controller, memory, predictor, or action score.

## Executed crossed-sweep fit

The completed source receipt is
`/Users/ember/paperbin/chreatures/integration/dynamics-v2-crossed-20260907/receipt.json`;
the native models and fit report are in its `gam/` child directory. The native
tensor skill surface scored leave-one-configuration-out RMSE `0.0127603`, versus
`0.0162789` for the additive native GAM and `0.0642475` for the fold training-mean
baseline. The joint surface therefore improves held-out interpolation in this
bounded design. This supported selecting a confirmatory query; it does not
identify a biological interaction.

All 27 saturation fractions were exactly zero, so no saturation response surface
is identifiable. Recovery-memory GAM holdout RMSE was `0.0601823`, worse than its
training-mean baseline RMSE `0.0412859`. Its predictions are retained as an
unreliable diagnostic and cannot support parameter choice.

The best crossed-design point was gain `1.4`, tau `0.04`, adaptation gain `0.05`,
with skill `0.821376`, temporal-probe MSE `0.175321`, and recovery memory
`0.142527`. The bounded grid then selected the unmeasured confirmation at gain
`1.4`, tau `0.04`, adaptation gain `0.0875`. Its predicted skill was `0.817397`;
the empirical leave-one-configuration-out error scale was RMSE `0.0127603`, giving
a deliberately plain two-RMSE heuristic range of `[0.791877, 0.842918]`. This was
not a calibrated confidence interval and did not claim improvement over the
crossed-design optimum.

That exact confirmation has now executed on the full graph with batch `12` for
`128` ticks. The receipt is
`/Users/ember/paperbin/chreatures/integration/dynamics-v2-crossed-20260907/gam-confirm.receipt.json`.
Observed skill was `0.815785`, only `0.001612` below the GAM prediction and well
inside the leave-one-configuration-out RMSE. Temporal-probe MSE was `0.180479`
against persistence MSE `0.840758`; recovery memory was `0.121848`, and saturation
remained zero. This validates the selected interpolation at one new parameter
configuration. It does not establish superiority to the crossed-design optimum
or turn the response surface into a physiological law.

The archived matched v1 control used the same twelve streams and probe and scored
skill `0.591408` at temporal-probe MSE `0.384153` against persistence MSE
`0.840758`. The best measured V2 setting's `0.821376` skill is therefore a useful
controlled improvement under the frozen audit, while remaining a response
calibration result rather than behaving-organism evidence.
