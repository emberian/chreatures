# Grandchild developmental GAM atlas

This atlas uses all 491,520 recorded resident transitions from the completed
four-episode grandchild run at source revision `9e110419`. It is analyst-only and
does not alter the resident, the deployed consequence bank, or future births.

The temporal split asks a harder question than random held-out rows. Episodes 0–2
and physical worlds 0–2 train the GAMs (276,480 transitions). World 3 from those
episodes validates physical-world transfer (92,160 transitions). The complete
fourth episode is unseen until the final evaluation (122,880 transitions). This is
one inherited lineage and seed, so episode transfer is evidence about this run,
not population-level development.

Pinned native `gamfit==0.1.259` minted two actual models. Physical effort is strongly
predictable from current physiology, neural/recurrent summaries, action history and
executed action: RMSE is `0.01878` on validation worlds and `0.02009` on the held-out
episode, against training-mean baselines `0.14827` and `0.13232`. Its two numerical
25-by-25 surfaces show the fitted motor-magnitude/fatigue and thrust/speed response
relationships.

The inherited rich-body consequence-law residual is moderately predictable. RMSE
is `0.30402` versus `0.35205` on validation worlds and `0.34299` versus `0.34510`
on the held-out episode. Direct episode summaries show mean standardized residual
falling from `0.5758` in episode 0 to `0.4660` and `0.4228`, then rising to `0.4805`
in the held-out episode. This pattern motivates a future resident-private residual
mechanism with explicit forgetting or phase context; it does not show monotonic
learning. Only the 271,366 transitions inside every reference-law domain enter this
fit.

The planned goal-progress model failed native REML certification after 200 outer
iterations: the projected gradient remained `0.6186` against a `0.00412` bound and
the Hessian was not positive semidefinite. No model or surface was minted. This
negative result is preserved in the report rather than replaced with an illustrative
curve. The run's independent behavior audit also reports that goal shaping was
76.1% of absolute physical reward on average, so goal-distance effects require a
matched objective/control experiment before becoming a mechanism.

The compact [atlas report](../integrations/gam_mechanisms/artifacts/grandchild_developmental_atlas_v1/developmental_atlas.json)
contains metrics, direct native prediction surfaces, feature normalization, packet
hashes and the complete-run receipt. [Provenance](../integrations/gam_mechanisms/artifacts/grandchild_developmental_atlas_v1/provenance.json)
records lifecycle ancestry and interpretation limits. Full `.gam` files live on
hbox bulk storage at
`/tank/chreatures/runs/analysis/grandchild-developmental-gam-atlas-v1/models`.
