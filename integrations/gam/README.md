# Native ecological transfer GAM

`transfer_atlas.py` authenticates completed population campaign files, fits the
pinned SauersML/gam native extension, and queries saved native `.gam` models.
The checked-in `artifacts/ecological_transfer_atlas_v1` run contains 160 actual
physical-life rows, eight fitted pre-run models, one realized-action diagnostic
model, a full supported/unranked scan of unobserved v2 pairs, and a native
population-core proposal-score artifact.

The proposal file uses the existing population-core score format and is scoped
to the frozen v2 search. The current v3 search correctly rejects that old search
state and these scores cannot be applied to new regulation genotypes. See
`docs/development/GENOTYPE_ENVIRONMENT_ATLAS.md` for interpretation and commands.
