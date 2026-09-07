# Genotype-by-environment transfer atlas

The ecological transfer atlas is an analyst-side descriptive model of 160
completed physical lives. Eighty principal-wave genotypes were each observed
once. The challenge wave added 80 transfers of ten selected genotypes across a
sparse ten-environment matrix. One complete 2,048-tick life is an observation
row. Eight residents shared each physical world and could interact, so life rows
are not generally statistically independent. Ticks, contacts, and actions within
a life are repeated measurements. The atlas makes no uncertainty-calibration
claim.

`scripts/fit_ecological_transfer_atlas.py` authenticates both evaluation files
against the checked-in campaign receipts and authenticates the post-ingest
search file. It resolves exact genome and environment records, rejects partial
lives, then fits `gamfit` 0.1.259 from SauersML/gam commit
`7c7eca8ac4826de95c8e743a20294bee132a9bcc`. Saved `.gam` files are native
library artifacts rather than Python regression substitutes.

The pre-run models use inherited morphology, metabolic capacity and allocation
parameters with five generated-environment descriptors. They predict a vector:
energy change, contact ticks, mechanical work rate, and material allocation
rate. Additive smooths are compared with small tensor interactions for reserve
capacity by resource density and leg length by terrain relief. Variant selection
uses whole validation genotype and environment groups. Separate whole groups
are held out for final reporting. Environment grouping also keeps every shared
world cohort on one side of a split. The artifact retains both fits even when an
interaction or an entire response fails to beat the mean baseline.

A separate diagnostic native GAM relates realized action means and physiology
to energy change. This model is useful for interpreting completed lives. It is
prohibited from ranking an unrun pair because those measurements do not exist
before execution.

Use `scripts/rank_ecological_transfers.py` with a request of the form:

```json
{
  "format": "chreatures-gxe-ranking-request-v1",
  "pairs": [
    {"candidate_sha256": "...", "environment_sha256": "..."}
  ]
}
```

The command loads the exact candidate and environment records from an
authenticated search state and invokes the saved native models. Every result
contains support diagnostics and a response vector with additive and interaction
estimates. It never emits a combined fitness. New genome or environment
mechanisms, out-of-range values, and distant feature vectors are returned as
unranked. In particular, new v7 enzyme or regional-flow mechanisms require new
physical observations before this atlas may evaluate them.

The rank command can also write the existing native population-core
`chreatures-population-challenge-scores-v1` contract. This requires an explicit
criterion and a separate policy receipt. Supported criteria either favor one
named response or favor a large additive-versus-interaction gap as an
information-seeking experiment. Scores are bounded to `[-1, 1]`, apply only to
supported pairs, and retain the full response vector in the sibling ranking
output. They are campaign scheduling hints, not a hidden aggregate fitness.

Environment descriptors, geometry-derived generation records, archive hashes,
and model estimates are not resident sensations. The atlas can propose diverse
physical experiments for the campaign analyst. It cannot establish a causal
effect, ecological fitness, feeding ability, survival, or a resident preference.

## Executed v1 result

The native fit used 75 training lives, 37 validation lives, and 48 final
held-out lives. Energy change was the only response to beat the train-mean
baseline on both grouped partitions, and the gains were small: validation RMSE
was 0.01482 versus 0.01509; held-out RMSE was 0.01702 versus 0.01722. All 160
energy changes were negative. The result therefore supports only a weak
descriptive energy-retention ranking inside the observed v2 mechanism domain.

Contact and mechanical-work models improved on validation but failed the final
group holdout. Allocation failed both comparisons. No tensor interaction earned
promotion. The retained interaction fits instead define an explicit disagreement
vector for choosing informative experiments. The realized-action/physiology
energy diagnostic reached validation RMSE 0.01118 and held-out RMSE 0.01581,
which is useful after a life has run but cannot be used before it runs.

The exhaustive current-v2 scan covered 650 previously unobserved pairs. Of
these, 323 passed the schema, range, and distance support checks. The companion
proposal artifact retains the 32 largest supported interaction disagreements;
its criterion is information collection, not predicted biological success.
