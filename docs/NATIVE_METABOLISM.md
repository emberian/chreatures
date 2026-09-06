# Native conservative metabolism

`MetabolicCohort` in `native/world-kernels` runs one shared, anonymous reaction
program over private compartment state in float64. It stores `N × K` resource
pools, `N` ATP values and capacities, and immutable-at-runtime `N × R` genotype
enzyme activities. Chemical species are columns; the kernel assigns them no
world meaning.

The constructor accepts these contiguous NumPy arrays:

- `stoichiometry[R,K]`: negative reactants and positive products.
- `elemental_composition[K,E]` and `chemical_energy[K]`.
- `atp_cost[R]`, `atp_yield[R]`, and `photon_cost[R]`.
- `half_saturation[R,K]` and `base_rates[R]`.
- `enzyme_activity[N,R]`, `pools[N,K]`, `atp[N]`, and `atp_capacity[N]`.
- `bulk_pool[K]` and scalar `bulk_atp` for conservative external exchange.

Construction rejects nonfinite or negative quantities, malformed dimensions,
ATP above capacity, reactions without a substrate, nonpositive half saturation
for a consumed substrate, elemental imbalance, and negative reaction heat. The
per-extent checks are

```text
delta_chemical_energy = sum(stoichiometry[k] * chemical_energy[k])
reaction_heat = ATP_cost + photon_cost - ATP_yield - delta_chemical_energy >= 0
```

`step(dt, light_energy_budget[N], mechanical_cost[N])` computes every proposed
extent from the beginning-of-step state. Its Monod mass-action factor is the
product of `pool / (half_saturation + pool)` over consumed resources. For each
compartment row and resource, the kernel divides availability by aggregate proposed
consumption. Each reaction receives the minimum factor only for resources it
consumes, plus the row ATP factor when it costs ATP and photon factor when it
costs photons. It then applies stoichiometry once. This proportional limiter
has no reaction iteration priority, and shortage of an unrelated resource does
not restrict a reaction.

ATP yield and cost are applied after extents. ATP above the explicit per-row
capacity becomes `atp_overflow_heat`. Mechanical cost is removed afterward;
the return reports paid and unmet work separately, and paid work is excluded
from heat. The returned NumPy ledger contains:

```text
extent[N,R], limiter[N,R]
photon_used[N]
reaction_atp_cost[N], reaction_atp_yield[N]
chemical_energy_delta[N]
reaction_heat[N], atp_overflow_heat[N], total_heat[N]
mechanical_paid[N], mechanical_unmet[N]
elemental_residual[N,E], energy_residual[N]
```

The energy residual checks photon input against chemical-energy change, ATP
change, reaction heat, overflow heat, and exported mechanical work. The object
also retains a six-column cumulative ledger and elapsed time.

`transfer(donor, receiver, resources[K], atp=0)` moves exact quantities. A
Python `None` endpoint denotes the owned bulk pool; one endpoint must differ
from the other. The operation validates all availability and receiver ATP
capacity before mutating anything. `split(parent, child, fraction)` requires an
empty child and subtracts the exact transferred resource and ATP values from
the parent. Enzyme inheritance and mutation remain outside this primitive.

`pay_work(row, amount)` performs an all-or-none ATP debit from one compartment
and records the same amount in the cumulative exported-work column. It rejects
invalid rows, nonfinite or negative amounts, and any amount above the exact ATP
available. Exported work is not added to heat. An assembly transaction can
snapshot metabolism, allocate tissue, pay work, attempt geometry commit, and
restore the snapshot if the later commit fails.

`program_sha256` hashes the complete shared program with named arrays,
dimensions, and Rust-order float64 bytes. `snapshot()` returns a versioned byte
record containing that hash, dimensions, pools, ATP, ATP capacities, enzyme
activities, bulk state, cumulative ledger, and time. `restore(bytes)` validates
the whole record into temporary arrays before replacing live state, and rejects
a different program or dimensions.

Build an isolated extension with the repository helper:

```bash
.venv/bin/python native/world-kernels/build_extension.py --output-dir /tmp/chreatures-world-kernels
```

World integration should map the checked common-chemistry JSON columns to this
numeric API and record `program_sha256`. Rows can represent body, gut, or
allocated structure compartments with different enzyme activity, preventing a
digestion reaction from consuming tissue held in another compartment.
Structural growth must reserve exact
resource columns and ATP through `transfer`; a scalar biomass request requires
a versioned composition mapping rather than creation of untracked mass.
