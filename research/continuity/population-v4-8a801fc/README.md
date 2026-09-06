# Frozen 8a801fc coupled continuation check

This is an isolated research fork of the eight-resident M2 world served on port
8782. The acquisition cloned a coherent world/neural pair at tick 10020 without
pausing, advancing, or writing to the authoritative deployment. The source world
UUID is retained inside the runtime because it addresses the resident neural
identities; the two executions have separate receipt-level research fork IDs and
make no claim to be the authoritative life.

`run_continuation_check.py` used frozen source revision `8a801fc`, frozen Metal
binary SHA-256 `e886a59e7797d11e68e6846449bb39f74cada5d775d0bfc45be783a944c73a6b`,
graph artifact SHA-256 `4a2df4b62208cb4021c6abe1e33c02f008f13d8964c90eebe8255a68a9b88df0`,
and the pinned cognitive/world native modules. It ran two sequential services on
unused loopback port 19782, restored all eight residents and their private neural,
cognitive, controller, RNG, body, chemical, ecological, acoustic, visitor,
history, and pending-boundary state, then advanced each fork by 32 physical ticks.

Both forks reached tick 10052. Their complete world checkpoint states were exact
with SHA-256 `ef8eeaf600fc9621f46186d1b98ff486f35e0e8d849182bc1e94fe8bd7ef4caa`.
Their complete neural snapshots were byte-identical with SHA-256
`716460bb004419c110c127a342583423d576b4bbd46a9a95aac3aed5423643ee`.
No runtime state fields were excluded. Only receipt-local fork IDs, service
incarnations, and elapsed wall time sit outside the comparison.

The evidence establishes deterministic continuation for this checkpoint on this
Apple M2 Max and frozen runtime. It does not establish cross-engine,
cross-machine, or cross-checkpoint equivalence. Exact inputs, per-run identities,
commands, hashes, durations, comparison details, and limitations are retained in
`receipt.json`.
