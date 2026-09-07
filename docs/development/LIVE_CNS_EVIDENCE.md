# Live CNS evidence graph

The live browser wave has one portable evidence graph built by the repository's
existing native Universal Weave adapter. It links three executed research
chains without treating a fit, export, or physical motion as proof of general
competence:

- A complete 27-setting, full-MaleCNS dynamics sweep feeds native GAM fits, a
  leave-one-setting-out prediction, and a separate execution of the proposed
  setting. The executed skill was close to the prediction but did not exceed
  the best measured setting. Recovery-memory prediction failed its baseline
  comparison and remains a diagnostic only.
- A 192-update temporal CNS research fit feeds a canonical intact/zero-edge
  audit, the 255 MB service export, a complete browser pack, a Dawn WebGPU
  numeric probe, and a joined CNS + Rust resident + MuJoCo replay. Its
  procedural frequency split reported positive trained temporal-probe skill.
  Removing graph edges reduced those six trained probe skills to approximately
  zero; this establishes dependence of those probe outputs on the graph, not a
  biological or behavioral result. A separate startup loss in the same audit
  exercised newly initialized broad sensory heads and reported poor future
  optic/body scores. Those values establish finite gradients and execution for
  untrained heads; they do not show that the trained temporal probe failed its
  held-out split. The graph also retains the failed first whole-life load and
  the executed repaired replay.
- Two actual full-CNS physical teacher episodes feed a 128-update controller
  fit, a native reference comparison, a trained browser pack, and matched
  fresh-life physical rollouts. The trained arm produced more varied actions.
  Displacement, path length, turning, and stopping changed differently across
  the three residents, so the recorded result is mixed. The teacher used
  evaluator-only geometry offline; neither controller received geometry or
  object labels as policy input. The graph retains the failed first held-out
  collection attempt. It also retains the first two-arm rollout receipt, whose
  envelope identity check reported a mismatch because only the source-revision
  header differed. Its decoded GPU bytes and canonical host numeric state did
  match. The refined successor records that distinction explicitly; it did not
  repair or replace different numeric initial states.
- A matched physical-screen comparison starts the current V2 runtime twice from
  byte-identical physical, CNS, and resident state. One arm decodes the 30-second
  film onto the emitting screen; the other keeps that physical screen black.
  Resident 0 supplies the 1,771-site retinal capture and neural-rate capture;
  action and physical summaries cover the three-resident batch. At an absolute
  paired-difference threshold of `1e-6`, 140,200 of 149,782 nonafferent neurons
  changed at least once across the 600 ticks. Actions differed on 567 ticks and
  the physical root trajectories diverged. These are stream-specific paired
  measurements, not evidence of a simultaneous response, recovered physiology,
  stimulus understanding, or learned motor competence. The resident controller
  in this run was initialized and untrained.

The public files are
[`live-cns-evidence.json`](../../site/assets/live-cns-evidence.json), the generic
portable projection, and
[`live-cns-evidence.weave.json`](../../site/assets/live-cns-evidence.weave.json),
the exact native serialization. The screen comparison also has a sanitized
[`compact receipt`](../../site/assets/live-cns-screen-response.json), its exact
[`compressed traces`](../../site/assets/live-cns-screen-response.traces.json.gz),
and a four-signal
[`static trace figure`](../../site/assets/live-cns-screen-response.svg). The
compact receipt preserves the SHA-256 of the original receipt while omitting
machine paths and decoder command lines. Every source blob is represented by
local SHA-256 and byte length; other bulk artifacts remain outside Git.

Rebuild from the frozen paperbin inputs with:

```sh
python3 integrations/export_live_cns_weave.py
```

The exporter verifies the receipt formats, source-to-fit links, all release
members, teacher episode hashes, trained artifacts, and matched-rollout
identities. It also checks the paired-run source and model identities, exact
initial-state hashes, 600-sample trace dimensions, resident scopes, and the
original receipt and trace hashes before invoking `cargo run --locked --release`
for the pinned Universal Weave revision. The Rust adapter validates the DAG,
serializes it, reloads it, and requires exact equality. Existing outputs are not
overwritten unless `--replace` is supplied. Later richer training data should
produce a new dated evidence chain rather than reinterpret these first results.
