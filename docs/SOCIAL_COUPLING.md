# Physical social coupling probe

`scripts/probe_social_coupling.py` tests whether one resident can causally
change another resident's ordinary local sensor stream and whether that change
reaches the canonical full MaleCNS circuit. It is a mechanism probe, not
evidence that a learned communication convention already exists.

The probe forks each active/silent pair from the same physical snapshot. The
listener receives no action. Its observation is encoded through the normal 351
ports and contains no resident ID, name, world position, object kind or scenario
label. IDs exist only outside the observation to address private physics and
neural state.

## Physical contrasts

The tone conditions place two residents 1.5 m apart. The active source emits
`signal_mid=1`; the silent source receives the otherwise identical action. A
static barrier is either on or away from their direct ray.

There are two deliberately distinct sound mechanisms:

- A legacy resident signal is an ephemeral three-tone event. Without an
  attached `Acoustics` engine it has distance attenuation but deliberately does
  not ray-test occlusion. Clear and blocked conditions therefore match.
- Attaching an empty physical `Acoustics` engine does not invent an emitter or
  energy. It activates the world's acoustic visibility path for resident
  signals. The barrier then applies the configured physical transmission of
  `0.16`.

The common-object condition uses the ordinary 3-D crawler traction and contact
mechanics. One resident walks into a movable resonant box for 60 steps while the
listener remains passive. The paired idle world starts from the same snapshot.
This exposes both motion through retinal proximity and mechanically excited
resonance through sound. The action and scenario facts never enter the listener
vector.

## Executed result

The canonical run used a private CPU `RemoteBrain` on hbox, loaded from
`/tank/chreatures/data/malecns/derived` and
`/tank/chreatures/data/ports/retinal-v1-maps.npz`. It instantiated fresh neural
state and did not call, add residents to, advance, or remove residents from any
live service. The graph was MaleCNS v1.0 with 165,122 neurons, 25,563,197 edges,
351 input ports and 384 readouts.

| Contrast | Listener input effect | Full-graph readout effect |
| --- | --- | --- |
| Legacy signal, clear | `sound/1` +0.2322965 | 3 rows; max 5.58e-7 |
| Legacy signal, blocked | same as clear | 3 rows; max 5.58e-7 |
| Physical visibility, clear | `sound/1` +0.2322965 | 3 rows; max 5.58e-7 |
| Physical visibility, blocked | `sound/1` +0.0371674 | 3 rows; max 9.01e-8 |
| Shared object motion | sound +0.0868297; two retinal proximity changes near +0.0028 | 160 rows; max 2.06e-5 |

The barrier reduced the resident tone to exactly 16% of the clear-path value.
The common box moved 8.921 mm. Its paired listener contrast changed only three
of 351 inputs: the resonator's tone and two central retinal proximity bins.

The direct tone reaches three small aggregate readouts after the circuit's two
recurrent substeps: navigation/other, mushroom-body/other and visual/other. This
is a real but weak path. The 384-row atlas has no dedicated auditory domain, so
the result should not be described as an auditory decoder. The shared visual
motion reaches 160 readouts, headed by named C2, L1, L3, L5, Tm4, L2 and Mi4
visual strata. This establishes usable causal coupling while identifying the
weak immediate tone observability that later sensory-motif work should address.

## Run

World-only checks work in the compact local environment:

```bash
.venv/bin/python scripts/probe_social_coupling.py --skip-neural
```

The full canonical probe belongs on a private hbox process:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/probe_social_coupling.py \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --output /tank/chreatures/probes/social-coupling.json
```

The script fails if tone routing leaves `sound/1`, physical occlusion stops
attenuating, source motion fails to move the shared object, the listener loses
the retinal or resonant consequence, or any physical contrast becomes
disconnected from all full-graph readouts.
