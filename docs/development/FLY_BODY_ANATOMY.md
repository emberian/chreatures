# MaleCNS to fly-body anatomy

This atlas replaces the synthetic twelve-hinge motor interpretation with an
actual articulated *Drosophila* body and the complete annotated MaleCNS motor
set. It is a build-time research contract for CNS V4. It does not change an
already-running V2 or V3 life.

The body is FlyGym/NeuroMechFly 2.1.0 at
`ca65a510c2afe6ac61c51df4f274c8d190c2f95f`, imported in author
yaw-pitch-roll order. It has 69 anatomical segments and 126 joint coordinates.
The geometry derives from micro-CT of one adult female fly, while MaleCNS is an
adult male from a different animal. Every CNS-to-body relationship is therefore
at least an inter-animal transfer. The body model supplies position servos and
synthetic adhesion; it does not supply anatomical muscle or tendon identities.

## Frozen physical and neural boundaries

The physical MJCF has 90 controls. The first 84 are position targets in this
fixed order: walking legs 42, head 3, pedicels 6, rostrum and haustellum 6,
abdomen 15, wings 6 and halteres 6. Controls 84:90 are left-front,
left-middle, left-hind, right-front, right-middle and right-hind adhesion.
Eyes, funiculi/aristae and distal tarsal axes remain passive, for 42 passive
joint coordinates. Exact machine IDs and physical ranges come from
`native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json`.

CNS MOTOR92 appends two native physiology controls to the physical M90:

| Outputs | Meaning | Neural support | Output range |
|---|---|---|---|
| 0:84 | Position target servos | Matching broad motor family and side where the body has a side | signed normalized `[-1,1]` |
| 84:90 | Synthetic adhesion, leg order | Same-leg motor family; explicitly engineered | `[0,1]` |
| 90 | Pharyngeal pump | `MN10`, `MN11D`, `MN11V`, `MN12D` | `[0,1]` |
| 91 | Salivary drive | `MN13` | `[0,1]` |

The physical target conversion maps outputs 0:84 through the author neutral
pose and servo limits. MOTOR92 is an effective servo and physiology interface,
not muscle recruitment. A target can move in either direction, so learned
position weights are signed. No joint direction is inferred from neuron side,
and no inverse “teacher muscle cohort” is fabricated from servo positions.

The decoder uses all 815 canonical MaleCNS motor rows. For each motor neuron,
`z = (rate - reference_rate) / rate_scale`; the fitted positive scale and
reference rate come from the actual training-world CNS calibration corpus. The
runtime computes `(weight * anatomical_mask) @ z + intercept` and reapplies the
mask on every forward pass. The anatomy artifact deliberately contains no
invented baseline rate, scale, activation time constant or decoder coefficient.

## Motor coverage and ambiguity

The MaleCNS motor partition in the canonical derived metadata is exact:

| Subclass | Rows | M92 mapping |
|---|---:|---|
| `fl` | 135 | Side-matched front-leg seven-axis bank and adhesion |
| `ml` | 116 | Side-matched middle-leg seven-axis bank and adhesion |
| `hl` | 130 | Side-matched hind-leg seven-axis bank and adhesion |
| `nm` | 44 | Three central head axes |
| `am` | 13 | Three pedicel axes on the annotated side |
| `pm` | 67 | 53 external/finer-unresolved rows to six mouthpart axes; 12 identified pump rows; 2 identified salivary rows |
| `ad` | 214 | Fifteen central abdominal axes |
| `wm` | 67 | Three wing axes on the annotated side |
| `hm` | 16 | Three haltere axes on the annotated side |
| `rm` | 7 | Unsupported because eye axes are passive |
| `xm` | 6 | Unsupported because the target identity is unresolved |

Thus 802 of 815 motor rows have at least one M92 structural edge. The seven
`rm` rows are PS349, GNG647 and GNG314 retina/eye motor annotations. The six
`xm` rows are MNxm01, MNxm02 and MNxm03 with DMetaN/PDMNa exits. Their mask
columns are zero. Making this explicit is preferable to assigning a body part
from side, exit nerve or a stable hash.

Broad family masks preserve control capacity for incompletely resolved head,
antenna, mouth and abdomen neurons. They do not claim that every member drives
every anatomical axis. Signed learned weights may specialize within the allowed
family and side.

The optional fine layer has 156 side-resolved descriptive cohorts covering 400
motor rows:

| Part | Cohorts | Rows | Evidence |
|---|---:|---:|---|
| Six legs | 102 | 328 | MANC/FANC target matching and serial homology |
| Wings | 46 | 62 | Published descriptive steering, power and indirect-flight types |
| Halteres | 8 | 10 | Published hDVM, hi1, hi2 and hiii2 types |

The layer excludes blank and systematic `MN...` labels. MANC authors matched T1
leg neurons to FANC and light microscopy, then transferred 198 T1 identities to
T2/T3 by serial homology. These are useful inter-animal anatomical constraints,
but are not direct measurements of MaleCNS muscle force, moment arm, recruitment
sign or servo-axis effect. The 156 cohorts remain diagnostics and possible
auxiliary losses rather than an exclusive runtime bottleneck.

## BODY807 afferent bank

The exact nonvisual sensory set is 11,798 rows. It includes peripheral CB/VNC
sensory classes and annotated sensory ascending/descending rows, including the
22 sensory-ascending leg chordotonal rows omitted by the old body selector. It
is disjoint from the 4,114 `ol_sensory` rows; 4,107 of those have optic anchors
in the separate retina interface and seven stay explicitly unsupported there.
Motor and body-afferent row sets are also disjoint.

BODY807 provides at least one structural edge to 10,609 of those rows; 1,189
rows remain explicit zero rows because their site, side or modality does not
support a defensible channel assignment. Of the 807 physical channels, 740 have
afferent support and 67 remain zero columns. Most zero columns are unavailable
joint-load modalities plus eye proprioception/contact and nonvisual irradiance.

BODY807 is ordered as follows:

| Offsets | Physical channels | Anatomical treatment |
|---|---|---|
| 0:32 | Four olfactory sites × eight local chemical features | Organ- and side-matched olfactory rows; chemical tuning learned |
| 32:40 | Eight mouth-local chemical features | Proboscis/pharyngeal gustatory and chemosensory rows |
| 40:46 | Two antennae × airflow xyz | Same-side `wind_gravity` rows |
| 46:62 | Sixteen effective acoustic bands, 40–1600 Hz | All annotated auditory rows; no fabricated per-cell frequency tuning |
| 62:68 | Thorax-local linear and angular velocity | Engineered inertial transduction to haltere and campaniform cohorts |
| 68:69 | Local irradiance | Unsupported in the nonvisual atlas; zero mask |
| 69:81 | Twelve internal regulatory signals | Engineered transduction to unresolved sensory rows; no receptor identity claim |
| 81:207 | 126 joint positions | Body-part/side proprioceptor cohorts; unsupported axes remain zero |
| 207:333 | 126 joint velocities | Chordotonal and corresponding mechanosensory cohorts |
| 333:459 | 126 generalized joint loads | Campaniform/load cohorts where annotated; otherwise zero |
| 459:495 | Six feet × force xyz and slip xyz | Same-leg touch/load rows; measured simulated contact and engineered transduction |
| 495:702 | 69 segments × contact force xyz | Segment/body-part tactile cohorts; eyes remain unsupported |
| 702:708 | Two halteres × inertial load xyz | Same-side haltere rows |
| 708:712 | Mouth contact normal xyz and fraction | Mouth mechanosensory rows |
| 712:804 | Persistent effective fatigue for MOTOR92 | Engineered same-body-part proprioceptive cohorts |
| 804:807 | Pump phase sine/cosine and actual flow | Pharyngeal mechanosensory rows |

The 126-joint modality views retain chordotonal, hair-plate and campaniform
distinctions. The MaleCNS annotations contain 392 leg chordotonal afferents when
the peripheral and sensory-ascending rows are both included. Lee et al.'s FANC
work supports claw position, hook motion-direction and club vibration roles,
but individual FANC identities are not copied onto MaleCNS neurons.

Antenna ports preserve olfaction, audition, wind/gravity, humidity, temperature
and contact cohorts. Maxillary palp ports preserve olfaction, taste and contact.
Proboscis and pharynx ports preserve external taste/contact and internal
taste/contact separately. Foot wrench and slip are measured simulated physical
signals with engineered transduction, not claims that a fly has a six-axis force
sensor. All BODY values enter cognition only by injection through
`atlas.body_mask` and full CNS recurrence; raw joint, contact, chemical or
physiology values remain host state and supervised targets.

## Evidence vocabulary

`measured` means an author dataset identifies a modality or peripheral site in
MaleCNS. `inter_animal_inferred` means an author annotation or this join crosses
animals, connectomes, sex or body axes. `engineered` identifies a simulation
actuator or transduction with no claimed biological identity. `unsupported`
means no defensible mapping is available and the corresponding mask is zero.

The body morphology itself is measured/author reconstructed. Its joint axes,
servo selection, gains, bounds, fatigue signals, adhesion and effective
physiology are model-inferred or engineered. No whole-body anatomical muscle
model currently exists in the selected author body. FlyMimic's released
muscle/tendon model covers the left front leg, so it is retained as future
evidence rather than extrapolated to the other five legs and nonleg anatomy.

## Artifacts

`research/fly_embodiment/fly-body-neural-atlas-v1.npz` contains:

- `atlas.body_rows u32[11798]` and `atlas.body_mask f32[11798,807]`;
- `atlas.motor_rows u32[815]` and `atlas.motor_mask f32[92,815]`;
- canonical motor IDs, types, subclasses and sides;
- exact M92 output IDs, targets, groups, ranges, evidence and mapping bases;
- `atlas.motor_cohort_index u32[815]`, using `4294967295` for finer-unresolved
  rows, plus `atlas.fine_cohort_mask u8[156,815]`;
- 60 named anatomical afferent-port cohorts and three 126-joint modality masks;
- source identity and SHA-256 scalar fields.

`research/fly_embodiment/body807-channel-schema.json` is the shared channel
ledger with exact names, units, raw ranges, evidence grades and bases.
`research/fly_embodiment/motor92-channel-schema.json` owns the exact neural
output order, range, activation and MJCF/native-physiology boundary.
`research/fly_embodiment/fly-body-neural-atlas-v1.manifest.json` records array
semantics, counts, hashes and every unsupported motor row.
`research/fly_embodiment/source_ledger.json` pins the primary papers, author
repositories, revisions, supplements and the implementation consequence of each
source.

The source ledger includes Cheong et al. (eLife, DOI
`10.7554/eLife.96084.3`), Azevedo et al. (Nature, DOI
`10.1038/s41586-024-07389-x`), the MaleCNS primary paper and author release
(Nature, DOI `10.1038/s41586-026-10735-w`), the author MANC/FANC comparative
repository, Lee et al.'s proprioception implementation, FlyGym and FlyMimic.

## Cross-schema identity

The neural semantic schemas and compiled physical calibration payloads have
different content and therefore different hashes. The runtime and CNS artifact
must carry both where they meet; one hash must never be relabeled as the other.

| Identity | SHA-256 | Exact scope |
|---|---|---|
| Body schema | `d8c3ff3d22b7f68ec8fb752ba210689ce6531820df57edc96c3bd305766a2d5a` | Canonical author-derived morphology, 126-DOF order and physical M90 schema file |
| Morphology asset set | `2da4b8004d89d2d89f211bd51079d524376a197abcdbbdd9512be1a86ecf6c94` | Model asset paths and content hashes used by the compiled fixture |
| CNS sensory schema | `97b48925c5580c883e1e06bac2d14ad458d84c8dec8ed8b25b143975dc0b1786` | Canonical BODY807 names, ranges, units and evidence |
| Physical sensory schema | `62738ff3c6da3971799b3574620cccf92d27d30d25db1ebfbff4ecf2dc4eb3b0` | Contact schema, compound-eye source, body anchors and engineered optic calibration |
| CNS actuator schema | `00a4a98097f90f6a511e9ee72741e2ee403a8017cfd9d0c0b8c474e195940928` | Canonical MOTOR92 order, ranges, activations and physical/native boundary |
| Physical actuator schema | `09685e5cfc3ed56d5c218498965c32b0cd61814c79644a60bd9dbf5e9991bff7` | Compiled resident M90 numeric actuator/control map |
| Optic calibration schema | `6e150bcffe27a8cfcb53444c4af2d6e1e01c832e33bac7194c6fdfa2dda3b946` | Engineered 1,771-ray equidistant projection from the author camera field of view |

In `physics.json`, compatibility fields `sensory_schema_sha256` and
`actuator_schema_sha256` now mean the CNS BODY807 and MOTOR92 semantic schemas.
The explicit `physical_sensory_schema_sha256` and
`physical_actuator_schema_sha256` fields preserve the compiled anchor,
calibration and numeric-control identities. CNS metadata uses the body-schema,
CNS-sensory and CNS-actuator hashes; it must not substitute either physical
fixture hash for those canonical semantic inputs.
