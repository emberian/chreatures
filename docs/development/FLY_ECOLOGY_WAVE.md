# Fly body, CNS V4, and connected ecology

The September 8 overnight wave starts with the user's correction: the connectome
originally controlled a fly. NeuroMechFly/FlyGym v2.1.0 at
`ca65a510c2afe6ac61c51df4f274c8d190c2f95f` supplies the new body foundation.
Its micro-CT adult female and the MaleCNS adult male are different specimens;
the transfer is explicit. This is the current build contract, not a claim that
the coupled system has already been trained or published.

## Body and time

Use the author's YAW-PITCH-ROLL rig: 69 anatomical segments and 126 hinge
coordinates. Preserve the six segmented legs and tarsi, head, antennae,
rostrum/haustellum, abdomen, wings and halteres. The active physical bank is 84
position servos plus six adhesion actuators. Distal tarsi, eyes and distal
antennal axes remain passive. Servos are effective engineered actuators, not
identified muscles. There is no supplied production gait or destination rule.

Physics uses the author model's millimetres, seconds and radians, with a
0.0001-second integration step. Its mass base unit is not explicitly declared
by the source; retain model mass units rather than silently asserting SI mass.
Ecology positions cross a millimetre-to-metre boundary. Material quantities
and chemical energy are separately declared synthetic conserved quantities.
Control and sensation advance every 0.01 seconds, after 100 physical steps.
CNS recurrence takes two 0.005-second substeps per control transition.

## Fixed CNS V4 interface

Full MaleCNS: 165122 neurons, 25563197 directed edges, 11752 cell types.
Optics retain 1771 RGB sites and 4107 anchored/explicitly unsupported receptor
rows. Body afferents use all 11798 annotated nonvisual sensory rows. Anatomical
masks distinguish supported correspondences and unknowns. Context12 enters
1314 descending rows; the private controller receives only CNS latent512 and
its own acknowledged context12. All injected rows are masked from the latent
readout. Neither geometry nor direct bodily truth enters private cognition.

The sensory vector is retina5313 followed by BODY807 (total6120). BODY blocks
are in the exact order below. Vectors are signed unless stated otherwise;
positions and contact identities stay host-only. Normalization is fitted on
training worlds and frozen in the afferent artifact.

| BODY offsets | Signals |
|---|---|
| 0:32 | Four olfactory sites (antenna L/R, maxillary palp L/R), eight chemical features each |
| 32:40 | Eight local gustatory chemical features at the mouth |
| 40:46 | Local airflow at the two antennae, three components each |
| 46:62 | Sixteen effective acoustic bands, logarithmically spaced 40–1600 Hz |
| 62:68 | Thorax-local linear and angular velocity |
| 68:69 | Local irradiance |
| 69:81 | Twelve engineered internal regulatory signals, listed below |
| 81:207 | The 126 joint positions, in the body schema's order |
| 207:333 | Joint angular velocities in the same order |
| 333:459 | Generalized joint loads in model force/torque units |
| 459:495 | Six feet: body-local contact force3 and surface-relative slip velocity3 |
| 495:702 | Contact force3 for each of 69 anatomical segments in its own frame |
| 702:708 | Two halteres' local effective inertial loads, three components each |
| 708:712 | Mouth-local contact normal3 and measured contact fraction |
| 712:804 | Persistent effective fatigue for all 92 actuator channels |
| 804:807 | Pump phase sine/cosine and actual transferred flow |

Internal signals are ATP fraction, gut fullness, carbon reserve fraction,
nitrogen reserve fraction, hydration, structural fraction, salivary reservoir
fraction, oxygen availability, normalized maintenance shortfall, acclimation
effort, reproductive investment fraction and internal concentration deviation.
These are engineered transductions of defined physiology, not twelve measured
fly receptor signals. Chemical features represent local physical concentrations,
not object class labels. Their pool composition and transduction are bound in
the world schema. Every sampled quantity must come from an implemented process;
an unavailable mechanism is labeled unsupported rather than fabricated.

## Motor pathway

All 815 identified motor neurons feed a signed, anatomically masked learned
decoder. For each motor neuron, `z=(rate-reference_rate)/rate_scale`, with
positive frozen scale and explicit reference rate. No folded large bias, hidden
clipping, positive-only weight transform or direct controller action exists.
The decoder computes `(weight * anatomical_mask) @ z + intercept`.

Outputs 0:84 are tanh-normalized position targets for the schema's active axes;
84:90 are sigmoid adhesion commands. The six adhesion correspondences are
engineered same-leg MN couplings. Outputs90 and91 are sigmoid pharyngeal pump
and salivary drive; they act through native finite-volume physiology rather
than fictional MJCF joints. Intake requires actual mouth contact. Pump and
salivary MN annotations are separated from proboscis positioning annotations.
Retinal and unresolved extrathoracic motor rows remain explicitly unmapped.
The 156 descriptive muscle cohorts are auxiliary anatomical evidence, not a
bottleneck that prevents the other known body parts from acquiring control.

The physical MJCF therefore has90 controls, while CNS MOTOR has92. Store both
delivered normalized commands and actual physical controls. The neutral pose
and author servo limits define the normalized-to-physical target conversion.

## One numerical and artifact contract

CHCNS4 replaces the production CHCNS3 loader. Canonical graph weights are the
exact IEEE binary16 bits obtained by rounding the measured normalized float32
graph once. Torch, Rust/Metal and WebGPU decode those identical bits to float32;
no backend silently runs a different graph. Record the source float32 hash,
rounding algorithm and canonical bit hash. Other learned tensors remain float32.
Keep explicit `motor.reference_rate[815]`, `motor.rate_scale[815]`, signed
`motor.weight[92,815]` and `motor.intercept[92]`. Bind morphology, sensory,
actuator, graph, anatomy and calibration-corpus identities into artifacts.

The rejected V3 decoder coefficients were not saved and cannot be recovered.
New calibration records centered tensors before export checks. Existing V3
research lives and the public frozen release retain their own engine; they are
not silently reinterpreted as flies under this contract.

## Coupled ecology and learning

The native ecology kernel advances connected material regions, conservative
transport, finite stores, inherited metabolic responses and resource-funded
growth together. Geometry proposals use prepare/commit/abort: failed physical
construction cannot spend material. Bodies encounter gradients, surfaces,
movable matter, other bodies and externally presented light/sound through their
senses. Local growth changes route clearance and subsequent material flow.

Train the CNS interfaces from actual body trajectories, including labeled
offline posture/locomotor teachers and real descending-context interventions.
Teachers are training mechanisms, not installed resident policies. Preserve
whole-world splits and unsuccessful episodes. Rich private memory, interaction
and ecological development progress alongside body learning.

Private learning uses 100 Hz acknowledgements and a 0.4-second achieved-goal
horizon. Its current eight-phase suffix spans 0.08 seconds; retention duration
and recurrent credit are separate and reported in seconds. Acquired endpoints
and per-phase consequence distributions come only from delivered experience.

Root integrates substantial batches before joined physical/GPU evaluation.
The user extended this wave through 10:00 PM EDT September 8 (02:00 UTC
September 9), and is now present to collaborate. Publish coherent model-bound
increments while continuing substantial implementation and training cycles;
the original 10 AM wind-down is superseded. Do not substitute isolated checks
or visual promises for integrated outcomes.
