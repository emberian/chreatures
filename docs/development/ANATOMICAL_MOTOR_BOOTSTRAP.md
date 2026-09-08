# Anatomical motor integration and a sensory training curriculum

Source audit: 2026-09-07, Chreatures `a278f45b495080e12f43ebb0390e8a75bf8a4f7a`.
This is a source/data audit and proposed next integration, **not an implemented
motor replacement or an executed tone-to-pose experiment**. The audit receipt is
[anatomical-motor-audit.json](../receipts/live-cns-v2/anatomical-motor-audit.json).
No existing resident, model weights or physics state changed during this audit.

## What currently runs

The live loop is physical retina/body samples -> learned sensory currents ->
measured MaleCNS recurrence -> learned global neural representation -> separate
Rust resident controller -> abstract commands -> supplied gait/PD -> MuJoCo.
`site/live/engine.js:69-80` couples these steps. The graph has 165,122 neurons,
25,563,197 directed pairs and 124,025,046 anatomical synapses represented by
counts. It is computed at runtime, not replaced by fitted response curves.

[The V2 equations](CNS_DYNAMICS_V2.md) are engineered graded activity dynamics
with type-shared operating points, gains, time constants and adaptation, plus a
bounded support variable. They are not recovered electrophysiology. Readout is
`Z=tanh(U V (mask * (r-r0)) + b)`: 512 output values through a rank-64 bottleneck.
All 15,340 injected afferent rows are excluded from readout. Motor identity does
not restrict the current readout; any unmasked cell can contribute.

The importer assigns acetylcholine +1; GABA, glutamate and histamine -1; every
other transmitter 0. This is an explicit modeling assumption, not receptor-level
physiology. **938,454 stored edges representing 2,864,102 synapses have zero
effective coupling**: dopamine, octopamine, serotonin and unavailable-transmitter
sources. Thus full stored topology must not be described as a fully functional
chemical signaling system. A new modulatory mechanism needs target/receptor and
kinetic assumptions; simply flipping these zeros to excitatory would not resolve
that gap. Anatomical edge counts and effective nonzero coupling are distinct.

The 43 nonvisual channels are a dense `43 -> 128 -> 11233` learned mixture across
body afferents. Sound does not preferentially enter identified auditory neurons.
Individual leg-joint angles and velocities are absent: the current proprioception
is principally body-root velocity, contact and interoception. Physics knows joint
state, but uses it in supplied PD control rather than delivering it to the CNS.

The resident emits 12 abstract axes, including thrust, yaw, posture, grip and oral
commands. Rust generates tripod targets and twelve hip/knee torques. The enlarged
body is an engineered hexapod, not a reconstructed muscle system. This makes the
CNS a mandatory sensory processor without yet making identified fly motor circuits
the final actuator pathway. The public resident controller is initialized; recent
trained candidates improved offline losses but were not promoted after worse
physical resource/effort outcomes. Neither anatomical presence nor activity
visualization establishes recovered motor competence.

## Anatomical resources already available

The selected graph includes 815 motor neurons (708 VNC, 107 central brain), 1,314
descending and 1,846 ascending neurons. Motor annotations include side for all
815, exit nerve for 814, soma neuromere for 727, and MANC type for 699. Type and
nerve labels are useful selectors, not themselves a completed muscle mapping.

There are 115 auditory-subclass cells, of which 114 are in the currently injected
sensory superclasses. There are 425 chordotonal-organ cells; 403 are `vnc_sensory`
and 22 `sensory_ascending`. The latter are not in the present body injection set.
Any changed selector must be a declared anatomical interface, not silently inferred
from the old dense MLP. Premotor cohorts should be derived from actual connectivity
to identified motor neurons and retained annotations, not an invented category.

The relevant external resources give complementary constraints:

- [Cheong et al., MANC premotor organization](https://doi.org/10.7554/eLife.96084):
  descending pathways, VNC premotor communities and motor targets. Direct DN-to-MN
  connections are uncommon, so decoding descending cells straight to torque would
  still skip much of the motor organization we want to use.
- [Azevedo et al., FANC reconstruction and motor atlas](https://doi.org/10.1038/s41586-024-07389-x):
  motor-neuron muscle targets. FANC and MANC are other animals/datasets; transfer
  requires explicit type matches and provenance, never equating their neuron IDs
  with MaleCNS IDs.
- [Lee et al., leg proprioceptive circuits](https://doi.org/10.1038/s41467-025-59302-3)
  and [author analysis code](https://github.com/sagrawal/Lee_2024): joint position,
  movement and vibration have distinct afferent pathways. This argues for typed
  sensory adapters rather than routing every body channel to every sensory cell.
- [Gorko et al., pose-targeted movement](https://doi.org/10.1038/s41586-024-07222-5):
  head-motor effects depend on posture and proprioceptive feedback. The useful
  training object is a closed sensorimotor loop, not an isolated pose label.
- [FlyMimic author code](https://github.com/gizemozd/FlyMimic/tree/9ea1131626cd76f7203b74076ef8f0e9cab30bef),
  pinned `9ea1131626cd76f7203b74076ef8f0e9cab30bef`, Apache-2.0: inspected the MJCF,
  muscle tracking task and PPO training source. The inspected MJCF has 15 named
  left-front-leg tendon actuators. This is source inspection, not local execution.
  The [author demonstration](https://gizemozd.github.io/fly_mimic/) explicitly uses
  direct torque support for middle/hind legs in its ground example. It does not
  establish a complete six-legged muscle-controlled animal. Preserve that scope.
- [FlyGM](https://arxiv.org/html/2602.17997v3): a connectome-structured model trained
  by expert imitation then reinforcement learning. Its graph is female FlyWire,
  not MaleCNS brain+cord. Its learned efferent decoder is useful algorithmic prior
  art, not a verified MaleCNS motor-neuron-to-muscle implementation.

Scry schema and an executed OpenAlex discovery query are retained under
`~/paperbin/chreatures/research/anatomical-motor-20260907/` (record
`db1a90ba-8e20-4547-9065-671a1bbd97f5`). That narrow query returned six older
records; it is not comprehensive literature coverage. Primary sources above were
then inspected directly. The directory also retains selected pinned FlyMimic
source files, their license and hashes; no foreign runtime was installed.

## Proposed coupled replacement

Build sensory, motor, body and learning changes together in a new engine/model
epoch. Keep the current engine in Git history and existing sealed lives, without
a production compatibility branch.

1. **Typed afferents.** Native per-joint angles, angular velocities, load and
   contact feed anatomically selected proprioceptive/tactile cohorts. Acoustic
   frequency/envelope/temporal features feed auditory cohorts. Learners fit gains,
   tuning and temporal adaptation within those mappings. No cue IDs, requested
   poses, world positions or task clocks become lifetime sensory truth. The
   current three amplitude bands are not a 50/100/200/400-Hz hearing model.
2. **Anatomical motor output.** Select identified motor cells by side, segment,
   exit nerve and resolved muscle target. Learn recruitment and activation laws
   within supported pools; muscle geometry supplies the direction of force.
   Preserve unknown mappings explicitly. Do not turn a CNS glutamate sign into
   a negative muscle activation or an arbitrary signed joint torque. Start from
   identifiable muscle groups and use an explicit synthetic target map where the
   body differs; label that map as engineered.
3. **Cognitive feedback through the CNS.** The private memory/goal organ can read
   CNS representations and write learned, bounded context currents into declared
   CNS populations. Actuator commands then come from motor-cell activity after
   recurrence. A parallel global-representation-to-torque head would retain the
   same missing motor pathway under a new name. Context access need not be forced
   through the present global rank-64 bottleneck; anatomically grouped learned
   interfaces can preserve separate sensorimotor information.
4. **Physical learning.** Replace the supplied gait in the new controller path
   with learned recruitment/coordination, while keeping honest mechanical joint
   limits, passive properties and actuator dynamics. MuJoCo/Rust implement the
   recurring world and body; Torch/ROCm handles training. Preserve full neural,
   muscle, controller and physical state at sequence boundaries.

Do not simultaneously free every edge, all neuron dynamics, muscle parameters
and sensory mappings on four pose labels. Fit identifiable interfaces with shared
priors first, then permit bounded efficacy/plasticity on measured edges. Hebbian
updates and evolutionary variation are distinct mechanisms requiring explicit
state and inheritance rules; neither already exists merely because this graph is
trainable. Target-specific modulation deserves its own actual mechanism in the
same organism, not a dopamine variable attached to a global reward scalar.

## Curriculum: a repertoire that the organism can use

A four-tone/four-pose task is a useful public interaction and a small part of a
broader curriculum. Its tones are sensory events; named poses are trainer targets.
It must not be a runtime tone-to-pose lookup or joint-position animation.

- Gather varied motor babbling and useful teacher trajectories: joint excursions,
  support transfer, posture hold/release, stepping, turning, stopping, reaching
  and recovery, including failed attempts. Unlabeled rollouts train sensory and
  action-conditioned predictions through the CNS.
- Use supervised demonstrations and privileged inverse-control targets to
  initialize viable muscle recruitment and coordination. Teacher information is
  confined to loss/collection machinery. Joint feedback returns through the CNS
  during both collection and autonomous execution.
- Learn tone/visual-cue associations with held poses and short movements, then
  vary duration, loudness, starting posture and perturbations. Cue sequences,
  pauses, distractors and reversals make personal memory relevant. Reusing the
  same repertoire for reaching, signaling and object handling ties this to the
  shared world, rather than producing a disconnected dance classifier.
- Continue with the resident's own goals and actual physical consequences.
  Imitating recorded commands cannot by itself teach autonomous selection or
  termination. Train on policy, retain failed encounters and diverse successful
  strategies, and keep ecology/social interaction advancing alongside motor work.

This mixes self-supervision, supervised initialization and reinforcement learning;
calling the four-tone classifier alone “semi-supervised” would be misleading.
One joined campaign should measure physical pose/trajectory control, reactions to
interruption, energy/effort and transfer across contexts. Motor-pool interventions
can then examine whether the intended pathway actually controls the body. This
campaign is not a prerequisite for building memory, development or ecology.
