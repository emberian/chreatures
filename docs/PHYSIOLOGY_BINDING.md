# External physiology binding

`PhysicsWorld.bind_physiology(engine)` assigns metabolism and fatigue to a
transient owner. Before each tick, `begin_step(actions, dt)` returns one paid
active-effort scale in `[0, 1]` for every resident. The scale gates propulsion,
joint and posture torque, grip force, and emitted signal amplitude. A zero scale
therefore supplies none of those active physical or acoustic outputs.

Gaze pitch remains a parameter of the retinal sensor calculation. It does not
move a modeled head joint and is not included in mechanical work. Adding head
dynamics and their energetic cost is a separate body-mechanism change.

After accepted substeps, `finish_step(actions, outcomes, contact_samples, dt)`
owns energy, gut, and fatigue updates. `mechanical_work` measures positive work
from resident propulsion, articulated and grip forces only. It excludes the
external hand, acoustic forces, and passive MuJoCo dynamics and is a diagnostic,
not a claim of conserved metabolic-to-mechanical energy.
