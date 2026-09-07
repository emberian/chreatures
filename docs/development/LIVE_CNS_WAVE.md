# A living tab: full CNS, embodied world, private history

Approved September 7, 2026. This wave replaces the public experience's dependence
on a recorded movie with an actual local simulation. The recording remains a
historical experiment with its original model identity. A browser session is a
new research life, with its own physical engine epoch and CNS/controller artifacts.

The implemented path is physical MuJoCo state → Rust anatomical retina and
43 body-local channels → learned afferents → all 165,122 MaleCNS neurons and
25,563,197 directed edges → masked, learned rank-64 readout → Z512 → the same
Rust private recurrent controller used by the native host → delivered motor
commands → physical consequences. Observer state, positions, human identity,
video captions and experimental records never enter a parallel controller path.

`native/browser-world` owns actual articulated physics and Rust sensation,
physiology, spatial signals and objects. `native/cognitive-core` is one pure
Rust resident implementation with optional PyO3 and a thin Wasm export in
`native/resident-runtime`. `site/live/cns-webgpu.js` implements the frozen V2
GPU equations in `CNS_DYNAMICS_V2.md`; the native Metal/Torch paths use that same
model. `site/live/engine.js` orders one coupled tick, commits physical motor
receipts, and checkpoints the entire interacting world. It performs no policy
or neural simulation arithmetic itself.

Three initial residents share only immutable weights. Each has its own full CNS
rate/adaptation/support, recurrent memory, goals, skills, RNG and delivered action
history. The browser engine uses a four-lane GPU capacity and a three-resident
physical fixture. Fullgraph recurrence continues while a selected resident's
measured soma positions display its live activity. Missing somas are excluded
from the display, not invented; their neurons still participate in computation.
Displayed quantities are normalized model rates/deviations, not measured Hz or
calcium. A fixed user-visible gain controls the display without changing the model.

Visitors interact through a physical screen, sound sources and movable objects.
A displayed video is sampled from its emitting surface using the same anatomical
rays as the rest of the world. Another object can occlude it. Film playback time
and simulation time are distinct; input frames become part of physical state.
Object insertion compiles a new physical model transactionally and preserves
existing state. The browser ecology is a declared new engine epoch, with explicit
limits documented in `IN_TAB_PHYSICS.md`; it does not claim equivalence with every
regional/lifecycle mechanism of the larger native ecology.

The service artifact is CHCNS2 float32. Browser export rounds only graph weights
and the large learned readout projection to binary16, retaining every edge and
row. Every other float tensor remains float32. The browser manifest records both
source identity and the numerical export, with per-array transport and decoded
hashes. Compressed model blobs belong in an immutable public release and the
Pages build artifact, not in source Git. The first export is about 123.5 MB,
including controller and measured soma data. WebGPU absence or device loss is
reported directly; the page does not substitute a movie for a running organism.

A life checkpoint contains the complete physical state and grown topology,
private ecology and pending events, CNS state and timing, controller private
memory/RNG, motor receipt boundary, screen state, artifact identities and observer
journal. Loading validates the envelope and component hashes, reconstructs a
candidate physical world, and commits the coupled state together. An ambiguous
mutation stops advancement. Saved lives remain tied to the original architecture;
a different model requires an explicit new life.

The root integration check runs the actual Wasm modules and the same WGSL through
headless native WebGPU, followed by an interactive physical scenario and exact
same-runtime continuation. This checks the implementation, not browser UI
performance, general learning or biological validity. Temporal fullgraph fitting,
GAM regime characterization and CNS-only resident skill training proceed in
parallel on the AMD nodes. Each keeps trained, initialized and demonstrated
behavior distinct. No isolated motor success is a prerequisite for building
memory, social interaction, physical development or ecology.

The publication loader also authenticates the three Wasm binaries and binds life
snapshots to the executable-byte manifest (host, shaders, bindings and binaries),
separately from informational Git revisions. A documentation-only build does not
change that runtime identity. During this addition, a joined restore exposed
MuJoCo's mutation of its factory options object. Each reconstructed world now
receives fresh options around the authenticated immutable Wasm bytes. The repaired
joined run restored the entire grown world exactly; its receipt and the failed
integration record are retained in `docs/receipts/live-cns-v2`.
