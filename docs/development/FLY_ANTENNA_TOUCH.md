# Physical antennal touch

The imported NeuroMechFly morphology has separate bilateral pedicel,
funiculus, and arista bodies. Each link has three author-defined joint axes;
the distal funiculus and arista axes remain passive. The visual author meshes
have collision disabled, so the earlier world could report airflow and joint
motion at these bodies but could not physically touch an object or another
fly with an antenna.

`native/fly-body/assets/antenna-contact-proxies-v1.json` derives six compact
collision shapes from the pinned simplified author meshes. Pedicel and
funiculus use 95-percent vertex ellipsoid fits. Arista uses a principal-axis
capsule whose radius is the 95th-percentile radial distance. These are
synthetic, stable contact approximations rather than measured cuticle,
bristles, compliance, or receptor fields. They contribute zero mass, leaving
the author segment mass and inertia unchanged.

The proxy contact time constant is 2 ms. This is an explicitly engineered
numerical compliance that lets the current 1 kHz physical packet observe a
brief collision; it is not a measured antennal material parameter. The exact
margin, impedance, friction, evidence grade, and limitation travel in the
proxy artifact and therefore in each compiled fixture identity.

Each proxy is attached to its actual articulated segment. Resident collision
bits exclude contact with the same fly and admit environment and conspecific
contact. MuJoCo therefore applies reciprocal forces to the antenna and the
other physical body. Existing BODY807 paths carry the result as joint
position, angular velocity, generalized load, and segment-local contact force.
No contact-source ID, object label, task, or behavior policy reaches the CNS.

The evidence and exact source revision are recorded in
`research/fly_embodiment/fly-antenna-touch-source-ledger.json`.

The joined native diagnostic uses zero MOTOR92 and holds an existing free
grain at antennal height with only its model-weight compensation. During the
next 10 ms control interval, resident 00's passive left arista reports BODY807
segment contact `[-0.00372422, -0.00122501, 0.00315742]` in model-force units,
its three distal joints deflect, and the grain acquires lateral velocity
`[8.13465, -2.92297]` mm/s. Restoring the pre-contact snapshot and applying the
same physical input reproduces BODY807, qpos, qvel, and sensordata byte exactly.
This demonstrates simulated reciprocal contact and transduction; it does not
establish biological force calibration or a learned response.

The exact diagnostic can be rerun against its staged B2 world with:

```sh
DYLD_LIBRARY_PATH=native/fly-world/vendor/mujoco-3.1.2/lib \
  .venv/bin/python native/fly-body/tools/check_antenna_contact.py \
  --world /path/to/b2-grain-probe/world.json \
  --native-host native/fly-world/target/release/chreatures-fly-world \
  --output /path/to/native-contact-replay-receipt.json
```
