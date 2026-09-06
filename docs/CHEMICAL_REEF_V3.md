# Chemical Reef v3

Chemical Reef v3 is a future-birth habitat and biosphere preset for studying
material-funded construction on exposed terrain. It does not alter saved reefs
or the earlier `chemical-reef` presets.

The three founder frames in
[`chemical-reef-v3.json`](../data/habitats/chemical-reef-v3.json) sit just
outside an island or wall edge. Branch axes begin upward while root axes begin
outward and downward. This gives new root segments a physical route along an
exposed boundary instead of starting inside a terrain volume. The frames are
attachment transforms, not goals or coordinates supplied to resident
controllers.

[`chemical-reef-v3.json`](../data/biosphere/chemical-reef-v3.json) uses the
shared v3 reef grammar and seven world-frame hemisphere rays. One ray points
upward and six cover the surrounding upper hemisphere. Their positive weights
sum to one. Each occluded ray retains the declared low transmission, so one
nearby leaf changes only the angular share it covers. Declarative scene lights
continue to use their physical pose, cone, distance attenuation, and ray
occlusion; a finite visitor light keeps its existing time and distance law.

Leaves capture light in proportion to grammar-supplied area and their remaining
live soft-tissue fraction. Development still requires the chemistry and ATP
paid by the founder compartment before a topology transaction can commit. The
light sampler does not expose geom identities, colony identities, or world
coordinates to a resident.

This preset does not claim collision-free future growth. MuJoCo resolves the
resulting physical shapes, while capsule-volume clearance belongs at the
transactional proposal boundary so a rejected segment cannot consume material
or repeatedly retry from the same obstructed bud.
