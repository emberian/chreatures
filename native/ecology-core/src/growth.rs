//! Local inherited colony development. All geometry here is host truth, never
//! an observation for fly cognition. Final collision acceptance belongs to the
//! physical host. A committed world step retains an attempted developmental
//! choice, even when placement failed. Aborted steps retain neither RNG changes
//! nor material transfers; material is spent only on realized construction.
use crate::model::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

const EPS: f64 = 1e-12;

/// Host collision-query directions for local colony development. The finite
/// spherical sampling is shared by native and Wasm hosts; it is not a fly
/// receptor layout or an organism observation. Keep individual surface hits:
/// two points on one large support are different attachment opportunities.
pub fn surface_probe_directions() -> Vec<[f64; 3]> {
    let mut rays = vec![[0., 0., 1.], [0., 0., -1.]];
    let angle = std::f64::consts::PI * (3. - 5_f64.sqrt());
    for i in 0..96 {
        let z = 1. - 2. * (i as f64 + 0.5) / 96.;
        let radius = (1. - z * z).sqrt();
        let phi = i as f64 * angle;
        rays.push([radius * phi.cos(), radius * phi.sin(), z]);
    }
    rays
}

pub(crate) fn seed_state(rng: u64) -> DevelopmentState {
    DevelopmentState {
        rng: rng ^ 0x6a09_e667_f3bc_c909,
        attempt_index: 0,
        apical_binding: None,
        lateral_cursor: 0,
    }
}

pub(crate) fn build(
    config: &EcologyConfig,
    state: &WorldState,
    input: &GrowthInput,
) -> EcologyResult<PendingGrowth> {
    validate(config, state, input)?;
    let mut out = GrowthProposals {
        format: "chreatures-ecology-growth-v2".into(),
        token: String::new(),
        step_index: state.step_index,
        construction_sites: Vec::new(),
        birth_sites: Vec::new(),
        clearance_queries: Vec::new(),
    };
    let mut transitions = Vec::new();
    let mut colonies = input.colonies.iter().collect::<Vec<_>>();
    colonies.sort_by(|a, b| a.organism_id.cmp(&b.organism_id));
    for local in colonies {
        let organism = state
            .organisms
            .iter()
            .find(|o| o.id == local.organism_id)
            .unwrap();
        // Mobile bodies do not have this growth mechanism, even if a host asks.
        if organism.anchored_region.is_none() {
            continue;
        }
        let Some(program) = &organism.genotype.development else {
            continue;
        };
        let mut rng = organism.development_state.rng;
        let owned = state
            .structures
            .iter()
            .filter(|s| s.active && s.owner_id == organism.id)
            .collect::<Vec<_>>();
        let stem = format!(
            "growth-{}-{}-{}",
            short_id(&organism.id),
            state.step_index,
            organism.development_state.attempt_index
        );
        if organism.development_credit_s + input.dt_s + EPS >= program.interval_s
            && owned.len() < program.maximum_structures
        {
            let lateral = !owned.is_empty() && random(&mut rng) < program.lateral_probability;
            let parent = if lateral {
                Some(owned[(organism.development_state.lateral_cursor as usize) % owned.len()])
            } else {
                owned
                    .iter()
                    .find(|s| {
                        organism.development_state.apical_binding.as_ref()
                            == Some(&s.physics_binding)
                    })
                    .copied()
                    .or_else(|| owned.last().copied())
            };
            let root_axis = unit(add(
                scale(
                    axis(local.orientation_xyzw),
                    program.directional_persistence,
                ),
                scale(
                    unit(local.surface_normal),
                    1. - program.directional_persistence,
                ),
            ));
            let (base, parent_axis, attachment) = match parent {
                Some(s) => {
                    let direction = unit(axis(s.orientation_xyzw));
                    let fraction = if lateral { 0.25 } else { 0.5 };
                    (
                        add(
                            s.position_m,
                            scale(
                                direction,
                                s.nominal_length_m * s.remaining_fraction.cbrt() * fraction,
                            ),
                        ),
                        direction,
                        s.physics_binding.clone(),
                    )
                }
                None => (
                    local.position_m,
                    root_axis,
                    organism.physics_binding.clone(),
                ),
            };
            let azimuth = random(&mut rng) * std::f64::consts::TAU;
            let transverse = tangent(parent_axis, azimuth);
            let branch = if lateral {
                add(
                    scale(parent_axis, program.branch_angle_rad.cos()),
                    scale(transverse, program.branch_angle_rad.sin()),
                )
            } else {
                parent_axis
            };
            let reach = program.nominal_length_m + program.nominal_radius_m;
            let mut avoid = [0.; 3];
            for surface in &local.nearby_surfaces {
                let normal = unit(surface.normal);
                let displacement = sub(base, surface.point_m);
                let distance = dot(displacement, normal);
                if norm(displacement) <= 2. * reach && distance < reach {
                    avoid = add(
                        avoid,
                        scale(normal, (1. - distance.max(0.) / reach).clamp(0., 1.)),
                    );
                }
            }
            for sample in &local.clearance_samples {
                if norm(sub(sample.origin_m, base)) <= reach && sample.free_distance_m < reach {
                    avoid = sub(
                        avoid,
                        scale(unit(sample.direction), 1. - sample.free_distance_m / reach),
                    );
                }
            }
            let direction = unit_or(
                add(
                    add(
                        scale(branch, 0.25 + program.directional_persistence),
                        scale(
                            unit_or(local.light_direction, [0.; 3]),
                            program.phototropism * local.light_intensity,
                        ),
                    ),
                    scale(avoid, program.contact_avoidance),
                ),
                branch,
            );
            let end = add(base, scale(direction, program.nominal_length_m));
            let center = scale(add(base, end), 0.5);
            let radius = program.nominal_radius_m;
            // Native known structures participate in avoidance/clearance; the
            // host additionally checks current fly, packet and world geometry.
            let intersects = state
                .structures
                .iter()
                .filter(|s| s.active && s.physics_binding != attachment)
                .any(|s| {
                    let half = scale(
                        unit(axis(s.orientation_xyzw)),
                        s.nominal_length_m * s.remaining_fraction.cbrt() * 0.5,
                    );
                    segment_distance(base, end, sub(s.position_m, half), add(s.position_m, half))
                        < radius + s.nominal_radius_m * s.remaining_fraction.cbrt() - EPS
                });
            if inside_capsule(config, base, end, radius) && !intersects {
                let site_id = format!("{stem}-branch");
                let region_id = nearest_region(config, center);
                let route_hint = nearby_route(config, base, end, radius);
                out.construction_sites.push(ConstructionSite {
                    site_id: site_id.clone(),
                    organism_id: organism.id.clone(),
                    region_id,
                    host_template_id: local.host_template_id.clone(),
                    physics_binding: format!("{stem}-geometry"),
                    position_m: center,
                    orientation_xyzw: orientation(direction),
                    route_hint,
                });
                out.clearance_queries.push(GrowthClearanceQuery {
                    site_id,
                    organism_id: organism.id.clone(),
                    kind: if lateral {
                        GrowthKind::Lateral
                    } else {
                        GrowthKind::Apical
                    },
                    from_m: base,
                    to_m: end,
                    radius_m: radius,
                    attachment_binding: Some(attachment),
                });
            }
        }
        if let (Some(reproduction), Some(template)) =
            (&organism.genotype.reproduction, &local.child_template_id)
        {
            if organism.reproduction_credit_s + input.dt_s + EPS >= reproduction.interval_s
                && organism.descendant_count < reproduction.maximum_descendants
            {
                let mut surfaces = local
                    .nearby_surfaces
                    .iter()
                    .filter(|s| {
                        s.attachable
                            && norm(sub(s.point_m, local.position_m))
                                <= reproduction.dispersal_distance_m
                            && norm(sub(s.point_m, local.position_m))
                                >= 2. * program.nominal_radius_m
                    })
                    .collect::<Vec<_>>();
                surfaces.sort_by(|a, b| a.surface_id.cmp(&b.surface_id));
                if !surfaces.is_empty() {
                    let surface = surfaces[(random(&mut rng) * surfaces.len() as f64).floor()
                        as usize
                        % surfaces.len()];
                    let normal = unit(surface.normal);
                    let from = add(surface.point_m, scale(normal, program.nominal_radius_m));
                    let to = add(from, scale(normal, program.nominal_length_m));
                    if inside_capsule(config, from, to, program.nominal_radius_m) {
                        let site_id = format!("{stem}-birth");
                        out.birth_sites.push(BirthSite {
                            site_id: site_id.clone(),
                            parent_id: organism.id.clone(),
                            child_id: format!("{stem}-colony"),
                            host_template_id: template.clone(),
                            physics_binding: format!("{stem}-anchor"),
                            anchored_region: Some(surface.region_id.clone()),
                            position_m: scale(add(from, to), 0.5),
                            orientation_xyzw: orientation(normal),
                            internal_volume_m3: organism.internal_volume_m3,
                            capacity: organism.capacity.clone(),
                            atp_capacity: organism.atp_capacity,
                            nominal_radius_m: program.nominal_radius_m,
                            nominal_length_m: program.nominal_length_m,
                        });
                        out.clearance_queries.push(GrowthClearanceQuery {
                            site_id,
                            organism_id: organism.id.clone(),
                            kind: GrowthKind::ColonyBirth,
                            from_m: from,
                            to_m: to,
                            radius_m: program.nominal_radius_m,
                            attachment_binding: None,
                        });
                    }
                }
            }
        }
        // Placement failure is a completed attempt, not an instruction to
        // repeat the identical draw forever. This successor is installed into
        // a pending world state below, so a physical abort still rolls it back.
        if rng != organism.development_state.rng {
            transitions.push(GrowthTransition {
                organism_id: organism.id.clone(),
                next_rng: rng,
            });
        }
    }
    let mut digest = Sha256::new();
    digest.update(serde_json::to_vec(config)?);
    digest.update(serde_json::to_vec(state)?);
    digest.update(serde_json::to_vec(input)?);
    digest.update(serde_json::to_vec(&out)?);
    out.token = format!("{:x}", digest.finalize());
    Ok(PendingGrowth {
        input: input.clone(),
        proposals: out,
        transitions,
    })
}

pub(crate) fn validate_selection(
    plan: Option<&PendingGrowth>,
    input: &TickInput,
) -> EcologyResult<()> {
    match (plan, input.growth_token.as_ref()) {
        (None, None) => Ok(()),
        (Some(plan), Some(token))
            if token == &plan.proposals.token && input.dt_s == plan.input.dt_s =>
        {
            if input
                .construction_sites
                .iter()
                .any(|site| !plan.proposals.construction_sites.contains(site))
                || input
                    .birth_sites
                    .iter()
                    .any(|site| !plan.proposals.birth_sites.contains(site))
            {
                return Err(EcologyError(
                    "accepted growth sites differ from the native proposal".into(),
                ));
            }
            Ok(())
        }
        _ => Err(EcologyError(
            "growth proposal token or time step differs".into(),
        )),
    }
}

pub(crate) fn install(
    plan: &PendingGrowth,
    state: &mut WorldState,
    creations: &[PhysicalCreation],
) {
    for transition in &plan.transitions {
        let successful = plan
            .proposals
            .clearance_queries
            .iter()
            .filter(|q| q.organism_id == transition.organism_id)
            .filter_map(|q| {
                let binding = plan
                    .proposals
                    .construction_sites
                    .iter()
                    .find(|s| s.site_id == q.site_id)
                    .map(|s| &s.physics_binding)
                    .or_else(|| {
                        plan.proposals
                            .birth_sites
                            .iter()
                            .find(|s| s.site_id == q.site_id)
                            .map(|s| &s.physics_binding)
                    })?;
                creations
                    .iter()
                    .any(|c| &c.physics_binding == binding)
                    .then_some((q, binding))
            })
            .collect::<Vec<_>>();
        let private = &mut state
            .organisms
            .iter_mut()
            .find(|o| o.id == transition.organism_id)
            .unwrap()
            .development_state;
        private.rng = transition.next_rng;
        private.attempt_index = private.attempt_index.saturating_add(1);
        for (query, binding) in successful {
            match query.kind {
                GrowthKind::Apical => private.apical_binding = Some(binding.clone()),
                GrowthKind::Lateral => {
                    private.lateral_cursor = private.lateral_cursor.saturating_add(1)
                }
                GrowthKind::ColonyBirth => {}
            }
        }
    }
}

fn validate(config: &EcologyConfig, state: &WorldState, input: &GrowthInput) -> EcologyResult<()> {
    if !input.dt_s.is_finite()
        || input.dt_s <= 0.
        || input.dt_s > 10.
        || input.colonies.len() > state.organisms.len()
    {
        return fail("growth input dimensions differ");
    }
    let mut ids = HashSet::new();
    for local in &input.colonies {
        if !ids.insert(&local.organism_id)
            || !state.organisms.iter().any(|o| o.id == local.organism_id)
            || !inside(config, local.position_m)
            || !finite(&local.orientation_xyzw)
            || norm4(local.orientation_xyzw) < 0.5
            || !finite(&local.surface_normal)
            || norm(local.surface_normal) < 0.5
            || !finite(&local.light_direction)
            || !local.light_intensity.is_finite()
            || !(0. ..=1.).contains(&local.light_intensity)
            || !valid_id(&local.host_template_id)
            || local
                .child_template_id
                .as_ref()
                .is_some_and(|s| !valid_id(s))
            || local.nearby_surfaces.len() > 256
            || local.clearance_samples.len() > 256
        {
            return fail("local colony pose or light differs");
        }
        let mut surfaces = HashSet::new();
        for s in &local.nearby_surfaces {
            if !valid_id(&s.surface_id)
                || !surfaces.insert(&s.surface_id)
                || !config.regions.iter().any(|r| r.id == s.region_id)
                || !inside(config, s.point_m)
                || !finite(&s.normal)
                || norm(s.normal) < 0.5
            {
                return fail("growth surface measurement differs");
            }
        }
        for ray in &local.clearance_samples {
            if !inside(config, ray.origin_m)
                || !finite(&ray.direction)
                || norm(ray.direction) < 0.5
                || !ray.free_distance_m.is_finite()
                || ray.free_distance_m < 0.
            {
                return fail("growth clearance measurement differs");
            }
        }
    }
    Ok(())
}

fn nearest_region(config: &EcologyConfig, point: [f64; 3]) -> String {
    config
        .regions
        .iter()
        .min_by(|a, b| norm(sub(a.center_m, point)).total_cmp(&norm(sub(b.center_m, point))))
        .unwrap()
        .id
        .clone()
}
fn nearby_route(
    config: &EcologyConfig,
    from: [f64; 3],
    to: [f64; 3],
    radius: f64,
) -> Option<String> {
    config
        .routes
        .iter()
        .filter_map(|route| {
            let a = config.regions.iter().find(|r| r.id == route.a)?.center_m;
            let b = config.regions.iter().find(|r| r.id == route.b)?.center_m;
            let distance = segment_distance(from, to, a, b);
            (distance <= radius + (route.cross_section_m2 / std::f64::consts::PI).sqrt())
                .then_some((distance, &route.id))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, id)| id.clone())
}
fn segment_distance(p: [f64; 3], q: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let d1 = sub(q, p);
    let d2 = sub(b, a);
    let r = sub(p, a);
    let aa = dot(d1, d1);
    let ee = dot(d2, d2);
    let ff = dot(d2, r);
    let (mut s, mut t);
    if aa < EPS * EPS && ee < EPS * EPS {
        return norm(r);
    }
    if aa < EPS * EPS {
        s = 0.;
        t = (ff / ee).clamp(0., 1.);
    } else {
        let cc = dot(d1, r);
        if ee < EPS * EPS {
            t = 0.;
            s = (-cc / aa).clamp(0., 1.);
        } else {
            let bb = dot(d1, d2);
            let denom = aa * ee - bb * bb;
            s = if denom > EPS * EPS * EPS * EPS {
                ((bb * ff - cc * ee) / denom).clamp(0., 1.)
            } else {
                0.
            };
            t = (bb * s + ff) / ee;
            if t < 0. {
                t = 0.;
                s = (-cc / aa).clamp(0., 1.);
            } else if t > 1. {
                t = 1.;
                s = ((bb - cc) / aa).clamp(0., 1.);
            }
        }
    }
    norm(sub(add(p, scale(d1, s)), add(a, scale(d2, t))))
}
fn axis(q: [f64; 4]) -> [f64; 3] {
    let n = norm4(q);
    let [x, y, z, w] = q.map(|v| v / n);
    [
        2. * (x * z + w * y),
        2. * (y * z - w * x),
        1. - 2. * (x * x + y * y),
    ]
}
fn orientation(v: [f64; 3]) -> [f64; 4] {
    let v = unit(v);
    if v[2] < -0.999999999 {
        return [1., 0., 0., 0.];
    }
    let q = [-v[1], v[0], 0., 1. + v[2]];
    let n = norm4(q);
    q.map(|x| x / n)
}
fn tangent(v: [f64; 3], phi: f64) -> [f64; 3] {
    let helper = if v[2].abs() < 0.9 {
        [0., 0., 1.]
    } else {
        [1., 0., 0.]
    };
    let x = unit(cross(v, helper));
    let y = cross(v, x);
    add(scale(x, phi.cos()), scale(y, phi.sin()))
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|v| v * s)
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn norm4(a: [f64; 4]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    unit_or(a, [0., 0., 1.])
}
fn unit_or(a: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > EPS {
        scale(a, 1. / n)
    } else {
        fallback
    }
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn finite(a: &[f64]) -> bool {
    a.iter().all(|x| x.is_finite())
}
fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && s.as_bytes()[0].is_ascii_alphabetic()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}
fn short_id(s: &str) -> String {
    let hash = Sha256::digest(s.as_bytes());
    format!("{:x}", hash)[..16].to_string()
}
fn inside(config: &EcologyConfig, p: [f64; 3]) -> bool {
    finite(&p)
        && (0..3).all(|i| {
            p[i] >= config.coordinate_contract.world_min_m[i]
                && p[i] <= config.coordinate_contract.world_max_m[i]
        })
}
fn inside_capsule(config: &EcologyConfig, a: [f64; 3], b: [f64; 3], r: f64) -> bool {
    [a, b].iter().all(|p| {
        (0..3).all(|i| {
            p[i] - r >= config.coordinate_contract.world_min_m[i]
                && p[i] + r <= config.coordinate_contract.world_max_m[i]
        })
    })
}
fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
fn fail<T>(message: &str) -> EcologyResult<T> {
    Err(EcologyError(message.into()))
}
