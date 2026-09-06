//! Transactional, resource-gated parametric growth grammar.
//!
//! A proposal runs against copies of the private turtle/bud and RNG state.
//! Only an exact resource receipt plus a committed physical transaction can
//! advance those copies. Rejection discards the candidate without consuming
//! random numbers or rewriting a bud.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

type SuccessorInput = (String, f64, f64, f64, f64, f64, [f64; 4]);
type LeafInput = (f64, f64, f64, f64, f64);
type RuleInput = (
    String,
    String,
    f64,
    f64,
    f64,
    f64,
    f64,
    [f64; 3],
    [f64; 3],
    f64,
    Vec<SuccessorInput>,
    (Option<LeafInput>, [f64; 5], [f64; 2]),
);
type InitialBudInput = (String, [f64; 3], [f64; 3], [f64; 3], f64);
type SignalInput = (u64, [f64; 4], [f64; 3], [f64; 3], f64, i32);
type SegmentOutput = (
    String,
    String,
    u64,
    Option<String>,
    [f64; 3],
    [f64; 3],
    f64,
    f64,
    f64,
    i32,
    [f64; 3],
);
type LeafOutput = (String, u64, [f64; 3], [f64; 4], [f64; 3], f64, f64);
type ProposalOutput = (
    String,
    u32,
    Vec<u64>,
    f64,
    [f64; 3],
    Vec<f64>,
    f64,
    Vec<SegmentOutput>,
    Vec<LeafOutput>,
);
type BudState = (
    u64,
    String,
    String,
    u32,
    f64,
    [f64; 3],
    [f64; 3],
    [f64; 3],
    [f64; 3],
    Option<String>,
    f64,
);
type KernelState = (u64, u64, u64, u32, f64, f64, [f64; 3], Vec<BudState>);
type CommitReceipt = (String, Vec<f64>, f64, Vec<f64>, f64);

#[derive(Clone)]
struct Successor {
    rule: usize,
    angle: f64,
    azimuth: f64,
    generation_phase: f64,
    scale: f64,
    probability: f64,
    response: [f64; 4],
}

#[derive(Clone)]
struct LeafRule {
    probability: f64,
    area: f64,
    aspect: f64,
    thickness: f64,
    areal_density: f64,
}

#[derive(Clone)]
struct Rule {
    name: String,
    role: String,
    length: f64,
    radius: f64,
    density: f64,
    radius_scale_exponent: f64,
    minimum_aspect_ratio: f64,
    minimum: [f64; 3],
    weights: [f64; 3],
    competition_gain: f64,
    successors: Vec<Successor>,
    leaf: Option<LeafRule>,
    guidance: [f64; 5],
    transport: [f64; 2],
}

#[derive(Clone)]
struct Bud {
    id: u64,
    rule: usize,
    generation: u32,
    scale: f64,
    position: [f64; 3],
    forward: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    parent_part: Option<String>,
    transport_resistance: f64,
}

#[derive(Clone)]
struct Segment {
    id: String,
    role: String,
    parent_bud: u64,
    parent_part: Option<String>,
    from: [f64; 3],
    to: [f64; 3],
    radius: f64,
    biomass: f64,
    transport_resistance: f64,
    attachment_geom: i32,
    attachment_point: [f64; 3],
}

#[derive(Clone)]
struct Leaf {
    id: String,
    parent_bud: u64,
    position: [f64; 3],
    quaternion: [f64; 4],
    size: [f64; 3],
    area: f64,
    biomass: f64,
}

#[derive(Clone)]
struct Proposal {
    token: String,
    generation: u32,
    activated: Vec<u64>,
    biomass: f64,
    kind_biomass: [f64; 3],
    resources: Vec<f64>,
    atp: f64,
    segments: Vec<Segment>,
    leaves: Vec<Leaf>,
}

impl Proposal {
    fn output(&self) -> ProposalOutput {
        (
            self.token.clone(),
            self.generation,
            self.activated.clone(),
            self.biomass,
            self.kind_biomass,
            self.resources.clone(),
            self.atp,
            self.segments
                .iter()
                .map(|value| {
                    (
                        value.id.clone(),
                        value.role.clone(),
                        value.parent_bud,
                        value.parent_part.clone(),
                        value.from,
                        value.to,
                        value.radius,
                        value.biomass,
                        value.transport_resistance,
                        value.attachment_geom,
                        value.attachment_point,
                    )
                })
                .collect(),
            self.leaves
                .iter()
                .map(|value| {
                    (
                        value.id.clone(),
                        value.parent_bud,
                        value.position,
                        value.quaternion,
                        value.size,
                        value.area,
                        value.biomass,
                    )
                })
                .collect(),
        )
    }
}

struct Pending {
    proposal: Proposal,
    buds: Vec<Bud>,
    rng_state: u64,
    next_bud: u64,
    next_part: u64,
}

#[pyclass]
pub struct GrowthKernel {
    grammar_hash: String,
    rules: Vec<Rule>,
    resource_names: Vec<String>,
    resource_composition: Vec<[f64; 3]>,
    atp_per_biomass: [f64; 3],
    variation: [f64; 3],
    cadence: f64,
    minimum_feature_size: f64,
    max_buds: usize,
    rng_state: u64,
    next_bud: u64,
    next_part: u64,
    generation: u32,
    clock: f64,
    next_due: f64,
    genotype: [f64; 3],
    buds: Vec<Bud>,
    pending: Option<Pending>,
    last_resolution_terminals: usize,
    last_capacity_rejections: usize,
    last_budget_rejections: usize,
    last_shape_rejections: usize,
}

#[pymethods]
impl GrowthKernel {
    #[new]
    #[pyo3(signature = (
        grammar_json, grammar_hash, seed, rules, initial_buds, resource_names,
        resource_composition, atp_per_biomass, variation, cadence,
        initial_delay, max_buds, minimum_feature_size
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        grammar_json: String,
        grammar_hash: String,
        seed: u64,
        rules: Vec<RuleInput>,
        initial_buds: Vec<InitialBudInput>,
        resource_names: Vec<String>,
        resource_composition: Vec<[f64; 3]>,
        atp_per_biomass: [f64; 3],
        variation: [f64; 3],
        cadence: f64,
        initial_delay: f64,
        max_buds: usize,
        minimum_feature_size: f64,
    ) -> PyResult<Self> {
        let computed_hash = sha256_hex(grammar_json.as_bytes());
        if grammar_hash != computed_hash {
            return Err(PyValueError::new_err(
                "grammar hash does not match canonical grammar",
            ));
        }
        if rules.is_empty()
            || rules.len() > 64
            || initial_buds.is_empty()
            || initial_buds.len() > max_buds
            || max_buds == 0
            || max_buds > 16_384
            || resource_names.is_empty()
            || resource_names.len() > 64
            || resource_names.len() != resource_composition.len()
            || atp_per_biomass
                .iter()
                .any(|value| !finite_range(*value, 0.0, 1.0e9))
            || variation
                .iter()
                .any(|value| !finite_range(*value, 0.0, 2.0))
            || !finite_range(cadence, 0.05, 1.0e9)
            || !finite_range(initial_delay, 0.0, 1.0e9)
            || !finite_range(minimum_feature_size, 0.002, 1.0)
        {
            return Err(PyValueError::new_err(
                "invalid growth grammar dimensions or limits",
            ));
        }
        let mut resource_set = HashSet::new();
        if resource_names
            .iter()
            .any(|name| name.is_empty() || name.len() > 80 || !resource_set.insert(name.clone()))
            || resource_composition
                .iter()
                .flatten()
                .any(|value| !finite_range(*value, 0.0, 1.0))
            || (0..3).any(|kind| {
                (resource_composition
                    .iter()
                    .map(|row| row[kind])
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    > 1.0e-12
            })
        {
            return Err(PyValueError::new_err("invalid growth resource composition"));
        }

        let mut names = HashMap::new();
        for (index, value) in rules.iter().enumerate() {
            if value.0.is_empty()
                || value.0.len() > 80
                || names.insert(value.0.clone(), index).is_some()
            {
                return Err(PyValueError::new_err("growth rule names must be unique"));
            }
        }
        let mut normalized_rules = Vec::with_capacity(rules.len());
        for value in &rules {
            let (
                name,
                role,
                length,
                radius,
                density,
                radius_scale_exponent,
                minimum_aspect_ratio,
                minimum,
                weights,
                competition_gain,
                successors,
                developmental,
            ) = value;
            let (leaf, guidance, transport) = developmental;
            if !matches!(role.as_str(), "branch" | "root")
                || !finite_range(*length, 0.001, 10.0)
                || !finite_range(*radius, 0.0005, 1.0)
                || !finite_range(*density, 0.001, 30_000.0)
                || !finite_range(*radius_scale_exponent, 0.25, 2.0)
                || !finite_range(*minimum_aspect_ratio, 0.0, 100.0)
                || minimum.iter().any(|item| !finite_range(*item, 0.0, 1.0))
                || weights.iter().any(|item| !finite_range(*item, 0.0, 1.0))
                || weights.iter().sum::<f64>() <= 0.0
                || !finite_range(*competition_gain, 0.0, 20.0)
                || !finite_range(guidance[0], -2.0, 2.0)
                || guidance[1..3]
                    .iter()
                    .any(|item| !finite_range(*item, 0.0, 4.0))
                || !finite_range(guidance[3], 0.002, 2.0)
                || !finite_range(guidance[4], 0.002, 2.0)
                || !finite_range(transport[0], 1.0e-12, 1.0e6)
                || !finite_range(transport[1], 1.0e-12, 1.0e15)
                || successors.len() > 16
            {
                return Err(PyValueError::new_err("invalid growth rule parameters"));
            }
            let mut normalized_successors = Vec::with_capacity(successors.len());
            for successor in successors {
                let target = *names
                    .get(&successor.0)
                    .ok_or_else(|| PyValueError::new_err("successor names an unknown rule"))?;
                if !finite_range(successor.1, -std::f64::consts::PI, std::f64::consts::PI)
                    || !finite_range(successor.2, -1000.0, 1000.0)
                    || !finite_range(successor.3, -1000.0, 1000.0)
                    || !finite_range(successor.4, 0.05, 4.0)
                    || !finite_range(successor.5, 0.0, 1.0)
                    || successor
                        .6
                        .iter()
                        .any(|item| !finite_range(*item, -8.0, 8.0))
                {
                    return Err(PyValueError::new_err("invalid growth successor"));
                }
                normalized_successors.push(Successor {
                    rule: target,
                    angle: successor.1,
                    azimuth: successor.2,
                    generation_phase: successor.3,
                    scale: successor.4,
                    probability: successor.5,
                    response: successor.6,
                });
            }
            let normalized_leaf = match leaf {
                Some(item) => {
                    if !finite_range(item.0, 0.0, 1.0)
                        || !finite_range(item.1, 1.0e-7, 10.0)
                        || !finite_range(item.2, 0.05, 20.0)
                        || !finite_range(item.3, 0.004, 0.2)
                        || !finite_range(item.4, 0.0001, 10_000.0)
                    {
                        return Err(PyValueError::new_err("invalid growth leaf rule"));
                    }
                    Some(LeafRule {
                        probability: item.0,
                        area: item.1,
                        aspect: item.2,
                        thickness: item.3,
                        areal_density: item.4,
                    })
                }
                None => None,
            };
            normalized_rules.push(Rule {
                name: name.clone(),
                role: role.clone(),
                length: *length,
                radius: *radius,
                density: *density,
                radius_scale_exponent: *radius_scale_exponent,
                minimum_aspect_ratio: *minimum_aspect_ratio,
                minimum: *minimum,
                weights: *weights,
                competition_gain: *competition_gain,
                successors: normalized_successors,
                leaf: normalized_leaf,
                guidance: *guidance,
                transport: *transport,
            });
        }

        let mut rng_state = mixed_seed(seed);
        let genotype = [
            (normal(&mut rng_state) * variation[0]).exp(),
            normal(&mut rng_state) * variation[1],
            (normal(&mut rng_state) * variation[2]).exp(),
        ];
        let mut buds = Vec::with_capacity(initial_buds.len());
        for (index, initial) in initial_buds.iter().enumerate() {
            let rule = *names
                .get(&initial.0)
                .ok_or_else(|| PyValueError::new_err("axiom names an unknown rule"))?;
            if !valid_vector(initial.1)
                || !finite_range(initial.4, f64::MIN_POSITIVE, 100.0)
                || scaled_radius(&normalized_rules[rule], initial.4, genotype[0])
                    < minimum_feature_size
            {
                return Err(PyValueError::new_err("invalid initial bud"));
            }
            let (forward, up, right) = frame(initial.2, initial.3)?;
            buds.push(Bud {
                id: index as u64 + 1,
                rule,
                generation: 0,
                scale: initial.4,
                position: initial.1,
                forward,
                up,
                right,
                parent_part: None,
                transport_resistance: 0.0,
            });
        }
        Ok(Self {
            grammar_hash,
            rules: normalized_rules,
            resource_names,
            resource_composition,
            atp_per_biomass,
            variation,
            cadence,
            minimum_feature_size,
            max_buds,
            rng_state,
            next_bud: buds.len() as u64 + 1,
            next_part: 1,
            generation: 0,
            clock: 0.0,
            next_due: initial_delay,
            genotype,
            buds,
            pending: None,
            last_resolution_terminals: 0,
            last_capacity_rejections: 0,
            last_budget_rejections: 0,
            last_shape_rejections: 0,
        })
    }

    #[getter]
    fn grammar_hash(&self) -> String {
        self.grammar_hash.clone()
    }

    #[getter]
    fn resource_names(&self) -> Vec<String> {
        self.resource_names.clone()
    }

    #[getter]
    fn is_due(&self) -> bool {
        !self.buds.is_empty() && self.pending.is_none() && self.clock + 1.0e-12 >= self.next_due
    }

    fn capacity(&self) -> (usize, usize, usize) {
        (
            self.buds.len(),
            self.max_buds,
            self.max_buds - self.buds.len(),
        )
    }

    fn invalidate_parent_parts(&mut self, part_ids: Vec<String>) -> PyResult<Vec<u64>> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot invalidate attachment ancestry during a pending proposal",
            ));
        }
        let parts: HashSet<String> = part_ids.into_iter().collect();
        if parts.iter().any(|value| value.is_empty()) {
            return Err(PyValueError::new_err("invalid developmental part identity"));
        }
        let mut removed = Vec::new();
        self.buds.retain(|bud| {
            let keep = bud
                .parent_part
                .as_ref()
                .is_none_or(|part| !parts.contains(part));
            if !keep {
                removed.push(bud.id);
            }
            keep
        });
        Ok(removed)
    }

    fn proposal_metrics(&self) -> (usize, usize, usize, usize, usize, usize, f64) {
        (
            self.buds.len(),
            self.max_buds,
            self.last_resolution_terminals,
            self.last_capacity_rejections,
            self.last_budget_rejections,
            self.last_shape_rejections,
            self.minimum_feature_size,
        )
    }

    fn elapse(&mut self, dt: f64) -> PyResult<f64> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot advance developmental time with a pending proposal",
            ));
        }
        if !finite_range(dt, 0.0, 1.0e6) {
            return Err(PyValueError::new_err(
                "growth elapsed time must be finite and nonnegative",
            ));
        }
        self.clock += dt;
        if !self.clock.is_finite() || self.clock > 1.0e15 {
            return Err(PyValueError::new_err("growth clock exceeds its bound"));
        }
        Ok(self.clock)
    }

    fn buds(&self) -> Vec<BudState> {
        self.bud_state()
    }

    #[pyo3(signature = (signals, structural_budget, max_new_shapes))]
    fn propose(
        &mut self,
        signals: Vec<SignalInput>,
        structural_budget: f64,
        max_new_shapes: usize,
    ) -> PyResult<Option<ProposalOutput>> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "a growth proposal is already pending",
            ));
        }
        if self.clock + 1.0e-12 < self.next_due {
            return Ok(None);
        }
        if !finite_range(structural_budget, 0.0, 1.0e9)
            || max_new_shapes == 0
            || max_new_shapes > 4096
        {
            return Err(PyValueError::new_err("invalid growth proposal budget"));
        }
        let mut signal_map = HashMap::with_capacity(signals.len());
        let valid_buds: HashSet<u64> = self.buds.iter().map(|bud| bud.id).collect();
        for (bud, values, surface_direction, world_up, surface_distance, surface_geom) in signals {
            if !valid_buds.contains(&bud)
                || values.iter().any(|value| !finite_range(*value, 0.0, 1.0))
                || !valid_vector(surface_direction)
                || !valid_vector(world_up)
                || !finite_range(surface_distance, 0.0, 100.0)
                || surface_geom < -1
                || signal_map
                    .insert(
                        bud,
                        (
                            values,
                            surface_direction,
                            world_up,
                            surface_distance,
                            surface_geom,
                        ),
                    )
                    .is_some()
            {
                return Err(PyValueError::new_err(
                    "invalid or duplicate local bud signal",
                ));
            }
        }

        self.last_resolution_terminals = 0;
        self.last_capacity_rejections = 0;
        self.last_budget_rejections = 0;
        self.last_shape_rejections = 0;
        let mut candidate_buds = self.buds.clone();
        let mut rng = self.rng_state;
        let mut next_bud = self.next_bud;
        let mut next_part = self.next_part;
        let mut segments = Vec::new();
        let mut leaves = Vec::new();
        let mut activated = Vec::new();
        let mut biomass = 0.0;
        let mut kind_biomass = [0.0; 3]; // branch, root, leaf

        let ordered = self.buds.clone();
        for bud in &ordered {
            let Some((values, surface_direction, world_up, surface_distance, surface_geom)) =
                signal_map.get(&bud.id)
            else {
                continue;
            };
            let rule = &self.rules[bud.rule];
            let transport_fraction = if bud.transport_resistance == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + bud.transport_resistance / rule.transport[1])
            };
            let clearance_competition =
                (1.0 - *surface_distance / rule.guidance[4]).clamp(0.0, 1.0);
            let local_values = [
                values[0],
                values[1] * transport_fraction,
                values[2],
                values[3].max(clearance_competition),
            ];
            if local_values[..3]
                .iter()
                .zip(rule.minimum)
                .any(|(value, minimum)| *value < minimum)
            {
                continue;
            }
            let weight_sum = rule.weights.iter().sum::<f64>();
            let local = local_values[..3]
                .iter()
                .zip(rule.weights)
                .map(|(value, weight)| *value * weight)
                .sum::<f64>()
                / weight_sum;
            let vigor = (local * (-rule.competition_gain * local_values[3]).exp()).clamp(0.12, 1.0);

            let mut trial_rng = rng;
            let length_noise = (normal(&mut trial_rng) * self.variation[0] * 0.25).exp();
            let mut length =
                rule.length * bud.scale * self.genotype[0] * length_noise * (0.55 + 0.45 * vigor);
            let radius = scaled_radius(rule, bud.scale, self.genotype[0]);
            let surface_proximity = (1.0 - *surface_distance / rule.guidance[3]).clamp(0.0, 1.0);
            let guidance = add(
                add(bud.forward, mul(*world_up, rule.guidance[0])),
                mul(
                    *surface_direction,
                    rule.guidance[2] * surface_proximity - rule.guidance[1] * clearance_competition,
                ),
            );
            let steered = normalize(guidance)?;
            let attach = *surface_geom >= 0
                && rule.guidance[2] > 0.0
                && *surface_distance <= rule.guidance[3].min(length + radius)
                && dot(steered, *surface_direction) > 0.25;
            let endpoint = if attach {
                length = (*surface_distance - radius).max(self.minimum_feature_size);
                add(bud.position, mul(*surface_direction, length))
            } else {
                add(bud.position, mul(steered, length))
            };
            let volume = std::f64::consts::PI * radius * radius * length
                + 4.0 * std::f64::consts::PI * radius.powi(3) / 3.0;
            let segment_biomass = volume * rule.density;
            let mut trial_leaves = Vec::new();
            let mut trial_biomass = segment_biomass;
            if let Some(leaf_rule) = &rule.leaf {
                if uniform(&mut trial_rng) <= leaf_rule.probability {
                    let leaf_noise = (normal(&mut trial_rng) * self.variation[2] * 0.25).exp();
                    let area = leaf_rule.area * bud.scale * self.genotype[2] * leaf_noise * vigor;
                    let full_length = (area * leaf_rule.aspect).sqrt();
                    let full_width = (area / leaf_rule.aspect).sqrt();
                    let leaf_biomass = area * leaf_rule.areal_density;
                    let rotation = [bud.forward, bud.right, mul(bud.up, -1.0)];
                    let size = [
                        0.5 * full_length,
                        0.5 * full_width,
                        0.5 * leaf_rule.thickness,
                    ];
                    if size.iter().all(|value| *value >= self.minimum_feature_size) {
                        trial_leaves.push(Leaf {
                            id: format!("leaf-{next_part}"),
                            parent_bud: bud.id,
                            position: endpoint,
                            quaternion: matrix_quaternion(rotation),
                            size,
                            area,
                            biomass: leaf_biomass,
                        });
                        trial_biomass += leaf_biomass;
                    } else {
                        self.last_resolution_terminals += 1;
                    }
                }
            }
            let mut trial_children = Vec::new();
            let aspect_ratio = length / (2.0 * radius);
            if aspect_ratio + 1.0e-14 >= rule.minimum_aspect_ratio {
                for successor in &rule.successors {
                    let response = successor
                        .response
                        .iter()
                        .zip(local_values)
                        .map(|(weight, signal)| weight * (signal - 0.5))
                        .sum::<f64>();
                    let probability = (successor.probability * response.exp()).min(1.0);
                    if uniform(&mut trial_rng) > probability {
                        continue;
                    }
                    let angle_noise = normal(&mut trial_rng) * self.variation[1] * 0.25;
                    let angle = successor.angle + self.genotype[1] + angle_noise;
                    let azimuth =
                        successor.azimuth + successor.generation_phase * f64::from(bud.generation);
                    let radial = add(mul(bud.right, azimuth.cos()), mul(bud.up, azimuth.sin()));
                    let child_forward =
                        normalize(add(mul(bud.forward, angle.cos()), mul(radial, angle.sin())))?;
                    let preferred_up = if dot(child_forward, bud.up).abs() < 0.95 {
                        bud.up
                    } else {
                        bud.right
                    };
                    let child_right = normalize(cross(child_forward, preferred_up))?;
                    let child_up = normalize(cross(child_right, child_forward))?;
                    let child_scale = bud.scale * successor.scale;
                    let child_rule = &self.rules[successor.rule];
                    let child_radius = scaled_radius(child_rule, child_scale, self.genotype[0]);
                    if child_radius < self.minimum_feature_size {
                        self.last_resolution_terminals += 1;
                        continue;
                    }
                    trial_children.push(Bud {
                        id: next_bud + trial_children.len() as u64,
                        rule: successor.rule,
                        generation: bud.generation.saturating_add(1),
                        scale: child_scale,
                        position: endpoint,
                        forward: child_forward,
                        up: child_up,
                        right: child_right,
                        parent_part: Some(format!("{}-{next_part}", rule.role)),
                        transport_resistance: bud.transport_resistance
                            + length / (rule.transport[0] * std::f64::consts::PI * radius * radius),
                    });
                }
            } else {
                // Build the funded terminal segment, but do not perpetuate a
                // lineage whose realized geometry is already bead-like.
                self.last_resolution_terminals += 1;
            }
            let shape_count = 1 + trial_leaves.len();
            let resulting_buds = candidate_buds.len() - 1 + trial_children.len();
            let transport_budget = if bud.transport_resistance == 0.0 {
                f64::INFINITY
            } else {
                self.cadence / bud.transport_resistance
            };
            if trial_biomass > transport_budget + 1.0e-14 {
                self.last_budget_rejections += 1;
                continue;
            }
            if biomass + trial_biomass > structural_budget + 1.0e-14 {
                self.last_budget_rejections += 1;
                continue;
            }
            if segments.len() + leaves.len() + shape_count > max_new_shapes {
                self.last_shape_rejections += 1;
                continue;
            }
            if resulting_buds > self.max_buds {
                self.last_capacity_rejections += 1;
                continue;
            }
            candidate_buds.retain(|value| value.id != bud.id);
            candidate_buds.extend(trial_children.iter().cloned());
            next_bud += trial_children.len() as u64;
            let segment_id = format!("{}-{next_part}", rule.role);
            next_part += 1;
            for leaf in &mut trial_leaves {
                leaf.id = format!("leaf-{next_part}");
                next_part += 1;
            }
            segments.push(Segment {
                id: segment_id,
                role: rule.role.clone(),
                parent_bud: bud.id,
                parent_part: bud.parent_part.clone(),
                from: bud.position,
                to: endpoint,
                radius,
                biomass: segment_biomass,
                transport_resistance: bud.transport_resistance
                    + length / (rule.transport[0] * std::f64::consts::PI * radius * radius),
                attachment_geom: if attach { *surface_geom } else { -1 },
                attachment_point: if attach { endpoint } else { [0.0; 3] },
            });
            leaves.extend(trial_leaves);
            biomass += trial_biomass;
            kind_biomass[if rule.role == "branch" { 0 } else { 1 }] += segment_biomass;
            kind_biomass[2] += trial_biomass - segment_biomass;
            activated.push(bud.id);
            rng = trial_rng;
        }
        if activated.is_empty() {
            return Ok(None);
        }
        candidate_buds.sort_by_key(|bud| bud.id);
        let resources: Vec<f64> = self
            .resource_composition
            .iter()
            .map(|row| (0..3).map(|kind| row[kind] * kind_biomass[kind]).sum())
            .collect();
        let atp = (0..3)
            .map(|kind| kind_biomass[kind] * self.atp_per_biomass[kind])
            .sum();
        let mut token_material = format!(
            "{}:{}:{}:{}:{}",
            self.grammar_hash,
            self.generation + 1,
            self.rng_state,
            self.next_part,
            biomass.to_bits(),
        );
        for id in &activated {
            token_material.push_str(&format!(":{id}"));
        }
        let proposal = Proposal {
            token: sha256_hex(token_material.as_bytes()),
            generation: self.generation + 1,
            activated,
            biomass,
            kind_biomass,
            resources,
            atp,
            segments,
            leaves,
        };
        let output = proposal.output();
        self.pending = Some(Pending {
            proposal,
            buds: candidate_buds,
            rng_state: rng,
            next_bud,
            next_part,
        });
        Ok(Some(output))
    }

    fn reject(&mut self, token: String) -> PyResult<()> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no growth proposal is pending"))?;
        if token != pending.proposal.token {
            return Err(PyValueError::new_err("growth transaction token differs"));
        }
        self.pending = None;
        Ok(())
    }

    #[pyo3(signature = (token, accepted_resources, accepted_atp, physical_committed))]
    fn commit(
        &mut self,
        token: String,
        accepted_resources: Vec<f64>,
        accepted_atp: f64,
        physical_committed: bool,
    ) -> PyResult<CommitReceipt> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no growth proposal is pending"))?;
        if token != pending.proposal.token {
            return Err(PyValueError::new_err("growth transaction token differs"));
        }
        if !physical_committed {
            return Err(PyValueError::new_err(
                "growth requires a committed physical transaction",
            ));
        }
        if accepted_resources.len() != pending.proposal.resources.len()
            || !accepted_atp.is_finite()
            || accepted_atp.to_bits() != pending.proposal.atp.to_bits()
            || accepted_resources
                .iter()
                .zip(&pending.proposal.resources)
                .any(|(accepted, requested)| {
                    !accepted.is_finite() || accepted.to_bits() != requested.to_bits()
                })
        {
            return Err(PyValueError::new_err(
                "accepted growth resources differ from the exact request",
            ));
        }
        let pending = self.pending.take().expect("pending checked above");
        let receipt = (
            pending.proposal.token.clone(),
            pending.proposal.resources.clone(),
            pending.proposal.atp,
            accepted_resources,
            accepted_atp,
        );
        self.buds = pending.buds;
        self.rng_state = pending.rng_state;
        self.next_bud = pending.next_bud;
        self.next_part = pending.next_part;
        self.generation = pending.proposal.generation;
        self.next_due = self.clock + self.cadence;
        Ok(receipt)
    }

    fn state(&self) -> KernelState {
        (
            self.rng_state,
            self.next_bud,
            self.next_part,
            self.generation,
            self.clock,
            self.next_due,
            self.genotype,
            self.bud_state(),
        )
    }

    #[pyo3(signature = (
        rng_state, next_bud, next_part, generation, clock, next_due,
        genotype, buds
    ))]
    #[allow(clippy::too_many_arguments)]
    fn restore_state(
        &mut self,
        rng_state: u64,
        next_bud: u64,
        next_part: u64,
        generation: u32,
        clock: f64,
        next_due: f64,
        genotype: [f64; 3],
        buds: Vec<BudState>,
    ) -> PyResult<()> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot restore over a pending growth proposal",
            ));
        }
        if rng_state == 0
            || next_bud == 0
            || next_part == 0
            || !finite_range(clock, 0.0, 1.0e15)
            || !finite_range(next_due, 0.0, 1.0e15)
            || genotype.iter().any(|value| !value.is_finite())
            || genotype[0] <= 0.0
            || genotype[2] <= 0.0
            || buds.len() > self.max_buds
        {
            return Err(PyValueError::new_err("invalid growth state scalars"));
        }
        if !genotype[1].is_finite() {
            return Err(PyValueError::new_err("invalid genotype angle"));
        }
        let rule_names: HashMap<&str, usize> = self
            .rules
            .iter()
            .enumerate()
            .map(|(index, rule)| (rule.name.as_str(), index))
            .collect();
        let mut ids = HashSet::new();
        let mut restored = Vec::with_capacity(buds.len());
        for value in buds {
            let (
                id,
                symbol,
                role,
                bud_generation,
                scale,
                position,
                forward,
                up,
                right,
                parent_part,
                transport_resistance,
            ) = value;
            let rule = *rule_names
                .get(symbol.as_str())
                .ok_or_else(|| PyValueError::new_err("growth state names an unknown rule"))?;
            if role != self.rules[rule].role
                || id == 0
                || !ids.insert(id)
                || !finite_range(scale, f64::MIN_POSITIVE, 100.0)
                || scaled_radius(&self.rules[rule], scale, self.genotype[0])
                    < self.minimum_feature_size
                || !valid_vector(position)
                || !orthonormal(forward, up, right)
                || parent_part.as_ref().is_some_and(|value| value.is_empty())
                || !finite_range(transport_resistance, 0.0, 1.0e15)
            {
                return Err(PyValueError::new_err("invalid growth bud state"));
            }
            restored.push(Bud {
                id,
                rule,
                generation: bud_generation,
                scale,
                position,
                forward,
                up,
                right,
                parent_part,
                transport_resistance,
            });
        }
        if ids.iter().any(|id| *id >= next_bud) {
            return Err(PyValueError::new_err("growth next bud counter is stale"));
        }
        restored.sort_by_key(|bud| bud.id);
        self.rng_state = rng_state;
        self.next_bud = next_bud;
        self.next_part = next_part;
        self.generation = generation;
        self.clock = clock;
        self.next_due = next_due;
        self.genotype = genotype;
        self.buds = restored;
        Ok(())
    }
}

impl GrowthKernel {
    fn bud_state(&self) -> Vec<BudState> {
        self.buds
            .iter()
            .map(|bud| {
                (
                    bud.id,
                    self.rules[bud.rule].name.clone(),
                    self.rules[bud.rule].role.clone(),
                    bud.generation,
                    bud.scale,
                    bud.position,
                    bud.forward,
                    bud.up,
                    bud.right,
                    bud.parent_part.clone(),
                    bud.transport_resistance,
                )
            })
            .collect()
    }
}

fn scaled_radius(rule: &Rule, scale: f64, length_genotype: f64) -> f64 {
    // One inherited power law controls radius throughout development.
    let scale_factor = scale.powf(rule.radius_scale_exponent);
    rule.radius * scale_factor * length_genotype.sqrt()
}

fn finite_range(value: f64, low: f64, high: f64) -> bool {
    value.is_finite() && value >= low && value <= high
}

fn valid_vector(value: [f64; 3]) -> bool {
    value
        .iter()
        .all(|item| item.is_finite() && item.abs() <= 1.0e6)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul(a: [f64; 3], value: f64) -> [f64; 3] {
    [a[0] * value, a[1] * value, a[2] * value]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(value: [f64; 3]) -> PyResult<[f64; 3]> {
    if !valid_vector(value) {
        return Err(PyValueError::new_err(
            "turtle frame contains a nonfinite vector",
        ));
    }
    let norm = dot(value, value).sqrt();
    if norm < 1.0e-10 {
        return Err(PyValueError::new_err("turtle frame contains a zero vector"));
    }
    Ok(mul(value, 1.0 / norm))
}

fn frame(forward: [f64; 3], up: [f64; 3]) -> PyResult<([f64; 3], [f64; 3], [f64; 3])> {
    let forward = normalize(forward)?;
    let right = normalize(cross(forward, up))?;
    let up = normalize(cross(right, forward))?;
    Ok((forward, up, right))
}

fn orthonormal(forward: [f64; 3], up: [f64; 3], right: [f64; 3]) -> bool {
    [forward, up, right]
        .iter()
        .all(|value| valid_vector(*value) && (dot(*value, *value) - 1.0).abs() <= 1.0e-9)
        && dot(forward, up).abs() <= 1.0e-9
        && dot(forward, right).abs() <= 1.0e-9
        && dot(up, right).abs() <= 1.0e-9
        && dot(cross(forward, up), right) >= 1.0 - 1.0e-9
}

fn mixed_seed(seed: u64) -> u64 {
    let mut value = seed ^ 0x9e3779b97f4a7c15;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    if value == 0 {
        0x2545f4914f6cdd1d
    } else {
        value
    }
}

fn random_u64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(0x2545f4914f6cdd1d)
}

fn uniform(state: &mut u64) -> f64 {
    let bits = random_u64(state) >> 11;
    (bits as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
}

fn normal(state: &mut u64) -> f64 {
    let u1 = uniform(state).max(f64::MIN_POSITIVE);
    let u2 = uniform(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn matrix_quaternion(columns: [[f64; 3]; 3]) -> [f64; 4] {
    let m00 = columns[0][0];
    let m01 = columns[1][0];
    let m02 = columns[2][0];
    let m10 = columns[0][1];
    let m11 = columns[1][1];
    let m12 = columns[2][1];
    let m20 = columns[0][2];
    let m21 = columns[1][2];
    let m22 = columns[2][2];
    let trace = m00 + m11 + m22;
    let mut q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    };
    let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    for value in &mut q {
        *value /= norm;
    }
    if q[0] < 0.0 {
        for value in &mut q {
            *value = -*value;
        }
    }
    q
}

// Small self-contained SHA-256 keeps the grammar identity native without adding
// a dependency to the simulation kernel crate.
fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = input.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for chunk in data.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (value, addend) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *value = value.wrapping_add(addend);
        }
    }
    h.iter().map(|value| format!("{value:08x}")).collect()
}
