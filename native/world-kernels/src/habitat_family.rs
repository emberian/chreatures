//! Seeded build-time generation of connected nursery habitats.
//!
//! Runtime organisms never receive the designer graph.  This kernel transforms
//! exact immutable habitat and biosphere templates and emits separate analyst
//! metadata for reproducibility and reachability review.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};

const FORMAT: &str = "chreatures-nursery-family-v1";
const ANALYST_FORMAT: &str = "chreatures-nursery-analyst-v1";
const MAX_NODES: usize = 32;
const MAX_EDGES: usize = 64;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Variation {
    horizontal_jitter_m: f64,
    elevation_jitter_m: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    minimum_ramp_width_m: f64,
    maximum_rise_over_run: f64,
    minimum_underpass_clearance_m: f64,
    platform_thickness_m: f64,
    minimum_spawn_clearance_m: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeSpec {
    position_m: [f64; 3],
    half_size_m: [f64; 2],
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeSpec {
    nodes: [usize; 2],
    width_m: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilySpec {
    id: String,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    underpass_nodes: Vec<usize>,
    canopy_nodes: Vec<usize>,
    landmark_nodes: Vec<usize>,
    gate_node: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingVariant {
    family: String,
    seed: u64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    format: String,
    habitat_template_sha256: String,
    biosphere_template_sha256: String,
    terrain_entity_ids: Vec<String>,
    variation: Variation,
    limits: Limits,
    families: Vec<FamilySpec>,
    training_variants: Vec<TrainingVariant>,
}

#[derive(Clone)]
struct Node {
    position: [f64; 3],
    half_size: [f64; 2],
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn symmetric(&mut self, extent: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        extent * (2.0 * unit - 1.0)
    }
}

fn sha256(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn finite(value: f64) -> bool {
    value.is_finite()
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn object_mut<'a>(value: &'a mut Value, label: &str) -> PyResult<&'a mut Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be an object")))
}

fn array_mut<'a>(value: &'a mut Value, key: &str) -> PyResult<&'a mut Vec<Value>> {
    object_mut(value, "document")?
        .get_mut(key)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PyValueError::new_err(format!("document requires array {key}")))
}

fn id(value: &Value) -> Option<&str> {
    value.as_object()?.get("id")?.as_str()
}

fn position_mut(value: &mut Value) -> PyResult<&mut Vec<Value>> {
    value
        .as_object_mut()
        .and_then(|item| item.get_mut("position"))
        .and_then(Value::as_array_mut)
        .filter(|value| value.len() == 3)
        .ok_or_else(|| PyValueError::new_err("physical entity requires position[3]"))
}

fn set_position(entity: &mut Value, position: [f64; 3]) -> PyResult<()> {
    *position_mut(entity)? = position.into_iter().map(Value::from).collect();
    Ok(())
}

fn translate_entity(entity: &mut Value, delta: [f64; 3]) -> PyResult<()> {
    let position = position_mut(entity)?;
    for axis in 0..3 {
        let value = position[axis]
            .as_f64()
            .ok_or_else(|| PyValueError::new_err("entity position must be numeric"))?;
        position[axis] = Value::from(value + delta[axis]);
    }
    Ok(())
}

fn entity_index(entities: &[Value]) -> HashMap<String, usize> {
    entities
        .iter()
        .enumerate()
        .filter_map(|(index, entity)| id(entity).map(|name| (name.to_owned(), index)))
        .collect()
}

fn ramp_quaternion(dx: f64, dy: f64, dz: f64) -> [f64; 4] {
    let yaw = dy.atan2(dx);
    let pitch = -dz.atan2((dx * dx + dy * dy).sqrt());
    let (sy, cy) = (0.5 * yaw).sin_cos();
    let (sp, cp) = (0.5 * pitch).sin_cos();
    [cy * cp, -sy * sp, cy * sp, sy * cp]
}

fn validate_config(config: &Config) -> PyResult<()> {
    if config.format != FORMAT
        || config.families.len() < 3
        || config.families.len() > 8
        || config.terrain_entity_ids.is_empty()
        || config.terrain_entity_ids.len() > 64
        || config.habitat_template_sha256.len() != 64
        || config.biosphere_template_sha256.len() != 64
        || !finite(config.variation.horizontal_jitter_m)
        || !(0.0..=0.25).contains(&config.variation.horizontal_jitter_m)
        || !finite(config.variation.elevation_jitter_m)
        || !(0.0..=0.08).contains(&config.variation.elevation_jitter_m)
        || !finite(config.limits.minimum_ramp_width_m)
        || !(0.45..=2.0).contains(&config.limits.minimum_ramp_width_m)
        || !finite(config.limits.maximum_rise_over_run)
        || !(0.05..=0.4).contains(&config.limits.maximum_rise_over_run)
        || !finite(config.limits.minimum_underpass_clearance_m)
        || !(0.35..=1.5).contains(&config.limits.minimum_underpass_clearance_m)
        || !finite(config.limits.platform_thickness_m)
        || !(0.03..=0.15).contains(&config.limits.platform_thickness_m)
        || !finite(config.limits.minimum_spawn_clearance_m)
        || !(0.35..=2.0).contains(&config.limits.minimum_spawn_clearance_m)
    {
        return Err(PyValueError::new_err(
            "invalid nursery-family configuration",
        ));
    }
    let mut names = HashSet::new();
    for family in &config.families {
        if !identifier(&family.id)
            || !names.insert(family.id.clone())
            || family.nodes.len() < 6
            || family.nodes.len() > MAX_NODES
            || family.edges.len() < family.nodes.len() - 1
            || family.edges.len() > MAX_EDGES
            || family.gate_node >= family.nodes.len()
        {
            return Err(PyValueError::new_err("invalid nursery topology family"));
        }
        for node in &family.nodes {
            if node.position_m.iter().any(|value| !finite(*value))
                || node.half_size_m.iter().any(|value| !finite(*value))
                || !(0.0..12.0).contains(&node.position_m[0])
                || !(0.0..8.0).contains(&node.position_m[1])
                || !(0.08..=1.4).contains(&node.position_m[2])
                || node
                    .half_size_m
                    .iter()
                    .any(|value| !(0.55..=1.4).contains(value))
                || node.position_m[0] - node.half_size_m[0] < 0.15
                || node.position_m[0] + node.half_size_m[0] > 11.85
                || node.position_m[1] - node.half_size_m[1] < 0.15
                || node.position_m[1] + node.half_size_m[1] > 7.85
            {
                return Err(PyValueError::new_err(
                    "nursery platform exceeds habitat bounds",
                ));
            }
        }
        let mut adjacency = vec![Vec::new(); family.nodes.len()];
        let mut edges = HashSet::new();
        for edge in &family.edges {
            let [a, b] = edge.nodes;
            if a >= family.nodes.len()
                || b >= family.nodes.len()
                || a == b
                || !finite(edge.width_m)
                || edge.width_m < config.limits.minimum_ramp_width_m
                || !edges.insert((a.min(b), a.max(b)))
            {
                return Err(PyValueError::new_err("invalid nursery ramp edge"));
            }
            let first = &family.nodes[a].position_m;
            let second = &family.nodes[b].position_m;
            let run = ((second[0] - first[0]).powi(2) + (second[1] - first[1]).powi(2)).sqrt();
            let slope = (second[2] - first[2]).abs() / run;
            if !slope.is_finite() || slope > config.limits.maximum_rise_over_run * 0.85 {
                return Err(PyValueError::new_err(
                    "base nursery ramp is too steep for jitter",
                ));
            }
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        let mut reached = vec![false; family.nodes.len()];
        let mut queue = VecDeque::from([0]);
        reached[0] = true;
        while let Some(node) = queue.pop_front() {
            for &next in &adjacency[node] {
                if !reached[next] {
                    reached[next] = true;
                    queue.push_back(next);
                }
            }
        }
        if reached.iter().any(|value| !value)
            || family
                .underpass_nodes
                .iter()
                .chain(&family.canopy_nodes)
                .chain(&family.landmark_nodes)
                .any(|node| *node >= family.nodes.len())
            || family.underpass_nodes.iter().any(|node| {
                family.nodes[*node].position_m[2] - config.limits.platform_thickness_m
                    < config.limits.minimum_underpass_clearance_m
            })
        {
            return Err(PyValueError::new_err("nursery designer graph is invalid"));
        }
    }
    if config.training_variants.is_empty()
        || config.training_variants.len() > 256
        || config
            .training_variants
            .iter()
            .any(|value| !names.contains(&value.family))
        || config
            .training_variants
            .iter()
            .map(|value| (&value.family, value.seed))
            .collect::<HashSet<_>>()
            .len()
            != config.training_variants.len()
    {
        return Err(PyValueError::new_err("invalid nursery training variants"));
    }
    Ok(())
}

fn platform_entity(index: usize, node: &Node, thickness: f64, material: &str) -> Value {
    let mut shapes = vec![json!({
        "type": "box",
        "size": [node.half_size[0], node.half_size[1], thickness * 0.5],
        "position": [0.0, 0.0, node.position[2] - thickness * 0.5]
    })];
    if node.position[2] > thickness + 0.12 {
        let support_half = 0.045;
        let support_height = (node.position[2] - thickness) * 0.5;
        for x in [-0.78, 0.78] {
            for y in [-0.72, 0.72] {
                shapes.push(json!({
                    "type": "box",
                    "size": [support_half, support_half, support_height],
                    "position": [x * node.half_size[0], y * node.half_size[1], support_height]
                }));
            }
        }
    }
    json!({
        "id": format!("family-platform-{index:02}"),
        "mobility": "static", "material": material,
        "physical_material": "masonry",
        "position": [node.position[0], node.position[1], 0.0],
        "shapes": shapes, "components": []
    })
}

fn ramp_entity(
    index: usize,
    first: &Node,
    second: &Node,
    width: f64,
    thickness: f64,
    material: &str,
) -> Value {
    let dx = second.position[0] - first.position[0];
    let dy = second.position[1] - first.position[1];
    let dz = second.position[2] - first.position[2];
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    json!({
        "id": format!("family-ramp-{index:02}"),
        "mobility": "static", "material": material,
        "physical_material": "masonry",
        "position": [
            0.5 * (first.position[0] + second.position[0]),
            0.5 * (first.position[1] + second.position[1]),
            0.5 * (first.position[2] + second.position[2]) - thickness * 0.5
        ],
        "quaternion": ramp_quaternion(dx, dy, dz),
        "shapes": [{"type":"box", "size":[length * 0.5, width * 0.5, thickness * 0.5]}],
        "components": []
    })
}

fn canopy_entity(index: usize, node: &Node) -> Value {
    let clearance = 0.58;
    let roof = node.position[2] + clearance;
    json!({
        "id": format!("family-canopy-{index:02}"),
        "mobility": "static", "material": "leaf", "physical_material": "timber",
        "position": [node.position[0], node.position[1], 0.0],
        "shapes": [
            {"type":"box", "size":[0.045,0.045,clearance*0.5], "position":[-node.half_size[0]*0.72,0.0,node.position[2]+clearance*0.5]},
            {"type":"box", "size":[0.045,0.045,clearance*0.5], "position":[ node.half_size[0]*0.72,0.0,node.position[2]+clearance*0.5]},
            {"type":"box", "size":[node.half_size[0]*0.78,node.half_size[1]*0.42,0.035], "position":[0.0,0.0,roof]}
        ],
        "components": [{"type":"shade", "radius":node.half_size[0], "strength":0.52}]
    })
}

fn landmark_entity(index: usize, node: &Node, material: &str) -> Value {
    json!({
        "id": format!("family-landmark-{index:02}"),
        "mobility":"static", "material":material, "physical_material":"timber",
        "position":[node.position[0], node.position[1], 0.0],
        "shapes":[{"type":"box", "size":[0.035,0.24,0.24], "position":[node.half_size[0]*0.72,0.0,node.position[2]+0.24]}],
        "components":[]
    })
}

#[pyclass]
pub struct HabitatFamily {
    config: Config,
    config_sha256: String,
}

#[pymethods]
impl HabitatFamily {
    #[new]
    fn new(config_json: String, config_sha256: String) -> PyResult<Self> {
        if sha256(config_json.as_bytes()) != config_sha256 {
            return Err(PyValueError::new_err(
                "nursery-family configuration hash differs",
            ));
        }
        let config: Config = serde_json::from_str(&config_json).map_err(|error| {
            PyValueError::new_err(format!("invalid nursery-family JSON: {error}"))
        })?;
        validate_config(&config)?;
        Ok(Self {
            config,
            config_sha256,
        })
    }

    fn families(&self) -> Vec<String> {
        self.config
            .families
            .iter()
            .map(|value| value.id.clone())
            .collect()
    }

    fn training_variants(&self) -> Vec<(String, u64)> {
        self.config
            .training_variants
            .iter()
            .map(|value| (value.family.clone(), value.seed))
            .collect()
    }

    fn generate(
        &self,
        habitat_json: String,
        biosphere_json: String,
        seed: u64,
        family_id: String,
    ) -> PyResult<(String, String, String)> {
        if sha256(habitat_json.as_bytes()) != self.config.habitat_template_sha256
            || sha256(biosphere_json.as_bytes()) != self.config.biosphere_template_sha256
        {
            return Err(PyValueError::new_err(
                "nursery source template hash differs",
            ));
        }
        let family = self
            .config
            .families
            .iter()
            .find(|value| value.id == family_id)
            .ok_or_else(|| PyValueError::new_err("unknown nursery topology family"))?;
        let mut habitat: Value = serde_json::from_str(&habitat_json)
            .map_err(|error| PyValueError::new_err(format!("invalid habitat template: {error}")))?;
        let biosphere: Value = serde_json::from_str(&biosphere_json).map_err(|error| {
            PyValueError::new_err(format!("invalid biosphere template: {error}"))
        })?;
        if habitat.get("version").and_then(Value::as_u64) != Some(1)
            || biosphere.get("format").and_then(Value::as_str)
                != Some("chreatures-biosphere-birth-v5")
        {
            return Err(PyValueError::new_err(
                "nursery templates use unsupported current schemas",
            ));
        }

        let family_hash = family_id.bytes().fold(0_u64, |value, byte| {
            value.wrapping_mul(131).wrapping_add(byte as u64)
        });
        let mut rng = SplitMix64::new(seed ^ family_hash.rotate_left(17));
        let mut nodes = Vec::with_capacity(family.nodes.len());
        for source in &family.nodes {
            nodes.push(Node {
                position: [
                    source.position_m[0] + rng.symmetric(self.config.variation.horizontal_jitter_m),
                    source.position_m[1] + rng.symmetric(self.config.variation.horizontal_jitter_m),
                    (source.position_m[2]
                        + rng.symmetric(self.config.variation.elevation_jitter_m))
                    .max(0.08),
                ],
                half_size: source.half_size_m,
            });
        }
        for node in &nodes {
            if node.position[0] - node.half_size[0] < 0.1
                || node.position[0] + node.half_size[0] > 11.9
                || node.position[1] - node.half_size[1] < 0.1
                || node.position[1] + node.half_size[1] > 7.9
            {
                return Err(PyValueError::new_err(
                    "seeded platform jitter exceeds habitat",
                ));
            }
        }
        if family.underpass_nodes.iter().any(|index| {
            nodes[*index].position[2] - self.config.limits.platform_thickness_m
                < self.config.limits.minimum_underpass_clearance_m
        }) {
            return Err(PyValueError::new_err(
                "seeded underpass violates its clearance limit",
            ));
        }
        let mut edge_metadata = Vec::new();
        for (index, edge) in family.edges.iter().enumerate() {
            let first = &nodes[edge.nodes[0]];
            let second = &nodes[edge.nodes[1]];
            let dx = second.position[0] - first.position[0];
            let dy = second.position[1] - first.position[1];
            let dz = second.position[2] - first.position[2];
            let run = (dx * dx + dy * dy).sqrt();
            let slope = dz.abs() / run;
            if slope > self.config.limits.maximum_rise_over_run {
                return Err(PyValueError::new_err("seeded ramp exceeds slope limit"));
            }
            edge_metadata.push(json!({
                "id":format!("family-ramp-{index:02}"), "nodes":edge.nodes,
                "run_m":run, "rise_m":dz, "rise_over_run":slope, "width_m":edge.width_m
            }));
        }

        let terrain: HashSet<&str> = self
            .config
            .terrain_entity_ids
            .iter()
            .map(String::as_str)
            .collect();
        let entities = array_mut(&mut habitat, "entities")?;
        let existing: HashSet<String> = entities.iter().filter_map(id).map(str::to_owned).collect();
        if existing.len() != entities.len()
            || terrain.iter().any(|name| !existing.contains(*name))
            || (0..nodes.len())
                .any(|index| existing.contains(&format!("family-platform-{index:02}")))
            || (0..family.edges.len())
                .any(|index| existing.contains(&format!("family-ramp-{index:02}")))
        {
            return Err(PyValueError::new_err(
                "nursery template identities are missing or duplicated",
            ));
        }
        entities.retain(|entity| id(entity).is_none_or(|name| !terrain.contains(name)));
        let materials = ["terracotta", "limestone", "darkstone"];
        for (index, node) in nodes.iter().enumerate() {
            entities.push(platform_entity(
                index,
                node,
                self.config.limits.platform_thickness_m,
                materials[(index + (rng.next_u64() as usize)) % materials.len()],
            ));
        }
        for (index, edge) in family.edges.iter().enumerate() {
            entities.push(ramp_entity(
                index,
                &nodes[edge.nodes[0]],
                &nodes[edge.nodes[1]],
                edge.width_m,
                self.config.limits.platform_thickness_m,
                materials[(index + 1) % materials.len()],
            ));
        }
        for (index, &node) in family.canopy_nodes.iter().enumerate() {
            entities.push(canopy_entity(index, &nodes[node]));
        }
        for (index, &node) in family.landmark_nodes.iter().enumerate() {
            entities.push(landmark_entity(
                index,
                &nodes[node],
                if index % 2 == 0 { "violet" } else { "cyan" },
            ));
        }
        let indices = entity_index(entities);
        let required = [
            "lift-approach",
            "pressure-lift",
            "passage-gate",
            "lift-frame",
            "counterweight-block",
            "balance-plank",
            "balance-fulcrum",
            "bell-frame",
            "rain-bell",
            "leaf-gate",
            "underdeck-light",
            "terrace-light",
        ];
        if required.iter().any(|name| !indices.contains_key(*name)) {
            return Err(PyValueError::new_err(
                "nursery mechanism identity is missing",
            ));
        }

        let gate = &nodes[family.gate_node];
        let gate_target = [gate.position[0], gate.position[1], gate.position[2]];
        let gate_delta = [gate_target[0] - 7.5, gate_target[1] - 3.85, gate_target[2]];
        for name in [
            "lift-approach",
            "pressure-lift",
            "passage-gate",
            "lift-frame",
            "counterweight-block",
        ] {
            translate_entity(&mut entities[indices[name]], gate_delta)?;
        }
        let cluster_specs = [
            (
                ["balance-plank", "balance-fulcrum"].as_slice(),
                [5.05, 6.55, 0.0],
                (family.gate_node + 2) % nodes.len(),
            ),
            (
                ["bell-frame", "rain-bell"].as_slice(),
                [1.4, 6.25, 0.0],
                (family.gate_node + 4) % nodes.len(),
            ),
        ];
        for (names, origin, target_index) in cluster_specs {
            let target = &nodes[target_index];
            let delta = [
                target.position[0] - origin[0],
                target.position[1] - origin[1],
                target.position[2],
            ];
            for name in names {
                translate_entity(&mut entities[indices[*name]], delta)?;
            }
        }
        let leaf_target = &nodes[(family.gate_node + 1) % nodes.len()];
        set_position(
            &mut entities[indices["leaf-gate"]],
            [
                leaf_target.position[0],
                leaf_target.position[1],
                leaf_target.position[2] + 0.34,
            ],
        )?;
        let low_light_node = family.underpass_nodes.first().copied().unwrap_or(0);
        let low = &nodes[low_light_node];
        set_position(
            &mut entities[indices["underdeck-light"]],
            [
                low.position[0],
                low.position[1],
                (low.position[2] - 0.24).max(0.12),
            ],
        )?;
        let high_index = nodes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.position[2].total_cmp(&b.1.position[2]))
            .map(|value| value.0)
            .unwrap_or(0);
        let high = &nodes[high_index];
        set_position(
            &mut entities[indices["terrace-light"]],
            [high.position[0], high.position[1], high.position[2] + 0.42],
        )?;
        let _ = entities;

        let bodies = array_mut(&mut habitat, "bodies")?;
        if bodies.len() != 6 {
            return Err(PyValueError::new_err(
                "nursery requires six existing resident genotypes",
            ));
        }
        let spawn_nodes: Vec<usize> = (0..nodes.len())
            .filter(|index| *index != family.gate_node)
            .collect();
        if spawn_nodes.len() < bodies.len() {
            return Err(PyValueError::new_err(
                "nursery topology has too few gate-clear spawn platforms",
            ));
        }
        let mut spawn_metadata = Vec::new();
        for (index, body) in bodies.iter_mut().enumerate() {
            let node_index = spawn_nodes[index];
            let node = &nodes[node_index];
            let lateral = if index % 2 == 0 { -0.22 } else { 0.22 };
            let position = [
                node.position[0],
                node.position[1] + lateral,
                node.position[2] + 0.12,
            ];
            set_position(body, position)?;
            spawn_metadata
                .push(json!({"resident":id(body), "node":node_index, "position_m":position}));
        }
        for first in 0..spawn_metadata.len() {
            for second in first + 1..spawn_metadata.len() {
                let a = spawn_metadata[first]["position_m"].as_array().unwrap();
                let b = spawn_metadata[second]["position_m"].as_array().unwrap();
                let dx = a[0].as_f64().unwrap() - b[0].as_f64().unwrap();
                let dy = a[1].as_f64().unwrap() - b[1].as_f64().unwrap();
                if (dx * dx + dy * dy).sqrt() < self.config.limits.minimum_spawn_clearance_m {
                    return Err(PyValueError::new_err(
                        "seeded resident spawns are not clear",
                    ));
                }
            }
        }

        let entities = array_mut(&mut habitat, "entities")?;
        let indices = entity_index(entities);
        let colony_slots: [[f64; 2]; 2] = [[-0.42, 0.34], [0.42, -0.34]];
        let mut colony_metadata = Vec::new();
        for colony in 0..12 {
            let node_index = (colony + 2) % nodes.len();
            let node = &nodes[node_index];
            let slot = colony_slots[(colony / nodes.len()) % colony_slots.len()];
            let position = [
                node.position[0]
                    + slot[0].clamp(-node.half_size[0] * 0.55, node.half_size[0] * 0.55),
                node.position[1]
                    + slot[1].clamp(-node.half_size[1] * 0.55, node.half_size[1] * 0.55),
                node.position[2] + 0.025,
            ];
            for suffix in ["branches", "roots", "leaves"] {
                let name = format!("reef-{:02}-{suffix}", colony + 1);
                let index = *indices.get(&name).ok_or_else(|| {
                    PyValueError::new_err("colony attachment identity is missing")
                })?;
                set_position(&mut entities[index], position)?;
            }
            colony_metadata.push(json!({"colony":format!("reef-{:02}",colony+1), "node":node_index, "position_m":position}));
        }
        let packet_slots = [[0.0, 0.0], [0.28, 0.24], [-0.28, -0.24]];
        let mut resource_metadata = Vec::new();
        for packet in 0..12 {
            let node_index = (packet + 5) % nodes.len();
            let node = &nodes[node_index];
            let slot = packet_slots[(packet / nodes.len()) % packet_slots.len()];
            let position = [
                node.position[0] + slot[0],
                node.position[1] + slot[1],
                node.position[2] + 0.10,
            ];
            let name = format!("living-packet-{packet:02}");
            let index = *indices
                .get(&name)
                .ok_or_else(|| PyValueError::new_err("finite material entity is missing"))?;
            set_position(&mut entities[index], position)?;
            resource_metadata
                .push(json!({"entity":name, "node":node_index, "position_m":position}));
        }
        for (offset, name) in [
            "tone-ball",
            "cyan-ball",
            "rattle-block",
            "stack-block-a",
            "stack-block-b",
        ]
        .iter()
        .enumerate()
        {
            if let Some(&index) = indices.get(*name) {
                let node = &nodes[(offset + 3) % nodes.len()];
                set_position(
                    &mut entities[index],
                    [
                        node.position[0],
                        node.position[1],
                        node.position[2] + 0.16 + 0.12 * (offset == 4) as u8 as f64,
                    ],
                )?;
            }
        }

        object_mut(&mut habitat, "habitat")?.insert(
            "name".to_owned(),
            Value::String(format!("nursery-{}-{seed}", family.id)),
        );
        let node_metadata: Vec<Value> = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| json!({
                "id":format!("family-platform-{index:02}"), "position_m":node.position,
                "half_size_m":node.half_size, "underpass":family.underpass_nodes.contains(&index)
            }))
            .collect();
        let analyst = json!({
            "format":ANALYST_FORMAT, "family":family.id, "seed":seed,
            "generator_config_sha256":self.config_sha256,
            "runtime_visible":false,
            "graph":{"nodes":node_metadata,"edges":edge_metadata,"connected":true},
            "underpass_nodes":family.underpass_nodes,
            "canopy_nodes":family.canopy_nodes,
            "landmark_nodes":family.landmark_nodes,
            "gate_node":family.gate_node,
            "resident_spawns":spawn_metadata,
            "colony_attachments":colony_metadata,
            "finite_resources":resource_metadata,
            "limits":{
                "maximum_rise_over_run":self.config.limits.maximum_rise_over_run,
                "minimum_ramp_width_m":self.config.limits.minimum_ramp_width_m,
                "minimum_underpass_clearance_m":self.config.limits.minimum_underpass_clearance_m,
                "minimum_spawn_clearance_m":self.config.limits.minimum_spawn_clearance_m
            }
        });
        let mut habitat_output = serde_json::to_string_pretty(&habitat)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        // The biosphere is an immutable source template. Preserve its exact
        // decimal spellings so build-time layout generation cannot shift a
        // chemical coefficient by a serialization round trip.
        let biosphere_output = biosphere_json;
        let mut analyst_output = serde_json::to_string_pretty(&analyst)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        habitat_output.push('\n');
        analyst_output.push('\n');
        Ok((habitat_output, biosphere_output, analyst_output))
    }
}
