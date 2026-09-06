//! Native build-time generation of inherited regional habitat genomes.
//!
//! Geometry, finite resource founders, and resident placement are emitted into
//! ordinary physical/biosphere templates. The designer graph and ancestry record
//! are returned separately and are never controller inputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

const FORMAT: &str = "chreatures-regional-habitat-family-v2";
const GENOME_FORMAT: &str = "chreatures-environment-genome-v2";
const RESIDENT_FORMAT: &str = "chreatures-regional-residents-v2";
const ANALYST_FORMAT: &str = "chreatures-regional-analyst-v2";
const RECORD_FORMAT: &str = "chreatures-environment-record-v2";
const MAX_REGIONS: usize = 48;
const MAX_EDGES: usize = 128;
const MAX_RESIDENTS: usize = 32;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalarRange {
    min: f64,
    max: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerRange {
    min: usize,
    max: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Capacity {
    residents: IntegerRange,
    regions: IntegerRange,
    width_m: ScalarRange,
    height_m: ScalarRange,
    depth_m: ScalarRange,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Geometry {
    margin_m: f64,
    platform_half_size_m: ScalarRange,
    platform_thickness_m: f64,
    ramp_width_m: ScalarRange,
    maximum_rise_over_run: f64,
    minimum_underpass_clearance_m: f64,
    minimum_spawn_clearance_m: f64,
    boundary_height_m: f64,
    boundary_thickness_m: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatchmentSpec {
    id: String,
    surface_height_m: f64,
    margin_m: f64,
    thickness_m: f64,
    wall_height_m: f64,
    material: String,
    physical_material: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MutationSpec {
    operator: String,
    recipe_sha256: String,
    scalar_fraction: f64,
    integer_delta: usize,
    resident_delta: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationCostLimits {
    physical_geoms: usize,
    regions: usize,
    edges: usize,
    movables: usize,
    compartments: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DescriptorNormalization {
    resource_mass_per_square_meter: f64,
    renewal_capacity_per_square_meter_second: f64,
    cycle_rank_per_region: f64,
    resident_physical_geoms: usize,
    generation_cost_limits: GenerationCostLimits,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DevelopmentalRuleAlleles {
    vertical_weight: f64,
    clearance_weight: f64,
    surface_weight: f64,
    surface_reach: f64,
    clearance_distance: f64,
    conductivity: f64,
    half_resistance: f64,
    response_light: f64,
    response_nutrient: f64,
    response_support: f64,
    response_competition: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DevelopmentalRuleAlleleRanges {
    vertical_weight: ScalarRange,
    clearance_weight: ScalarRange,
    surface_weight: ScalarRange,
    surface_reach: ScalarRange,
    clearance_distance: ScalarRange,
    conductivity: ScalarRange,
    half_resistance: ScalarRange,
    response_light: ScalarRange,
    response_nutrient: ScalarRange,
    response_support: ScalarRange,
    response_competition: ScalarRange,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MechanismCluster {
    entity_ids: Vec<String>,
    anchor_m: [f64; 3],
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Archetype {
    id: String,
    region_count: IntegerRange,
    lane_count: IntegerRange,
    width_m: ScalarRange,
    height_m: ScalarRange,
    depth_m: ScalarRange,
    elevation_span_m: ScalarRange,
    loop_fraction: ScalarRange,
    shelter_fraction: ScalarRange,
    underpass_fraction: ScalarRange,
    landmark_fraction: ScalarRange,
    resource_scale: ScalarRange,
    movable_count: IntegerRange,
    terrace_bias: ScalarRange,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingSeed {
    archetype: String,
    seed: u64,
    resident_count: usize,
    epoch: u64,
    profile_sha256: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    format: String,
    habitat_template_sha256: String,
    biosphere_template_sha256: String,
    replace_entity_ids: Vec<String>,
    mechanism_clusters: Vec<MechanismCluster>,
    movable_template_ids: Vec<String>,
    capacity: Capacity,
    geometry: Geometry,
    catchment: CatchmentSpec,
    mutation: MutationSpec,
    descriptor_normalization: DescriptorNormalization,
    developmental_rule_alleles: DevelopmentalRuleAlleleRanges,
    archetypes: Vec<Archetype>,
    training_genomes: Vec<TrainingSeed>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Variation {
    operator: String,
    seed: u64,
    recipe_sha256: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    archetype: String,
    seed: u64,
    resident_count: usize,
    dimensions_m: [f64; 3],
    region_count: usize,
    lane_count: usize,
    elevation_span_m: f64,
    loop_fraction: f64,
    shelter_fraction: f64,
    underpass_fraction: f64,
    landmark_fraction: f64,
    resource_scale: f64,
    movable_count: usize,
    terrace_bias: f64,
    developmental_rules: DevelopmentalRuleAlleles,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Genome {
    format: String,
    version: u64,
    sha256: String,
    parents: Vec<String>,
    environment_parents: Vec<String>,
    variation: Variation,
    epoch: u64,
    profile_sha256: String,
    parameters: Parameters,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentBundle {
    format: String,
    residents: Vec<ResidentFounder>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentFounder {
    body: Value,
    mobile: Value,
    exchange: Value,
    founders: FounderCompartments,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FounderCompartments {
    body: Value,
    gut: Value,
    structure: Value,
    gland: Value,
    brood: Value,
}

#[derive(Clone)]
struct Node {
    position: [f64; 3],
    half_size: [f64; 2],
    lane: usize,
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
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
    fn symmetric(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
    fn range(&mut self, range: &ScalarRange) -> f64 {
        range.min + self.unit() * (range.max - range.min)
    }
    fn integer(&mut self, range: &IntegerRange) -> usize {
        range.min + (self.next_u64() as usize % (range.max - range.min + 1))
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}
fn valid_range(value: &ScalarRange, low: f64, high: f64) -> bool {
    value.min.is_finite()
        && value.max.is_finite()
        && low <= value.min
        && value.min <= value.max
        && value.max <= high
}
fn valid_integer(value: &IntegerRange, low: usize, high: usize) -> bool {
    low <= value.min && value.min <= value.max && value.max <= high
}
fn object<'a>(value: &'a Value, label: &str) -> PyResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be an object")))
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
fn canonical<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| PyValueError::new_err(error.to_string()))
}
fn genome_hash(genome: &Genome) -> PyResult<String> {
    let mut unsigned = genome.clone();
    unsigned.sha256.clear();
    let value =
        serde_json::to_value(unsigned).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(sha256(canonical(&value)?.as_bytes()))
}
fn parse_genome(text: &str) -> PyResult<Genome> {
    let value: Genome = serde_json::from_str(text)
        .map_err(|e| PyValueError::new_err(format!("invalid environment genome: {e}")))?;
    if value.format != GENOME_FORMAT
        || value.version != 2
        || !valid_sha(&value.sha256)
        || !valid_sha(&value.profile_sha256)
        || value.parents.len() > 2
        || value.parents.iter().any(|v| !valid_sha(v))
        || value.environment_parents.len() > 2
        || value.environment_parents.iter().any(|v| !valid_sha(v))
        || genome_hash(&value)? != value.sha256
    {
        return Err(PyValueError::new_err("environment genome identity differs"));
    }
    Ok(value)
}
fn pretty<T: Serialize>(value: &T) -> PyResult<String> {
    let mut result =
        serde_json::to_string_pretty(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    result.push('\n');
    Ok(result)
}
fn set_position(value: &mut Value, position: [f64; 3]) -> PyResult<()> {
    let target = object_mut(value, "physical entity")?
        .get_mut("position")
        .and_then(Value::as_array_mut)
        .filter(|v| v.len() == 3)
        .ok_or_else(|| PyValueError::new_err("physical entity requires position[3]"))?;
    *target = position.into_iter().map(Value::from).collect();
    Ok(())
}
fn translate(value: &mut Value, delta: [f64; 3]) -> PyResult<()> {
    let map = object_mut(value, "physical entity")?;
    let target = map
        .get_mut("position")
        .and_then(Value::as_array_mut)
        .filter(|v| v.len() == 3)
        .ok_or_else(|| PyValueError::new_err("physical entity requires position[3]"))?;
    for axis in 0..3 {
        target[axis] = Value::from(
            target[axis]
                .as_f64()
                .ok_or_else(|| PyValueError::new_err("position must be numeric"))?
                + delta[axis],
        );
    }
    Ok(())
}
fn entity_index(values: &[Value]) -> HashMap<String, usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| id(v).map(|s| (s.to_owned(), i)))
        .collect()
}
fn ramp_quaternion(dx: f64, dy: f64, dz: f64) -> [f64; 4] {
    let yaw = dy.atan2(dx);
    let pitch = -dz.atan2((dx * dx + dy * dy).sqrt());
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    [cy * cp, -sy * sp, cy * sp, sy * cp]
}

fn validate_config(config: &Config) -> PyResult<()> {
    if config.format != FORMAT
        || !valid_sha(&config.habitat_template_sha256)
        || !valid_sha(&config.biosphere_template_sha256)
        || !valid_integer(&config.capacity.residents, 1, MAX_RESIDENTS)
        || !valid_integer(&config.capacity.regions, 6, MAX_REGIONS)
        || !valid_range(&config.capacity.width_m, 8.0, 30.0)
        || !valid_range(&config.capacity.height_m, 6.0, 24.0)
        || !valid_range(&config.capacity.depth_m, 3.0, 8.0)
        || config.replace_entity_ids.is_empty()
        || config.replace_entity_ids.len() > 96
        || config.mechanism_clusters.is_empty()
        || config.mechanism_clusters.len() > 16
        || config.movable_template_ids.is_empty()
        || !valid_range(&config.geometry.platform_half_size_m, 0.45, 2.5)
        || !valid_range(&config.geometry.ramp_width_m, 0.45, 2.5)
        || !(0.8..=2.5).contains(&config.geometry.margin_m)
        || !(0.03..=0.2).contains(&config.geometry.platform_thickness_m)
        || !(0.08..=0.4).contains(&config.geometry.maximum_rise_over_run)
        || !(0.35..=1.8).contains(&config.geometry.minimum_underpass_clearance_m)
        || !(0.24..=1.5).contains(&config.geometry.minimum_spawn_clearance_m)
        || !(0.2..=2.0).contains(&config.geometry.boundary_height_m)
        || !(0.03..=0.25).contains(&config.geometry.boundary_thickness_m)
        || !identifier(&config.catchment.id)
        || !(-2.5..=-0.2).contains(&config.catchment.surface_height_m)
        || !(2.0..=12.0).contains(&config.catchment.margin_m)
        || !(0.03..=0.25).contains(&config.catchment.thickness_m)
        || !(0.2..=2.5).contains(&config.catchment.wall_height_m)
        || !identifier(&config.catchment.material)
        || !identifier(&config.catchment.physical_material)
        || config.mutation.operator != "bounded-regional-perturbation-v2"
        || !valid_sha(&config.mutation.recipe_sha256)
        || !(0.0..=0.3).contains(&config.mutation.scalar_fraction)
        || config.mutation.integer_delta > 8
        || config.mutation.resident_delta > 8
        || !config
            .descriptor_normalization
            .resource_mass_per_square_meter
            .is_finite()
        || config
            .descriptor_normalization
            .resource_mass_per_square_meter
            <= 0.0
        || !config
            .descriptor_normalization
            .renewal_capacity_per_square_meter_second
            .is_finite()
        || config
            .descriptor_normalization
            .renewal_capacity_per_square_meter_second
            <= 0.0
        || !config
            .descriptor_normalization
            .cycle_rank_per_region
            .is_finite()
        || config.descriptor_normalization.cycle_rank_per_region <= 0.0
        || !(1..=64).contains(&config.descriptor_normalization.resident_physical_geoms)
        || config
            .descriptor_normalization
            .generation_cost_limits
            .physical_geoms
            < config.capacity.residents.max
                * config.descriptor_normalization.resident_physical_geoms
        || config
            .descriptor_normalization
            .generation_cost_limits
            .regions
            < config.capacity.regions.max
        || config.descriptor_normalization.generation_cost_limits.edges < MAX_EDGES
        || config
            .descriptor_normalization
            .generation_cost_limits
            .movables
            < 24
        || config
            .descriptor_normalization
            .generation_cost_limits
            .compartments
            == 0
        || allele_ranges(&config.developmental_rule_alleles)
            .iter()
            .any(|range| !valid_range(range, 0.5, 1.5))
        || config.archetypes.len() < 3
        || config.archetypes.len() > 12
        || config.training_genomes.is_empty()
        || config.training_genomes.len() > 256
    {
        return Err(PyValueError::new_err(
            "invalid regional habitat-family configuration",
        ));
    }
    let mut ids = HashSet::new();
    for value in config
        .replace_entity_ids
        .iter()
        .chain(
            config
                .mechanism_clusters
                .iter()
                .flat_map(|v| v.entity_ids.iter()),
        )
        .chain(&config.movable_template_ids)
    {
        if !identifier(value) {
            return Err(PyValueError::new_err("invalid configured entity id"));
        }
    }
    if config.mechanism_clusters.iter().any(|cluster| {
        cluster.entity_ids.is_empty()
            || cluster.entity_ids.len() > 12
            || cluster.anchor_m.iter().any(|v| !v.is_finite())
    }) {
        return Err(PyValueError::new_err("invalid mechanism cluster"));
    }
    for a in &config.archetypes {
        let resident_capacity_regions =
            (config.capacity.residents.max + 3) / 4 + config.mechanism_clusters.len() + 4;
        if !identifier(&a.id)
            || !ids.insert(a.id.clone())
            || !valid_integer(
                &a.region_count,
                config.capacity.regions.min,
                config.capacity.regions.max,
            )
            || !valid_integer(&a.lane_count, 2, 6)
            || !valid_range(
                &a.width_m,
                config.capacity.width_m.min,
                config.capacity.width_m.max,
            )
            || !valid_range(
                &a.height_m,
                config.capacity.height_m.min,
                config.capacity.height_m.max,
            )
            || !valid_range(
                &a.depth_m,
                config.capacity.depth_m.min,
                config.capacity.depth_m.max,
            )
            || !valid_range(&a.elevation_span_m, 0.15, 2.4)
            || !valid_range(&a.loop_fraction, 0.0, 0.8)
            || !valid_range(&a.shelter_fraction, 0.0, 0.8)
            || !valid_range(&a.underpass_fraction, 0.0, 0.8)
            || !valid_range(&a.landmark_fraction, 0.0, 0.8)
            || !valid_range(&a.resource_scale, 0.5, 1.6)
            || !valid_integer(&a.movable_count, 1, 24)
            || !valid_range(&a.terrace_bias, -1.0, 1.0)
            || a.region_count.max < resident_capacity_regions
        {
            return Err(PyValueError::new_err("invalid regional archetype"));
        }
    }
    let mut variants = HashSet::new();
    for t in &config.training_genomes {
        if !ids.contains(&t.archetype)
            || !valid_sha(&t.profile_sha256)
            || t.resident_count < config.capacity.residents.min
            || t.resident_count > config.capacity.residents.max
            || !variants.insert((t.archetype.clone(), t.seed, t.resident_count))
        {
            return Err(PyValueError::new_err("invalid regional training genome"));
        }
    }
    Ok(())
}

fn clamp(value: f64, range: &ScalarRange) -> f64 {
    value.max(range.min).min(range.max)
}
fn perturb(value: f64, range: &ScalarRange, fraction: f64, rng: &mut SplitMix64) -> f64 {
    clamp(
        value + rng.symmetric() * (range.max - range.min) * fraction,
        range,
    )
}
fn perturb_integer(
    value: usize,
    range: &IntegerRange,
    delta: usize,
    rng: &mut SplitMix64,
) -> usize {
    let d = (rng.next_u64() % (2 * delta as u64 + 1)) as isize - delta as isize;
    (value as isize + d).clamp(range.min as isize, range.max as isize) as usize
}

fn allele_ranges(value: &DevelopmentalRuleAlleleRanges) -> [&ScalarRange; 11] {
    [
        &value.vertical_weight,
        &value.clearance_weight,
        &value.surface_weight,
        &value.surface_reach,
        &value.clearance_distance,
        &value.conductivity,
        &value.half_resistance,
        &value.response_light,
        &value.response_nutrient,
        &value.response_support,
        &value.response_competition,
    ]
}

fn allele_values(value: &DevelopmentalRuleAlleles) -> [f64; 11] {
    [
        value.vertical_weight,
        value.clearance_weight,
        value.surface_weight,
        value.surface_reach,
        value.clearance_distance,
        value.conductivity,
        value.half_resistance,
        value.response_light,
        value.response_nutrient,
        value.response_support,
        value.response_competition,
    ]
}

fn sample_developmental_rules(
    ranges: &DevelopmentalRuleAlleleRanges,
    rng: &mut SplitMix64,
) -> DevelopmentalRuleAlleles {
    DevelopmentalRuleAlleles {
        vertical_weight: rng.range(&ranges.vertical_weight),
        clearance_weight: rng.range(&ranges.clearance_weight),
        surface_weight: rng.range(&ranges.surface_weight),
        surface_reach: rng.range(&ranges.surface_reach),
        clearance_distance: rng.range(&ranges.clearance_distance),
        conductivity: rng.range(&ranges.conductivity),
        half_resistance: rng.range(&ranges.half_resistance),
        response_light: rng.range(&ranges.response_light),
        response_nutrient: rng.range(&ranges.response_nutrient),
        response_support: rng.range(&ranges.response_support),
        response_competition: rng.range(&ranges.response_competition),
    }
}

fn mutate_developmental_rules(
    value: &DevelopmentalRuleAlleles,
    ranges: &DevelopmentalRuleAlleleRanges,
    fraction: f64,
    rng: &mut SplitMix64,
) -> DevelopmentalRuleAlleles {
    DevelopmentalRuleAlleles {
        vertical_weight: perturb(
            value.vertical_weight,
            &ranges.vertical_weight,
            fraction,
            rng,
        ),
        clearance_weight: perturb(
            value.clearance_weight,
            &ranges.clearance_weight,
            fraction,
            rng,
        ),
        surface_weight: perturb(value.surface_weight, &ranges.surface_weight, fraction, rng),
        surface_reach: perturb(value.surface_reach, &ranges.surface_reach, fraction, rng),
        clearance_distance: perturb(
            value.clearance_distance,
            &ranges.clearance_distance,
            fraction,
            rng,
        ),
        conductivity: perturb(value.conductivity, &ranges.conductivity, fraction, rng),
        half_resistance: perturb(
            value.half_resistance,
            &ranges.half_resistance,
            fraction,
            rng,
        ),
        response_light: perturb(value.response_light, &ranges.response_light, fraction, rng),
        response_nutrient: perturb(
            value.response_nutrient,
            &ranges.response_nutrient,
            fraction,
            rng,
        ),
        response_support: perturb(
            value.response_support,
            &ranges.response_support,
            fraction,
            rng,
        ),
        response_competition: perturb(
            value.response_competition,
            &ranges.response_competition,
            fraction,
            rng,
        ),
    }
}

fn platform_entity(index: usize, node: &Node, g: &Geometry, material: &str) -> Value {
    let mut shapes = vec![
        json!({"type":"box","size":[node.half_size[0],node.half_size[1],g.platform_thickness_m*0.5],"position":[0,0,node.position[2]-g.platform_thickness_m*0.5]}),
    ];
    if node.position[2] > g.platform_thickness_m + 0.12 {
        let h = (node.position[2] - g.platform_thickness_m) * 0.5;
        for x in [-0.78, 0.78] {
            for y in [-0.72, 0.72] {
                shapes.push(json!({"type":"box","size":[0.05,0.05,h],"position":[x*node.half_size[0],y*node.half_size[1],h]}));
            }
        }
    }
    json!({"id":format!("region-platform-{index:02}"),"mobility":"static","material":material,"physical_material":"masonry","position":[node.position[0],node.position[1],0],"shapes":shapes,"components":[]})
}
fn ramp_entity(
    index: usize,
    a: &Node,
    b: &Node,
    width: f64,
    g: &Geometry,
    material: &str,
) -> Value {
    let dx = b.position[0] - a.position[0];
    let dy = b.position[1] - a.position[1];
    let dz = b.position[2] - a.position[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    json!({"id":format!("region-ramp-{index:02}"),"mobility":"static","material":material,"physical_material":"masonry","position":[(a.position[0]+b.position[0])*0.5,(a.position[1]+b.position[1])*0.5,(a.position[2]+b.position[2])*0.5-g.platform_thickness_m*0.5],"quaternion":ramp_quaternion(dx,dy,dz),"shapes":[{"type":"box","size":[len*0.5,width*0.5,g.platform_thickness_m*0.5]}],"components":[]})
}
fn canopy_entity(index: usize, node: &Node, clearance: f64) -> Value {
    let roof = node.position[2] + clearance;
    json!({"id":format!("region-shelter-{index:02}"),"mobility":"static","material":"leaf","physical_material":"timber","position":[node.position[0],node.position[1],0],"shapes":[
      {"type":"box","size":[0.05,0.05,clearance*0.5],"position":[-node.half_size[0]*0.72,0,node.position[2]+clearance*0.5]},
      {"type":"box","size":[0.05,0.05,clearance*0.5],"position":[node.half_size[0]*0.72,0,node.position[2]+clearance*0.5]},
      {"type":"box","size":[node.half_size[0]*0.82,node.half_size[1]*0.48,0.04],"position":[0,0,roof]}],"components":[{"type":"shade","radius":node.half_size[0],"strength":0.55}]})
}
fn landmark_entity(index: usize, node: &Node, material: &str) -> Value {
    json!({"id":format!("region-landmark-{index:02}"),"mobility":"static","material":material,"physical_material":"timber","position":[node.position[0],node.position[1],0],"shapes":[{"type":"cylinder","size":[0.055,0.3],"position":[node.half_size[0]*0.65,0,node.position[2]+0.3]}],"components":[]})
}
fn boundary_entities(width: f64, height: f64, g: &Geometry) -> Vec<Value> {
    vec![
        json!({"id":"region-ground","mobility":"static","material":"loam","physical_material":"earth","position":[width*0.5,height*0.5,-0.04],"shapes":[{"type":"box","size":[width*0.5,height*0.5,0.04]}],"components":[]}),
        json!({"id":"region-west-wall","mobility":"static","material":"darkstone","physical_material":"masonry","position":[0,height*0.5,g.boundary_height_m*0.5],"shapes":[{"type":"box","size":[g.boundary_thickness_m*0.5,height*0.5,g.boundary_height_m*0.5]}],"components":[]}),
        json!({"id":"region-east-wall","mobility":"static","material":"darkstone","physical_material":"masonry","position":[width,height*0.5,g.boundary_height_m*0.5],"shapes":[{"type":"box","size":[g.boundary_thickness_m*0.5,height*0.5,g.boundary_height_m*0.5]}],"components":[]}),
        json!({"id":"region-south-wall","mobility":"static","material":"darkstone","physical_material":"masonry","position":[width*0.5,0,g.boundary_height_m*0.5],"shapes":[{"type":"box","size":[width*0.5,g.boundary_thickness_m*0.5,g.boundary_height_m*0.5]}],"components":[]}),
        json!({"id":"region-north-wall","mobility":"static","material":"darkstone","physical_material":"masonry","position":[width*0.5,height,g.boundary_height_m*0.5],"shapes":[{"type":"box","size":[width*0.5,g.boundary_thickness_m*0.5,g.boundary_height_m*0.5]}],"components":[]}),
    ]
}
fn catchment_entity(width: f64, height: f64, c: &CatchmentSpec) -> Value {
    let hx = width * 0.5 + c.margin_m;
    let hy = height * 0.5 + c.margin_m;
    let x = width * 0.5;
    let y = height * 0.5;
    let wh = c.wall_height_m * 0.5;
    let t = c.thickness_m * 0.5;
    json!({"id":c.id,"mobility":"static","material":c.material,"physical_material":c.physical_material,"position":[x,y,0],"shapes":[
      {"type":"box","size":[hx,hy,t],"position":[0,0,c.surface_height_m-t]},
      {"type":"box","size":[t,hy,wh],"position":[-hx,0,c.surface_height_m+wh]},
      {"type":"box","size":[t,hy,wh],"position":[hx,0,c.surface_height_m+wh]},
      {"type":"box","size":[hx,t,wh],"position":[0,-hy,c.surface_height_m+wh]},
      {"type":"box","size":[hx,t,wh],"position":[0,hy,c.surface_height_m+wh]}],"components":[]})
}

fn remap_rows(value: &mut Value, map: &HashMap<usize, usize>) -> PyResult<()> {
    match value {
        Value::Object(items) => {
            for (key, nested) in items {
                if (key == "row" || key.ends_with("_row")) && nested.is_u64() {
                    let old = nested.as_u64().unwrap() as usize;
                    if let Some(new) = map.get(&old) {
                        *nested = Value::from(*new);
                    }
                } else {
                    remap_rows(nested, map)?;
                }
            }
        }
        Value::Array(items) => {
            for nested in items {
                remap_rows(nested, map)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn validate_compartment(value: &Value) -> PyResult<()> {
    let keys: HashSet<&str> = object(value, "founder compartment")?
        .keys()
        .map(String::as_str)
        .collect();
    if keys != HashSet::from(["enzymes", "pools", "atp", "atp_capacity"]) {
        return Err(PyValueError::new_err("founder compartment fields differ"));
    }
    Ok(())
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
            return Err(PyValueError::new_err("regional configuration hash differs"));
        }
        let config: Config = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid regional configuration: {e}")))?;
        validate_config(&config)?;
        Ok(Self {
            config,
            config_sha256,
        })
    }
    fn archetypes(&self) -> Vec<String> {
        self.config
            .archetypes
            .iter()
            .map(|v| v.id.clone())
            .collect()
    }
    fn training_genomes(&self) -> Vec<(String, u64, usize, u64, String)> {
        self.config
            .training_genomes
            .iter()
            .map(|v| {
                (
                    v.archetype.clone(),
                    v.seed,
                    v.resident_count,
                    v.epoch,
                    v.profile_sha256.clone(),
                )
            })
            .collect()
    }
    fn initial_genome(
        &self,
        seed: u64,
        archetype: String,
        resident_count: usize,
        profile_sha256: String,
        epoch: u64,
    ) -> PyResult<String> {
        if !valid_sha(&profile_sha256)
            || resident_count < self.config.capacity.residents.min
            || resident_count > self.config.capacity.residents.max
        {
            return Err(PyValueError::new_err(
                "invalid environment founder identity",
            ));
        }
        let a = self
            .config
            .archetypes
            .iter()
            .find(|v| v.id == archetype)
            .ok_or_else(|| PyValueError::new_err("unknown regional archetype"))?;
        let mut rng = SplitMix64::new(
            seed ^ archetype
                .bytes()
                .fold(0u64, |x, b| x.wrapping_mul(131).wrapping_add(b as u64)),
        );
        let minimum_regions = (resident_count + 3) / 4 + self.config.mechanism_clusters.len() + 4;
        let parameters = Parameters {
            archetype,
            seed,
            resident_count,
            dimensions_m: [
                rng.range(&a.width_m),
                rng.range(&a.height_m),
                rng.range(&a.depth_m),
            ],
            region_count: rng.integer(&a.region_count).max(minimum_regions),
            lane_count: rng.integer(&a.lane_count),
            elevation_span_m: rng.range(&a.elevation_span_m),
            loop_fraction: rng.range(&a.loop_fraction),
            shelter_fraction: rng.range(&a.shelter_fraction),
            underpass_fraction: rng.range(&a.underpass_fraction),
            landmark_fraction: rng.range(&a.landmark_fraction),
            resource_scale: rng.range(&a.resource_scale),
            movable_count: rng.integer(&a.movable_count),
            terrace_bias: rng.range(&a.terrace_bias),
            developmental_rules: sample_developmental_rules(
                &self.config.developmental_rule_alleles,
                &mut rng,
            ),
        };
        let mut genome = Genome {
            format: GENOME_FORMAT.into(),
            version: 2,
            sha256: String::new(),
            parents: vec![],
            environment_parents: vec![],
            variation: Variation {
                operator: "initial-regional-sample-v2".into(),
                seed,
                recipe_sha256: self.config.mutation.recipe_sha256.clone(),
            },
            epoch,
            profile_sha256,
            parameters,
        };
        self.validate_parameters(&genome.parameters)?;
        genome.sha256 = genome_hash(&genome)?;
        pretty(&genome)
    }
    fn mutate_genome(
        &self,
        parent_json: String,
        parent_environment_record_sha256: String,
        variation_seed: u64,
    ) -> PyResult<String> {
        let parent = parse_genome(&parent_json)?;
        if !valid_sha(&parent_environment_record_sha256) {
            return Err(PyValueError::new_err(
                "parent environment record identity is invalid",
            ));
        }
        self.validate_parameters(&parent.parameters)?;
        let a = self
            .config
            .archetypes
            .iter()
            .find(|v| v.id == parent.parameters.archetype)
            .ok_or_else(|| PyValueError::new_err("unknown parent archetype"))?;
        let mut rng = SplitMix64::new(
            variation_seed ^ u64::from_str_radix(&parent.sha256[..16], 16).unwrap(),
        );
        let mut p = parent.parameters.clone();
        p.seed = variation_seed;
        p.resident_count = perturb_integer(
            p.resident_count,
            &self.config.capacity.residents,
            self.config.mutation.resident_delta,
            &mut rng,
        );
        p.dimensions_m = [
            perturb(
                p.dimensions_m[0],
                &a.width_m,
                self.config.mutation.scalar_fraction,
                &mut rng,
            ),
            perturb(
                p.dimensions_m[1],
                &a.height_m,
                self.config.mutation.scalar_fraction,
                &mut rng,
            ),
            perturb(
                p.dimensions_m[2],
                &a.depth_m,
                self.config.mutation.scalar_fraction,
                &mut rng,
            ),
        ];
        p.region_count = perturb_integer(
            p.region_count,
            &a.region_count,
            self.config.mutation.integer_delta,
            &mut rng,
        );
        p.region_count = p
            .region_count
            .max((p.resident_count + 3) / 4 + self.config.mechanism_clusters.len() + 4);
        p.lane_count = perturb_integer(p.lane_count, &a.lane_count, 1, &mut rng);
        p.elevation_span_m = perturb(
            p.elevation_span_m,
            &a.elevation_span_m,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.loop_fraction = perturb(
            p.loop_fraction,
            &a.loop_fraction,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.shelter_fraction = perturb(
            p.shelter_fraction,
            &a.shelter_fraction,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.underpass_fraction = perturb(
            p.underpass_fraction,
            &a.underpass_fraction,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.landmark_fraction = perturb(
            p.landmark_fraction,
            &a.landmark_fraction,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.resource_scale = perturb(
            p.resource_scale,
            &a.resource_scale,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.movable_count = perturb_integer(
            p.movable_count,
            &a.movable_count,
            self.config.mutation.integer_delta,
            &mut rng,
        );
        p.terrace_bias = perturb(
            p.terrace_bias,
            &a.terrace_bias,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        p.developmental_rules = mutate_developmental_rules(
            &p.developmental_rules,
            &self.config.developmental_rule_alleles,
            self.config.mutation.scalar_fraction,
            &mut rng,
        );
        let mut genome = Genome {
            format: GENOME_FORMAT.into(),
            version: 2,
            sha256: String::new(),
            parents: vec![parent.sha256],
            environment_parents: vec![parent_environment_record_sha256],
            variation: Variation {
                operator: self.config.mutation.operator.clone(),
                seed: variation_seed,
                recipe_sha256: self.config.mutation.recipe_sha256.clone(),
            },
            epoch: parent.epoch,
            profile_sha256: parent.profile_sha256,
            parameters: p,
        };
        self.validate_parameters(&genome.parameters)?;
        genome.sha256 = genome_hash(&genome)?;
        pretty(&genome)
    }
    fn generate(
        &self,
        habitat_json: String,
        biosphere_json: String,
        genome_json: String,
        residents_json: String,
    ) -> PyResult<(String, String, String)> {
        if sha256(habitat_json.as_bytes()) != self.config.habitat_template_sha256
            || sha256(biosphere_json.as_bytes()) != self.config.biosphere_template_sha256
        {
            return Err(PyValueError::new_err(
                "regional source template hash differs",
            ));
        }
        let genome = parse_genome(&genome_json)?;
        self.validate_parameters(&genome.parameters)?;
        let residents: ResidentBundle = serde_json::from_str(&residents_json)
            .map_err(|e| PyValueError::new_err(format!("invalid resident bundle: {e}")))?;
        if residents.format != RESIDENT_FORMAT
            || residents.residents.len() != genome.parameters.resident_count
        {
            return Err(PyValueError::new_err(
                "resident bundle count differs from environment genome",
            ));
        }
        self.generate_values(habitat_json, biosphere_json, genome, residents)
    }
}

impl HabitatFamily {
    fn validate_parameters(&self, p: &Parameters) -> PyResult<()> {
        let a = self
            .config
            .archetypes
            .iter()
            .find(|v| v.id == p.archetype)
            .ok_or_else(|| PyValueError::new_err("unknown regional archetype"))?;
        let minimum_regions = (p.resident_count + 3) / 4 + self.config.mechanism_clusters.len() + 4;
        if p.resident_count < self.config.capacity.residents.min
            || p.resident_count > self.config.capacity.residents.max
            || p.region_count < a.region_count.min
            || p.region_count > a.region_count.max
            || p.region_count < minimum_regions
            || p.lane_count < a.lane_count.min
            || p.lane_count > a.lane_count.max
            || !p.dimensions_m.iter().all(|v| v.is_finite())
            || !(a.width_m.min..=a.width_m.max).contains(&p.dimensions_m[0])
            || !(a.height_m.min..=a.height_m.max).contains(&p.dimensions_m[1])
            || !(a.depth_m.min..=a.depth_m.max).contains(&p.dimensions_m[2])
            || !(a.elevation_span_m.min..=a.elevation_span_m.max).contains(&p.elevation_span_m)
            || !(a.loop_fraction.min..=a.loop_fraction.max).contains(&p.loop_fraction)
            || !(a.shelter_fraction.min..=a.shelter_fraction.max).contains(&p.shelter_fraction)
            || !(a.underpass_fraction.min..=a.underpass_fraction.max)
                .contains(&p.underpass_fraction)
            || !(a.landmark_fraction.min..=a.landmark_fraction.max).contains(&p.landmark_fraction)
            || !(a.resource_scale.min..=a.resource_scale.max).contains(&p.resource_scale)
            || p.movable_count < a.movable_count.min
            || p.movable_count > a.movable_count.max
            || !(a.terrace_bias.min..=a.terrace_bias.max).contains(&p.terrace_bias)
            || allele_values(&p.developmental_rules)
                .iter()
                .zip(allele_ranges(&self.config.developmental_rule_alleles))
                .any(|(value, range)| {
                    !value.is_finite() || !(range.min..=range.max).contains(value)
                })
        {
            return Err(PyValueError::new_err(
                "environment genome parameters exceed archetype",
            ));
        }
        Ok(())
    }

    fn generate_values(
        &self,
        habitat_json: String,
        biosphere_json: String,
        genome: Genome,
        residents: ResidentBundle,
    ) -> PyResult<(String, String, String)> {
        let p = &genome.parameters;
        let width = p.dimensions_m[0];
        let height = p.dimensions_m[1];
        let mut rng = SplitMix64::new(p.seed ^ 0xC6A4A7935BD1E995);
        let columns = (p.region_count + p.lane_count - 1) / p.lane_count;
        let margin = self.config.geometry.margin_m;
        let usable_x = width - 2.0 * margin;
        let usable_y = height - 2.0 * margin;
        if columns < 2 || usable_x <= 0.0 || usable_y <= 0.0 {
            return Err(PyValueError::new_err(
                "regional dimensions cannot place connected graph",
            ));
        }
        let mut nodes = Vec::with_capacity(p.region_count);
        for index in 0..p.region_count {
            let lane = index % p.lane_count;
            let column = index / p.lane_count;
            let xn = column as f64 / (columns - 1) as f64;
            let yn = (lane as f64 + 0.5) / p.lane_count as f64;
            let x = margin + xn * usable_x + rng.symmetric() * (usable_x / columns as f64) * 0.12;
            let y =
                margin + yn * usable_y + rng.symmetric() * (usable_y / p.lane_count as f64) * 0.12;
            let wave = ((xn * std::f64::consts::PI * 2.0) + (lane as f64 * 0.73)).sin() * 0.18;
            let terrace = (xn * 3.0 + p.terrace_bias).floor() / 3.0;
            let z = 0.1
                + p.elevation_span_m
                    * (0.58 * terrace.max(0.0) + wave + 0.16 * (lane % 2) as f64).clamp(0.0, 1.0);
            let size = rng.range(&self.config.geometry.platform_half_size_m);
            nodes.push(Node {
                position: [x, y, z],
                half_size: [
                    size,
                    (size * (0.78 + 0.2 * rng.unit()))
                        .min(self.config.geometry.platform_half_size_m.max),
                ],
                lane,
            });
        }
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for lane in 0..p.lane_count {
            let lane_nodes: Vec<usize> = (0..p.region_count)
                .filter(|i| i % p.lane_count == lane)
                .collect();
            for pair in lane_nodes.windows(2) {
                edges.push((pair[0], pair[1]));
            }
        }
        for column in 0..columns {
            let base = column * p.lane_count;
            let end = (base + p.lane_count).min(p.region_count);
            for i in base..end.saturating_sub(1) {
                if (column + i) % 2 == 0 || rng.unit() < 0.55 {
                    edges.push((i, i + 1));
                }
            }
        }
        // Connect any component to its nearest reached node, then add bounded loops.
        loop {
            let reached = reachable(p.region_count, &edges);
            if reached.iter().all(|v| *v) {
                break;
            }
            let b = reached.iter().position(|v| !*v).unwrap();
            let (a, _) = nodes
                .iter()
                .enumerate()
                .filter(|(i, _)| reached[*i])
                .filter(|(i, _)| {
                    segment_clears_other_nodes(
                        *i,
                        b,
                        &nodes,
                        self.config.geometry.ramp_width_m.max * 0.5 + 0.15,
                    )
                })
                .map(|(i, n)| (i, dist2(n, &nodes[b])))
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .ok_or_else(|| {
                    PyValueError::new_err("regional graph cannot connect without crossing a region")
                })?;
            edges.push((a, b));
        }
        let possible = (p.region_count * (p.region_count - 1) / 2).min(MAX_EDGES);
        let target = (edges.len()
            + ((possible - edges.len()) as f64 * p.loop_fraction * 0.12) as usize)
            .min(MAX_EDGES);
        let mut loop_attempts = 0usize;
        while edges.len() < target && loop_attempts < MAX_EDGES * 8 {
            loop_attempts += 1;
            let a = rng.next_u64() as usize % p.region_count;
            let b = rng.next_u64() as usize % p.region_count;
            if a != b
                && !edges
                    .iter()
                    .any(|(x, y)| (*x == a && *y == b) || (*x == b && *y == a))
                && dist2(&nodes[a], &nodes[b]).sqrt() < width.max(height) * 0.42
                && segment_clears_other_nodes(
                    a,
                    b,
                    &nodes,
                    self.config.geometry.ramp_width_m.max * 0.5 + 0.15,
                )
            {
                edges.push((a, b));
            }
        }
        // Relax adjacent elevations until every physical ramp is within slope.
        for _ in 0..24 {
            let mut changed = false;
            for &(a, b) in &edges {
                let dx = nodes[b].position[0] - nodes[a].position[0];
                let dy = nodes[b].position[1] - nodes[a].position[1];
                let run = (dx * dx + dy * dy).sqrt();
                let limit = run * self.config.geometry.maximum_rise_over_run * 0.88;
                let dz = nodes[b].position[2] - nodes[a].position[2];
                if dz.abs() > limit {
                    let excess = (dz.abs() - limit) * 0.5;
                    let sign = dz.signum();
                    nodes[a].position[2] += sign * excess;
                    nodes[b].position[2] -= sign * excess;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let underpasses: BTreeSet<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.position[2]
                    >= self.config.geometry.minimum_underpass_clearance_m
                        + self.config.geometry.platform_thickness_m
            })
            .filter(|_| rng.unit() < p.underpass_fraction)
            .map(|(i, _)| i)
            .collect();
        let mechanism_nodes: BTreeSet<usize> = (0..self.config.mechanism_clusters.len())
            .map(|i| (i * 7 + 3) % nodes.len())
            .collect();
        let ecology_nodes: Vec<usize> = (0..nodes.len())
            .filter(|i| !mechanism_nodes.contains(i))
            .collect();
        let spawn_node_count = (p.resident_count + 3) / 4;
        if ecology_nodes.len() < spawn_node_count {
            return Err(PyValueError::new_err(
                "regional graph lacks collision-clear resident regions",
            ));
        }
        let spawn_nodes: Vec<usize> = ecology_nodes[..spawn_node_count].to_vec();
        let spawn_node_set: BTreeSet<usize> = spawn_nodes.iter().copied().collect();
        let content_nodes: Vec<usize> = ecology_nodes
            .iter()
            .copied()
            .filter(|i| !spawn_node_set.contains(i))
            .collect();
        if content_nodes.len() < 4 {
            return Err(PyValueError::new_err(
                "regional graph lacks structure-free ecology regions",
            ));
        }
        let shelters: BTreeSet<usize> = (0..p.region_count)
            .filter(|i| !mechanism_nodes.contains(i) && !spawn_node_set.contains(i))
            .filter(|_| rng.unit() < p.shelter_fraction)
            .collect();
        let landmarks: BTreeSet<usize> = (0..p.region_count)
            .filter(|i| !mechanism_nodes.contains(i) && !spawn_node_set.contains(i))
            .filter(|_| rng.unit() < p.landmark_fraction)
            .collect();

        let mut habitat: Value = serde_json::from_str(&habitat_json)
            .map_err(|e| PyValueError::new_err(format!("invalid habitat template: {e}")))?;
        let mut biosphere: Value = serde_json::from_str(&biosphere_json)
            .map_err(|e| PyValueError::new_err(format!("invalid biosphere template: {e}")))?;
        apply_developmental_rules(&mut biosphere, &p.developmental_rules)?;
        object_mut(&mut habitat, "habitat")?.insert("size".into(), json!(p.dimensions_m));
        let entities = array_mut(&mut habitat, "entities")?;
        let existing: HashSet<String> = entities.iter().filter_map(id).map(str::to_owned).collect();
        if self
            .config
            .replace_entity_ids
            .iter()
            .any(|v| !existing.contains(v))
            || self
                .config
                .mechanism_clusters
                .iter()
                .flat_map(|v| v.entity_ids.iter())
                .any(|v| !existing.contains(v))
            || self
                .config
                .movable_template_ids
                .iter()
                .any(|v| !existing.contains(v))
            || existing.contains(&self.config.catchment.id)
        {
            return Err(PyValueError::new_err(
                "regional template identities are missing or duplicated",
            ));
        }
        let solar_components = entities
            .iter()
            .find(|value| id(value) == Some("ground"))
            .and_then(Value::as_object)
            .and_then(|value| value.get("components"))
            .and_then(Value::as_array)
            .filter(|components| {
                components.iter().any(|component| {
                    component.get("type").and_then(Value::as_str) == Some("light")
                        && component.get("directional").and_then(Value::as_bool) == Some(true)
                })
            })
            .cloned()
            .ok_or_else(|| PyValueError::new_err("regional template solar light is missing"))?;
        let replacement: HashSet<&str> = self
            .config
            .replace_entity_ids
            .iter()
            .map(String::as_str)
            .collect();
        entities.retain(|v| id(v).is_none_or(|name| !replacement.contains(name)));
        entities.extend(boundary_entities(width, height, &self.config.geometry));
        let ground = entities
            .iter_mut()
            .find(|value| id(value) == Some("region-ground"))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| PyValueError::new_err("generated regional ground is missing"))?;
        ground.insert("components".into(), Value::Array(solar_components));
        entities.push(catchment_entity(width, height, &self.config.catchment));
        let mats = ["terracotta", "limestone", "darkstone"];
        for (i, node) in nodes.iter().enumerate() {
            entities.push(platform_entity(
                i,
                node,
                &self.config.geometry,
                mats[(i + rng.next_u64() as usize) % 3],
            ));
        }
        let mut edge_meta = Vec::new();
        for (i, &(a, b)) in edges.iter().enumerate() {
            let dx = nodes[b].position[0] - nodes[a].position[0];
            let dy = nodes[b].position[1] - nodes[a].position[1];
            let dz = nodes[b].position[2] - nodes[a].position[2];
            let run = (dx * dx + dy * dy).sqrt();
            let slope = dz.abs() / run;
            if slope > self.config.geometry.maximum_rise_over_run {
                return Err(PyValueError::new_err("generated ramp exceeds slope bound"));
            }
            let rw = self.config.geometry.ramp_width_m.min
                + rng.unit()
                    * (self.config.geometry.ramp_width_m.max
                        - self.config.geometry.ramp_width_m.min);
            entities.push(ramp_entity(
                i,
                &nodes[a],
                &nodes[b],
                rw,
                &self.config.geometry,
                mats[(i + 1) % 3],
            ));
            edge_meta.push(json!({"id":format!("region-ramp-{i:02}"),"nodes":[a,b],"run_m":run,"rise_m":dz,"rise_over_run":slope,"width_m":rw}));
        }
        for &i in &shelters {
            entities.push(canopy_entity(
                i,
                &nodes[i],
                self.config.geometry.minimum_underpass_clearance_m,
            ));
        }
        for &i in &landmarks {
            entities.push(landmark_entity(
                i,
                &nodes[i],
                if i % 2 == 0 { "violet" } else { "cyan" },
            ));
        }
        let indices = entity_index(entities);
        // Relocate physical mechanisms as coherent clusters around graph regions.
        for (offset, cluster) in self.config.mechanism_clusters.iter().enumerate() {
            let n = &nodes[(offset * 7 + 3) % nodes.len()];
            let delta = [
                n.position[0] - cluster.anchor_m[0],
                n.position[1] - cluster.anchor_m[1],
                n.position[2] - cluster.anchor_m[2],
            ];
            for name in &cluster.entity_ids {
                let idx = *indices
                    .get(name)
                    .ok_or_else(|| PyValueError::new_err("mechanism identity vanished"))?;
                translate(&mut entities[idx], delta)?;
            }
        }
        // Retain a bounded set of template movable objects and clone additional
        // construction pieces from those same physical definitions.
        let movable_templates: Vec<Value> = self
            .config
            .movable_template_ids
            .iter()
            .map(|name| entities[*indices.get(name).unwrap()].clone())
            .collect();
        for (i, name) in self.config.movable_template_ids.iter().enumerate() {
            if i >= p.movable_count {
                entities.retain(|v| id(v) != Some(name));
            } else {
                let n = &nodes[content_nodes[(i * 5 + 1) % content_nodes.len()]];
                let idx = entity_index(entities)[name];
                set_position(
                    &mut entities[idx],
                    [n.position[0], n.position[1], n.position[2] + 0.18],
                )?;
            }
        }
        for i in self.config.movable_template_ids.len()..p.movable_count {
            let mut value = movable_templates[i % movable_templates.len()].clone();
            let clone_id = format!("regional-building-{i:02}");
            let map = object_mut(&mut value, "movable clone")?;
            map.insert("id".into(), Value::String(clone_id.clone()));
            if let Some(components) = map.get_mut("components").and_then(Value::as_array_mut) {
                for (component_index, component) in components.iter_mut().enumerate() {
                    if let Some(component) = component.as_object_mut() {
                        if component.contains_key("id") {
                            component.insert(
                                "id".into(),
                                Value::String(format!("{clone_id}-component-{component_index:02}")),
                            );
                        }
                    }
                }
            }
            let n = &nodes[content_nodes[(i * 5 + 1) % content_nodes.len()]];
            set_position(
                &mut value,
                [n.position[0], n.position[1], n.position[2] + 0.18],
            )?;
            entities.push(value);
        }

        // Bind the existing conserved chemistry to this region: colony and
        // material entities move; resource scale changes only declared founder
        // pools, never runtime replenishment.
        let indices = entity_index(entities);
        let colonies = object(&biosphere, "biosphere")?
            .get("colonies")
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err("biosphere colonies missing"))?
            .clone();
        let colony_count = colonies.len();
        for (i, colony) in colonies.iter().enumerate() {
            let cid = object(colony, "colony")?
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| PyValueError::new_err("colony id missing"))?;
            let n = &nodes[content_nodes[(i * 11 + 2) % content_nodes.len()]];
            for role in ["branches", "roots", "leaves"] {
                let eid = format!("{cid}-{role}");
                let idx = *indices
                    .get(&eid)
                    .ok_or_else(|| PyValueError::new_err("colony physical binding missing"))?;
                set_position(
                    &mut entities[idx],
                    [n.position[0], n.position[1], n.position[2] + 0.025],
                )?;
            }
        }
        let material_objects = object(&biosphere, "biosphere")?
            .get("material_objects")
            .and_then(Value::as_object)
            .and_then(|v| v.get("objects"))
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err("material objects missing"))?
            .clone();
        let active_packets: Vec<(String, usize)> = material_objects
            .iter()
            .filter_map(|v| {
                let m = v.as_object()?;
                let e = m.get("entity")?.as_str()?;
                if e.starts_with("living-packet-") {
                    Some((e.to_owned(), m.get("row")?.as_u64()? as usize))
                } else {
                    None
                }
            })
            .collect();
        let active_packet_count = active_packets.len();
        for (i, (eid, _)) in active_packets.iter().enumerate() {
            let n = &nodes[content_nodes[(i * 13 + 5) % content_nodes.len()]];
            let idx = *indices
                .get(eid)
                .ok_or_else(|| PyValueError::new_err("finite packet physical binding missing"))?;
            set_position(
                &mut entities[idx],
                [
                    n.position[0] + if i % 2 == 0 { 0.62 } else { -0.62 },
                    n.position[1] + 0.5,
                    n.position[2] + 0.14,
                ],
            )?;
        }

        // Replace resident rows transactionally in the birth document. Colony
        // rows stay first; all non-mobile rows are shifted consistently.
        let old_mobiles = object(&biosphere, "biosphere")?
            .get("mobiles")
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err("biosphere mobiles missing"))?;
        let mut old_mobile_rows = HashSet::new();
        for mobile in old_mobiles {
            let m = object(mobile, "mobile")?;
            for key in [
                "body_row",
                "gut_row",
                "structure_row",
                "gland_row",
                "brood_row",
            ] {
                if let Some(row) = m.get(key).and_then(Value::as_u64) {
                    old_mobile_rows.insert(row as usize);
                }
            }
        }
        let old_compartments = object(&biosphere, "biosphere")?
            .get("compartments")
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err("biosphere compartments missing"))?
            .clone();
        let colony_limit = colonies
            .iter()
            .flat_map(|v| {
                let m = v.as_object().unwrap();
                [m.get("body_row"), m.get("structure_row")]
            })
            .filter_map(|v| v.and_then(Value::as_u64))
            .max()
            .ok_or_else(|| PyValueError::new_err("colony rows missing"))?
            as usize;
        let mut new_compartments = old_compartments[..=colony_limit].to_vec();
        let mut row_map: HashMap<usize, usize> = (0..=colony_limit).map(|i| (i, i)).collect();
        let mut new_mobiles = Vec::new();
        let mut new_exchange = Vec::new();
        let mut new_bodies = Vec::new();
        let mut resident_ids = HashSet::new();
        for founder in residents.residents {
            for c in [
                &founder.founders.body,
                &founder.founders.gut,
                &founder.founders.structure,
                &founder.founders.gland,
                &founder.founders.brood,
            ] {
                validate_compartment(c)?;
            }
            let body = founder.body;
            let rid = id(&body)
                .ok_or_else(|| PyValueError::new_err("resident body id missing"))?
                .to_owned();
            if !identifier(&rid) || !resident_ids.insert(rid.clone()) {
                return Err(PyValueError::new_err(
                    "resident ids are invalid or duplicated",
                ));
            }
            let mut mobile = founder.mobile;
            let mm = object_mut(&mut mobile, "mobile")?;
            if mm.keys().any(|k| k.ends_with("_row")) || mm.get("id").is_some() {
                return Err(PyValueError::new_err(
                    "resident mobile rows/id are generator-owned",
                ));
            }
            mm.insert("id".into(), Value::String(rid.clone()));
            for (key, c) in [
                ("body_row", founder.founders.body),
                ("gut_row", founder.founders.gut),
                ("structure_row", founder.founders.structure),
                ("gland_row", founder.founders.gland),
                ("brood_row", founder.founders.brood),
            ] {
                mm.insert(key.into(), Value::from(new_compartments.len()));
                new_compartments.push(c);
            }
            let mut exchange = founder.exchange;
            let em = object_mut(&mut exchange, "exchange mobile")?;
            if em.get("id").is_some() || em.keys().any(|k| k.ends_with("_row")) {
                return Err(PyValueError::new_err(
                    "resident exchange identity/rows are generator-owned",
                ));
            }
            em.insert("id".into(), Value::String(rid));
            new_bodies.push(body);
            new_mobiles.push(mobile);
            new_exchange.push(exchange);
        }
        for (old, value) in old_compartments.iter().enumerate().skip(colony_limit + 1) {
            if old_mobile_rows.contains(&old) {
                continue;
            }
            row_map.insert(old, new_compartments.len());
            new_compartments.push(value.clone());
        }
        remap_rows(&mut biosphere, &row_map)?;
        let bm = object_mut(&mut biosphere, "biosphere")?;
        bm.insert(
            "format".into(),
            Value::String("chreatures-biosphere-birth-v6".into()),
        );
        bm.insert("compartments".into(), Value::Array(new_compartments));
        bm.insert("mobiles".into(), Value::Array(new_mobiles));
        let illumination = object_mut(
            bm.get_mut("illumination_cycle")
                .ok_or_else(|| PyValueError::new_err("illumination cycle missing"))?,
            "illumination cycle",
        )?;
        illumination.insert("light_entity".into(), Value::String("region-ground".into()));
        illumination.insert("center_m".into(), json!([width * 0.5, height * 0.5, 0.0]));
        let exchange = object_mut(
            bm.get_mut("exchange")
                .ok_or_else(|| PyValueError::new_err("exchange missing"))?,
            "exchange",
        )?;
        exchange.insert(
            "format".into(),
            Value::String("chreatures-ecological-exchange-v4".into()),
        );
        exchange.insert("mobiles".into(), Value::Array(new_exchange));
        // Apply resource scale after row remapping.
        let objects = bm
            .get("material_objects")
            .and_then(Value::as_object)
            .and_then(|v| v.get("objects"))
            .and_then(Value::as_array)
            .unwrap()
            .clone();
        let compartments = bm
            .get_mut("compartments")
            .and_then(Value::as_array_mut)
            .unwrap();
        for item in objects {
            let m = item.as_object().unwrap();
            let eid = m.get("entity").and_then(Value::as_str).unwrap_or("");
            if eid.starts_with("living-packet-") {
                let row = m.get("row").and_then(Value::as_u64).unwrap() as usize;
                let pools = object_mut(&mut compartments[row], "packet compartment")?
                    .get_mut("pools")
                    .and_then(Value::as_object_mut)
                    .unwrap();
                for amount in pools.values_mut() {
                    *amount = Value::from(amount.as_f64().unwrap() * p.resource_scale);
                }
            }
        }

        // Place arbitrary resident cohorts on the structure-free regions reserved
        // before any canopies, landmarks, or mechanism clusters were emitted.
        // The pairwise distance check is a hard geometric precondition before
        // MuJoCo construction.
        let slots = [[0.0, -0.4], [-0.45, 0.0], [0.45, 0.0], [0.0, 0.45]];
        let mut spawn_meta = Vec::new();
        let mut spawn_positions = Vec::new();
        for (index, body) in new_bodies.iter_mut().enumerate() {
            let node_index = spawn_nodes[index % spawn_nodes.len()];
            let layer = index / spawn_nodes.len();
            if layer >= slots.len() {
                return Err(PyValueError::new_err(
                    "regional platforms cannot place resident capacity",
                ));
            }
            let node = &nodes[node_index];
            let position = [
                node.position[0] + slots[layer][0],
                node.position[1] + slots[layer][1],
                node.position[2] + 0.22,
            ];
            set_position(body, position)?;
            spawn_positions.push(position);
            spawn_meta.push(json!({"resident":id(body),"node":node_index,"position_m":position}));
        }
        for a in 0..spawn_positions.len() {
            for b in a + 1..spawn_positions.len() {
                let dx = spawn_positions[a][0] - spawn_positions[b][0];
                let dy = spawn_positions[a][1] - spawn_positions[b][1];
                if (dx * dx + dy * dy).sqrt() < self.config.geometry.minimum_spawn_clearance_m {
                    return Err(PyValueError::new_err(
                        "generated resident spawns are not collision-clear",
                    ));
                }
            }
        }
        *array_mut(&mut habitat, "bodies")? = new_bodies;
        let hm = object_mut(&mut habitat, "habitat")?;
        hm.insert(
            "name".into(),
            Value::String(format!("regional-{}-{}", p.archetype, &genome.sha256[..12])),
        );

        let node_meta:Vec<Value>=nodes.iter().enumerate().map(|(i,n)|json!({"id":format!("region-platform-{i:02}"),"position_m":n.position,"half_size_m":n.half_size,"lane":n.lane,"underpass":underpasses.contains(&i),"sheltered":shelters.contains(&i)})).collect();
        let habitat_output = pretty(&habitat)?;
        let biosphere_output = pretty(&biosphere)?;
        let topology_sha256 = sha256(habitat_output.as_bytes());
        let resource_sha256 = sha256(biosphere_output.as_bytes());
        let area_m2 = width * height;
        let finite_material = finite_material_mass(&biosphere)?;
        let renewal = renewal_capacity(&biosphere)?;
        let minimum_z = nodes
            .iter()
            .map(|node| node.position[2])
            .fold(f64::INFINITY, f64::min);
        let maximum_z = nodes
            .iter()
            .map(|node| node.position[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let cycle_rank = edges.len().saturating_add(1).saturating_sub(nodes.len());
        let normalization = &self.config.descriptor_normalization;
        let descriptors = json!({
            "regional_scale": (area_m2 / (self.config.capacity.width_m.max * self.config.capacity.height_m.max)).sqrt().clamp(0.0, 1.0),
            "elevation_relief": ((maximum_z - minimum_z) / self.config.capacity.depth_m.max).clamp(0.0, 1.0),
            "resource_density": (finite_material / area_m2 / normalization.resource_mass_per_square_meter).clamp(0.0, 1.0),
            "renewal_rate": (renewal / area_m2 / normalization.renewal_capacity_per_square_meter_second).clamp(0.0, 1.0),
            "connectivity": (cycle_rank as f64 / nodes.len() as f64 / normalization.cycle_rank_per_region).clamp(0.0, 1.0),
        });
        let physical_geoms =
            declared_physical_geoms(&habitat, normalization.resident_physical_geoms)?;
        let compartments = object(&biosphere, "biosphere")?
            .get("compartments")
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err("biosphere compartments are missing"))?
            .len();
        let limits = &normalization.generation_cost_limits;
        if physical_geoms > limits.physical_geoms
            || nodes.len() > limits.regions
            || edges.len() > limits.edges
            || p.movable_count > limits.movables
            || compartments > limits.compartments
        {
            return Err(PyValueError::new_err(
                "generated regional environment exceeds declared generation cost",
            ));
        }
        let normalized_cost = (physical_geoms as f64 / limits.physical_geoms as f64
            + nodes.len() as f64 / limits.regions as f64
            + edges.len() as f64 / limits.edges as f64
            + p.movable_count as f64 / limits.movables as f64
            + compartments as f64 / limits.compartments as f64)
            / 5.0;
        let generation_cost = json!({
            "physical_geoms": physical_geoms,
            "regions": nodes.len(),
            "edges": edges.len(),
            "movables": p.movable_count,
            "compartments": compartments,
            "normalized": normalized_cost,
        });
        let mut record = json!({"format":RECORD_FORMAT,"sha256":"","genome_sha256":genome.sha256.clone(),"genome_parents":genome.parents.clone(),"parents":genome.environment_parents.clone(),"variation":genome.variation.clone(),"topology_sha256":topology_sha256,"resource_sha256":resource_sha256,"profile_sha256":genome.profile_sha256.clone(),"epoch":genome.epoch,"descriptors":descriptors,"generation_cost":generation_cost});
        let record_sha = sha256(serde_json::to_string(&record).unwrap().as_bytes());
        record["sha256"] = Value::String(record_sha);
        let analyst = json!({"format":ANALYST_FORMAT,"runtime_visible":false,"environment_genome":genome,"environment_record":record,"graph":{"nodes":node_meta,"edges":edge_meta,"connected":true},"resident_spawns":spawn_meta,"underpass_nodes":underpasses,"shelter_nodes":shelters,"landmark_nodes":landmarks,"resource_budget":{"founder_scale":p.resource_scale,"finite_material_element_equivalents":finite_material,"renewal_capacity_per_second":renewal,"finite_packets":active_packet_count,"colonies":colony_count,"movable_building_materials":p.movable_count},"limits":{"maximum_rise_over_run":self.config.geometry.maximum_rise_over_run,"minimum_underpass_clearance_m":self.config.geometry.minimum_underpass_clearance_m,"minimum_spawn_clearance_m":self.config.geometry.minimum_spawn_clearance_m},"generator_config_sha256":self.config_sha256});
        Ok((habitat_output, biosphere_output, pretty(&analyst)?))
    }
}

fn dist2(a: &Node, b: &Node) -> f64 {
    let dx = a.position[0] - b.position[0];
    let dy = a.position[1] - b.position[1];
    dx * dx + dy * dy
}
fn scale_grammar_number(
    values: &mut Map<String, Value>,
    key: &str,
    factor: f64,
    minimum: f64,
    maximum: f64,
) -> PyResult<()> {
    let source = values
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| PyValueError::new_err(format!("growth grammar {key} is missing")))?;
    let scaled = source * factor;
    if !scaled.is_finite() || !(minimum..=maximum).contains(&scaled) {
        return Err(PyValueError::new_err(format!(
            "inherited growth grammar {key} exceeds its physical bound"
        )));
    }
    values.insert(key.into(), Value::from(scaled));
    Ok(())
}

fn apply_developmental_rules(
    biosphere: &mut Value,
    alleles: &DevelopmentalRuleAlleles,
) -> PyResult<()> {
    let colonies = object_mut(biosphere, "biosphere")?
        .get_mut("colonies")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PyValueError::new_err("biosphere colonies are missing"))?;
    for colony in colonies {
        let grammar = object_mut(colony, "colony")?
            .get_mut("grammar")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| PyValueError::new_err("colony growth grammar is missing"))?;
        if grammar.get("version").and_then(Value::as_u64) != Some(4) {
            return Err(PyValueError::new_err(
                "current regional colonies require growth grammar v4",
            ));
        }
        let rules = grammar
            .get_mut("rules")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| PyValueError::new_err("growth grammar rules are missing"))?;
        for rule in rules.values_mut() {
            let rule = object_mut(rule, "growth rule")?;
            let guidance = rule
                .get_mut("guidance")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| PyValueError::new_err("growth guidance is missing"))?;
            scale_grammar_number(
                guidance,
                "vertical_weight",
                alleles.vertical_weight,
                -2.0,
                2.0,
            )?;
            scale_grammar_number(
                guidance,
                "clearance_weight",
                alleles.clearance_weight,
                0.0,
                4.0,
            )?;
            scale_grammar_number(guidance, "surface_weight", alleles.surface_weight, 0.0, 4.0)?;
            scale_grammar_number(guidance, "surface_reach", alleles.surface_reach, 0.002, 2.0)?;
            scale_grammar_number(
                guidance,
                "clearance_distance",
                alleles.clearance_distance,
                0.002,
                2.0,
            )?;
            let transport = rule
                .get_mut("transport")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| PyValueError::new_err("growth transport is missing"))?;
            scale_grammar_number(
                transport,
                "conductivity",
                alleles.conductivity,
                1.0e-12,
                1.0e6,
            )?;
            scale_grammar_number(
                transport,
                "half_resistance",
                alleles.half_resistance,
                1.0e-12,
                1.0e15,
            )?;
            let successors = rule
                .get_mut("successors")
                .and_then(Value::as_array_mut)
                .ok_or_else(|| PyValueError::new_err("growth successors are missing"))?;
            for successor in successors {
                let response = object_mut(successor, "growth successor")?
                    .get_mut("response")
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| PyValueError::new_err("growth response is missing"))?;
                for (key, factor) in [
                    ("light", alleles.response_light),
                    ("nutrient", alleles.response_nutrient),
                    ("support", alleles.response_support),
                    ("competition", alleles.response_competition),
                ] {
                    scale_grammar_number(response, key, factor, -8.0, 8.0)?;
                }
            }
        }
    }
    Ok(())
}

fn finite_material_mass(biosphere: &Value) -> PyResult<f64> {
    let root = object(biosphere, "biosphere")?;
    let pools = root
        .get("chemistry")
        .and_then(Value::as_object)
        .and_then(|value| value.get("pools"))
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err("biosphere chemistry pools are missing"))?;
    let mut weights = HashMap::new();
    for pool in pools {
        let pool = object(pool, "chemistry pool")?;
        let name = pool
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| PyValueError::new_err("chemistry pool name is missing"))?;
        let composition = pool
            .get("composition")
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err("chemistry pool composition is missing"))?;
        let mut weight = 0.0;
        for value in composition {
            let amount = value
                .as_f64()
                .ok_or_else(|| PyValueError::new_err("pool composition must be numeric"))?;
            if !amount.is_finite() || amount < 0.0 {
                return Err(PyValueError::new_err("pool composition must be finite"));
            }
            weight += amount;
        }
        weights.insert(name, weight);
    }
    if !weights.contains_key("fermentate") {
        return Err(PyValueError::new_err(
            "current regional chemistry requires fermentate",
        ));
    }
    let compartments = root
        .get("compartments")
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err("biosphere compartments are missing"))?;
    let objects = root
        .get("material_objects")
        .and_then(Value::as_object)
        .and_then(|value| value.get("objects"))
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err("material objects are missing"))?;
    let mut rows = HashSet::new();
    let mut total = 0.0;
    for item in objects {
        let row = object(item, "material object")?
            .get("row")
            .and_then(Value::as_u64)
            .ok_or_else(|| PyValueError::new_err("material object row is missing"))?
            as usize;
        if row >= compartments.len() || !rows.insert(row) {
            return Err(PyValueError::new_err(
                "material object rows must be valid and distinct",
            ));
        }
        let inventory = object(&compartments[row], "material compartment")?
            .get("pools")
            .and_then(Value::as_object)
            .ok_or_else(|| PyValueError::new_err("material pools are missing"))?;
        for (name, value) in inventory {
            let amount = value
                .as_f64()
                .ok_or_else(|| PyValueError::new_err("material pool must be numeric"))?;
            let weight = weights
                .get(name.as_str())
                .ok_or_else(|| PyValueError::new_err("material pool is not in chemistry"))?;
            if !amount.is_finite() || amount < 0.0 {
                return Err(PyValueError::new_err("material pool must be finite"));
            }
            total += amount * weight;
        }
    }
    Ok(total)
}

fn renewal_capacity(biosphere: &Value) -> PyResult<f64> {
    let colonies = object(biosphere, "biosphere")?
        .get("colonies")
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err("biosphere colonies are missing"))?;
    let mut total = 0.0;
    for colony in colonies {
        let colony = object(colony, "colony")?;
        let grammar_version = colony
            .get("grammar")
            .and_then(Value::as_object)
            .and_then(|grammar| grammar.get("version"))
            .and_then(Value::as_u64);
        if grammar_version != Some(4) {
            return Err(PyValueError::new_err(
                "current regional colonies require growth grammar v4",
            ));
        }
        let area = colony.get("seed_capture_area").and_then(Value::as_f64);
        let flux = colony.get("photon_flux").and_then(Value::as_f64);
        match (area, flux) {
            (Some(area), Some(flux))
                if area.is_finite() && flux.is_finite() && area >= 0.0 && flux >= 0.0 =>
            {
                total += area * flux;
            }
            _ => return Err(PyValueError::new_err("colony renewal capacity is invalid")),
        }
    }
    Ok(total)
}

fn declared_physical_geoms(habitat: &Value, resident_geoms: usize) -> PyResult<usize> {
    let root = object(habitat, "habitat")?;
    let entities = root
        .get("entities")
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err("habitat entities are missing"))?;
    let entity_geoms = entities.iter().try_fold(0usize, |total, entity| {
        let entity = object(entity, "physical entity")?;
        let count = match entity.get("shapes") {
            None => 0,
            Some(Value::Array(shapes)) => shapes.len(),
            Some(_) => {
                return Err(PyValueError::new_err(
                    "physical entity shapes must be an array",
                ))
            }
        };
        total
            .checked_add(count)
            .ok_or_else(|| PyValueError::new_err("physical geom count overflow"))
    })?;
    let bodies = root
        .get("bodies")
        .and_then(Value::as_array)
        .ok_or_else(|| PyValueError::new_err("habitat bodies are missing"))?
        .len();
    entity_geoms
        .checked_add(bodies.saturating_mul(resident_geoms))
        .ok_or_else(|| PyValueError::new_err("physical geom count overflow"))
}

fn segment_clears_other_nodes(a: usize, b: usize, nodes: &[Node], clearance: f64) -> bool {
    let ax = nodes[a].position[0];
    let ay = nodes[a].position[1];
    let bx = nodes[b].position[0];
    let by = nodes[b].position[1];
    let dx = bx - ax;
    let dy = by - ay;
    let length2 = dx * dx + dy * dy;
    nodes.iter().enumerate().all(|(i, node)| {
        if i == a || i == b {
            return true;
        }
        let t = if length2 > 0.0 {
            (((node.position[0] - ax) * dx + (node.position[1] - ay) * dy) / length2)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rx = node.position[0] - (ax + t * dx);
        let ry = node.position[1] - (ay + t * dy);
        let required = node.half_size[0].max(node.half_size[1]) + clearance;
        rx * rx + ry * ry >= required * required
    })
}
fn reachable(count: usize, edges: &[(usize, usize)]) -> Vec<bool> {
    let mut adjacent = vec![Vec::new(); count];
    for &(a, b) in edges {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut reached = vec![false; count];
    let mut q = VecDeque::from([0]);
    reached[0] = true;
    while let Some(a) = q.pop_front() {
        for &b in &adjacent[a] {
            if !reached[b] {
                reached[b] = true;
                q.push_back(b);
            }
        }
    }
    reached
}
