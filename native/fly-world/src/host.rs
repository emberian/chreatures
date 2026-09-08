// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::ffi::{Dimensions, Physics};
use chreatures_browser_world::{AeroWorld, FlyWorldConfig, WorldCore};
use chreatures_ecology_core::{
    RouteGeometryPlan, RouteGeometryState, ROUTE_ENDPOINT_PROBE_RADIUS_M,
};
use chreatures_fly_aerodynamics::{
    neuromechfly_geometry::{DERIVATION_SHA256, ELEMENT_COUNT, SOURCE_STL_SHA256},
    AeroConfig, ModelUnitScale,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const ENGINE: &str = "mujoco-3.12.0-neuromechfly-cns-v4";
const SITES: usize = 1771;
const BODY_CHANNELS: usize = 807;
const MOTOR_CHANNELS: usize = 92;
const PHYSICAL_CONTROLS: usize = 90;
const CONTROL_DT: f64 = 0.01;
const PHYSICS_STEPS: usize = 100;
const CONTACT_SAMPLES: usize = 10;
const CONTACT_STRIDE: usize = 20;
const AERO_SCHEMA_SHA256: &str = "88893421111b6339091cb2c97368704f1ee51fcbf4cbdcbf3a793f16cf78d9fd";
const AERO_IDENTITY_JSON: &str =
    include_str!("../../fly-aerodynamics/assets/aerodynamics-model-v1.json");

#[derive(Clone, Deserialize)]
struct MeshAsset {
    path: String,
    sha256: String,
}
#[derive(Clone, Deserialize)]
struct Counts {
    nq: usize,
    nv: usize,
    nu: usize,
    nbody: usize,
    njnt: usize,
    ngeom: usize,
    nmesh: usize,
    nsite: usize,
    nsensor: usize,
    nsensordata: usize,
}
#[derive(Clone, Deserialize)]
struct Keyframe {
    key_id: i32,
}
#[derive(Clone, Deserialize)]
struct Illumination {
    sky_direction_world: [f64; 3],
    sky_intensity: f64,
    screen_intensity: f64,
    photon_energy_per_second: f64,
}
#[derive(Clone, Deserialize)]
struct AeroGeometryIdentity {
    derivation_sha256: String,
    source_stl_sha256: String,
    elements_per_wing: usize,
}
#[derive(Clone, Deserialize)]
struct AeroConfigFields {
    air_density_kg_m3: f64,
    kinematic_viscosity_m2_s: f64,
    lift_scale: f64,
    drag_scale: f64,
    minimum_speed_m_s: f64,
}
#[derive(Clone, Deserialize)]
struct AeroUnitFields {
    meters_per_length_unit: f64,
    kilograms_per_mass_unit: f64,
    seconds_per_time_unit: f64,
    mass_evidence: String,
}
#[derive(Clone, Deserialize)]
struct AeroExecutionIdentity {
    required_physics_dt_s: f64,
    force_application: String,
    snapshot_state: String,
}
#[derive(Clone, Deserialize)]
struct AeroIdentity {
    format: String,
    model_id: String,
    aerodynamic_schema_sha256: String,
    geometry: AeroGeometryIdentity,
    default_config: AeroConfigFields,
    model_unit_scale: AeroUnitFields,
    execution: AeroExecutionIdentity,
    controller_boundary: String,
}
#[derive(Clone, Deserialize)]
struct Entity {
    id: String,
    #[serde(default)]
    physics_binding: Option<String>,
    body: usize,
    #[serde(default)]
    free: bool,
    geoms: Vec<usize>,
}
#[derive(Clone, Deserialize)]
struct GeomMap {
    name: String,
}
#[derive(Clone, Deserialize)]
struct HostFields {
    scene_xml: String,
    source_mjcf_sha256: String,
    #[serde(default)]
    scene_xml_sha256: String,
    mesh_assets: Vec<MeshAsset>,
    compiled_counts: Counts,
    neutral_keyframe: Keyframe,
    entities: Vec<Entity>,
    residents: Vec<Value>,
    #[serde(default)]
    geom_map: Vec<GeomMap>,
    illumination: Illumination,
    ray_distance_mm: f64,
    screen_geom: usize,
    ecology_capacity: Value,
    wing_aerodynamics: AeroIdentity,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct GrowthStats {
    proposed: u64,
    accepted: u64,
    blocked: u64,
    offspring_rejected: u64,
}
#[derive(Clone, Serialize, Deserialize, Default)]
struct AeroStats {
    force_norm_sum_n: f64,
    power_against_air_sum_w: f64,
    active_elements: u64,
    evaluations: u64,
}
#[derive(Serialize, Deserialize)]
struct Snapshot {
    format: String,
    model: String,
    fixture: Value,
    xml: String,
    physics: Vec<f64>,
    core: String,
    frame: Vec<f32>,
    width: usize,
    height: usize,
    physics_sensed: bool,
    route_open: Vec<f64>,
    route_flow: Vec<f64>,
    route_geometry: RouteGeometryState,
    topology_revision: u64,
    clearance_memory: HashMap<String, Vec<Value>>,
    growth: GrowthStats,
    last_illumination: Vec<Value>,
    geom_size: Vec<f64>,
    geom_pos: Vec<f64>,
    geom_rgba: Vec<f64>,
    geom_contype: Vec<i32>,
    geom_conaffinity: Vec<i32>,
    force_range: Vec<f64>,
    gain: Vec<f64>,
    base_force_range: Vec<f64>,
    base_gain: Vec<f64>,
    visitor_forces: Vec<(usize, [f64; 3])>,
    visitor_counter: u64,
    last_aero: AeroStats,
}

pub struct SensorySample {
    pub optic: Vec<f32>,
    pub body: Vec<f32>,
}
pub struct ResearchSample {
    pub optic: Vec<f32>,
    pub body: Vec<f32>,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub body_positions: Vec<f64>,
    pub body_quaternions: Vec<f64>,
    pub body_rotations: Vec<f64>,
    pub sensor_data: Vec<f64>,
    pub controls: Vec<f64>,
    pub entity_positions: Vec<f32>,
    pub body_map: Value,
    pub ecology: Value,
    pub actuator_state: Value,
    pub time: f64,
}
struct PhysicsPacket {
    positions: Vec<f64>,
    rotations: Vec<f64>,
    velocities: Vec<f64>,
    contacts: Vec<f64>,
    offsets: Vec<u32>,
}
struct PreparedPhysical {
    physics: Physics,
    fixture_value: Value,
    host: HostFields,
    config: FlyWorldConfig,
    xml: String,
    base_force_range: Vec<f64>,
    base_gain: Vec<f64>,
    topology_changed: bool,
}

pub struct NativeFlyWorld {
    scene_dir: PathBuf,
    fixture_bytes: Vec<u8>,
    fixture_value: Value,
    host: HostFields,
    config: FlyWorldConfig,
    xml: String,
    physics: Physics,
    core: WorldCore,
    base_force_range: Vec<f64>,
    base_gain: Vec<f64>,
    retinal_sites: [Vec<usize>; 2],
    growth_rays: Vec<[f64; 3]>,
    frame: Vec<f32>,
    width: usize,
    height: usize,
    physics_sensed: bool,
    paused: bool,
    route_open: Vec<f64>,
    route_flow: Vec<f64>,
    route_plan: RouteGeometryPlan,
    route_geometry: RouteGeometryState,
    topology_revision: u64,
    clearance_memory: HashMap<String, Vec<Value>>,
    growth: GrowthStats,
    last_illumination: Vec<Value>,
    clearance_scratch: Option<Physics>,
    visitor_forces: Vec<(usize, [f64; 3])>,
    visitor_counter: u64,
    aero_world: AeroWorld,
    unit_scale: ModelUnitScale,
    aero_positions: Vec<f64>,
    aero_com_positions: Vec<f64>,
    aero_rotations: Vec<f64>,
    aero_velocities: Vec<f64>,
    aero_selected_velocities: Vec<f64>,
    aero_forces: Vec<f64>,
    last_aero: AeroStats,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        v.map(|x| x / n)
    } else {
        [0.0, 0.0, 1.0]
    }
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|x| x * s)
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn vec3(values: &[f64], index: usize) -> [f64; 3] {
    values[index * 3..index * 3 + 3].try_into().unwrap()
}
fn mat_vec(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}
fn mat_t_vec(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}
fn quat_from_z(v: [f64; 3]) -> [f64; 4] {
    let [x, y, z] = normalize(v);
    if z < -0.999999 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let s = (2.0 * (1.0 + z)).sqrt();
        [-y / s, x / s, 0.0, s / 2.0]
    }
}
fn xml_escape(v: &str) -> String {
    v.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}
fn value_f64_3(v: &Value, name: &str) -> Result<[f64; 3], String> {
    let a = v
        .as_array()
        .ok_or_else(|| format!("{name} is not an array"))?;
    if a.len() != 3 {
        return Err(format!("{name} shape differs"));
    }
    let out = [
        a[0].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
        a[1].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
        a[2].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
    ];
    if finite(&out) {
        Ok(out)
    } else {
        Err(format!("{name} nonfinite"))
    }
}
fn value_f64_4(v: &Value, name: &str) -> Result<[f64; 4], String> {
    let a = v
        .as_array()
        .ok_or_else(|| format!("{name} is not an array"))?;
    if a.len() != 4 {
        return Err(format!("{name} shape differs"));
    }
    let out = [
        a[0].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
        a[1].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
        a[2].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
        a[3].as_f64()
            .ok_or_else(|| format!("{name} scalar differs"))?,
    ];
    if finite(&out) {
        Ok(out)
    } else {
        Err(format!("{name} nonfinite"))
    }
}

fn validated_aerodynamics(identity: &AeroIdentity) -> Result<(AeroConfig, ModelUnitScale), String> {
    if identity.format != "chreatures.fly-aerodynamics.v1"
        || identity.model_id != "DICKINSON_TRANSLATIONAL_QS_24_STRIPS_V1"
        || identity.aerodynamic_schema_sha256 != AERO_SCHEMA_SHA256
        || identity.geometry.derivation_sha256 != DERIVATION_SHA256
        || identity.geometry.source_stl_sha256 != SOURCE_STL_SHA256
        || identity.geometry.elements_per_wing != ELEMENT_COUNT
        || (identity.execution.required_physics_dt_s - CONTROL_DT / PHYSICS_STEPS as f64).abs()
            > 1.0e-15
        || identity.execution.force_application != "24_WORLD_POINT_FORCES_PER_WING_PER_SUBSTEP"
        || identity.execution.snapshot_state != "NONE_STATELESS"
        || identity.controller_boundary != "PHYSICAL_EFFECTS_ONLY_NO_DIRECT_CNS_INPUT"
        || identity.model_unit_scale.mass_evidence != "INFERRED_GRAM_CONFIGURABLE"
    {
        return Err("wing aerodynamics identity differs".into());
    }
    let config = AeroConfig {
        air_density_kg_m3: identity.default_config.air_density_kg_m3,
        kinematic_viscosity_m2_s: identity.default_config.kinematic_viscosity_m2_s,
        lift_scale: identity.default_config.lift_scale,
        drag_scale: identity.default_config.drag_scale,
        minimum_speed_m_s: identity.default_config.minimum_speed_m_s,
    };
    let units = ModelUnitScale {
        meters_per_length_unit: identity.model_unit_scale.meters_per_length_unit,
        kilograms_per_mass_unit: identity.model_unit_scale.kilograms_per_mass_unit,
        seconds_per_time_unit: identity.model_unit_scale.seconds_per_time_unit,
    };
    units.validate().map_err(|e| e.to_string())?;
    if !(config.air_density_kg_m3.is_finite() && config.air_density_kg_m3 > 0.0)
        || !(config.kinematic_viscosity_m2_s.is_finite() && config.kinematic_viscosity_m2_s > 0.0)
        || !(config.lift_scale.is_finite() && config.lift_scale >= 0.0)
        || !(config.drag_scale.is_finite() && config.drag_scale >= 0.0)
        || !(config.minimum_speed_m_s.is_finite() && config.minimum_speed_m_s >= 0.0)
    {
        return Err("wing aerodynamics parameters differ".into());
    }
    Ok((config, units))
}
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl NativeFlyWorld {
    pub fn open(scene_path: impl AsRef<Path>, seed: u32) -> Result<Self, String> {
        let scene_path = fs::canonicalize(scene_path).map_err(|e| format!("scene: {e}"))?;
        let scene_dir = scene_path
            .parent()
            .ok_or("scene has no directory")?
            .to_owned();
        let fixture_bytes = fs::read(&scene_path).map_err(|e| format!("fixture: {e}"))?;
        let fixture_value: Value =
            serde_json::from_slice(&fixture_bytes).map_err(|e| format!("fixture JSON: {e}"))?;
        let host: HostFields = serde_json::from_value(fixture_value.clone())
            .map_err(|e| format!("host fixture: {e}"))?;
        let config: FlyWorldConfig = serde_json::from_value(fixture_value.clone())
            .map_err(|e| format!("core fixture: {e}"))?;
        if config.engine != ENGINE
            || host.scene_xml_sha256 != host.source_mjcf_sha256
            || host.ray_distance_mm <= 0.0
        {
            return Err("native fly fixture identity differs".into());
        }
        let canonical_aero: Value = serde_json::from_str(AERO_IDENTITY_JSON)
            .map_err(|e| format!("embedded wing aerodynamics identity: {e}"))?;
        if fixture_value.get("wing_aerodynamics") != Some(&canonical_aero) {
            return Err("wing aerodynamics identity object differs".into());
        }
        let (_aero_config, unit_scale) = validated_aerodynamics(&host.wing_aerodynamics)?;
        let aero_world = AeroWorld::new(std::str::from_utf8(&fixture_bytes).unwrap())?;
        let xml_path = scene_dir.join(&host.scene_xml);
        let xml_bytes = fs::read(&xml_path).map_err(|e| format!("scene XML: {e}"))?;
        if sha256(&xml_bytes) != host.source_mjcf_sha256 {
            return Err("scene XML digest differs".into());
        }
        for asset in &host.mesh_assets {
            let bytes = fs::read(scene_dir.join(&asset.path))
                .map_err(|e| format!("mesh {}: {e}", asset.path))?;
            if sha256(&bytes) != asset.sha256 {
                return Err(format!("mesh digest differs: {}", asset.path));
            }
        }
        let xml = String::from_utf8(xml_bytes).map_err(|_| "scene XML is not UTF-8")?;
        if xml.contains("<pair ") {
            return Err(
                "explicit MuJoCo contact pairs are unsupported by collision-solid route sampling"
                    .into(),
            );
        }
        let mut physics = Physics::load(&xml_path)?;
        physics.reset_keyframe(host.neutral_keyframe.key_id)?;
        physics.forward()?;
        Self::validate_dimensions(physics.dimensions(), &host.compiled_counts)?;
        if physics.dimensions().npair != 0 {
            return Err(
                "explicit MuJoCo contact pairs are unsupported by collision-solid route sampling"
                    .into(),
            );
        }
        let base_force_range = physics.num(crate::ffi::NumField::ActuatorForceRange)?;
        let base_gain = physics.num(crate::ffi::NumField::ActuatorGainPrm)?;
        let core = WorldCore::new(std::str::from_utf8(&fixture_bytes).unwrap(), seed)?;
        let route_open = config
            .ecology
            .routes
            .iter()
            .map(|route| route.base_open_fraction)
            .collect::<Vec<_>>();
        let route_flow = vec![0.0; route_open.len()];
        let route_plan = RouteGeometryPlan::new(&config.ecology.regions, &config.ecology.routes)
            .map_err(|e| e.to_string())?;
        let route_geometry = RouteGeometryState::new(&route_plan);
        let mut retinal_sites = [Vec::new(), Vec::new()];
        for (site, supported) in config.supported_sites.iter().enumerate() {
            if *supported {
                retinal_sites[(config.anatomical_sites[site][0] - 1) as usize].push(site);
            }
        }
        if retinal_sites.iter().map(Vec::len).sum::<usize>() != 1486 {
            return Err("supported retinal atlas differs".into());
        }
        let growth_rays: Vec<[f64; 3]> = serde_json::from_str(&core.growth_probe_directions())
            .map_err(|e| format!("growth probes: {e}"))?;
        if growth_rays.len() != 98 {
            return Err("shared growth probe directions differ".into());
        }
        if aero_world.resident_count() != config.bodies.len() {
            return Err("wing aerodynamic resident count differs".into());
        }
        let nbody = physics.dimensions().nbody;
        let wing_count = aero_world.wing_count();
        Ok(Self {
            scene_dir,
            fixture_bytes,
            fixture_value,
            host,
            config,
            xml,
            physics,
            core,
            base_force_range,
            base_gain,
            retinal_sites,
            frame: vec![0.0; 12],
            width: 2,
            height: 2,
            growth_rays,
            physics_sensed: false,
            paused: false,
            route_open,
            route_flow,
            route_plan,
            route_geometry,
            topology_revision: 0,
            clearance_memory: HashMap::new(),
            growth: GrowthStats::default(),
            last_illumination: Vec::new(),
            clearance_scratch: None,
            visitor_forces: Vec::new(),
            visitor_counter: 0,
            aero_world,
            unit_scale,
            aero_positions: vec![0.0; nbody * 3],
            aero_com_positions: vec![0.0; nbody * 3],
            aero_rotations: vec![0.0; nbody * 9],
            aero_velocities: vec![0.0; nbody * 6],
            aero_selected_velocities: vec![0.0; wing_count * 6],
            aero_forces: vec![0.0; nbody * 6],
            last_aero: AeroStats::default(),
        })
    }

    fn validate_dimensions(actual: Dimensions, expected: &Counts) -> Result<(), String> {
        let wanted = [
            expected.nq,
            expected.nv,
            expected.nu,
            expected.nbody,
            expected.njnt,
            expected.ngeom,
            expected.nmesh,
            expected.nsite,
            expected.nsensor,
            expected.nsensordata,
        ];
        let got = [
            actual.nq,
            actual.nv,
            actual.nu,
            actual.nbody,
            actual.njnt,
            actual.ngeom,
            actual.nmesh,
            actual.nsite,
            actual.nsensor,
            actual.nsensordata,
        ];
        if got != wanted
            || (actual.timestep - 0.0001).abs() > 1e-12
            || actual.version != 3_012_000
            || actual.integrator != 0
        {
            return Err("native MuJoCo compiled dimensions/version differ".into());
        }
        Ok(())
    }
    pub fn residents(&self) -> usize {
        self.config.bodies.len()
    }
    pub fn engine(&self) -> &str {
        &self.config.engine
    }
    pub fn time(&self) -> f64 {
        self.core.time()
    }
    pub fn fixture_sha256(&self) -> String {
        sha256(&self.fixture_bytes)
    }
    pub fn scene_sha256(&self) -> String {
        self.host.source_mjcf_sha256.clone()
    }
    pub fn ready_fixture(&self) -> Value {
        json!({
        "source_revision":self.fixture_value["source_revision"], "source_mjcf_sha256":self.host.source_mjcf_sha256,
        "atlas_sha256":self.config.atlas_sha256,"morphology_sha256":self.config.morphology_sha256,
        "sensory_schema_sha256":self.config.sensory_schema_sha256,"actuator_schema_sha256":self.config.actuator_schema_sha256,
        "bodies":self.fixture_value["bodies"],"entities":self.fixture_value["entities"],
        "wing_aerodynamics":self.fixture_value["wing_aerodynamics"] })
    }
    pub fn set_screen(&mut self, frame: &[f32], width: usize, height: usize) -> Result<(), String> {
        if width == 0
            || height == 0
            || width > 2048
            || height > 2048
            || frame.len() != width * height * 3
            || frame
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("invalid physical screen frame".into());
        }
        self.frame = frame.to_vec();
        self.width = width;
        self.height = height;
        Ok(())
    }
    pub fn visitor_sound(
        &mut self,
        position: [f64; 3],
        frequency: f64,
        envelope: f64,
        duration: f64,
    ) -> Result<(), String> {
        self.core
            .visitor_sound(&position, frequency, envelope, duration)
    }
    pub fn queue_visitor_force(&mut self, entity_id: &str, force: [f64; 3]) -> Result<(), String> {
        if !finite(&force) || norm(force) > 60.0 {
            return Err("visitor force outside finite bound".into());
        }
        let body = self
            .host
            .entities
            .iter()
            .find(|e| e.id == entity_id && e.free)
            .map(|e| e.body)
            .ok_or("movable physical entity missing")?;
        self.visitor_forces.retain(|v| v.0 != body);
        self.visitor_forces.push((body, force));
        self.visitor_counter += 1;
        Ok(())
    }
    pub fn set_routes(&mut self, open: &[f64], flow: &[f64]) -> Result<(), String> {
        if open.len() != self.route_open.len()
            || flow.len() != open.len()
            || !finite(open)
            || !finite(flow)
            || open.iter().any(|v| !(0.0..=1.0).contains(v))
        {
            return Err("route measurement contract differs".into());
        }
        self.core.set_route_measurements(open, flow)?;
        self.route_open = open.to_vec();
        self.route_flow = flow.to_vec();
        Ok(())
    }

    fn geom_normal(&self, geom: usize, point: [f64; 3]) -> Result<Option<[f64; 3]>, String> {
        let centers = self.physics.num(crate::ffi::NumField::GeomXpos)?;
        let rotations = self.physics.num(crate::ffi::NumField::GeomXmat)?;
        let sizes = self.physics.num(crate::ffi::NumField::GeomSize)?;
        let types = self.physics.int(crate::ffi::IntField::GeomType)?;
        let center = vec3(&centers, geom);
        let m = &rotations[geom * 9..geom * 9 + 9];
        let local = mat_t_vec(m, sub(point, center));
        let size = vec3(&sizes, geom);
        let local_normal = match types[geom] {
            0 => Some([0.0, 0.0, 1.0]),
            2 => Some(normalize(local)),
            4 => Some(normalize([
                local[0] / (size[0] * size[0]),
                local[1] / (size[1] * size[1]),
                local[2] / (size[2] * size[2]),
            ])),
            6 => {
                let axis = (0..3)
                    .max_by(|a, b| {
                        (local[*a] / size[*a])
                            .abs()
                            .total_cmp(&(local[*b] / size[*b]).abs())
                    })
                    .unwrap();
                let mut n = [0.0; 3];
                n[axis] = local[axis].signum();
                Some(n)
            }
            3 => {
                let z = local[2].clamp(-size[1], size[1]);
                Some(normalize([local[0], local[1], local[2] - z]))
            }
            5 => {
                let radial = (local[0] * local[0] + local[1] * local[1]).sqrt();
                if (local[2].abs() - size[1]).abs() < (radial - size[0]).abs() {
                    Some([0.0, 0.0, local[2].signum()])
                } else {
                    Some(normalize([local[0], local[1], 0.0]))
                }
            }
            _ => None,
        };
        Ok(local_normal.map(|n| normalize(mat_vec(m, n))))
    }
    fn support_on_geom(
        &self,
        geom: usize,
        direction: [f64; 3],
    ) -> Result<Option<([f64; 3], [f64; 3])>, String> {
        let centers = self.physics.num(crate::ffi::NumField::GeomXpos)?;
        let rotations = self.physics.num(crate::ffi::NumField::GeomXmat)?;
        let sizes = self.physics.num(crate::ffi::NumField::GeomSize)?;
        let types = self.physics.int(crate::ffi::IntField::GeomType)?;
        let center = vec3(&centers, geom);
        let m = &rotations[geom * 9..geom * 9 + 9];
        let d = mat_t_vec(m, normalize(direction));
        let s = vec3(&sizes, geom);
        let local = match types[geom] {
            2 => Some(scale(d, s[0])),
            4 => {
                let den = (0..3)
                    .map(|i| s[i] * s[i] * d[i] * d[i])
                    .sum::<f64>()
                    .sqrt();
                Some(std::array::from_fn(|i| s[i] * s[i] * d[i] / den))
            }
            6 => Some(std::array::from_fn(
                |i| if d[i] < 0.0 { -s[i] } else { s[i] },
            )),
            3 => {
                let mut p = scale(d, s[0]);
                p[2] += if d[2] < 0.0 { -s[1] } else { s[1] };
                Some(p)
            }
            _ => None,
        };
        if let Some(local) = local {
            let point = add(center, mat_vec(m, local));
            let normal = self
                .geom_normal(geom, point)?
                .ok_or("support normal missing")?;
            Ok(Some((point, normal)))
        } else {
            Ok(None)
        }
    }
    fn light_at(
        &mut self,
        position_m: [f64; 3],
        surface_normal: [f64; 3],
        hidden: &[i32],
    ) -> Result<Value, String> {
        let normal = normalize(surface_normal);
        let origin = add(scale(position_m, 1000.0), scale(normal, 0.002));
        let sky_dir = normalize(self.host.illumination.sky_direction_world);
        let (sky_geoms, sky_d) =
            self.physics
                .multi_ray(origin, &sky_dir, hidden, self.host.ray_distance_mm)?;
        let sky = if sky_geoms[0] < 0 || sky_d[0] > self.host.ray_distance_mm {
            self.host.illumination.sky_intensity * dot(normal, sky_dir).max(0.0)
        } else {
            0.0
        };
        let screen_pos = vec3(
            &self.physics.num(crate::ffi::NumField::GeomXpos)?,
            self.host.screen_geom,
        );
        let screen_v = sub(screen_pos, origin);
        let screen_distance = norm(screen_v);
        let screen_dir = normalize(screen_v);
        let (screen_geoms, _) =
            self.physics
                .multi_ray(origin, &screen_dir, hidden, screen_distance + 0.001)?;
        let luminance = self.frame.iter().map(|v| *v as f64).sum::<f64>() / self.frame.len() as f64;
        let screen = if screen_geoms[0] == self.host.screen_geom as i32 {
            self.host.illumination.screen_intensity * luminance * dot(normal, screen_dir).max(0.0)
        } else {
            0.0
        };
        let intensity = (sky + screen).min(1.0);
        let direction = normalize(add(scale(sky_dir, sky), scale(screen_dir, screen)));
        Ok(
            json!({"intensity":intensity,"direction":direction,"sky_intensity":sky,"screen_intensity":screen,"available_energy_per_s":intensity*self.host.illumination.photon_energy_per_second}),
        )
    }
    fn irradiance(&mut self) -> Result<Vec<f64>, String> {
        let positions = self.physics.num(crate::ffi::NumField::BodyXpos)?;
        let rotations = self.physics.num(crate::ffi::NumField::BodyXmat)?;
        let mut out = Vec::with_capacity(self.residents());
        for row in 0..self.residents() {
            let head = self.config.bodies[row].head;
            let normal = normalize([
                rotations[head * 9 + 2],
                rotations[head * 9 + 5],
                rotations[head * 9 + 8],
            ]);
            let hidden = self.hidden_bodies(row);
            out.push(
                self.light_at(scale(vec3(&positions, head), 0.001), normal, &hidden)?["intensity"]
                    .as_f64()
                    .unwrap(),
            )
        }
        Ok(out)
    }
    fn nearest_region(&self, p: [f64; 3]) -> Result<String, String> {
        let regions = self
            .fixture_value
            .pointer("/ecology/regions")
            .and_then(Value::as_array)
            .ok_or("ecology regions missing")?;
        let mut best = None;
        for region in regions {
            let center = value_f64_3(&region["center_m"], "region center")?;
            let d = norm(sub(center, p));
            if best.as_ref().map(|(_, x)| d < *x).unwrap_or(true) {
                best = Some((
                    region["id"].as_str().ok_or("region id missing")?.to_owned(),
                    d,
                ))
            }
        }
        best.map(|x| x.0).ok_or("no ecology region".into())
    }
    fn load_xml_text(&self, xml: &str, label: &str) -> Result<Physics, String> {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = self.scene_dir.join(format!(
            ".fly-world-{}-{id}-{label}.xml",
            std::process::id()
        ));
        let result = (|| {
            fs::write(&path, xml).map_err(|e| format!("temporary MJCF: {e}"))?;
            Physics::load(&path)
        })();
        let _ = fs::remove_file(path);
        result
    }
    fn ensure_clearance_scratch(&mut self) -> Result<(), String> {
        if self.clearance_scratch.is_some() {
            return Ok(());
        }
        let proxies=(0..2).map(|i|format!("<body name=\"clearance-proxy-{i}\" pos=\"0 0 0\"><geom name=\"clearance-proxy-{i}:geom\" type=\"capsule\" size=\"0.001 0.001\" contype=\"0\" conaffinity=\"0\"/></body>")).collect::<String>()+&format!("<body name=\"route-endpoint-proxy\" pos=\"0 0 0\"><geom name=\"route-endpoint-proxy:geom\" type=\"sphere\" size=\"{}\" contype=\"0\" conaffinity=\"0\"/></body>",ROUTE_ENDPOINT_PROBE_RADIUS_M*1000.0);
        let xml = self
            .xml
            .replacen("</worldbody>", &format!("{proxies}</worldbody>"), 1);
        let mut scratch = self.load_xml_text(&xml, "clearance")?;
        let d = scratch.dimensions();
        let current = self.physics.dimensions();
        if d.nq != current.nq
            || d.nv != current.nv
            || d.nbody != current.nbody + 3
            || d.ngeom != current.ngeom + 3
        {
            return Err("clearance scratch changed prior compiled addresses".into());
        }
        for i in 0..2 {
            if scratch.geom_name_to_id(&format!("clearance-proxy-{i}:geom"))? != current.ngeom + i {
                return Err("clearance proxy address differs".into());
            }
        }
        scratch.copy_prefix_from(&self.physics)?;
        self.clearance_scratch = Some(scratch);
        Ok(())
    }
    fn place_clearance(&mut self, entries: &[(Value, Value)]) -> Result<(), String> {
        self.ensure_clearance_scratch()?;
        let current = self.physics.dimensions().ngeom;
        let scratch = self.clearance_scratch.as_mut().unwrap();
        scratch.copy_prefix_from(&self.physics)?;
        for i in 0..2 {
            let geom = current + i;
            if let Some((_, query)) = entries.get(i) {
                let from = value_f64_3(&query["from_m"], "growth from")?;
                let to = value_f64_3(&query["to_m"], "growth to")?;
                let vector = sub(to, from);
                let q = quat_from_z(vector);
                scratch.write_num_at(
                    crate::ffi::NumField::GeomSize,
                    geom * 3,
                    &[
                        query["radius_m"].as_f64().ok_or("growth radius missing")? * 1000.0,
                        norm(vector) * 500.0,
                        0.0,
                    ],
                )?;
                scratch.write_num_at(
                    crate::ffi::NumField::GeomPos,
                    geom * 3,
                    &scale(add(from, to), 500.0),
                )?;
                scratch.write_num_at(
                    crate::ffi::NumField::GeomQuat,
                    geom * 4,
                    &[q[3], q[0], q[1], q[2]],
                )?
            } else {
                scratch.write_num_at(crate::ffi::NumField::GeomPos, geom * 3, &[1e6; 3])?;
                scratch.write_num_at(
                    crate::ffi::NumField::GeomSize,
                    geom * 3,
                    &[0.001, 0.001, 0.0],
                )?;
                scratch.write_num_at(
                    crate::ffi::NumField::GeomQuat,
                    geom * 4,
                    &[1.0, 0.0, 0.0, 0.0],
                )?
            }
        }
        scratch.set_const()?;
        scratch.copy_prefix_from(&self.physics)?;
        scratch.forward()
    }
    fn measure_routes_if_due(&mut self) -> Result<(), String> {
        let tick = (self.core.time() / CONTROL_DT).round() as u64;
        if !self
            .route_geometry
            .refresh_due(&self.route_plan, tick, self.topology_revision)
            .map_err(|e| e.to_string())?
        {
            return Ok(());
        }
        self.ensure_clearance_scratch()?;
        let source_geoms = self.physics.dimensions().ngeom;
        let proxy = source_geoms + 2;
        let scratch = self.clearance_scratch.as_mut().unwrap();
        scratch.copy_prefix_from(&self.physics)?;
        let endpoint_mm = self
            .route_plan
            .endpoints_m
            .iter()
            .flat_map(|v| v.iter().map(|x| x * 1000.0))
            .collect::<Vec<_>>();
        let inside = scratch.endpoints_inside(proxy, source_geoms, &endpoint_mm)?;
        let origins = self
            .route_plan
            .rays
            .iter()
            .flat_map(|ray| {
                self.route_plan.endpoints_m[ray.origin_endpoint]
                    .iter()
                    .map(|x| x * 1000.0)
            })
            .collect::<Vec<_>>();
        let directions = self
            .route_plan
            .rays
            .iter()
            .flat_map(|ray| ray.direction)
            .collect::<Vec<_>>();
        let cutoff = self
            .route_plan
            .rays
            .iter()
            .map(|ray| ray.max_distance_m)
            .fold(0.0, f64::max)
            * 1000.0;
        let (hits, raw) = self.physics.rays(&origins, &directions, cutoff)?;
        let distances = self
            .route_plan
            .rays
            .iter()
            .enumerate()
            .map(|(i, ray)| {
                if hits[i] < 0 {
                    ray.max_distance_m
                } else {
                    (raw[i] * 0.001).min(ray.max_distance_m)
                }
            })
            .collect::<Vec<_>>();
        self.route_geometry
            .refresh(
                &self.route_plan,
                tick,
                self.topology_revision,
                &self.route_plan.sha256,
                &distances,
                &inside,
            )
            .map_err(|e| e.to_string())?;
        self.route_open = self
            .route_geometry
            .openness(&self.route_plan, tick, self.topology_revision)
            .map_err(|e| e.to_string())?
            .to_vec();
        self.route_flow.fill(0.0);
        self.core
            .set_route_measurements(&self.route_open, &self.route_flow)
    }
    fn verify_growth(&mut self, proposal: &Value) -> Result<(Vec<Value>, Vec<Value>), String> {
        if proposal["format"] != "chreatures-ecology-growth-v2"
            || proposal["token"].as_str().is_none()
        {
            return Err("native growth proposal differs".into());
        }
        let constructions = proposal["construction_sites"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let births = proposal["birth_sites"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        self.growth.proposed += constructions.len() as u64;
        let ecology: Value =
            serde_json::from_str(&self.core.ecology_observe()).map_err(|e| e.to_string())?;
        let colonies = ecology["organisms"]
            .as_array()
            .map(|a| a.iter().filter(|v| !v["anchored_region"].is_null()).count())
            .unwrap_or(0);
        let max_colonies = self.host.ecology_capacity["max_colonies"]
            .as_u64()
            .unwrap_or(64) as usize;
        let max_geoms = self.host.ecology_capacity["max_geoms"]
            .as_u64()
            .unwrap_or(2048) as usize;
        let births = births
            .into_iter()
            .filter(|v| {
                !v["anchored_region"].is_null() && v["host_template_id"] == "anchored-colony"
            })
            .take(max_colonies.saturating_sub(colonies))
            .collect::<Vec<_>>();
        let mut sites = constructions;
        sites.extend(births.iter().cloned());
        sites.truncate(max_geoms.saturating_sub(self.physics.dimensions().ngeom));
        let queries = proposal["clearance_queries"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut accepted = Vec::new();
        for pair in sites.chunks(2) {
            let mut entries = Vec::new();
            for site in pair {
                let id = site["site_id"].as_str().ok_or("growth site id missing")?;
                let query = queries
                    .iter()
                    .find(|q| q["site_id"] == id)
                    .ok_or("growth clearance query missing")?;
                entries.push((site.clone(), query.clone()))
            }
            self.place_clearance(&entries)?;
            let source_ngeom = self.physics.dimensions().ngeom;
            let contype = self.physics.int(crate::ffi::IntField::GeomContype)?;
            let conaff = self.physics.int(crate::ffi::IntField::GeomConaffinity)?;
            for (i, (site, query)) in entries.iter().enumerate() {
                let binding = query["attachment_binding"].as_str();
                let excluded = self
                    .host
                    .entities
                    .iter()
                    .find(|e| {
                        Some(e.id.as_str()) == binding || e.physics_binding.as_deref() == binding
                    })
                    .map(|e| e.geoms.iter().copied().collect::<HashSet<_>>())
                    .unwrap_or_default();
                let mut blocked = None;
                for geom in 0..source_ngeom {
                    if excluded.contains(&geom) || (contype[geom] == 0 && conaff[geom] == 0) {
                        continue;
                    }
                    let (distance, points) = self
                        .clearance_scratch
                        .as_mut()
                        .unwrap()
                        .geom_distance(source_ngeom + i, geom, 100.0)?;
                    if distance < -1e-6 {
                        blocked = Some(points[3..6].try_into().unwrap());
                        break;
                    }
                }
                if blocked.is_none() && entries.len() == 2 {
                    let (distance, points) = self
                        .clearance_scratch
                        .as_mut()
                        .unwrap()
                        .geom_distance(source_ngeom, source_ngeom + 1, 100.0)?;
                    if distance < -1e-6 {
                        blocked = Some(if i == 0 {
                            points[3..6].try_into().unwrap()
                        } else {
                            points[0..3].try_into().unwrap()
                        })
                    }
                }
                if let Some(witness) = blocked {
                    self.growth.blocked += 1;
                    let from = value_f64_3(&query["from_m"], "growth from")?;
                    let toward = sub(scale(witness, 0.001), from);
                    let failure = json!({"origin_m":from,"direction":normalize(toward),"free_distance_m":norm(toward).max(0.0)});
                    let key = query["organism_id"]
                        .as_str()
                        .ok_or("growth organism missing")?
                        .to_owned();
                    let memory = self.clearance_memory.entry(key).or_default();
                    memory.retain(|v| {
                        value_f64_3(&v["origin_m"], "memory origin")
                            .map(|p| norm(sub(p, from)) > 1e-9)
                            .unwrap_or(false)
                    });
                    memory.push(failure);
                    if memory.len() > 32 {
                        memory.remove(0);
                    }
                } else {
                    accepted.push(site.clone())
                }
            }
        }
        let birth_ids = births
            .iter()
            .filter_map(|v| v["site_id"].as_str())
            .collect::<HashSet<_>>();
        let accepted_births = accepted
            .iter()
            .filter(|v| birth_ids.contains(v["site_id"].as_str().unwrap_or("")))
            .cloned()
            .collect::<Vec<_>>();
        let accepted_constructions = accepted
            .into_iter()
            .filter(|v| !birth_ids.contains(v["site_id"].as_str().unwrap_or("")))
            .collect::<Vec<_>>();
        Ok((accepted_constructions, accepted_births))
    }
    fn prepare_growth(&mut self, dt: f64) -> Result<(), String> {
        let ecology: Value =
            serde_json::from_str(&self.core.ecology_observe()).map_err(|e| e.to_string())?;
        let mut colonies = Vec::new();
        let mut photons = Vec::new();
        let mut illumination = Vec::new();
        for organism in ecology["organisms"]
            .as_array()
            .ok_or("ecology organisms missing")?
        {
            if organism["anchored_region"].is_null()
                || organism
                    .pointer("/genotype/development")
                    .map(Value::is_null)
                    .unwrap_or(true)
            {
                continue;
            }
            let binding = organism["physics_binding"]
                .as_str()
                .ok_or("colony binding missing")?;
            let entity = self
                .host
                .entities
                .iter()
                .find(|e| e.id == binding || e.physics_binding.as_deref() == Some(binding))
                .cloned()
                .ok_or("colony physical binding missing")?;
            let mut support = None;
            for geom in &entity.geoms {
                if let Some(v) = self.support_on_geom(*geom, [0., 0., 1.])? {
                    support = Some(v);
                    break;
                }
            }
            let (point, normal) = support.ok_or("colony support geometry unsupported")?;
            let position = scale(point, 0.001);
            let origin = scale(add(point, scale(normal, 0.002)), 0.001);
            let max_ray = organism
                .pointer("/genotype/reproduction/dispersal_distance_m")
                .and_then(Value::as_f64)
                .unwrap_or(0.006)
                .max(0.006);
            let body_ids = self.physics.int(crate::ffi::IntField::GeomBodyId)?;
            let ct = self.physics.int(crate::ffi::IntField::GeomContype)?;
            let ca = self.physics.int(crate::ffi::IntField::GeomConaffinity)?;
            let mut clearance = Vec::new();
            let mut nearby = Vec::new();
            let ray_directions = self
                .growth_rays
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect::<Vec<_>>();
            let (ray_geoms, ray_distances) = self.physics.multi_ray(
                scale(origin, 1000.0),
                &ray_directions,
                &[entity.body as i32],
                max_ray * 1000.0,
            )?;
            for probe in 0..self.growth_rays.len() {
                let direction = self.growth_rays[probe];
                let raw = ray_distances[probe];
                let distance = if ray_geoms[probe] >= 0 {
                    (raw * 0.001).min(max_ray)
                } else {
                    max_ray
                };
                let geom = if ray_geoms[probe] >= 0 && raw * 0.001 <= max_ray {
                    ray_geoms[probe]
                } else {
                    -1
                };
                clearance.push(
                    json!({"origin_m":origin,"direction":direction,"free_distance_m":distance}),
                );
                if geom < 0 {
                    continue;
                }
                let point_mm = scale(add(origin, scale(direction, distance)), 1000.0);
                if let Some(mut n) = self.geom_normal(geom as usize, point_mm)? {
                    if -dot(n, direction) < 0.0 {
                        n = scale(n, -1.0)
                    }
                    let moving = self
                        .config
                        .bodies
                        .iter()
                        .any(|b| b.segments.contains(&(body_ids[geom as usize] as usize)));
                    let free = self
                        .host
                        .entities
                        .iter()
                        .any(|e| e.body == body_ids[geom as usize] as usize && e.free);
                    nearby.push(json!({"surface_id":format!("surface-{geom}-probe-{probe}"),"region_id":self.nearest_region(scale(point_mm,0.001))?,"point_m":scale(point_mm,0.001),"normal":n,"attachable":!moving&&!free&&(ct[geom as usize]!=0||ca[geom as usize]!=0)}));
                }
            }
            if let Some(memory) = self
                .clearance_memory
                .get(organism["id"].as_str().unwrap_or(""))
            {
                clearance.extend(memory.iter().cloned())
            }
            clearance.truncate(256);
            let measured = self.light_at(position, normal, &[entity.body as i32])?;
            colonies.push(json!({"organism_id":organism["id"],"position_m":position,"orientation_xyzw":quat_from_z(normal),"surface_normal":normal,"light_direction":measured["direction"],"light_intensity":measured["intensity"],"nearby_surfaces":nearby,"clearance_samples":clearance,"host_template_id":"fiber-capsule","child_template_id":"anchored-colony"}));
            photons.push(json!({"organism_id":organism["id"],"available_energy_per_s":measured["available_energy_per_s"]}));
            illumination.push(json!({"organism_id":organism["id"],"intensity":measured["intensity"],"direction":measured["direction"],"sky_intensity":measured["sky_intensity"],"screen_intensity":measured["screen_intensity"],"available_energy_per_s":measured["available_energy_per_s"]}));
        }
        let proposal: Value = serde_json::from_str(
            &self
                .core
                .propose_growth(&json!({"dt_s":dt,"colonies":colonies}).to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let token = proposal["token"]
            .as_str()
            .ok_or("growth token missing")?
            .to_owned();
        match self.verify_growth(&proposal) {
            Ok((construction_sites, birth_sites)) => {
                self.growth.accepted += (construction_sites.len() + birth_sites.len()) as u64;
                self.core.set_ecology_sites(&json!({"growth_token":token,"construction_sites":construction_sites,"birth_sites":birth_sites,"photon_exposures":photons}).to_string())?;
                self.last_illumination = illumination;
                Ok(())
            }
            Err(e) => {
                let _ = self.core.discard_growth(&token);
                Err(e)
            }
        }
    }

    fn capture(&mut self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        Ok((
            self.physics.num(crate::ffi::NumField::BodyXpos)?,
            self.physics.num(crate::ffi::NumField::BodyXmat)?,
            self.physics.xbody_velocities()?,
            self.physics.contacts()?,
        ))
    }
    fn sampled_packet(
        samples: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
        nbody: usize,
    ) -> Result<PhysicsPacket, String> {
        let mut positions = Vec::with_capacity(samples.len() * nbody * 3);
        let mut rotations = Vec::with_capacity(samples.len() * nbody * 9);
        let mut velocities = Vec::with_capacity(samples.len() * nbody * 6);
        let mut contacts = Vec::new();
        let mut offsets = Vec::with_capacity(samples.len() + 1);
        offsets.push(0);
        for (p, r, v, c) in samples {
            if p.len() != nbody * 3
                || r.len() != nbody * 9
                || v.len() != nbody * 6
                || c.len() % CONTACT_STRIDE != 0
            {
                return Err("native sampled physical packet differs".into());
            }
            positions.extend(p);
            rotations.extend(r);
            velocities.extend(v);
            contacts.extend(c);
            offsets.push(
                (contacts.len() / CONTACT_STRIDE)
                    .try_into()
                    .map_err(|_| "contact packet too large")?,
            );
        }
        Ok(PhysicsPacket {
            positions,
            rotations,
            velocities,
            contacts,
            offsets,
        })
    }
    fn joint_loads(&self) -> Result<Vec<f64>, String> {
        let mut out = self.physics.num(crate::ffi::NumField::QfrcActuator)?;
        let constraint = self.physics.num(crate::ffi::NumField::QfrcConstraint)?;
        let applied = self.physics.num(crate::ffi::NumField::QfrcApplied)?;
        if out.len() != constraint.len() || out.len() != applied.len() {
            return Err("joint load arrays differ".into());
        }
        for i in 0..out.len() {
            out[i] += constraint[i] + applied[i]
        }
        Ok(out)
    }
    fn colors(&self) -> Result<Vec<f64>, String> {
        let dimensions = self.physics.dimensions();
        let matid = self.physics.int(crate::ffi::IntField::GeomMatId)?;
        let geom = self.physics.num(crate::ffi::NumField::GeomRgba)?;
        let material = self.physics.num(crate::ffi::NumField::MatRgba)?;
        let mut out = vec![0.0; dimensions.ngeom * 4];
        for id in 0..dimensions.ngeom {
            let source = if matid[id] >= 0 {
                &material[matid[id] as usize * 4..matid[id] as usize * 4 + 4]
            } else {
                &geom[id * 4..id * 4 + 4]
            };
            out[id * 4..id * 4 + 4].copy_from_slice(source);
        }
        Ok(out)
    }
    fn hidden_bodies(&self, row: usize) -> Vec<i32> {
        let resident = &self.config.bodies[row];
        let mut bodies = resident
            .segments
            .iter()
            .map(|v| *v as i32)
            .collect::<HashSet<_>>();
        for (geom, item) in self.host.geom_map.iter().enumerate() {
            if item.name.starts_with(&format!("{}/", resident.id)) {
                bodies.insert(self.config.geoms[geom].body as i32);
            }
        }
        bodies.into_iter().collect()
    }
    pub fn sample(&mut self) -> Result<SensorySample, String> {
        if self.paused {
            return Err("native world paused after failed mutation".into());
        }
        let positions = self.physics.num(crate::ffi::NumField::BodyXpos)?;
        let rotations = self.physics.num(crate::ffi::NumField::BodyXmat)?;
        let geom_positions = self.physics.num(crate::ffi::NumField::GeomXpos)?;
        let geom_rotations = self.physics.num(crate::ffi::NumField::GeomXmat)?;
        let sizes = self.physics.num(crate::ffi::NumField::GeomSize)?;
        let colors = self.colors()?;
        let mut optic = vec![0.0f32; self.residents() * SITES * 3];
        for row in 0..self.residents() {
            let rays = self.core.rays(row, &positions, &rotations)?;
            let mut hits = vec![-1i32; SITES * 2];
            let mut distances = vec![self.host.ray_distance_mm; SITES * 2];
            let hidden = self.hidden_bodies(row);
            for eye in 0..2 {
                let start = eye * (3 + SITES * 3);
                let origin: [f64; 3] = rays[start..start + 3].try_into().unwrap();
                let directions = self.retinal_sites[eye]
                    .iter()
                    .flat_map(|site| {
                        rays[start + 3 + site * 3..start + 6 + site * 3]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>();
                let (ray_hits, ray_distances) = self.physics.multi_ray(
                    origin,
                    &directions,
                    &hidden,
                    self.host.ray_distance_mm,
                )?;
                for (i, site) in self.retinal_sites[eye].iter().enumerate() {
                    hits[eye * SITES + site] = ray_hits[i];
                    distances[eye * SITES + site] = ray_distances[i];
                }
            }
            let value = self.core.retina(
                row,
                &rays,
                &hits,
                &distances,
                &geom_positions,
                &geom_rotations,
                &sizes,
                &colors,
                &self.frame,
                self.width,
                self.height,
            )?;
            optic[row * SITES * 3..(row + 1) * SITES * 3].copy_from_slice(&value);
        }
        if !self.physics_sensed {
            let packet =
                Self::sampled_packet(vec![self.capture()?], self.physics.dimensions().nbody)?;
            let qpos = self.physics.num(crate::ffi::NumField::Qpos)?;
            let qvel = self.physics.num(crate::ffi::NumField::Qvel)?;
            let loads = self.joint_loads()?;
            let irradiance = self.irradiance()?;
            self.core.sense_physics(
                &qpos,
                &qvel,
                &loads,
                &packet.positions,
                &packet.rotations,
                &packet.velocities,
                &packet.contacts,
                &packet.offsets,
                &irradiance,
            )?;
            self.physics_sensed = true;
        }
        let body = self.core.afferents();
        if body.len() != self.residents() * BODY_CHANNELS
            || body.iter().any(|v| !v.is_finite())
            || optic.iter().any(|v| !v.is_finite())
        {
            return Err("nonfinite native CNS sensory packet".into());
        }
        Ok(SensorySample { optic, body })
    }
    pub fn research_sample(&mut self) -> Result<ResearchSample, String> {
        let sensory = self.sample()?;
        let body = self.core.afferents();
        if body != sensory.body {
            return Err("research BODY cache differs".into());
        }
        let geom_positions = self.physics.num(crate::ffi::NumField::GeomXpos)?;
        let mut entity_positions = Vec::with_capacity(self.host.entities.len() * 3);
        for entity in &self.host.entities {
            if let Some(geom) = entity.geoms.first() {
                entity_positions.extend(vec3(&geom_positions, *geom).map(|v| v as f32))
            } else {
                entity_positions.extend([0.0; 3])
            }
        }
        let mut ecology: Value =
            serde_json::from_str(&self.core.ecology_observe()).map_err(|e| e.to_string())?;
        ecology["native_host"] = json!({"growth":self.growth,"illumination":self.last_illumination,"aerodynamics":self.last_aero,"retinal_rays_per_resident":1486,"route_plan_sha256":self.route_plan.sha256,"route_open_fraction":self.route_open,"topology_revision":self.topology_revision});
        Ok(ResearchSample {
            optic: sensory.optic,
            body: sensory.body,
            qpos: self.physics.num(crate::ffi::NumField::Qpos)?,
            qvel: self.physics.num(crate::ffi::NumField::Qvel)?,
            body_positions: self.physics.num(crate::ffi::NumField::BodyXpos)?,
            body_quaternions: self.physics.num(crate::ffi::NumField::BodyXquat)?,
            body_rotations: self.physics.num(crate::ffi::NumField::BodyXmat)?,
            sensor_data: self.physics.num(crate::ffi::NumField::SensorData)?,
            controls: self.physics.num(crate::ffi::NumField::Ctrl)?,
            entity_positions,
            body_map: json!({"bodies":self.fixture_value["bodies"],"residents":self.host.residents}),
            ecology,
            actuator_state: serde_json::from_str(&self.core.actuator_state())
                .map_err(|e| e.to_string())?,
            time: self.time(),
        })
    }
    fn apply_controls(&mut self, commands: &[f64]) -> Result<(), String> {
        let controls = self.core.actuation(commands)?;
        let capacity = self.core.actuator_capacity()?;
        if controls.len() != self.residents() * PHYSICAL_CONTROLS
            || capacity.len() != controls.len()
        {
            return Err("native actuation shape differs".into());
        }
        let mut ctrl = vec![0.0; self.physics.dimensions().nu];
        let mut force = self.base_force_range.clone();
        let mut gain = self.base_gain.clone();
        let gain_width = gain.len() / self.physics.dimensions().nu;
        for (row, body) in self.config.bodies.iter().enumerate() {
            for k in 0..PHYSICAL_CONTROLS {
                let actuator = body.actuators[k];
                let c = capacity[row * PHYSICAL_CONTROLS + k];
                if !(0.0..=1.0).contains(&c) {
                    return Err("native actuator capacity outside bounds".into());
                }
                ctrl[actuator] = controls[row * PHYSICAL_CONTROLS + k];
                if k < 84 {
                    force[actuator * 2] *= c;
                    force[actuator * 2 + 1] *= c
                } else {
                    for j in 0..gain_width {
                        gain[actuator * gain_width + j] *= c
                    }
                }
            }
        }
        self.physics.write_num(crate::ffi::NumField::Ctrl, &ctrl)?;
        self.physics
            .write_num(crate::ffi::NumField::ActuatorForceRange, &force)?;
        self.physics
            .write_num(crate::ffi::NumField::ActuatorGainPrm, &gain)
    }
    fn apply_aerodynamics(&mut self, stats: &mut AeroStats) -> Result<(), String> {
        self.physics
            .read_num_into(crate::ffi::NumField::BodyXpos, &mut self.aero_positions)?;
        self.physics.read_num_into(
            crate::ffi::NumField::BodyXipos,
            &mut self.aero_com_positions,
        )?;
        self.physics
            .read_num_into(crate::ffi::NumField::BodyXmat, &mut self.aero_rotations)?;
        self.physics.xbody_velocities_selected(
            self.aero_world.wing_bodies(),
            &mut self.aero_selected_velocities,
        )?;
        for (row, &body) in self.aero_world.wing_bodies().iter().enumerate() {
            self.aero_velocities[body as usize * 6..body as usize * 6 + 6]
                .copy_from_slice(&self.aero_selected_velocities[row * 6..row * 6 + 6]);
        }
        self.aero_world.evaluate_bulk(
            &self.aero_positions,
            &self.aero_com_positions,
            &self.aero_rotations,
            &self.aero_velocities,
        )?;
        let model_force_to_newton = self.unit_scale.kilograms_per_mass_unit
            * self.unit_scale.meters_per_length_unit
            / (self.unit_scale.seconds_per_time_unit * self.unit_scale.seconds_per_time_unit);
        for row in self.aero_world.diagnostics().chunks_exact(15) {
            stats.force_norm_sum_n += norm(row[0..3].try_into().unwrap()) * model_force_to_newton;
            stats.power_against_air_sum_w += row[12];
            stats.active_elements += row[14] as u64;
            stats.evaluations += 1;
        }
        self.aero_forces.fill(0.0);
        for (body, force) in &self.visitor_forces {
            self.aero_forces[body * 6..body * 6 + 3].copy_from_slice(force);
        }
        for (row, &body) in self.aero_world.wing_bodies().iter().enumerate() {
            for axis in 0..6 {
                self.aero_forces[body as usize * 6 + axis] +=
                    self.aero_world.wrenches()[row * 6 + axis];
            }
        }
        self.physics
            .write_num(crate::ffi::NumField::XfrcApplied, &self.aero_forces)
    }
    fn apply_ecology_proposal(
        &mut self,
        proposal: &Value,
    ) -> Result<Option<PreparedPhysical>, String> {
        let creations = proposal["physical_creations"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let removals = proposal["physical_removals"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let changes = proposal["geom_updates"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if creations.is_empty() && removals.is_empty() && changes.is_empty() {
            return Ok(None);
        }
        let mut fragments = String::new();
        for item in &creations {
            let kind = item["kind"].as_str().ok_or("creation kind missing")?;
            let template = item["host_template_id"]
                .as_str()
                .ok_or("creation template missing")?;
            let valid = (kind == "constructed_geometry" && template == "fiber-capsule")
                || (kind == "offspring_body"
                    && template == "anchored-colony"
                    && item["material_store"]["kind"] == "organism");
            if !valid {
                return Err(format!(
                    "unsupported native physical creation {kind}/{template}"
                ));
            }
            let binding = xml_escape(
                item["physics_binding"]
                    .as_str()
                    .ok_or("creation binding missing")?,
            );
            let p = scale(
                value_f64_3(&item["position_m"], "creation position")?,
                1000.0,
            );
            let q = value_f64_4(&item["orientation_xyzw"], "creation orientation")?;
            let radius = item["nominal_radius_m"]
                .as_f64()
                .ok_or("creation radius missing")?;
            let length = item["nominal_length_m"]
                .as_f64()
                .ok_or("creation length missing")?;
            if radius <= 0.0 || length <= 0.0 || radius > 1.0 || length > 1.0 {
                return Err("creation dimensions outside bounds".into());
            }
            let color = if kind == "offspring_body" {
                "0.24 0.52 0.31 1"
            } else {
                "0.42 0.31 0.16 1"
            };
            fragments.push_str(&format!("<body name=\"ecology:{binding}\" pos=\"{} {} {}\" quat=\"{} {} {} {}\"><geom name=\"ecology:{binding}:geom\" type=\"capsule\" size=\"{} {}\" rgba=\"{color}\" friction=\"0.9 0.02 0.004\"/></body>",p[0],p[1],p[2],q[3],q[0],q[1],q[2],radius*1000.0,length*500.0));
        }
        let xml = if fragments.is_empty() {
            self.xml.clone()
        } else {
            self.xml
                .replacen("</worldbody>", &format!("{fragments}</worldbody>"), 1)
        };
        let mut fixture = self.fixture_value.clone();
        fixture["source_mjcf_sha256"] = Value::String(sha256(xml.as_bytes()));
        let mut candidate = self.load_xml_text(&xml, "candidate")?;
        let old = self.physics.dimensions();
        let new = candidate.dimensions();
        if new.nq != old.nq
            || new.nv != old.nv
            || new.nbody != old.nbody + creations.len()
            || new.ngeom != old.ngeom + creations.len()
        {
            return Err("physical construction changed prior compiled addresses".into());
        }
        candidate.copy_prefix_from(&self.physics)?;
        for item in &creations {
            let binding = item["physics_binding"].as_str().unwrap();
            let body = candidate.body_name_to_id(&format!("ecology:{binding}"))?;
            let geom = candidate.geom_name_to_id(&format!("ecology:{binding}:geom"))?;
            if body < old.nbody || geom < old.ngeom {
                return Err("physical construction was not append-only".into());
            }
            fixture["geoms"].as_array_mut().ok_or("fixture geoms missing")?.push(json!({"id":geom,"body":body,"size":[item["nominal_radius_m"].as_f64().unwrap()*1000.0,item["nominal_length_m"].as_f64().unwrap()*500.0,0.0]}));
            fixture["entities"].as_array_mut().ok_or("fixture entities missing")?.push(json!({"id":binding,"physics_binding":binding,"body":body,"free":false,"geoms":[geom]}));
            fixture["material_bindings"].as_array_mut().ok_or("fixture material bindings missing")?.push(json!({"body":body,"geoms":[geom],"store":item["material_store"],"exposed":true}));
        }
        for removal in &removals {
            let binding = removal["physics_binding"]
                .as_str()
                .ok_or("removal binding missing")?;
            let entity = fixture["entities"]
                .as_array()
                .and_then(|a| {
                    a.iter()
                        .find(|e| e["id"] == binding || e["physics_binding"] == binding)
                })
                .ok_or("physical removal binding missing")?;
            for geom in entity["geoms"].as_array().ok_or("removal geoms missing")? {
                let id = geom.as_u64().ok_or("removal geom differs")? as usize;
                let mut ct = candidate.int(crate::ffi::IntField::GeomContype)?;
                let mut ca = candidate.int(crate::ffi::IntField::GeomConaffinity)?;
                ct[id] = 0;
                ca[id] = 0;
                candidate.write_int(crate::ffi::IntField::GeomContype, &ct)?;
                candidate.write_int(crate::ffi::IntField::GeomConaffinity, &ca)?;
                let mut rgba = candidate.num(crate::ffi::NumField::GeomRgba)?;
                rgba[id * 4 + 3] = 0.0;
                candidate.write_num(crate::ffi::NumField::GeomRgba, &rgba)?
            }
        }
        for change in &changes {
            let geom = change
                .get("geom")
                .or_else(|| change.get("reserved_geom"))
                .and_then(Value::as_u64)
                .ok_or("geom update address missing")? as usize;
            if geom >= new.ngeom {
                return Err("geom update address outside model".into());
            }
            if let Some(v) = change.get("size") {
                candidate.write_num_at(
                    crate::ffi::NumField::GeomSize,
                    geom * 3,
                    &value_f64_3(v, "geom size")?,
                )?
            }
            if let Some(v) = change.get("position") {
                candidate.write_num_at(
                    crate::ffi::NumField::GeomPos,
                    geom * 3,
                    &value_f64_3(v, "geom position")?,
                )?
            }
            if let Some(v) = change.get("rgba") {
                candidate.write_num_at(
                    crate::ffi::NumField::GeomRgba,
                    geom * 4,
                    &value_f64_4(v, "geom rgba")?,
                )?
            }
            if change.get("contype").is_some() || change.get("conaffinity").is_some() {
                let mut ct = candidate.int(crate::ffi::IntField::GeomContype)?;
                let mut ca = candidate.int(crate::ffi::IntField::GeomConaffinity)?;
                if let Some(v) = change["contype"].as_i64() {
                    ct[geom] = i32::try_from(v).map_err(|_| "geom contype outside i32")?
                }
                if let Some(v) = change["conaffinity"].as_i64() {
                    ca[geom] = i32::try_from(v).map_err(|_| "geom conaffinity outside i32")?
                }
                candidate.write_int(crate::ffi::IntField::GeomContype, &ct)?;
                candidate.write_int(crate::ffi::IntField::GeomConaffinity, &ca)?
            }
        }
        candidate.set_const()?;
        candidate.forward()?;
        let (records, distances) = candidate.contacts_with_distances()?;
        if records
            .chunks_exact(CONTACT_STRIDE)
            .zip(&distances)
            .any(|(c, d)| *d < -0.003 && (c[0] as usize >= old.ngeom || c[1] as usize >= old.ngeom))
        {
            return Err("constructed geometry penetrates physical geometry".into());
        }
        fixture["compiled_counts"] = json!({"nq":new.nq,"nv":new.nv,"nu":new.nu,"nbody":new.nbody,"njnt":new.njnt,"ngeom":new.ngeom,"nmesh":new.nmesh,"nsite":new.nsite,"nsensor":new.nsensor,"nsensordata":new.nsensordata});
        let host: HostFields = serde_json::from_value(fixture.clone())
            .map_err(|e| format!("updated host fixture: {e}"))?;
        let config: FlyWorldConfig = serde_json::from_value(fixture.clone())
            .map_err(|e| format!("updated core fixture: {e}"))?;
        if !creations.is_empty() {
            self.core.rebind_physics(&fixture.to_string())?
        }
        let topology_changed = !creations.is_empty()
            || !removals.is_empty()
            || changes
                .iter()
                .any(|v| v.get("contype").is_some() || v.get("conaffinity").is_some());
        Ok(Some(PreparedPhysical {
            physics: candidate,
            fixture_value: fixture,
            host,
            config,
            xml,
            base_force_range: self.base_force_range.clone(),
            base_gain: self.base_gain.clone(),
            topology_changed,
        }))
    }
    fn commit_physical(&mut self, transaction: Option<PreparedPhysical>) {
        if let Some(t) = transaction {
            let changed = t.topology_changed;
            self.physics = t.physics;
            self.fixture_value = t.fixture_value;
            self.host = t.host;
            self.config = t.config;
            self.xml = t.xml;
            self.base_force_range = t.base_force_range;
            self.base_gain = t.base_gain;
            self.clearance_scratch = None;
            self.resize_aero_buffers();
            if changed {
                self.topology_revision = self.topology_revision.wrapping_add(1);
            }
        }
    }
    fn resize_aero_buffers(&mut self) {
        let nbody = self.physics.dimensions().nbody;
        self.aero_positions.resize(nbody * 3, 0.0);
        self.aero_com_positions.resize(nbody * 3, 0.0);
        self.aero_rotations.resize(nbody * 9, 0.0);
        self.aero_velocities.resize(nbody * 6, 0.0);
        self.aero_forces.resize(nbody * 6, 0.0);
    }
    pub fn advance(&mut self, commands: &[f64], dt: f64) -> Result<(), String> {
        if self.paused {
            return Err("native world paused after failed mutation".into());
        }
        if dt != CONTROL_DT
            || commands.len() != self.residents() * MOTOR_CHANNELS
            || !finite(commands)
        {
            return Err("native advance contract differs".into());
        }
        for (i, value) in commands.iter().enumerate() {
            let channel = i % MOTOR_CHANNELS;
            if (channel < 84 && !(-1.0..=1.0).contains(value))
                || (channel >= 84 && !(0.0..=1.0).contains(value))
            {
                return Err("native motor command outside channel bounds".into());
            }
        }
        let result = (|| {
            self.measure_routes_if_due()?;
            self.prepare_growth(dt)?;
            self.apply_controls(commands)?;
            let mut samples = Vec::with_capacity(CONTACT_SAMPLES);
            let mut aero = AeroStats::default();
            for step in 0..PHYSICS_STEPS {
                // Split stepping refreshes current-state kinematics before the
                // host supplies external physical forces. The fixture uses the
                // Euler integrator required by MuJoCo's step1/step2 contract.
                self.physics.step1()?;
                self.physics.clear_applied_forces()?;
                self.apply_aerodynamics(&mut aero)?;
                self.physics.step2()?;
                if (step + 1) % (PHYSICS_STEPS / CONTACT_SAMPLES) == 0 {
                    samples.push(self.capture()?);
                }
            }
            self.physics.forward()?;
            if (self.physics.time() - self.core.time() - dt).abs() > 1e-8
                || self
                    .physics
                    .num(crate::ffi::NumField::Qpos)?
                    .iter()
                    .any(|v| !v.is_finite())
            {
                return Err("native physical clock/finite violation".into());
            }
            let packet = Self::sampled_packet(samples, self.physics.dimensions().nbody)?;
            let qpos = self.physics.num(crate::ffi::NumField::Qpos)?;
            let qvel = self.physics.num(crate::ffi::NumField::Qvel)?;
            let loads = self.joint_loads()?;
            let irradiance = self.irradiance()?;
            let proposal_text = self.core.prepare_advance(
                commands,
                &qpos,
                &qvel,
                &loads,
                &packet.positions,
                &packet.rotations,
                &packet.velocities,
                &packet.contacts,
                &packet.offsets,
                &irradiance,
                dt,
            )?;
            let proposal: Value =
                serde_json::from_str(&proposal_text).map_err(|e| e.to_string())?;
            let token = proposal["token"]
                .as_str()
                .ok_or("ecology proposal token missing")?
                .to_owned();
            let transaction = self.apply_ecology_proposal(&proposal)?;
            let created = proposal["physical_creations"]
                .as_array()
                .map(|v| {
                    v.iter()
                        .filter_map(|x| x["proposal_id"].as_str())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let removed = proposal["physical_removals"]
                .as_array()
                .map(|v| {
                    v.iter()
                        .filter_map(|x| x["proposal_id"].as_str())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let receipt = json!({"token":token,"created":created,"removed":removed});
            if let Err(error) = self.core.commit_advance(&receipt.to_string()) {
                drop(transaction);
                let _ = self.core.abort_advance(&token);
                return Err(error);
            }
            self.commit_physical(transaction);
            self.last_aero = aero;
            self.visitor_forces.clear();
            self.physics_sensed = true;
            Ok(())
        })();
        if result.is_err() {
            self.paused = true
        }
        result
    }
    pub fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        if self.paused {
            return Err("cannot checkpoint paused native world".into());
        }
        let value = Snapshot {
            format: "chreatures-native-fly-world-v1".into(),
            model: self.host.source_mjcf_sha256.clone(),
            fixture: self.fixture_value.clone(),
            xml: self.xml.clone(),
            physics: self.physics.state()?,
            core: self.core.snapshot()?,
            frame: self.frame.clone(),
            width: self.width,
            height: self.height,
            physics_sensed: self.physics_sensed,
            route_open: self.route_open.clone(),
            route_flow: self.route_flow.clone(),
            route_geometry: self.route_geometry.clone(),
            topology_revision: self.topology_revision,
            clearance_memory: self.clearance_memory.clone(),
            growth: self.growth.clone(),
            last_illumination: self.last_illumination.clone(),
            geom_size: self.physics.num(crate::ffi::NumField::GeomSize)?,
            geom_pos: self.physics.num(crate::ffi::NumField::GeomPos)?,
            geom_rgba: self.physics.num(crate::ffi::NumField::GeomRgba)?,
            geom_contype: self.physics.int(crate::ffi::IntField::GeomContype)?,
            geom_conaffinity: self.physics.int(crate::ffi::IntField::GeomConaffinity)?,
            force_range: self.physics.num(crate::ffi::NumField::ActuatorForceRange)?,
            gain: self.physics.num(crate::ffi::NumField::ActuatorGainPrm)?,
            base_force_range: self.base_force_range.clone(),
            base_gain: self.base_gain.clone(),
            visitor_forces: self.visitor_forces.clone(),
            visitor_counter: self.visitor_counter,
            last_aero: self.last_aero.clone(),
        };
        serde_json::to_vec(&value).map_err(|e| e.to_string())
    }
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
        let saved: Snapshot =
            serde_json::from_slice(bytes).map_err(|e| format!("native checkpoint: {e}"))?;
        if saved.format != "chreatures-native-fly-world-v1"
            || saved.width == 0
            || saved.height == 0
            || saved.frame.len() != saved.width * saved.height * 3
        {
            return Err("native checkpoint contract differs".into());
        }
        let host: HostFields = serde_json::from_value(saved.fixture.clone())
            .map_err(|e| format!("checkpoint host: {e}"))?;
        let config: FlyWorldConfig = serde_json::from_value(saved.fixture.clone())
            .map_err(|e| format!("checkpoint core: {e}"))?;
        let aero_world = AeroWorld::new(&saved.fixture.to_string())?;
        let mut core = WorldCore::new(&saved.fixture.to_string(), 0)?;
        if host.source_mjcf_sha256 != saved.model || sha256(saved.xml.as_bytes()) != saved.model {
            return Err("checkpoint physical model identity differs".into());
        }
        let mut physics = self.load_xml_text(&saved.xml, "restore")?;
        Self::validate_dimensions(physics.dimensions(), &host.compiled_counts)?;
        physics.write_num(crate::ffi::NumField::GeomSize, &saved.geom_size)?;
        physics.write_num(crate::ffi::NumField::GeomPos, &saved.geom_pos)?;
        physics.write_num(crate::ffi::NumField::GeomRgba, &saved.geom_rgba)?;
        physics.write_int(crate::ffi::IntField::GeomContype, &saved.geom_contype)?;
        physics.write_int(
            crate::ffi::IntField::GeomConaffinity,
            &saved.geom_conaffinity,
        )?;
        physics.set_const()?;
        physics.set_state(&saved.physics)?;
        physics.write_num(crate::ffi::NumField::ActuatorForceRange, &saved.force_range)?;
        physics.write_num(crate::ffi::NumField::ActuatorGainPrm, &saved.gain)?;
        physics.forward()?;
        saved
            .route_geometry
            .validate(&self.route_plan)
            .map_err(|e| e.to_string())?;
        core.restore(&saved.core)?;
        if (physics.time() - core.time()).abs() > 1e-8 {
            return Err("checkpoint physical/core clock differs".into());
        }
        self.physics = physics;
        self.core = core;
        self.aero_world = aero_world;
        self.aero_selected_velocities
            .resize(self.aero_world.wing_count() * 6, 0.0);
        self.resize_aero_buffers();
        self.fixture_value = saved.fixture;
        self.host = host;
        self.config = config;
        self.xml = saved.xml;
        self.frame = saved.frame;
        self.width = saved.width;
        self.height = saved.height;
        self.physics_sensed = saved.physics_sensed;
        self.route_open = saved.route_open;
        self.route_flow = saved.route_flow;
        self.route_geometry = saved.route_geometry;
        self.topology_revision = saved.topology_revision;
        self.clearance_memory = saved.clearance_memory;
        self.growth = saved.growth;
        self.last_illumination = saved.last_illumination;
        self.base_force_range = saved.base_force_range;
        self.base_gain = saved.base_gain;
        self.clearance_scratch = None;
        self.visitor_forces = saved.visitor_forces;
        self.visitor_counter = saved.visitor_counter;
        self.last_aero = saved.last_aero;
        self.paused = false;
        Ok(())
    }
}
