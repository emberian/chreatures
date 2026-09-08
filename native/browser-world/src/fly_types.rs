use serde::{Deserialize, Serialize};

pub const SITES: usize = 1771;
pub const CHANNELS: usize = 807;
pub const MOTOR_CHANNELS: usize = 92;
pub const PHYSICAL_CONTROLS: usize = 90;
pub const JOINTS: usize = 126;
pub const SEGMENTS: usize = 69;
pub const DT: f64 = 0.01;
pub const ENGINE: &str = "mujoco-3.12.0-neuromechfly-cns-v4";

#[derive(Clone, Serialize, Deserialize)]
pub struct Site {
    pub body: usize,
    /// Local author-model millimetres.
    pub position: [f64; 3],
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Eye {
    pub body: usize,
    pub position: [f64; 3],
    /// Local eye frame in its parent body, row-major.
    pub rotation: [f64; 9],
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Mouth {
    pub body: usize,
    pub position: [f64; 3],
    pub contact_radius_mm: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Body {
    pub id: String,
    pub root: usize,
    pub head: usize,
    pub qpos: Vec<usize>,
    pub dofs: Vec<usize>,
    pub segments: Vec<usize>,
    pub actuators: Vec<usize>,
    pub neutral: Vec<f64>,
    pub control_ranges: Vec<[f64; 2]>,
    pub eyes: [Eye; 2],
    pub olfactory_sites: [Site; 4],
    pub mouth: Mouth,
    pub feet: [Vec<usize>; 6],
    pub halteres: [usize; 2],
    pub wings: [usize; 2],
    pub wing_centroid_local_mm: [[f64; 3]; 2],
    pub wing_source_gain: [f64; 2],
    pub ecology_id: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Geom {
    pub id: usize,
    pub body: usize,
    pub size: [f64; 3],
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MaterialBinding {
    pub body: usize,
    pub geoms: Vec<usize>,
    pub store: chreatures_ecology_core::StoreId,
    /// Chemical concentrations are sampled only from actual local contact here.
    pub exposed: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub engine: String,
    pub source_mjcf_sha256: String,
    pub atlas_sha256: String,
    pub morphology_sha256: String,
    pub sensory_schema_sha256: String,
    pub actuator_schema_sha256: String,
    pub anatomical_sites: Vec<[i16; 3]>,
    pub supported_sites: Vec<bool>,
    /// Site directions in each eye's local frame. Measured hex membership does
    /// not make this supplied optical calibration measured biological angles.
    pub retinal_directions: Vec<[f64; 3]>,
    pub bodies: Vec<Body>,
    pub geoms: Vec<Geom>,
    pub screen_geom: usize,
    pub material_bindings: Vec<MaterialBinding>,
    pub ecology: chreatures_ecology_core::EcologyConfig,
    pub airflow_mm_s: [f64; 3],
    pub volatile_fraction: [f64; 8],
    pub interoception: chreatures_ecology_core::FlyInteroceptionSpec,
    pub ray_distance_mm: f64,
    /// Explicit engineered conversion; model mechanical work is not SI joules.
    pub atp_per_model_work: f64,
    pub acoustics: crate::fly_acoustics::AcousticConfig,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Resident {
    pub fatigue: Vec<f64>,
    pub pump_phase: f64,
    pub pump_flow: f64,
    pub salivary_flow: f64,
    pub work: f64,
    pub last_commands: Vec<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Emission {
    pub origin_mm: [f64; 3],
    pub born: f64,
    pub frequency_hz: f64,
    pub envelope: f64,
    pub duration: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Saved {
    pub format: String,
    pub engine: String,
    pub model: String,
    pub atlas: String,
    pub time: f64,
    pub rng: u64,
    pub residents: Vec<Resident>,
    pub emissions: Vec<Emission>,
    pub memory: String,
    pub afferents: Vec<f32>,
}

pub struct PhysicalPacket<'a> {
    pub qpos: &'a [f64],
    pub qvel: &'a [f64],
    pub loads: &'a [f64],
    pub positions: &'a [f64],
    pub rotations: &'a [f64],
    pub velocities: &'a [f64],
    pub contacts: &'a [f64],
    pub offsets: &'a [u32],
    pub irradiance: &'a [f64],
}

pub fn rotate(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[i * 3 + j] * v[j]).sum())
}
pub fn local(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[j * 3 + i] * v[j]).sum())
}
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
pub fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}
