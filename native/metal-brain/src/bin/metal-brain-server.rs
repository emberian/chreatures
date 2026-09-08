#![allow(unexpected_cfgs)] // objc 0.2 macros still probe the historical cargo-clippy cfg.

#[path = "../mps_matrix.rs"]
mod mps_matrix;

use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use mps_matrix::{command_buffer_error, command_buffer_gpu_milliseconds, recommended_row_bytes};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{self, BufRead, Read, Seek, SeekFrom, Write},
    mem::size_of,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SHADER: &str = include_str!("../brain.metal");
const ARTIFACT_MAGIC: &[u8; 8] = b"CHCNS5\0\0";
const SNAPSHOT_MAGIC: &[u8; 9] = b"CNSSTATE5";
const FORMAT: &str = "chreatures-cns-service-v5";
const SNAPSHOT_FORMAT: &str = "chreatures-cns-state-v5";
const N: usize = 165_122;
const E: usize = 25_563_197;
const SITES: usize = 1_771;
const RECEPTORS: usize = 4_107;
const RECEPTOR_TYPES: usize = 10;
const SITE_EDGES: usize = 4_669;
const BODY_TARGETS: usize = 11_798;
const NEURON_TYPES: usize = 11_752;
const BODY_INPUTS: usize = 807;
const CONTEXT_INPUTS: usize = 12;
const CONTEXT_TARGETS: usize = 1_314;
const MOTOR_OUTPUTS: usize = 92;
const MOTOR_TARGETS: usize = 815;
const INPUTS: usize = 6_120;
const LATENT: usize = 512;
const READOUT_RANK: usize = 64;
const MAX_CAPACITY: usize = 32;
const MODULATOR_FAMILIES: usize = 3;
const PLASTIC_EDGES: usize = 4_184;
const PLASTIC_TARGETS: usize = 2;
const PLASTICITY_CONTRACT: &str = "gamma1pedc-cue-before-ppl-v1";
const PARAMETER_ORDER: [&str; 26] = [
    "optic.spectral_logits",
    "optic.gain_raw",
    "optic.bias",
    "body.mean",
    "body.scale",
    "body.weight",
    "body.bias",
    "context.weight",
    "context.bias",
    "dynamics.baseline_raw",
    "dynamics.recurrent_gain_raw",
    "dynamics.tau_raw",
    "dynamics.adaptation_gain_raw",
    "dynamics.adaptation_tau_raw",
    "dynamics.release_tau_raw",
    "dynamics.release_use_raw",
    "dynamics.mod_gain_raw",
    "dynamics.mod_adaptation_raw",
    "dynamics.modulation_tau_raw",
    "readout.projection.weight",
    "readout.output.weight",
    "readout.output.bias",
    "motor.reference_rate",
    "motor.rate_scale",
    "motor.weight",
    "motor.intercept",
];
const ARRAY_NAMES: [&str; 47] = [
    "graph.crow",
    "graph.col",
    "graph.weight_bits",
    "graph.channel",
    "atlas.receptor_rows",
    "atlas.receptor_type",
    "atlas.receptor_ptr",
    "atlas.site_indices",
    "atlas.site_weight",
    "atlas.body_rows",
    "atlas.body_mask",
    "atlas.context_rows",
    "atlas.motor_rows",
    "atlas.motor_mask",
    "atlas.neuron_type",
    "optic.spectral_logits",
    "optic.gain_raw",
    "optic.bias",
    "body.mean",
    "body.scale",
    "body.weight",
    "body.bias",
    "context.weight",
    "context.bias",
    "dynamics.baseline_raw",
    "dynamics.recurrent_gain_raw",
    "dynamics.tau_raw",
    "dynamics.adaptation_gain_raw",
    "dynamics.adaptation_tau_raw",
    "dynamics.release_tau_raw",
    "dynamics.release_use_raw",
    "dynamics.mod_gain_raw",
    "dynamics.mod_adaptation_raw",
    "dynamics.modulation_tau_raw",
    "afferent.neutral_drive",
    "readout.projection.weight",
    "readout.output.weight",
    "readout.output.bias",
    "motor.reference_rate",
    "motor.rate_scale",
    "motor.weight",
    "motor.intercept",
    "plasticity.edge_positions",
    "plasticity.target_ptr",
    "plasticity.target_rows",
    "plasticity.dan_rows",
    "plasticity.rule",
];

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: u32,
    dt: f32,
    final_step: u32,
    active_mask: u32,
    capacity: u32,
    tiles: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Dimensions {
    neurons: usize,
    edges: usize,
    sites: usize,
    receptors: usize,
    receptor_types: usize,
    site_edges: usize,
    body_targets: usize,
    neuron_types: usize,
    body_inputs: usize,
    context_inputs: usize,
    context_targets: usize,
    motor_outputs: usize,
    motor_targets: usize,
    latent: usize,
    readout_rank: usize,
    modulator_families: usize,
    plastic_edges: usize,
    plastic_targets: usize,
}

fn dimensions() -> Dimensions {
    Dimensions {
        neurons: N,
        edges: E,
        sites: SITES,
        receptors: RECEPTORS,
        receptor_types: RECEPTOR_TYPES,
        site_edges: SITE_EDGES,
        body_targets: BODY_TARGETS,
        neuron_types: NEURON_TYPES,
        body_inputs: BODY_INPUTS,
        context_inputs: CONTEXT_INPUTS,
        context_targets: CONTEXT_TARGETS,
        motor_outputs: MOTOR_OUTPUTS,
        motor_targets: MOTOR_TARGETS,
        latent: LATENT,
        readout_rank: READOUT_RANK,
        modulator_families: MODULATOR_FAMILIES,
        plastic_edges: PLASTIC_EDGES,
        plastic_targets: PLASTIC_TARGETS,
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ArtifactMetadata {
    format: String,
    graph_sha256: String,
    atlas_sha256: String,
    anatomy_sha256: String,
    morphology_sha256: String,
    sensory_schema_sha256: String,
    actuator_schema_sha256: String,
    motor_calibration_sha256: String,
    graph_source_weight_sha256: String,
    graph_quantization: Value,
    plasticity_sha256: String,
    plasticity_contract: String,
    readout_mask_sha256: String,
    adapter_sha256: String,
    dimensions: Dimensions,
    parameter_order: Vec<String>,
    training_status: String,
    #[serde(default)]
    provenance: Value,
    array_sha256: BTreeMap<String, String>,
}

impl ArtifactMetadata {
    fn validate(&self, raw: &Value) -> Result<(), String> {
        if self.format != FORMAT {
            return Err(format!("artifact format must be {FORMAT}"));
        }
        if self.dimensions != dimensions() {
            return Err("artifact dimensions differ from the current full MaleCNS contract".into());
        }
        if self.parameter_order != PARAMETER_ORDER {
            return Err("artifact parameter_order differs from the current contract".into());
        }
        if !matches!(
            self.training_status.as_str(),
            "initialized-untrained" | "trained"
        ) {
            return Err("artifact training_status must be initialized-untrained or trained".into());
        }
        for hash in [
            &self.graph_sha256,
            &self.atlas_sha256,
            &self.anatomy_sha256,
            &self.morphology_sha256,
            &self.sensory_schema_sha256,
            &self.actuator_schema_sha256,
            &self.motor_calibration_sha256,
            &self.graph_source_weight_sha256,
            &self.plasticity_sha256,
            &self.readout_mask_sha256,
            &self.adapter_sha256,
        ] {
            require_hash(hash)?;
        }
        if self.graph_quantization
            != json!({
                "storage": "ieee-754-binary16-bits-little-endian",
                "rounding": "round-to-nearest-ties-to-even",
                "compute": "decode-once-to-float32",
            })
        {
            return Err("artifact graph_quantization differs from the V4 contract".into());
        }
        if self.plasticity_contract != PLASTICITY_CONTRACT {
            return Err("artifact plasticity_contract differs from CNS V5".into());
        }
        let expected: BTreeSet<_> = ARRAY_NAMES.iter().map(|x| x.to_string()).collect();
        if self.array_sha256.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err("artifact array_sha256 keys differ from ARRAY_SPECS".into());
        }
        for hash in self.array_sha256.values() {
            require_hash(hash)?;
        }
        let mut projection = serde_json::Map::new();
        let object = raw
            .as_object()
            .ok_or_else(|| "artifact metadata must be a JSON object".to_string())?;
        for name in [
            "format",
            "graph_sha256",
            "atlas_sha256",
            "anatomy_sha256",
            "morphology_sha256",
            "sensory_schema_sha256",
            "actuator_schema_sha256",
            "motor_calibration_sha256",
            "graph_source_weight_sha256",
            "graph_quantization",
            "plasticity_sha256",
            "plasticity_contract",
            "readout_mask_sha256",
            "dimensions",
            "parameter_order",
            "array_sha256",
        ] {
            projection.insert(
                name.to_string(),
                object
                    .get(name)
                    .ok_or_else(|| format!("artifact metadata missing {name}"))?
                    .clone(),
            );
        }
        let digest = sha256_hex(&serde_json::to_vec(&Value::Object(projection)).unwrap());
        if digest != self.adapter_sha256 {
            return Err(
                "artifact adapter_sha256 does not match its canonical identity projection".into(),
            );
        }
        Ok(())
    }
}

fn require_hash(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        Ok(())
    } else {
        Err("artifact identities must be lowercase SHA-256 strings".into())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct HashedReader {
    file: File,
    hash: Sha256,
}

impl HashedReader {
    fn open(path: &Path) -> Result<Self, String> {
        Ok(Self {
            file: File::open(path).map_err(|e| format!("open CNS artifact: {e}"))?,
            hash: Sha256::new(),
        })
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<(), String> {
        self.file
            .read_exact(bytes)
            .map_err(|e| format!("truncated CNS artifact: {e}"))?;
        self.hash.update(bytes);
        Ok(())
    }

    fn array<T: Copy + Default>(
        &mut self,
        metadata: &ArtifactMetadata,
        name: &str,
        len: usize,
    ) -> Result<Vec<T>, String> {
        let mut values = vec![T::default(); len];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, size_of::<T>() * len)
        };
        self.read_exact(bytes)?;
        if sha256_hex(bytes) != metadata.array_sha256[name] {
            return Err(format!("CNS tensor checksum differs: {name}"));
        }
        Ok(values)
    }

    fn finish(mut self) -> Result<String, String> {
        let mut trailing = [0u8; 1];
        if self
            .file
            .read(&mut trailing)
            .map_err(|e| format!("read CNS artifact trailer: {e}"))?
            != 0
        {
            return Err("CNS artifact has trailing bytes".into());
        }
        Ok(format!("{:x}", self.hash.finalize()))
    }
}

fn buf<T: Copy>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr() as *const _,
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn zeros(device: &Device, float4_count: usize) -> Buffer {
    device.new_buffer(
        (float4_count * size_of::<[f32; 4]>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn copy<T: Copy>(buffer: &Buffer, len: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len).to_vec() }
}

fn bind(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    values: &[&Buffer],
) {
    enc.set_compute_pipeline_state(pipeline);
    for (index, buffer) in values.iter().enumerate() {
        enc.set_buffer(index as u64, Some(buffer), 0);
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits as u32) >> 10) & 0x1f;
    let fraction = (bits as u32) & 0x03ff;
    let decoded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((127 - 15 - shift + 1) << 23) | ((normalized & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(decoded)
}

fn all_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(format!("CNS tensor is nonfinite: {name}"))
    }
}

fn pad_matrix_rows(
    values: &[f32],
    rows: usize,
    columns: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, String> {
    if row_bytes < columns * size_of::<f32>() || row_bytes % size_of::<f32>() != 0 {
        return Err("MPS recommended an invalid matrix row stride".into());
    }
    let stride = row_bytes / size_of::<f32>();
    let mut padded = vec![0.0; rows * stride];
    for row in 0..rows {
        padded[row * stride..row * stride + columns]
            .copy_from_slice(&values[row * columns..(row + 1) * columns]);
    }
    Ok(padded)
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Step {
        dt: f64,
        active_mask: u32,
        sensory: Vec<f32>,
        context: Vec<f32>,
        #[serde(default)]
        research_current: Option<Vec<f32>>,
        #[serde(default)]
        selected_neuron_indices: Vec<u32>,
    },
    Reset {
        mask: u32,
    },
    Snapshot {
        path: String,
        metadata: String,
    },
    InspectSnapshot {
        path: String,
    },
    Restore {
        path: String,
        mask: u32,
    },
    CaptureRates {
        path: String,
        slot: u32,
    },
    Metadata,
    Shutdown,
}

#[derive(Debug, Deserialize, Serialize)]
struct SnapshotHeader {
    format: String,
    neurons: usize,
    capacity: usize,
    storage_tiles: usize,
    service_artifact_sha256: String,
    graph_sha256: String,
    atlas_sha256: String,
    anatomy_sha256: String,
    morphology_sha256: String,
    sensory_schema_sha256: String,
    actuator_schema_sha256: String,
    motor_calibration_sha256: String,
    graph_source_weight_sha256: String,
    graph_quantization: Value,
    plasticity_sha256: String,
    plasticity_contract: String,
    readout_mask_sha256: String,
    adapter_sha256: String,
    metadata: String,
}

struct SnapshotData {
    header: SnapshotHeader,
    times: Vec<f64>,
    neuronal_state: Option<[Vec<[f32; 4]>; 7]>,
    plastic_state: Option<[Vec<[f32; 4]>; 2]>,
}

fn validate_physical_state(
    state: &[Vec<[f32; 4]>; 7],
    capacity: usize,
    tiles: usize,
) -> Result<(), String> {
    let [rate, adapt, support, release, modulation0, modulation1, modulation2] = state;
    for row in 0..N {
        for slot in 0..capacity {
            let index = row * tiles + slot / 4;
            let lane = slot % 4;
            if !(0.0..=1.0).contains(&rate[index][lane])
                || !(-1.0..=1.0).contains(&adapt[index][lane])
                || !(0.65..=1.0).contains(&support[index][lane])
                || !(0.2..=1.0).contains(&release[index][lane])
                || !modulation0[index][lane].is_finite()
                || !modulation1[index][lane].is_finite()
                || !modulation2[index][lane].is_finite()
            {
                return Err("snapshot CNS state is nonfinite or outside physical bounds".into());
            }
        }
    }
    Ok(())
}

fn validate_plastic_state(
    state: &[Vec<[f32; 4]>; 2],
    maximum_depression: &[f32; 2],
    capacity: usize,
    tiles: usize,
) -> Result<(), String> {
    for edge in 0..PLASTIC_EDGES {
        let target = usize::from(edge >= 2_048);
        for slot in 0..capacity {
            let index = edge * tiles + slot / 4;
            let lane = slot % 4;
            if !(-maximum_depression[target]..=0.0).contains(&state[0][index][lane])
                || !(0.0..=1.0).contains(&state[1][index][lane])
            {
                return Err("snapshot plastic state is nonfinite or outside V5 bounds".into());
            }
        }
    }
    Ok(())
}

struct Engine {
    device: Device,
    queue: metal::CommandQueue,
    capacity: usize,
    tiles: usize,
    simd_rows: bool,
    metadata: ArtifactMetadata,
    artifact_sha256: String,
    rec: [Buffer; 4],
    receptor: [Buffer; 5],
    body_rows: Buffer,
    body_mask: Buffer,
    context_rows: Buffer,
    motor_rows: Buffer,
    motor_mask: Buffer,
    optic_spectral: Buffer,
    optic_gain: Buffer,
    optic_bias: Buffer,
    body_mean: Buffer,
    body_scale: Buffer,
    body_weight: Buffer,
    body_bias: Buffer,
    context_weight: Buffer,
    context_bias: Buffer,
    neuron_type: Buffer,
    dynamics: [Buffer; 10],
    neutral_drive: Buffer,
    readout_projection_weight: Buffer,
    readout_projection_stride: Buffer,
    readout_output_weight: Buffer,
    readout_output_bias: Buffer,
    motor_weight: Buffer,
    motor_reference_rate: Buffer,
    motor_rate_scale: Buffer,
    motor_intercept: Buffer,
    plastic_source_rows: Buffer,
    plastic_baseline_weight: Buffer,
    plastic_target_ptr: Buffer,
    plastic_target_rows: Buffer,
    plastic_dan_rows: Buffer,
    plastic_rule: Buffer,
    maximum_depression: [f32; 2],
    sensory: Buffer,
    context: Buffer,
    rate: [Buffer; 2],
    adapt: Buffer,
    support: Buffer,
    release: Buffer,
    modulation: [Buffer; 2],
    efficacy: Buffer,
    eligibility: Buffer,
    drive: Buffer,
    readout_hidden: Buffer,
    latent: Buffer,
    motor: Buffer,
    physiology_partial: Buffer,
    physiology: Buffer,
    times: Vec<f64>,
    poisoned: Option<String>,
    k_clear: ComputePipelineState,
    k_optic: ComputePipelineState,
    k_body_scatter: ComputePipelineState,
    k_context_scatter: ComputePipelineState,
    k_rec: ComputePipelineState,
    k_finalize: ComputePipelineState,
    k_plasticity: ComputePipelineState,
    k_motor: ComputePipelineState,
    k_gather: ComputePipelineState,
    k_projection: ComputePipelineState,
    k_readout_output: ComputePipelineState,
    k_phys: ComputePipelineState,
    k_phys_final: ComputePipelineState,
}

impl Engine {
    fn load(path: &Path, simd_rows: bool, capacity: usize) -> Result<Self, String> {
        if !(1..=MAX_CAPACITY).contains(&capacity) {
            return Err(format!("capacity must be in 1..={MAX_CAPACITY}"));
        }
        let tiles = capacity.div_ceil(4);
        let mut reader = HashedReader::open(path)?;
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != ARTIFACT_MAGIC {
            return Err("artifact header differs; only CHCNS5 is accepted".into());
        }
        let mut length = [0u8; 4];
        reader.read_exact(&mut length)?;
        let metadata_len = u32::from_le_bytes(length) as usize;
        if metadata_len == 0 || metadata_len > 16_000_000 {
            return Err("artifact metadata length is invalid".into());
        }
        let mut metadata_bytes = vec![0u8; metadata_len];
        reader.read_exact(&mut metadata_bytes)?;
        let raw_metadata: Value = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| format!("invalid artifact metadata JSON: {e}"))?;
        let metadata: ArtifactMetadata = serde_json::from_value(raw_metadata.clone())
            .map_err(|e| format!("invalid artifact metadata: {e}"))?;
        metadata.validate(&raw_metadata)?;

        let graph_crow = reader.array::<u32>(&metadata, "graph.crow", N + 1)?;
        let graph_col = reader.array::<u32>(&metadata, "graph.col", E)?;
        let graph_weight_bits = reader.array::<u16>(&metadata, "graph.weight_bits", E)?;
        let graph_weight: Vec<f32> = graph_weight_bits
            .iter()
            .map(|&bits| f16_bits_to_f32(bits))
            .collect();
        let graph_channel = reader.array::<u32>(&metadata, "graph.channel", N)?;
        let receptor_rows = reader.array::<u32>(&metadata, "atlas.receptor_rows", RECEPTORS)?;
        let receptor_type = reader.array::<u32>(&metadata, "atlas.receptor_type", RECEPTORS)?;
        let receptor_ptr = reader.array::<u32>(&metadata, "atlas.receptor_ptr", RECEPTORS + 1)?;
        let site_indices = reader.array::<u32>(&metadata, "atlas.site_indices", SITE_EDGES)?;
        let site_weight = reader.array::<f32>(&metadata, "atlas.site_weight", SITE_EDGES)?;
        let body_rows = reader.array::<u32>(&metadata, "atlas.body_rows", BODY_TARGETS)?;
        let body_mask =
            reader.array::<f32>(&metadata, "atlas.body_mask", BODY_TARGETS * BODY_INPUTS)?;
        let context_rows = reader.array::<u32>(&metadata, "atlas.context_rows", CONTEXT_TARGETS)?;
        let motor_rows = reader.array::<u32>(&metadata, "atlas.motor_rows", MOTOR_TARGETS)?;
        let motor_mask =
            reader.array::<f32>(&metadata, "atlas.motor_mask", MOTOR_OUTPUTS * MOTOR_TARGETS)?;
        let neuron_type = reader.array::<u32>(&metadata, "atlas.neuron_type", N)?;
        let spectral_logits =
            reader.array::<f32>(&metadata, "optic.spectral_logits", RECEPTOR_TYPES * 3)?;
        let gain_raw = reader.array::<f32>(&metadata, "optic.gain_raw", RECEPTOR_TYPES)?;
        let optic_bias = reader.array::<f32>(&metadata, "optic.bias", RECEPTOR_TYPES)?;
        let body_mean = reader.array::<f32>(&metadata, "body.mean", BODY_INPUTS)?;
        let body_scale = reader.array::<f32>(&metadata, "body.scale", BODY_INPUTS)?;
        let body_weight =
            reader.array::<f32>(&metadata, "body.weight", BODY_TARGETS * BODY_INPUTS)?;
        let body_bias = reader.array::<f32>(&metadata, "body.bias", BODY_TARGETS)?;
        let context_weight = reader.array::<f32>(
            &metadata,
            "context.weight",
            CONTEXT_TARGETS * CONTEXT_INPUTS,
        )?;
        let context_bias = reader.array::<f32>(&metadata, "context.bias", CONTEXT_TARGETS)?;
        let dynamics_baseline_raw =
            reader.array::<f32>(&metadata, "dynamics.baseline_raw", NEURON_TYPES)?;
        let dynamics_recurrent_gain_raw =
            reader.array::<f32>(&metadata, "dynamics.recurrent_gain_raw", NEURON_TYPES)?;
        let dynamics_tau_raw = reader.array::<f32>(&metadata, "dynamics.tau_raw", NEURON_TYPES)?;
        let dynamics_adaptation_gain_raw =
            reader.array::<f32>(&metadata, "dynamics.adaptation_gain_raw", NEURON_TYPES)?;
        let dynamics_adaptation_tau_raw =
            reader.array::<f32>(&metadata, "dynamics.adaptation_tau_raw", NEURON_TYPES)?;
        let dynamics_release_tau_raw =
            reader.array::<f32>(&metadata, "dynamics.release_tau_raw", NEURON_TYPES)?;
        let dynamics_release_use_raw =
            reader.array::<f32>(&metadata, "dynamics.release_use_raw", NEURON_TYPES)?;
        let dynamics_mod_gain_raw =
            reader.array::<f32>(&metadata, "dynamics.mod_gain_raw", NEURON_TYPES * 3)?;
        let dynamics_mod_adaptation_raw =
            reader.array::<f32>(&metadata, "dynamics.mod_adaptation_raw", NEURON_TYPES * 3)?;
        let dynamics_modulation_tau_raw =
            reader.array::<f32>(&metadata, "dynamics.modulation_tau_raw", 3)?;
        let neutral_drive = reader.array::<f32>(&metadata, "afferent.neutral_drive", N)?;
        let mut readout_projection =
            reader.array::<f32>(&metadata, "readout.projection.weight", READOUT_RANK * N)?;
        let readout_output =
            reader.array::<f32>(&metadata, "readout.output.weight", LATENT * READOUT_RANK)?;
        let readout_output_bias = reader.array::<f32>(&metadata, "readout.output.bias", LATENT)?;
        let motor_reference_rate =
            reader.array::<f32>(&metadata, "motor.reference_rate", MOTOR_TARGETS)?;
        let motor_rate_scale = reader.array::<f32>(&metadata, "motor.rate_scale", MOTOR_TARGETS)?;
        let motor_weight =
            reader.array::<f32>(&metadata, "motor.weight", MOTOR_OUTPUTS * MOTOR_TARGETS)?;
        let motor_intercept = reader.array::<f32>(&metadata, "motor.intercept", MOTOR_OUTPUTS)?;
        let plastic_edge_positions =
            reader.array::<u32>(&metadata, "plasticity.edge_positions", PLASTIC_EDGES)?;
        let plastic_target_ptr =
            reader.array::<u32>(&metadata, "plasticity.target_ptr", PLASTIC_TARGETS + 1)?;
        let plastic_target_rows =
            reader.array::<u32>(&metadata, "plasticity.target_rows", PLASTIC_TARGETS)?;
        let plastic_dan_rows =
            reader.array::<u32>(&metadata, "plasticity.dan_rows", PLASTIC_TARGETS)?;
        let plastic_rule =
            reader.array::<f32>(&metadata, "plasticity.rule", PLASTIC_TARGETS * 3)?;
        let artifact_sha256 = reader.finish()?;

        if graph_crow.first() != Some(&0)
            || graph_crow.last() != Some(&(E as u32))
            || graph_crow.windows(2).any(|x| x[0] > x[1])
        {
            return Err("invalid graph.crow".into());
        }
        if graph_weight_bits
            .iter()
            .any(|&bits| bits & 0x7c00 == 0x7c00)
        {
            return Err("graph.weight_bits must decode to finite IEEE binary16".into());
        }
        if receptor_ptr.first() != Some(&0)
            || receptor_ptr.last() != Some(&(SITE_EDGES as u32))
            || receptor_ptr.windows(2).any(|x| x[0] > x[1])
        {
            return Err("invalid atlas.receptor_ptr".into());
        }
        for (name, values, bound) in [
            ("graph.col", graph_col.as_slice(), N),
            ("atlas.receptor_rows", receptor_rows.as_slice(), N),
            ("atlas.body_rows", body_rows.as_slice(), N),
            ("atlas.context_rows", context_rows.as_slice(), N),
            ("atlas.motor_rows", motor_rows.as_slice(), N),
            ("graph.channel", graph_channel.as_slice(), 5),
            (
                "atlas.receptor_type",
                receptor_type.as_slice(),
                RECEPTOR_TYPES,
            ),
            ("atlas.site_indices", site_indices.as_slice(), SITES),
            ("atlas.neuron_type", neuron_type.as_slice(), NEURON_TYPES),
        ] {
            if values.iter().any(|&x| x as usize >= bound) {
                return Err(format!("out-of-range CNS index: {name}"));
            }
        }
        if graph_col
            .iter()
            .zip(&graph_weight_bits)
            .any(|(&source, &bits)| graph_channel[source as usize] == 0 && bits & 0x7fff != 0)
        {
            return Err("unknown-transmitter graph edges must remain zero".into());
        }
        let mut injected = BTreeSet::new();
        for (name, rows) in [
            ("atlas.receptor_rows", receptor_rows.as_slice()),
            ("atlas.body_rows", body_rows.as_slice()),
            ("atlas.context_rows", context_rows.as_slice()),
            ("atlas.motor_rows", motor_rows.as_slice()),
        ] {
            if rows.windows(2).any(|x| x[0] >= x[1]) {
                return Err(format!("{name} must be sorted and unique"));
            }
        }
        for &row in receptor_rows.iter().chain(&body_rows).chain(&context_rows) {
            if !injected.insert(row) {
                return Err("injected rows overlap".into());
            }
        }
        if motor_rows.iter().any(|row| injected.contains(row)) {
            return Err("motor rows overlap injected rows".into());
        }
        if site_weight.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err("atlas.site_weight must be finite and positive".into());
        }
        for endpoints in receptor_ptr.windows(2) {
            let (start, stop) = (endpoints[0] as usize, endpoints[1] as usize);
            if start < stop {
                let sum: f32 = site_weight[start..stop].iter().sum();
                if (sum - 1.0).abs() > 2e-6 {
                    return Err("supported receptor site weights must sum to one".into());
                }
            }
        }
        for (name, values) in [
            ("optic.spectral_logits", spectral_logits.as_slice()),
            ("optic.gain_raw", gain_raw.as_slice()),
            ("optic.bias", optic_bias.as_slice()),
            ("body.mean", body_mean.as_slice()),
            ("body.scale", body_scale.as_slice()),
            ("atlas.body_mask", body_mask.as_slice()),
            ("atlas.motor_mask", motor_mask.as_slice()),
            ("body.weight", body_weight.as_slice()),
            ("body.bias", body_bias.as_slice()),
            ("context.weight", context_weight.as_slice()),
            ("context.bias", context_bias.as_slice()),
            ("dynamics.baseline_raw", dynamics_baseline_raw.as_slice()),
            (
                "dynamics.recurrent_gain_raw",
                dynamics_recurrent_gain_raw.as_slice(),
            ),
            ("dynamics.tau_raw", dynamics_tau_raw.as_slice()),
            (
                "dynamics.adaptation_gain_raw",
                dynamics_adaptation_gain_raw.as_slice(),
            ),
            (
                "dynamics.adaptation_tau_raw",
                dynamics_adaptation_tau_raw.as_slice(),
            ),
            (
                "dynamics.release_tau_raw",
                dynamics_release_tau_raw.as_slice(),
            ),
            (
                "dynamics.release_use_raw",
                dynamics_release_use_raw.as_slice(),
            ),
            ("dynamics.mod_gain_raw", dynamics_mod_gain_raw.as_slice()),
            (
                "dynamics.mod_adaptation_raw",
                dynamics_mod_adaptation_raw.as_slice(),
            ),
            (
                "dynamics.modulation_tau_raw",
                dynamics_modulation_tau_raw.as_slice(),
            ),
            ("afferent.neutral_drive", neutral_drive.as_slice()),
            ("readout.projection.weight", readout_projection.as_slice()),
            ("readout.output.weight", readout_output.as_slice()),
            ("readout.output.bias", readout_output_bias.as_slice()),
            ("motor.reference_rate", motor_reference_rate.as_slice()),
            ("motor.rate_scale", motor_rate_scale.as_slice()),
            ("motor.weight", motor_weight.as_slice()),
            ("motor.intercept", motor_intercept.as_slice()),
        ] {
            all_finite(name, values)?;
        }
        if body_scale.iter().any(|x| *x <= 0.0) {
            return Err("body.scale must be positive".into());
        }
        if motor_rate_scale.iter().any(|x| *x <= 0.0) {
            return Err("motor.rate_scale must be positive".into());
        }
        if motor_reference_rate
            .iter()
            .any(|x| !(0.0..=1.0).contains(x))
        {
            return Err("motor.reference_rate must be in [0,1]".into());
        }
        let plastic_hashes: BTreeMap<String, String> = ARRAY_NAMES[42..]
            .iter()
            .map(|&name| (name.to_string(), metadata.array_sha256[name].clone()))
            .collect();
        let plasticity_sha256 = sha256_hex(
            &serde_json::to_vec(&plastic_hashes)
                .map_err(|e| format!("serialize plasticity identity: {e}"))?,
        );
        if plasticity_sha256 != metadata.plasticity_sha256 {
            return Err("plasticity_sha256 differs from its canonical tensor identity".into());
        }
        if plastic_target_ptr != [0, 2_048, PLASTIC_EDGES as u32]
            || plastic_target_rows != [655, 1_306]
            || plastic_dan_rows != [1_774, 1_235]
            || plastic_rule.len() != 6
            || plastic_rule.iter().any(|x| !x.is_finite())
        {
            return Err(
                "plasticity selector, targets, DAN rows or rules differ from CNS V5".into(),
            );
        }
        for rule in plastic_rule.chunks_exact(3) {
            if !(0.05..=5.0).contains(&rule[0])
                || !(0.0..=1.0).contains(&rule[1])
                || !(0.0..=0.95).contains(&rule[2])
            {
                return Err("plasticity.rule values are outside CNS V5 bounds".into());
            }
        }
        if plastic_edge_positions.windows(2).any(|x| x[0] >= x[1]) {
            return Err("plasticity.edge_positions must preserve unique canonical order".into());
        }
        let mut plastic_source_rows = Vec::with_capacity(PLASTIC_EDGES);
        let mut plastic_baseline_weight = Vec::with_capacity(PLASTIC_EDGES);
        for target in 0..PLASTIC_TARGETS {
            let row = plastic_target_rows[target] as usize;
            let row_start = graph_crow[row] as usize;
            let row_stop = graph_crow[row + 1] as usize;
            for &position in &plastic_edge_positions
                [plastic_target_ptr[target] as usize..plastic_target_ptr[target + 1] as usize]
            {
                let position = position as usize;
                if position < row_start || position >= row_stop {
                    return Err("plasticity edge position does not target its declared MBON".into());
                }
                let source = graph_col[position];
                if graph_channel[source as usize] != 1 {
                    return Err("plasticity edge source does not use the fast channel".into());
                }
                plastic_source_rows.push(source);
                let weight = graph_weight[position];
                if weight == 0.0 {
                    return Err("plasticity baseline weights must be nonzero".into());
                }
                plastic_baseline_weight.push(weight);
            }
        }
        if plastic_source_rows
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != 3_623
        {
            return Err("plasticity KC source count differs from the audited selector".into());
        }
        if body_mask
            .iter()
            .chain(&motor_mask)
            .any(|&x| x != 0.0 && x != 1.0)
        {
            return Err("structural masks must be binary".into());
        }

        let mut readout_mask = vec![1u8; N];
        for &row in receptor_rows.iter().chain(&body_rows).chain(&context_rows) {
            readout_mask[row as usize] = 0;
        }
        if sha256_hex(&readout_mask) != metadata.readout_mask_sha256 {
            return Err("readout_mask_sha256 differs from the enforced afferent mask".into());
        }
        let mut expected_neutral = vec![0.0f32; N];
        for receptor in 0..RECEPTORS {
            if receptor_ptr[receptor] != receptor_ptr[receptor + 1] {
                expected_neutral[receptor_rows[receptor] as usize] =
                    sigmoid(optic_bias[receptor_type[receptor] as usize]);
            }
        }
        for (afferent, &row) in body_rows.iter().enumerate() {
            expected_neutral[row as usize] = sigmoid(body_bias[afferent]);
        }
        if neutral_drive
            .iter()
            .zip(&expected_neutral)
            .any(|(&actual, &expected)| !actual.is_finite() || (actual - expected).abs() > 2e-7)
        {
            return Err("afferent.neutral_drive differs from the V4 afferent baseline".into());
        }
        for projection_row in 0..READOUT_RANK {
            let base = projection_row * N;
            for &afferent in receptor_rows.iter().chain(&body_rows).chain(&context_rows) {
                readout_projection[base + afferent as usize] = 0.0;
            }
        }

        let mut spectral = vec![0f32; RECEPTOR_TYPES * 3];
        for ty in 0..RECEPTOR_TYPES {
            let logits = &spectral_logits[ty * 3..ty * 3 + 3];
            let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = logits.iter().map(|x| (*x - peak).exp()).sum();
            for channel in 0..3 {
                spectral[ty * 3 + channel] = (logits[channel] - peak).exp() / denom;
            }
        }
        let optic_gain: Vec<f32> = gain_raw.iter().map(|&x| softplus(x)).collect();
        let expand = |raw: &[f32], transform: fn(f32) -> f32| -> Vec<f32> {
            neuron_type
                .iter()
                .map(|&ty| transform(raw[ty as usize]))
                .collect()
        };
        let dyn_baseline = expand(&dynamics_baseline_raw, |x| 0.05 + 0.4 * sigmoid(x));
        let dyn_recurrent_gain = expand(&dynamics_recurrent_gain_raw, |x| 0.5 + 1.5 * sigmoid(x));
        let dyn_tau = expand(&dynamics_tau_raw, |x| 0.02 + 0.23 * sigmoid(x));
        let dyn_adaptation_gain = expand(&dynamics_adaptation_gain_raw, |x| 0.5 * sigmoid(x));
        let dyn_adaptation_tau = expand(&dynamics_adaptation_tau_raw, |x| 0.25 + 4.75 * sigmoid(x));
        let dyn_release_tau = expand(&dynamics_release_tau_raw, |x| 0.05 + 1.95 * sigmoid(x));
        let dyn_release_use = expand(&dynamics_release_use_raw, |x| 0.01 + 0.49 * sigmoid(x));
        let modulation_tau: Vec<f32> = dynamics_modulation_tau_raw
            .iter()
            .map(|&x| 0.1 + 4.9 * sigmoid(x))
            .collect();

        let device = Device::system_default().ok_or_else(|| "no Metal device".to_string())?;
        let library = device
            .new_library_with_source(SHADER, &CompileOptions::new())
            .map_err(|e| format!("compile Metal CNS kernels: {e}"))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState, String> {
            let function = library
                .get_function(name, None)
                .map_err(|e| format!("load Metal kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("build Metal kernel {name}: {e}"))
        };

        let z = vec![[0f32; 4]; N * tiles];
        let mut initial_rate = vec![[0f32; 4]; N * tiles];
        let mut ones = vec![[1f32; 4]; N * tiles];
        for row in 0..N {
            for slot in 0..capacity {
                initial_rate[row * tiles + slot / 4][slot % 4] = dyn_baseline[row];
            }
        }
        if capacity % 4 != 0 {
            for row in 0..N {
                for lane in capacity % 4..4 {
                    ones[row * tiles + tiles - 1][lane] = 0.0;
                }
            }
        }
        let rec = [
            buf(&device, &graph_crow),
            buf(&device, &graph_col),
            buf(&device, &graph_weight),
            buf(&device, &graph_channel),
        ];
        let receptor = [
            buf(&device, &receptor_rows),
            buf(&device, &receptor_type),
            buf(&device, &receptor_ptr),
            buf(&device, &site_indices),
            buf(&device, &site_weight),
        ];
        let body_rows_buffer = buf(&device, &body_rows);
        let body_mask_buffer = buf(&device, &body_mask);
        let context_rows_buffer = buf(&device, &context_rows);
        let motor_rows_buffer = buf(&device, &motor_rows);
        let motor_mask_buffer = buf(&device, &motor_mask);
        let optic_spectral_buffer = buf(&device, &spectral);
        let optic_gain_buffer = buf(&device, &optic_gain);
        let optic_bias_buffer = buf(&device, &optic_bias);
        let body_mean_buffer = buf(&device, &body_mean);
        let body_scale_buffer = buf(&device, &body_scale);
        let body_weight_buffer = buf(&device, &body_weight);
        let body_bias_buffer = buf(&device, &body_bias);
        let context_weight_buffer = buf(&device, &context_weight);
        let context_bias_buffer = buf(&device, &context_bias);
        let neuron_type_buffer = buf(&device, &neuron_type);
        let dynamics = [
            buf(&device, &dyn_baseline),
            buf(&device, &dyn_recurrent_gain),
            buf(&device, &dyn_tau),
            buf(&device, &dyn_adaptation_gain),
            buf(&device, &dyn_adaptation_tau),
            buf(&device, &dyn_release_tau),
            buf(&device, &dyn_release_use),
            buf(&device, &dynamics_mod_gain_raw),
            buf(&device, &dynamics_mod_adaptation_raw),
            buf(&device, &modulation_tau),
        ];
        let neutral_drive_buffer = buf(&device, &neutral_drive);
        let projection_row_bytes = recommended_row_bytes(N);
        let projection_weight_padded =
            pad_matrix_rows(&readout_projection, READOUT_RANK, N, projection_row_bytes)?;
        let projection_weight_buffer = buf(&device, &projection_weight_padded);
        let projection_stride_buffer =
            buf(&device, &[(projection_row_bytes / size_of::<f32>()) as u32]);
        let readout_output_weight_buffer = buf(&device, &readout_output);
        let readout_output_bias_buffer = buf(&device, &readout_output_bias);
        let motor_weight_buffer = buf(&device, &motor_weight);
        let motor_reference_rate_buffer = buf(&device, &motor_reference_rate);
        let motor_rate_scale_buffer = buf(&device, &motor_rate_scale);
        let motor_intercept_buffer = buf(&device, &motor_intercept);
        let plastic_source_rows_buffer = buf(&device, &plastic_source_rows);
        let plastic_baseline_weight_buffer = buf(&device, &plastic_baseline_weight);
        let plastic_target_ptr_buffer = buf(&device, &plastic_target_ptr);
        let plastic_target_rows_buffer = buf(&device, &plastic_target_rows);
        let plastic_dan_rows_buffer = buf(&device, &plastic_dan_rows);
        let plastic_rule_buffer = buf(&device, &plastic_rule);
        let sensory = zeros(&device, INPUTS * tiles);
        let context = zeros(&device, CONTEXT_INPUTS * tiles);
        let rate = [buf(&device, &initial_rate), buf(&device, &initial_rate)];
        let adapt = buf(&device, &z);
        let support = buf(&device, &ones);
        let release = buf(&device, &ones);
        let modulation = [
            buf(&device, &vec![[0f32; 4]; N * tiles * 3]),
            buf(&device, &vec![[0f32; 4]; N * tiles * 3]),
        ];
        let plastic_zero = vec![[0f32; 4]; PLASTIC_EDGES * tiles];
        let efficacy = buf(&device, &plastic_zero);
        let eligibility = buf(&device, &plastic_zero);
        let drive = zeros(&device, N * tiles);
        let readout_hidden = zeros(&device, READOUT_RANK * tiles);
        let latent = zeros(&device, LATENT * tiles);
        let motor = zeros(&device, MOTOR_OUTPUTS * tiles);
        let physiology_partial = zeros(&device, N.div_ceil(256) * tiles * 3);
        let physiology = zeros(&device, 3 * tiles);
        Ok(Self {
            queue: device.new_command_queue(),
            capacity,
            tiles,
            simd_rows,
            metadata,
            artifact_sha256,
            rec,
            receptor,
            body_rows: body_rows_buffer,
            body_mask: body_mask_buffer,
            context_rows: context_rows_buffer,
            motor_rows: motor_rows_buffer,
            motor_mask: motor_mask_buffer,
            optic_spectral: optic_spectral_buffer,
            optic_gain: optic_gain_buffer,
            optic_bias: optic_bias_buffer,
            body_mean: body_mean_buffer,
            body_scale: body_scale_buffer,
            body_weight: body_weight_buffer,
            body_bias: body_bias_buffer,
            context_weight: context_weight_buffer,
            context_bias: context_bias_buffer,
            neuron_type: neuron_type_buffer,
            dynamics,
            neutral_drive: neutral_drive_buffer,
            readout_projection_weight: projection_weight_buffer,
            readout_projection_stride: projection_stride_buffer,
            readout_output_weight: readout_output_weight_buffer,
            readout_output_bias: readout_output_bias_buffer,
            motor_weight: motor_weight_buffer,
            motor_reference_rate: motor_reference_rate_buffer,
            motor_rate_scale: motor_rate_scale_buffer,
            motor_intercept: motor_intercept_buffer,
            plastic_source_rows: plastic_source_rows_buffer,
            plastic_baseline_weight: plastic_baseline_weight_buffer,
            plastic_target_ptr: plastic_target_ptr_buffer,
            plastic_target_rows: plastic_target_rows_buffer,
            plastic_dan_rows: plastic_dan_rows_buffer,
            plastic_rule: plastic_rule_buffer,
            maximum_depression: [plastic_rule[2], plastic_rule[5]],
            sensory,
            context,
            rate,
            adapt,
            support,
            release,
            modulation,
            efficacy,
            eligibility,
            drive,
            readout_hidden,
            latent,
            motor,
            physiology_partial,
            physiology,
            times: vec![0.0; capacity],
            poisoned: None,
            k_clear: pipeline("clear_drive")?,
            k_optic: pipeline("project_optic")?,
            k_body_scatter: pipeline("v5_project_body_masked")?,
            k_context_scatter: pipeline("v5_project_context_zero_neutral")?,
            k_rec: pipeline("v5_csr_dynamics_f16")?,
            k_finalize: pipeline("v5_finalize_private_state")?,
            k_plasticity: pipeline("v5_advance_plasticity")?,
            k_motor: pipeline("v5_motor92_centered")?,
            k_gather: pipeline("gather_rates")?,
            k_projection: pipeline("dense_projection")?,
            k_readout_output: pipeline("dense_readout_output")?,
            k_phys: pipeline("physiology_partials")?,
            k_phys_final: pipeline("physiology_final")?,
            device,
        })
    }

    fn valid_mask(&self) -> u32 {
        if self.capacity == 32 {
            u32::MAX
        } else {
            (1u32 << self.capacity) - 1
        }
    }

    fn params(&self, dt: f32, mask: u32, final_step: bool) -> Params {
        Params {
            n: N as u32,
            dt,
            final_step: u32::from(final_step),
            active_mask: mask,
            capacity: self.capacity as u32,
            tiles: self.tiles as u32,
        }
    }

    fn grid(enc: &metal::ComputeCommandEncoderRef, pipeline: &ComputePipelineState, count: usize) {
        enc.dispatch_threads(
            MTLSize::new(count as u64, 1, 1),
            MTLSize::new(pipeline.thread_execution_width(), 1, 1),
        );
    }

    fn encode_compute(
        &self,
        cb: &metal::CommandBufferRef,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: &Buffer,
        count: usize,
    ) {
        let enc = cb.new_compute_command_encoder();
        bind(enc, pipeline, buffers);
        enc.set_buffer(8, Some(params), 0);
        Self::grid(enc, pipeline, count);
        enc.end_encoding();
    }

    fn step(
        &mut self,
        dt: f64,
        mask: u32,
        sensory: &[f32],
        context: &[f32],
        research_current: Option<&[f32]>,
        selected_indices: &[u32],
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f64>,
            f64,
            [f64; 4],
        ),
        String,
    > {
        if let Some(reason) = &self.poisoned {
            return Err(format!(
                "CNS state is poisoned after a failed GPU tick ({reason}); restore a full coherent snapshot or cold-reset all slots"
            ));
        }
        if !dt.is_finite() || (dt - 0.01).abs() > 1e-12 {
            return Err("CNS V4 requires the fixed 0.01-second control interval".into());
        }
        if mask & !self.valid_mask() != 0 {
            return Err("active_mask exceeds configured capacity".into());
        }
        if sensory.len() != INPUTS * self.capacity || sensory.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "sensory must be finite channel-major [{INPUTS},{}]",
                self.capacity
            ));
        }
        if sensory[..SITES * 3 * self.capacity]
            .iter()
            .any(|&x| !(0.0..=1.0).contains(&x))
        {
            return Err("optic RGB must be in [0,1]".into());
        }
        if context.len() != CONTEXT_INPUTS * self.capacity
            || context.iter().any(|x| !x.is_finite() || x.abs() > 1.0)
        {
            return Err(format!(
                "context must be finite, signed [-1,1], channel-major [{CONTEXT_INPUTS},{}]",
                self.capacity
            ));
        }
        if context.iter().any(|&x| !(-1.0..=1.0).contains(&x)) {
            return Err("context must be in [-1,1]".into());
        }
        if let Some(current) = research_current {
            if std::env::var_os("CHREATURES_CNS_RESEARCH_INTERVENTION").as_deref()
                != Some(std::ffi::OsStr::new("1"))
            {
                return Err("research_current is disabled in the resident service".into());
            }
            if current.len() != N * self.capacity || current.iter().any(|x| !x.is_finite()) {
                return Err(format!(
                    "research_current must be finite channel-major [{N},{}]",
                    self.capacity
                ));
            }
        }
        if selected_indices.len() > 8192 || selected_indices.iter().any(|&x| x as usize >= N) {
            return Err("selected_neuron_indices are invalid".into());
        }
        let unique: BTreeSet<_> = selected_indices.iter().copied().collect();
        if unique.len() != selected_indices.len() {
            return Err("selected_neuron_indices must be unique".into());
        }

        let mut packed = vec![[0f32; 4]; INPUTS * self.tiles];
        for row in 0..INPUTS {
            for resident in 0..self.capacity {
                packed[row * self.tiles + resident / 4][resident % 4] =
                    sensory[row * self.capacity + resident];
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                self.sensory.contents() as *mut [f32; 4],
                packed.len(),
            );
        }
        if let Some(current) = research_current {
            let mut packed_drive = vec![[0f32; 4]; N * self.tiles];
            for row in 0..N {
                for resident in 0..self.capacity {
                    packed_drive[row * self.tiles + resident / 4][resident % 4] =
                        current[row * self.capacity + resident];
                }
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    packed_drive.as_ptr(),
                    self.drive.contents() as *mut [f32; 4],
                    packed_drive.len(),
                );
            }
        }
        let mut packed_context = vec![[0f32; 4]; CONTEXT_INPUTS * self.tiles];
        for row in 0..CONTEXT_INPUTS {
            for resident in 0..self.capacity {
                packed_context[row * self.tiles + resident / 4][resident % 4] =
                    context[row * self.capacity + resident];
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed_context.as_ptr(),
                self.context.contents() as *mut [f32; 4],
                packed_context.len(),
            );
        }
        let selected_index_buffer =
            (!selected_indices.is_empty()).then(|| buf(&self.device, selected_indices));
        let selected_buffer = (!selected_indices.is_empty())
            .then(|| zeros(&self.device, selected_indices.len() * self.tiles));
        let gpu_dt = dt as f32;
        let p0 = buf(&self.device, &[self.params(gpu_dt, mask, false)]);
        let p1 = buf(&self.device, &[self.params(gpu_dt, mask, true)]);
        let cb_afferents = self.queue.new_command_buffer();
        if research_current.is_none() {
            self.encode_compute(
                cb_afferents,
                &self.k_clear,
                &[&self.drive],
                &p0,
                N * self.tiles,
            );
            {
                let enc = cb_afferents.new_compute_command_encoder();
                bind(
                    enc,
                    &self.k_optic,
                    &[
                        &self.receptor[0],
                        &self.receptor[1],
                        &self.receptor[2],
                        &self.receptor[3],
                        &self.receptor[4],
                        &self.sensory,
                        &self.optic_spectral,
                        &self.optic_gain,
                    ],
                );
                enc.set_buffer(8, Some(&p0), 0);
                enc.set_buffer(9, Some(&self.optic_bias), 0);
                enc.set_buffer(10, Some(&self.drive), 0);
                enc.set_buffer(11, Some(&self.neutral_drive), 0);
                Self::grid(enc, &self.k_optic, RECEPTORS * self.tiles);
                enc.end_encoding();
            }
            {
                let enc = cb_afferents.new_compute_command_encoder();
                bind(
                    enc,
                    &self.k_body_scatter,
                    &[
                        &self.sensory,
                        &self.body_mean,
                        &self.body_scale,
                        &self.body_weight,
                        &self.body_mask,
                        &self.body_bias,
                        &self.body_rows,
                        &self.drive,
                    ],
                );
                enc.set_buffer(
                    0,
                    Some(&self.sensory),
                    (SITES * 3 * self.tiles * size_of::<[f32; 4]>()) as u64,
                );
                enc.set_buffer(8, Some(&p0), 0);
                Self::grid(enc, &self.k_body_scatter, BODY_TARGETS * self.tiles);
                enc.end_encoding();
            }
            self.encode_compute(
                cb_afferents,
                &self.k_context_scatter,
                &[
                    &self.context,
                    &self.context_weight,
                    &self.context_bias,
                    &self.context_rows,
                    &self.drive,
                ],
                &p0,
                CONTEXT_TARGETS * self.tiles,
            );
        }
        let cb_recurrence = self.queue.new_command_buffer();
        for (params, input, output, modulation_in, modulation_out) in [
            (
                &p0,
                &self.rate[0],
                &self.rate[1],
                &self.modulation[0],
                &self.modulation[1],
            ),
            (
                &p1,
                &self.rate[1],
                &self.rate[0],
                &self.modulation[1],
                &self.modulation[0],
            ),
        ] {
            let enc = cb_recurrence.new_compute_command_encoder();
            bind(
                enc,
                &self.k_rec,
                &[
                    &self.rec[0],
                    &self.rec[1],
                    &self.rec[2],
                    &self.rec[3],
                    input,
                    output,
                    modulation_in,
                    modulation_out,
                ],
            );
            enc.set_buffer(8, Some(params), 0);
            enc.set_buffer(9, Some(&self.adapt), 0);
            enc.set_buffer(10, Some(&self.support), 0);
            enc.set_buffer(11, Some(&self.release), 0);
            enc.set_buffer(12, Some(&self.drive), 0);
            for i in 0..4 {
                enc.set_buffer(13 + i as u64, Some(&self.dynamics[i]), 0);
            }
            enc.set_buffer(17, Some(&self.neuron_type), 0);
            for i in 7..10 {
                enc.set_buffer(11 + i as u64, Some(&self.dynamics[i]), 0);
            }
            enc.set_buffer(21, Some(&self.neutral_drive), 0);
            enc.set_buffer(22, Some(&self.plastic_source_rows), 0);
            enc.set_buffer(23, Some(&self.plastic_baseline_weight), 0);
            enc.set_buffer(24, Some(&self.plastic_target_ptr), 0);
            enc.set_buffer(25, Some(&self.plastic_target_rows), 0);
            enc.set_buffer(26, Some(&self.efficacy), 0);
            Self::grid(enc, &self.k_rec, N * self.tiles);
            enc.end_encoding();
        }
        self.encode_compute(
            cb_recurrence,
            &self.k_finalize,
            &[
                &self.rate[0],
                &self.adapt,
                &self.support,
                &self.release,
                &self.dynamics[0],
                &self.dynamics[4],
                &self.dynamics[5],
                &self.dynamics[6],
            ],
            &p1,
            N * self.tiles,
        );
        {
            let enc = cb_recurrence.new_compute_command_encoder();
            bind(
                enc,
                &self.k_plasticity,
                &[
                    &self.rate[0],
                    &self.release,
                    &self.dynamics[0],
                    &self.plastic_source_rows,
                    &self.plastic_target_ptr,
                    &self.plastic_dan_rows,
                    &self.plastic_rule,
                    &self.efficacy,
                ],
            );
            enc.set_buffer(8, Some(&p1), 0);
            enc.set_buffer(9, Some(&self.eligibility), 0);
            Self::grid(enc, &self.k_plasticity, PLASTIC_EDGES * self.tiles);
            enc.end_encoding();
        }
        let cb_readout = self.queue.new_command_buffer();
        {
            let enc = cb_readout.new_compute_command_encoder();
            bind(
                enc,
                &self.k_projection,
                &[
                    &self.readout_projection_weight,
                    &self.rate[0],
                    &self.dynamics[0],
                    &self.readout_hidden,
                ],
            );
            enc.set_buffer(8, Some(&p1), 0);
            enc.set_buffer(9, Some(&self.readout_projection_stride), 0);
            enc.dispatch_thread_groups(
                MTLSize::new((READOUT_RANK * self.tiles) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        self.encode_compute(
            cb_readout,
            &self.k_readout_output,
            &[
                &self.readout_output_weight,
                &self.readout_hidden,
                &self.readout_output_bias,
                &self.latent,
            ],
            &p1,
            LATENT * self.tiles,
        );
        self.encode_compute(
            cb_readout,
            &self.k_motor,
            &[
                &self.rate[0],
                &self.motor_rows,
                &self.motor_reference_rate,
                &self.motor_rate_scale,
                &self.motor_weight,
                &self.motor_mask,
                &self.motor_intercept,
                &self.motor,
            ],
            &p1,
            MOTOR_OUTPUTS * self.tiles,
        );
        let cb_observer = self.queue.new_command_buffer();
        if let (Some(indices), Some(selected)) = (&selected_index_buffer, &selected_buffer) {
            self.encode_compute(
                cb_observer,
                &self.k_gather,
                &[&self.rate[0], indices, selected],
                &p1,
                selected_indices.len() * self.tiles,
            );
        }
        {
            let enc = cb_observer.new_compute_command_encoder();
            bind(
                enc,
                &self.k_phys,
                &[&self.rate[0], &self.support, &self.physiology_partial],
            );
            enc.set_buffer(8, Some(&p1), 0);
            enc.dispatch_threads(
                MTLSize::new((N.div_ceil(256) * self.tiles * 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        self.encode_compute(
            cb_observer,
            &self.k_phys_final,
            &[&self.physiology_partial, &self.physiology],
            &p1,
            self.tiles,
        );

        let start = Instant::now();
        let command_buffers = [cb_afferents, cb_recurrence, cb_readout, cb_observer];
        for command_buffer in command_buffers {
            command_buffer.commit();
        }
        command_buffers[3].wait_until_completed();
        let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        let unpack = |buffer: &Buffer, rows: usize| {
            let raw = copy::<[f32; 4]>(buffer, rows * self.tiles);
            let mut result = Vec::with_capacity(rows * self.capacity);
            for row in 0..rows {
                for resident in 0..self.capacity {
                    result.push(raw[row * self.tiles + resident / 4][resident % 4]);
                }
            }
            result
        };
        for command_buffer in command_buffers {
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                let reason = command_buffer_error(command_buffer)
                    .unwrap_or_else(|| format!("status {:?}", command_buffer.status()));
                self.poisoned = Some(reason.clone());
                return Err(format!(
                    "Metal CNS tick failed and state is now poisoned: {reason}"
                ));
            }
        }
        let phase_ms = command_buffers.map(command_buffer_gpu_milliseconds);
        let latent = unpack(&self.latent, LATENT);
        let motor = unpack(&self.motor, MOTOR_OUTPUTS);
        let selected = selected_buffer
            .as_ref()
            .map_or_else(Vec::new, |b| unpack(b, selected_indices.len()));
        let physiology = unpack(&self.physiology, 3);
        if latent
            .iter()
            .chain(selected.iter())
            .chain(motor.iter())
            .chain(physiology.iter())
            .any(|x| !x.is_finite())
        {
            self.poisoned = Some("nonfinite GPU output".into());
            return Err(
                "Metal CNS tick produced nonfinite output and state is now poisoned".into(),
            );
        }
        for slot in 0..self.capacity {
            if mask & (1u32 << slot) != 0 {
                self.times[slot] += dt;
            }
        }
        Ok((
            latent,
            motor,
            selected,
            physiology,
            self.times.clone(),
            gpu_ms,
            phase_ms,
        ))
    }

    fn reset(&mut self, mask: u32) -> Result<(), String> {
        if mask == 0 || mask & !self.valid_mask() != 0 {
            return Err("reset mask must select slots within capacity".into());
        }
        let baseline = copy::<f32>(&self.dynamics[0], N);
        unsafe {
            let rate0 = self.rate[0].contents() as *mut [f32; 4];
            let rate1 = self.rate[1].contents() as *mut [f32; 4];
            let adapt = self.adapt.contents() as *mut [f32; 4];
            let support = self.support.contents() as *mut [f32; 4];
            let release = self.release.contents() as *mut [f32; 4];
            for row in 0..N {
                for slot in 0..self.capacity {
                    if mask & (1u32 << slot) != 0 {
                        let index = row * self.tiles + slot / 4;
                        (*rate0.add(index))[slot % 4] = baseline[row];
                        (*rate1.add(index))[slot % 4] = baseline[row];
                        (*adapt.add(index))[slot % 4] = 0.0;
                        (*support.add(index))[slot % 4] = 1.0;
                        (*release.add(index))[slot % 4] = 1.0;
                        for modulation_buffer in &self.modulation {
                            let modulation = modulation_buffer.contents() as *mut [f32; 4];
                            for family in 0..3 {
                                (*modulation.add((row * 3 + family) * self.tiles + slot / 4))
                                    [slot % 4] = 0.0;
                            }
                        }
                    }
                }
            }
            let efficacy = self.efficacy.contents() as *mut [f32; 4];
            let eligibility = self.eligibility.contents() as *mut [f32; 4];
            for edge in 0..PLASTIC_EDGES {
                for slot in 0..self.capacity {
                    if mask & (1u32 << slot) != 0 {
                        let index = edge * self.tiles + slot / 4;
                        (*efficacy.add(index))[slot % 4] = 0.0;
                        (*eligibility.add(index))[slot % 4] = 0.0;
                    }
                }
            }
        }
        for slot in 0..self.capacity {
            if mask & (1u32 << slot) != 0 {
                self.times[slot] = 0.0;
            }
        }
        if mask == self.valid_mask() {
            self.poisoned = None;
        }
        Ok(())
    }

    fn identity(&self) -> Value {
        json!({
            "format": FORMAT,
            "graph_sha256": self.metadata.graph_sha256,
            "atlas_sha256": self.metadata.atlas_sha256,
            "anatomy_sha256": self.metadata.anatomy_sha256,
            "morphology_sha256": self.metadata.morphology_sha256,
            "sensory_schema_sha256": self.metadata.sensory_schema_sha256,
            "actuator_schema_sha256": self.metadata.actuator_schema_sha256,
            "motor_calibration_sha256": self.metadata.motor_calibration_sha256,
            "graph_source_weight_sha256": self.metadata.graph_source_weight_sha256,
            "graph_quantization": self.metadata.graph_quantization,
            "plasticity_sha256": self.metadata.plasticity_sha256,
            "plasticity_contract": self.metadata.plasticity_contract,
            "readout_mask_sha256": self.metadata.readout_mask_sha256,
            "adapter_sha256": self.metadata.adapter_sha256,
            "service_artifact_sha256": self.artifact_sha256,
            "sensory_dim": INPUTS,
            "latent_dim": LATENT,
            "context_dim": CONTEXT_INPUTS,
            "motor_dim": MOTOR_OUTPUTS,
        })
    }

    fn runtime_metadata(&self) -> Value {
        json!({
            "device": self.device.name(),
            "neurons": N,
            "inputs": INPUTS,
            "readouts": LATENT,
            "context_inputs": CONTEXT_INPUTS,
            "motor_outputs": MOTOR_OUTPUTS,
            "kernel": "v5-row-f32-decoded-private-plasticity",
            "requested_kernel": if self.simd_rows { "simd" } else { "row" },
            "capacity": self.capacity,
            "storage_tiles": self.tiles,
            "dynamics": "private-gamma1pedc-plasticity-v5-two-0.005s-substeps",
            "readout_rank": READOUT_RANK,
            "snapshot_format": SNAPSHOT_FORMAT,
            "sensory_order": "channel-major optic_site_rgb_then_body807",
            "training_status": self.metadata.training_status,
            "research_intervention_enabled": std::env::var_os("CHREATURES_CNS_RESEARCH_INTERVENTION").as_deref() == Some(std::ffi::OsStr::new("1")),
            "provenance": self.metadata.provenance,
            "cns_adapter": self.identity(),
        })
    }

    fn snapshot_header(&self, metadata: String) -> SnapshotHeader {
        SnapshotHeader {
            format: SNAPSHOT_FORMAT.into(),
            neurons: N,
            capacity: self.capacity,
            storage_tiles: self.tiles,
            service_artifact_sha256: self.artifact_sha256.clone(),
            graph_sha256: self.metadata.graph_sha256.clone(),
            atlas_sha256: self.metadata.atlas_sha256.clone(),
            anatomy_sha256: self.metadata.anatomy_sha256.clone(),
            morphology_sha256: self.metadata.morphology_sha256.clone(),
            sensory_schema_sha256: self.metadata.sensory_schema_sha256.clone(),
            actuator_schema_sha256: self.metadata.actuator_schema_sha256.clone(),
            motor_calibration_sha256: self.metadata.motor_calibration_sha256.clone(),
            graph_source_weight_sha256: self.metadata.graph_source_weight_sha256.clone(),
            graph_quantization: self.metadata.graph_quantization.clone(),
            plasticity_sha256: self.metadata.plasticity_sha256.clone(),
            plasticity_contract: self.metadata.plasticity_contract.clone(),
            readout_mask_sha256: self.metadata.readout_mask_sha256.clone(),
            adapter_sha256: self.metadata.adapter_sha256.clone(),
            metadata,
        }
    }

    fn validate_snapshot_header(&self, header: &SnapshotHeader) -> Result<(), String> {
        if header.format != SNAPSHOT_FORMAT
            || header.neurons != N
            || header.capacity != self.capacity
            || header.storage_tiles != self.tiles
            || header.service_artifact_sha256 != self.artifact_sha256
            || header.graph_sha256 != self.metadata.graph_sha256
            || header.atlas_sha256 != self.metadata.atlas_sha256
            || header.anatomy_sha256 != self.metadata.anatomy_sha256
            || header.morphology_sha256 != self.metadata.morphology_sha256
            || header.sensory_schema_sha256 != self.metadata.sensory_schema_sha256
            || header.actuator_schema_sha256 != self.metadata.actuator_schema_sha256
            || header.motor_calibration_sha256 != self.metadata.motor_calibration_sha256
            || header.graph_source_weight_sha256 != self.metadata.graph_source_weight_sha256
            || header.graph_quantization != self.metadata.graph_quantization
            || header.plasticity_sha256 != self.metadata.plasticity_sha256
            || header.plasticity_contract != self.metadata.plasticity_contract
            || header.readout_mask_sha256 != self.metadata.readout_mask_sha256
            || header.adapter_sha256 != self.metadata.adapter_sha256
        {
            return Err("snapshot identity or shape differs from the loaded CNS service".into());
        }
        Ok(())
    }

    fn snapshot(&self, path: &Path, metadata: String) -> Result<(), String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!("refusing to snapshot poisoned CNS state: {reason}"));
        }
        let header = self.snapshot_header(metadata);
        let header_bytes =
            serde_json::to_vec(&header).map_err(|e| format!("serialize snapshot header: {e}"))?;
        if header_bytes.is_empty() || header_bytes.len() > 16_000_000 {
            return Err("snapshot metadata length is invalid".into());
        }
        let times = unsafe {
            std::slice::from_raw_parts(
                self.times.as_ptr() as *const u8,
                self.times.len() * size_of::<f64>(),
            )
        };
        let modulation = copy::<[f32; 4]>(&self.modulation[0], N * self.tiles * 3);
        let modulation_family = |family: usize| -> Vec<[f32; 4]> {
            let mut values = vec![[0.0; 4]; N * self.tiles];
            for row in 0..N {
                for tile in 0..self.tiles {
                    values[row * self.tiles + tile] =
                        modulation[(row * 3 + family) * self.tiles + tile];
                }
            }
            values
        };
        let state = [
            copy::<[f32; 4]>(&self.rate[0], N * self.tiles),
            copy::<[f32; 4]>(&self.adapt, N * self.tiles),
            copy::<[f32; 4]>(&self.support, N * self.tiles),
            copy::<[f32; 4]>(&self.release, N * self.tiles),
            modulation_family(0),
            modulation_family(1),
            modulation_family(2),
        ];
        let plastic_state = [
            copy::<[f32; 4]>(&self.efficacy, PLASTIC_EDGES * self.tiles),
            copy::<[f32; 4]>(&self.eligibility, PLASTIC_EDGES * self.tiles),
        ];
        validate_physical_state(&state, self.capacity, self.tiles)?;
        validate_plastic_state(
            &plastic_state,
            &self.maximum_depression,
            self.capacity,
            self.tiles,
        )?;
        exclusive_atomic_write(path, |file| {
            file.write_all(SNAPSHOT_MAGIC)?;
            file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
            file.write_all(&header_bytes)?;
            file.write_all(times)?;
            for values in &state {
                file.write_all(unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr() as *const u8,
                        std::mem::size_of_val(values.as_slice()),
                    )
                })?;
            }
            for values in &plastic_state {
                file.write_all(unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr() as *const u8,
                        std::mem::size_of_val(values.as_slice()),
                    )
                })?;
            }
            Ok(())
        })
    }

    fn read_snapshot(&self, path: &Path, load_state: bool) -> Result<SnapshotData, String> {
        let mut file = File::open(path).map_err(|e| format!("open snapshot: {e}"))?;
        let mut magic = [0u8; 9];
        file.read_exact(&mut magic)
            .map_err(|e| format!("read snapshot header: {e}"))?;
        if &magic != SNAPSHOT_MAGIC {
            return Err("snapshot header differs; only CNSSTATE5 is accepted".into());
        }
        let mut length = [0u8; 8];
        file.read_exact(&mut length)
            .map_err(|e| format!("read snapshot metadata length: {e}"))?;
        let header_len = usize::try_from(u64::from_le_bytes(length))
            .map_err(|_| "snapshot metadata length exceeds platform")?;
        if header_len == 0 || header_len > 16_000_000 {
            return Err("snapshot metadata length is invalid".into());
        }
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("read snapshot metadata: {e}"))?;
        let header: SnapshotHeader = serde_json::from_slice(&header_bytes)
            .map_err(|e| format!("invalid snapshot metadata: {e}"))?;
        self.validate_snapshot_header(&header)?;
        let mut times = vec![0f64; self.capacity];
        file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                times.as_mut_ptr() as *mut u8,
                times.len() * size_of::<f64>(),
            )
        })
        .map_err(|e| format!("read snapshot times: {e}"))?;
        if times.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err("snapshot times are invalid".into());
        }
        let neuronal_bytes = N * self.tiles * size_of::<[f32; 4]>();
        let plastic_bytes = PLASTIC_EDGES * self.tiles * size_of::<[f32; 4]>();
        let neuronal_state = if load_state {
            let mut arrays: [Vec<[f32; 4]>; 7] = std::array::from_fn(|_| Vec::new());
            for values in &mut arrays {
                values.resize(N * self.tiles, [0f32; 4]);
                file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, neuronal_bytes)
                })
                .map_err(|e| format!("read snapshot CNS state: {e}"))?;
                if values.iter().flatten().any(|x| !x.is_finite()) {
                    return Err("snapshot CNS state is nonfinite".into());
                }
            }
            Some(arrays)
        } else {
            file.seek(SeekFrom::Current((7 * neuronal_bytes) as i64))
                .map_err(|e| format!("inspect snapshot state length: {e}"))?;
            None
        };
        if let Some(values) = &neuronal_state {
            validate_physical_state(values, self.capacity, self.tiles)?;
        }
        let plastic_state = if load_state {
            let mut arrays: [Vec<[f32; 4]>; 2] = std::array::from_fn(|_| Vec::new());
            for values in &mut arrays {
                values.resize(PLASTIC_EDGES * self.tiles, [0f32; 4]);
                file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, plastic_bytes)
                })
                .map_err(|e| format!("read snapshot plastic state: {e}"))?;
            }
            validate_plastic_state(&arrays, &self.maximum_depression, self.capacity, self.tiles)?;
            Some(arrays)
        } else {
            file.seek(SeekFrom::Current((2 * plastic_bytes) as i64))
                .map_err(|e| format!("inspect snapshot plastic state length: {e}"))?;
            None
        };
        let expected_end = 9
            + 8
            + header_len as u64
            + (self.capacity * size_of::<f64>()) as u64
            + (7 * neuronal_bytes + 2 * plastic_bytes) as u64;
        if file
            .stream_position()
            .map_err(|e| format!("inspect snapshot position: {e}"))?
            != expected_end
            || file
                .metadata()
                .map_err(|e| format!("inspect snapshot size: {e}"))?
                .len()
                != expected_end
        {
            return Err("snapshot is truncated or has trailing state bytes".into());
        }
        Ok(SnapshotData {
            header,
            times,
            neuronal_state,
            plastic_state,
        })
    }

    fn restore(&mut self, path: &Path, mask: u32) -> Result<String, String> {
        if mask == 0 || mask & !self.valid_mask() != 0 {
            return Err("restore mask must select slots within capacity".into());
        }
        let loaded = self.read_snapshot(path, true)?;
        let state = loaded.neuronal_state.as_ref().unwrap();
        for (buffer, values) in [&self.rate[0], &self.adapt, &self.support, &self.release]
            .into_iter()
            .zip(state[..4].iter())
        {
            unsafe {
                let target = buffer.contents() as *mut [f32; 4];
                for row in 0..N {
                    for slot in 0..self.capacity {
                        if mask & (1u32 << slot) != 0 {
                            let index = row * self.tiles + slot / 4;
                            (*target.add(index))[slot % 4] = values[index][slot % 4];
                        }
                    }
                }
            }
        }
        unsafe {
            for modulation_buffer in &self.modulation {
                let target = modulation_buffer.contents() as *mut [f32; 4];
                for family in 0..3 {
                    for row in 0..N {
                        for slot in 0..self.capacity {
                            if mask & (1u32 << slot) != 0 {
                                let target_index = (row * 3 + family) * self.tiles + slot / 4;
                                let source_index = row * self.tiles + slot / 4;
                                (*target.add(target_index))[slot % 4] =
                                    state[4 + family][source_index][slot % 4];
                            }
                        }
                    }
                }
            }
            let plastic = loaded.plastic_state.as_ref().unwrap();
            for (buffer, values) in [
                (&self.efficacy, &plastic[0]),
                (&self.eligibility, &plastic[1]),
            ] {
                let target = buffer.contents() as *mut [f32; 4];
                for edge in 0..PLASTIC_EDGES {
                    for slot in 0..self.capacity {
                        if mask & (1u32 << slot) != 0 {
                            let index = edge * self.tiles + slot / 4;
                            (*target.add(index))[slot % 4] = values[index][slot % 4];
                        }
                    }
                }
            }
        }
        for slot in 0..self.capacity {
            if mask & (1u32 << slot) != 0 {
                self.times[slot] = loaded.times[slot];
            }
        }
        if mask == self.valid_mask() {
            self.poisoned = None;
        }
        Ok(loaded.header.metadata)
    }

    fn capture_rates(&self, path: &Path, slot: usize) -> Result<(), String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!("refusing to capture poisoned CNS state: {reason}"));
        }
        if slot >= self.capacity {
            return Err("capture slot exceeds configured capacity".into());
        }
        let raw = copy::<[f32; 4]>(&self.rate[0], N * self.tiles);
        let values: Vec<f32> = (0..N)
            .map(|row| raw[row * self.tiles + slot / 4][slot % 4])
            .collect();
        exclusive_atomic_write(path, |file| {
            file.write_all(unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    values.len() * size_of::<f32>(),
                )
            })
        })
    }
}

fn exclusive_atomic_write(
    path: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "refusing to overwrite existing file: {}",
            path.display()
        ));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| "output path has no filename".to_string())?
        .to_string_lossy();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("clock error: {e}"))?
        .as_nanos();
    let temporary: PathBuf = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| format!("create temporary output: {e}"))?;
        write(&mut file).map_err(|e| format!("write output: {e}"))?;
        file.sync_all().map_err(|e| format!("sync output: {e}"))?;
        std::fs::hard_link(&temporary, path)
            .map_err(|e| format!("publish output without overwrite: {e}"))?;
        let directory_sync = OpenOptions::new()
            .read(true)
            .open(parent)
            .and_then(|directory| directory.sync_all());
        if let Err(e) = directory_sync {
            return Err(format!("sync output directory: {e}"));
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result
}

fn ok_reply(fields: Value) -> Value {
    let mut object = match fields {
        Value::Object(x) => x,
        _ => serde_json::Map::new(),
    };
    object.insert("ok".into(), Value::Bool(true));
    Value::Object(object)
}

fn error_reply(error: impl ToString) -> Value {
    json!({"ok": false, "error": error.to_string()})
}

fn main() {
    let mut args = std::env::args().skip(1);
    let artifact = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("usage: metal-brain-server SERVICE_ARTIFACT (row|simd) CAPACITY");
            std::process::exit(2);
        }
    };
    let kernel = args.next().unwrap_or_default();
    if !matches!(kernel.as_str(), "row" | "simd") {
        eprintln!("kernel must be row or simd");
        std::process::exit(2);
    }
    let capacity = match args.next().and_then(|x| x.parse::<usize>().ok()) {
        Some(x) if (1..=MAX_CAPACITY).contains(&x) => x,
        _ => {
            eprintln!("capacity must be an integer in 1..={MAX_CAPACITY}");
            std::process::exit(2);
        }
    };
    if args.next().is_some() {
        eprintln!("unexpected command-line argument");
        std::process::exit(2);
    }
    let mut engine = match Engine::load(Path::new(&artifact), kernel == "simd", capacity) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("CNS service startup failed: {error}");
            std::process::exit(1);
        }
    };
    println!("{}", ok_reply(engine.runtime_metadata()));
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let response = match line {
            Err(error) => error_reply(error),
            Ok(line) => match serde_json::from_str::<Request>(&line) {
                Err(error) => error_reply(error),
                Ok(Request::Step {
                    dt,
                    active_mask,
                    sensory,
                    context,
                    research_current,
                    selected_neuron_indices,
                }) => match engine.step(
                    dt,
                    active_mask,
                    &sensory,
                    &context,
                    research_current.as_deref(),
                    &selected_neuron_indices,
                ) {
                    Ok((latent, motor, selected, physiology, times, gpu_ms, phase_ms)) => {
                        ok_reply(json!({
                            "latent": latent,
                            "motor": motor,
                            "selected_rates": if selected_neuron_indices.is_empty() { None } else { Some(selected) },
                            "physiology": physiology,
                            "times": times,
                            "gpu_ms": gpu_ms,
                            "gpu_phase_ms": {
                                "afferents": phase_ms[0],
                                "recurrence": phase_ms[1],
                                "readout": phase_ms[2],
                                "observer": phase_ms[3],
                            },
                        }))
                    }
                    Err(error) => error_reply(error),
                },
                Ok(Request::Reset { mask }) => match engine.reset(mask) {
                    Ok(()) => ok_reply(json!({})),
                    Err(error) => error_reply(error),
                },
                Ok(Request::Snapshot { path, metadata }) => {
                    match engine.snapshot(Path::new(&path), metadata) {
                        Ok(()) => ok_reply(json!({})),
                        Err(error) => error_reply(error),
                    }
                }
                Ok(Request::InspectSnapshot { path }) => {
                    match engine.read_snapshot(Path::new(&path), true) {
                        Ok(snapshot) => ok_reply(json!({
                            "metadata": snapshot.header.metadata,
                            "times": snapshot.times,
                            "snapshot_identity": snapshot.header,
                        })),
                        Err(error) => error_reply(error),
                    }
                }
                Ok(Request::Restore { path, mask }) => match engine.restore(Path::new(&path), mask)
                {
                    Ok(metadata) => ok_reply(json!({"metadata": metadata, "times": engine.times})),
                    Err(error) => error_reply(error),
                },
                Ok(Request::CaptureRates { path, slot }) => {
                    match engine.capture_rates(Path::new(&path), slot as usize) {
                        Ok(()) => ok_reply(
                            json!({"path": path, "slot": slot, "neurons": N, "dtype": "<f4"}),
                        ),
                        Err(error) => error_reply(error),
                    }
                }
                Ok(Request::Metadata) => ok_reply(engine.runtime_metadata()),
                Ok(Request::Shutdown) => {
                    println!("{}", ok_reply(json!({})));
                    break;
                }
            },
        };
        println!("{response}");
    }
}
