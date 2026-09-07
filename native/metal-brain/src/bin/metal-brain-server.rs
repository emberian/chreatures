#![allow(unexpected_cfgs)] // objc 0.2 macros still probe the historical cargo-clippy cfg.

#[path = "../mps_matrix.rs"]
mod mps_matrix;

use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use mps_matrix::{
    command_buffer_error, command_buffer_gpu_milliseconds, recommended_row_bytes,
    MatrixMultiplication,
};
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
const ARTIFACT_MAGIC: &[u8; 8] = b"CHCNS1\0\0";
const SNAPSHOT_MAGIC: &[u8; 9] = b"CNSSTATE1";
const FORMAT: &str = "chreatures-cns-service-v1";
const SNAPSHOT_FORMAT: &str = "chreatures-cns-state-v1";
const N: usize = 165_122;
const E: usize = 25_563_197;
const SITES: usize = 1_771;
const RECEPTORS: usize = 4_107;
const RECEPTOR_TYPES: usize = 10;
const SITE_EDGES: usize = 4_669;
const BODY_TARGETS: usize = 11_233;
const NEURON_TYPES: usize = 11_752;
const BODY_INPUTS: usize = 43;
const BODY_HIDDEN: usize = 128;
const INPUTS: usize = 5_356;
const LATENT: usize = 512;
const MAX_CAPACITY: usize = 32;
const PARAMETER_ORDER: [&str; 16] = [
    "optic.spectral_logits",
    "optic.gain_raw",
    "optic.bias",
    "body.mean",
    "body.scale",
    "body.input.weight",
    "body.input.bias",
    "body.output.weight",
    "body.output.bias",
    "dynamics.bias_raw",
    "dynamics.tau_raw",
    "dynamics.source_raw",
    "dynamics.target_raw",
    "dynamics.excitability_raw",
    "readout.weight",
    "readout.bias",
];
const ARRAY_NAMES: [&str; 26] = [
    "graph.crow",
    "graph.col",
    "graph.weight",
    "atlas.receptor_rows",
    "atlas.receptor_type",
    "atlas.receptor_ptr",
    "atlas.site_indices",
    "atlas.site_weight",
    "atlas.body_rows",
    "atlas.neuron_type",
    "optic.spectral_logits",
    "optic.gain_raw",
    "optic.bias",
    "body.mean",
    "body.scale",
    "body.input.weight",
    "body.input.bias",
    "body.output.weight",
    "body.output.bias",
    "dynamics.bias_raw",
    "dynamics.tau_raw",
    "dynamics.source_raw",
    "dynamics.target_raw",
    "dynamics.excitability_raw",
    "readout.weight",
    "readout.bias",
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
    body_hidden: usize,
    inputs: usize,
    latent: usize,
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
        body_hidden: BODY_HIDDEN,
        inputs: INPUTS,
        latent: LATENT,
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ArtifactMetadata {
    format: String,
    graph_sha256: String,
    atlas_sha256: String,
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
            &self.readout_mask_sha256,
            &self.adapter_sha256,
        ] {
            require_hash(hash)?;
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
    readout_mask_sha256: String,
    adapter_sha256: String,
    metadata: String,
}

struct SnapshotData {
    header: SnapshotHeader,
    times: Vec<f64>,
    state: Option<[Vec<[f32; 4]>; 3]>,
}

fn validate_physical_state(
    state: &[Vec<[f32; 4]>; 3],
    capacity: usize,
    tiles: usize,
) -> Result<(), String> {
    let [rate, adapt, support] = state;
    for row in 0..N {
        for slot in 0..capacity {
            let index = row * tiles + slot / 4;
            let lane = slot % 4;
            if !(0.0..=1.0).contains(&rate[index][lane])
                || !(0.0..=1.0).contains(&adapt[index][lane])
                || !(0.65..=1.0).contains(&support[index][lane])
            {
                return Err("snapshot CNS state is nonfinite or outside physical bounds".into());
            }
        }
    }
    Ok(())
}

struct Engine {
    body_input_mm: MatrixMultiplication,
    body_output_mm: MatrixMultiplication,
    device: Device,
    queue: metal::CommandQueue,
    capacity: usize,
    tiles: usize,
    simd_rows: bool,
    metadata: ArtifactMetadata,
    artifact_sha256: String,
    rec: [Buffer; 3],
    receptor: [Buffer; 5],
    body_rows: Buffer,
    optic_spectral: Buffer,
    optic_gain: Buffer,
    optic_bias: Buffer,
    body_mean: Buffer,
    body_scale: Buffer,
    // MPSMatrix retains these backing allocations indirectly; fields make that lifetime explicit.
    _body_input_weight: Buffer,
    body_input_bias: Buffer,
    _body_output_weight: Buffer,
    body_output_bias: Buffer,
    dynamics: [Buffer; 5],
    readout_weight: Buffer,
    readout_stride: Buffer,
    readout_bias: Buffer,
    sensory: Buffer,
    body_normalized: Buffer,
    body_hidden_pre: Buffer,
    body_hidden: Buffer,
    body_current_pre: Buffer,
    rate: [Buffer; 2],
    adapt: Buffer,
    support: Buffer,
    drive: Buffer,
    latent_pre: Buffer,
    latent: Buffer,
    physiology_partial: Buffer,
    physiology: Buffer,
    times: Vec<f64>,
    poisoned: Option<String>,
    k_clear: ComputePipelineState,
    k_optic: ComputePipelineState,
    k_normalize_body: ComputePipelineState,
    k_tanh_bias: ComputePipelineState,
    k_body_scatter: ComputePipelineState,
    k_rec: ComputePipelineState,
    k_gather: ComputePipelineState,
    k_readout: ComputePipelineState,
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
            return Err("artifact header differs; only CHCNS1 is accepted".into());
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
        let graph_weight = reader.array::<f32>(&metadata, "graph.weight", E)?;
        let receptor_rows = reader.array::<u32>(&metadata, "atlas.receptor_rows", RECEPTORS)?;
        let receptor_type = reader.array::<u32>(&metadata, "atlas.receptor_type", RECEPTORS)?;
        let receptor_ptr = reader.array::<u32>(&metadata, "atlas.receptor_ptr", RECEPTORS + 1)?;
        let site_indices = reader.array::<u32>(&metadata, "atlas.site_indices", SITE_EDGES)?;
        let site_weight = reader.array::<f32>(&metadata, "atlas.site_weight", SITE_EDGES)?;
        let body_rows = reader.array::<u32>(&metadata, "atlas.body_rows", BODY_TARGETS)?;
        let neuron_type = reader.array::<u32>(&metadata, "atlas.neuron_type", N)?;
        let spectral_logits =
            reader.array::<f32>(&metadata, "optic.spectral_logits", RECEPTOR_TYPES * 3)?;
        let gain_raw = reader.array::<f32>(&metadata, "optic.gain_raw", RECEPTOR_TYPES)?;
        let optic_bias = reader.array::<f32>(&metadata, "optic.bias", RECEPTOR_TYPES)?;
        let body_mean = reader.array::<f32>(&metadata, "body.mean", BODY_INPUTS)?;
        let body_scale = reader.array::<f32>(&metadata, "body.scale", BODY_INPUTS)?;
        let body_input_weight =
            reader.array::<f32>(&metadata, "body.input.weight", BODY_HIDDEN * BODY_INPUTS)?;
        let body_input_bias = reader.array::<f32>(&metadata, "body.input.bias", BODY_HIDDEN)?;
        let body_output_weight =
            reader.array::<f32>(&metadata, "body.output.weight", BODY_TARGETS * BODY_HIDDEN)?;
        let body_output_bias = reader.array::<f32>(&metadata, "body.output.bias", BODY_TARGETS)?;
        let dynamics_bias_raw =
            reader.array::<f32>(&metadata, "dynamics.bias_raw", NEURON_TYPES)?;
        let dynamics_tau_raw = reader.array::<f32>(&metadata, "dynamics.tau_raw", NEURON_TYPES)?;
        let dynamics_source_raw =
            reader.array::<f32>(&metadata, "dynamics.source_raw", NEURON_TYPES)?;
        let dynamics_target_raw =
            reader.array::<f32>(&metadata, "dynamics.target_raw", NEURON_TYPES)?;
        let dynamics_excitability_raw =
            reader.array::<f32>(&metadata, "dynamics.excitability_raw", NEURON_TYPES)?;
        let mut readout_weight = reader.array::<f32>(&metadata, "readout.weight", LATENT * N)?;
        let readout_bias = reader.array::<f32>(&metadata, "readout.bias", LATENT)?;
        let artifact_sha256 = reader.finish()?;

        if graph_crow.first() != Some(&0)
            || graph_crow.last() != Some(&(E as u32))
            || graph_crow.windows(2).any(|x| x[0] > x[1])
        {
            return Err("invalid graph.crow".into());
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
        if receptor_rows.windows(2).any(|x| x[0] >= x[1])
            || body_rows.windows(2).any(|x| x[0] >= x[1])
        {
            return Err("afferent rows must be unique and ascending".into());
        }
        let receptor_set: BTreeSet<u32> = receptor_rows.iter().copied().collect();
        if body_rows.iter().any(|row| receptor_set.contains(row)) {
            return Err("optic and body afferents overlap".into());
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
            ("graph.weight", graph_weight.as_slice()),
            ("optic.spectral_logits", spectral_logits.as_slice()),
            ("optic.gain_raw", gain_raw.as_slice()),
            ("optic.bias", optic_bias.as_slice()),
            ("body.mean", body_mean.as_slice()),
            ("body.scale", body_scale.as_slice()),
            ("body.input.weight", body_input_weight.as_slice()),
            ("body.input.bias", body_input_bias.as_slice()),
            ("body.output.weight", body_output_weight.as_slice()),
            ("body.output.bias", body_output_bias.as_slice()),
            ("dynamics.bias_raw", dynamics_bias_raw.as_slice()),
            ("dynamics.tau_raw", dynamics_tau_raw.as_slice()),
            ("dynamics.source_raw", dynamics_source_raw.as_slice()),
            ("dynamics.target_raw", dynamics_target_raw.as_slice()),
            (
                "dynamics.excitability_raw",
                dynamics_excitability_raw.as_slice(),
            ),
            ("readout.weight", readout_weight.as_slice()),
            ("readout.bias", readout_bias.as_slice()),
        ] {
            all_finite(name, values)?;
        }
        if body_scale.iter().any(|x| *x <= 0.0) {
            return Err("body.scale must be positive".into());
        }

        let mut readout_mask = vec![1u8; N];
        for &row in receptor_rows.iter().chain(body_rows.iter()) {
            readout_mask[row as usize] = 0;
        }
        if sha256_hex(&readout_mask) != metadata.readout_mask_sha256 {
            return Err("readout_mask_sha256 differs from the enforced afferent mask".into());
        }
        for latent_row in 0..LATENT {
            let base = latent_row * N;
            for &afferent in receptor_rows.iter().chain(body_rows.iter()) {
                readout_weight[base + afferent as usize] = 0.0;
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
        let dyn_bias = expand(&dynamics_bias_raw, |x| 0.5 * x.tanh());
        let dyn_tau = expand(&dynamics_tau_raw, |x| 0.025 + 0.475 * sigmoid(x));
        let dyn_source = expand(&dynamics_source_raw, |x| 0.5 + sigmoid(x));
        let dyn_target = expand(&dynamics_target_raw, |x| 0.5 + sigmoid(x));
        let dyn_excitability = expand(&dynamics_excitability_raw, |x| 0.5 + sigmoid(x));

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
        let mut ones = vec![[1f32; 4]; N * tiles];
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
        ];
        let receptor = [
            buf(&device, &receptor_rows),
            buf(&device, &receptor_type),
            buf(&device, &receptor_ptr),
            buf(&device, &site_indices),
            buf(&device, &site_weight),
        ];
        let body_rows_buffer = buf(&device, &body_rows);
        let optic_spectral_buffer = buf(&device, &spectral);
        let optic_gain_buffer = buf(&device, &optic_gain);
        let optic_bias_buffer = buf(&device, &optic_bias);
        let body_mean_buffer = buf(&device, &body_mean);
        let body_scale_buffer = buf(&device, &body_scale);
        let body_input_row_bytes = recommended_row_bytes(BODY_INPUTS);
        let body_input_weight_padded = pad_matrix_rows(
            &body_input_weight,
            BODY_HIDDEN,
            BODY_INPUTS,
            body_input_row_bytes,
        )?;
        let body_input_weight_buffer = buf(&device, &body_input_weight_padded);
        let body_input_bias_buffer = buf(&device, &body_input_bias);
        let body_output_weight_buffer = buf(&device, &body_output_weight);
        let body_output_bias_buffer = buf(&device, &body_output_bias);
        let dynamics = [
            buf(&device, &dyn_bias),
            buf(&device, &dyn_tau),
            buf(&device, &dyn_source),
            buf(&device, &dyn_target),
            buf(&device, &dyn_excitability),
        ];
        let readout_row_bytes = recommended_row_bytes(N);
        let readout_weight_padded = pad_matrix_rows(&readout_weight, LATENT, N, readout_row_bytes)?;
        let readout_weight_buffer = buf(&device, &readout_weight_padded);
        let readout_stride_buffer = buf(&device, &[(readout_row_bytes / size_of::<f32>()) as u32]);
        let readout_bias_buffer = buf(&device, &readout_bias);
        let sensory = zeros(&device, INPUTS * tiles);
        let body_normalized = zeros(&device, BODY_INPUTS * tiles);
        let body_hidden_pre = zeros(&device, BODY_HIDDEN * tiles);
        let body_hidden = zeros(&device, BODY_HIDDEN * tiles);
        let body_current_pre = zeros(&device, BODY_TARGETS * tiles);
        let rate = [buf(&device, &z), buf(&device, &z)];
        let adapt = buf(&device, &z);
        let support = buf(&device, &ones);
        let drive = zeros(&device, N * tiles);
        let latent_pre = zeros(&device, LATENT * tiles);
        let latent = zeros(&device, LATENT * tiles);
        let physiology_partial = zeros(&device, N.div_ceil(256) * tiles * 3);
        let physiology = zeros(&device, 3 * tiles);
        let row_bytes = tiles * size_of::<[f32; 4]>();
        let body_input_mm = MatrixMultiplication::new(
            &device,
            &body_input_weight_buffer,
            BODY_HIDDEN,
            BODY_INPUTS,
            body_input_row_bytes,
            0,
            &body_normalized,
            capacity,
            row_bytes,
            0,
            &body_hidden_pre,
            row_bytes,
        )?;
        let body_output_mm = MatrixMultiplication::new(
            &device,
            &body_output_weight_buffer,
            BODY_TARGETS,
            BODY_HIDDEN,
            BODY_HIDDEN * size_of::<f32>(),
            0,
            &body_hidden,
            capacity,
            row_bytes,
            0,
            &body_current_pre,
            row_bytes,
        )?;
        Ok(Self {
            body_input_mm,
            body_output_mm,
            queue: device.new_command_queue(),
            capacity,
            tiles,
            simd_rows,
            metadata,
            artifact_sha256,
            rec,
            receptor,
            body_rows: body_rows_buffer,
            optic_spectral: optic_spectral_buffer,
            optic_gain: optic_gain_buffer,
            optic_bias: optic_bias_buffer,
            body_mean: body_mean_buffer,
            body_scale: body_scale_buffer,
            _body_input_weight: body_input_weight_buffer,
            body_input_bias: body_input_bias_buffer,
            _body_output_weight: body_output_weight_buffer,
            body_output_bias: body_output_bias_buffer,
            dynamics,
            readout_weight: readout_weight_buffer,
            readout_stride: readout_stride_buffer,
            readout_bias: readout_bias_buffer,
            sensory,
            body_normalized,
            body_hidden_pre,
            body_hidden,
            body_current_pre,
            rate,
            adapt,
            support,
            drive,
            latent_pre,
            latent,
            physiology_partial,
            physiology,
            times: vec![0.0; capacity],
            poisoned: None,
            k_clear: pipeline("clear_drive")?,
            k_optic: pipeline("project_optic")?,
            k_normalize_body: pipeline("normalize_body")?,
            k_tanh_bias: pipeline("tanh_bias")?,
            k_body_scatter: pipeline("sigmoid_bias_scatter_body")?,
            k_rec: pipeline(if simd_rows {
                "csr_rate_simd"
            } else {
                "csr_rate"
            })?,
            k_gather: pipeline("gather_rates")?,
            k_readout: pipeline("dense_readout")?,
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
        selected_indices: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>, f64, [f64; 4]), String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!(
                "CNS state is poisoned after a failed GPU tick ({reason}); restore a full coherent snapshot or cold-reset all slots"
            ));
        }
        if !dt.is_finite() || dt <= 0.0 || dt > 0.2 {
            return Err("dt must be finite and in (0,0.2]".into());
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
        let selected_index_buffer =
            (!selected_indices.is_empty()).then(|| buf(&self.device, selected_indices));
        let selected_buffer = (!selected_indices.is_empty())
            .then(|| zeros(&self.device, selected_indices.len() * self.tiles));
        let gpu_dt = dt as f32;
        let p0 = buf(&self.device, &[self.params(gpu_dt, mask, false)]);
        let p1 = buf(&self.device, &[self.params(gpu_dt, mask, true)]);
        let hidden_rows = buf(&self.device, &[BODY_HIDDEN as u32]);
        let latent_rows = buf(&self.device, &[LATENT as u32]);
        let cb_afferents = self.queue.new_command_buffer();

        self.encode_compute(
            cb_afferents,
            &self.k_clear,
            &[&self.drive],
            &p0,
            N * self.tiles,
        );
        self.encode_compute(
            cb_afferents,
            &self.k_normalize_body,
            &[
                &self.sensory,
                &self.body_mean,
                &self.body_scale,
                &self.body_normalized,
            ],
            &p0,
            BODY_INPUTS * self.tiles,
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
            Self::grid(enc, &self.k_optic, RECEPTORS * self.tiles);
            enc.end_encoding();
        }
        self.body_input_mm.encode(cb_afferents);
        {
            let enc = cb_afferents.new_compute_command_encoder();
            bind(
                enc,
                &self.k_tanh_bias,
                &[
                    &self.body_hidden_pre,
                    &self.body_input_bias,
                    &self.body_hidden,
                ],
            );
            enc.set_buffer(8, Some(&p0), 0);
            enc.set_buffer(9, Some(&hidden_rows), 0);
            Self::grid(enc, &self.k_tanh_bias, BODY_HIDDEN * self.tiles);
            enc.end_encoding();
        }
        self.body_output_mm.encode(cb_afferents);
        self.encode_compute(
            cb_afferents,
            &self.k_body_scatter,
            &[
                &self.body_current_pre,
                &self.body_output_bias,
                &self.body_rows,
                &self.drive,
            ],
            &p0,
            BODY_TARGETS * self.tiles,
        );
        let cb_recurrence = self.queue.new_command_buffer();
        for (params, input, output) in [
            (&p0, &self.rate[0], &self.rate[1]),
            (&p1, &self.rate[1], &self.rate[0]),
        ] {
            let enc = cb_recurrence.new_compute_command_encoder();
            bind(
                enc,
                &self.k_rec,
                &[
                    &self.rec[0],
                    &self.rec[1],
                    &self.rec[2],
                    input,
                    output,
                    &self.adapt,
                    &self.support,
                    &self.drive,
                ],
            );
            enc.set_buffer(8, Some(params), 0);
            for (slot, parameter) in self.dynamics.iter().enumerate() {
                enc.set_buffer(9 + slot as u64, Some(parameter), 0);
            }
            if self.simd_rows {
                enc.dispatch_threads(
                    MTLSize::new((N * self.tiles * 32) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            } else {
                Self::grid(enc, &self.k_rec, N * self.tiles);
            }
            enc.end_encoding();
        }
        let cb_readout = self.queue.new_command_buffer();
        {
            let enc = cb_readout.new_compute_command_encoder();
            bind(
                enc,
                &self.k_readout,
                &[&self.readout_weight, &self.rate[0], &self.latent_pre],
            );
            enc.set_buffer(8, Some(&p1), 0);
            enc.set_buffer(9, Some(&self.readout_stride), 0);
            enc.dispatch_thread_groups(
                MTLSize::new((LATENT * self.tiles) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        {
            let enc = cb_readout.new_compute_command_encoder();
            bind(
                enc,
                &self.k_tanh_bias,
                &[&self.latent_pre, &self.readout_bias, &self.latent],
            );
            enc.set_buffer(8, Some(&p1), 0);
            enc.set_buffer(9, Some(&latent_rows), 0);
            Self::grid(enc, &self.k_tanh_bias, LATENT * self.tiles);
            enc.end_encoding();
        }
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
        let selected = selected_buffer
            .as_ref()
            .map_or_else(Vec::new, |b| unpack(b, selected_indices.len()));
        let physiology = unpack(&self.physiology, 3);
        if latent
            .iter()
            .chain(selected.iter())
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
        unsafe {
            let rate0 = self.rate[0].contents() as *mut [f32; 4];
            let rate1 = self.rate[1].contents() as *mut [f32; 4];
            let adapt = self.adapt.contents() as *mut [f32; 4];
            let support = self.support.contents() as *mut [f32; 4];
            for row in 0..N {
                for slot in 0..self.capacity {
                    if mask & (1u32 << slot) != 0 {
                        let index = row * self.tiles + slot / 4;
                        (*rate0.add(index))[slot % 4] = 0.0;
                        (*rate1.add(index))[slot % 4] = 0.0;
                        (*adapt.add(index))[slot % 4] = 0.0;
                        (*support.add(index))[slot % 4] = 1.0;
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
            "readout_mask_sha256": self.metadata.readout_mask_sha256,
            "adapter_sha256": self.metadata.adapter_sha256,
            "service_artifact_sha256": self.artifact_sha256,
            "sensory_dim": INPUTS,
            "latent_dim": LATENT,
        })
    }

    fn runtime_metadata(&self) -> Value {
        json!({
            "device": self.device.name(),
            "neurons": N,
            "inputs": INPUTS,
            "readouts": LATENT,
            "kernel": if self.simd_rows { "simd" } else { "row" },
            "capacity": self.capacity,
            "storage_tiles": self.tiles,
            "sensory_order": "optic_site_major_rgb_then_body43",
            "training_status": self.metadata.training_status,
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
        let state = [
            copy::<[f32; 4]>(&self.rate[0], N * self.tiles),
            copy::<[f32; 4]>(&self.adapt, N * self.tiles),
            copy::<[f32; 4]>(&self.support, N * self.tiles),
        ];
        validate_physical_state(&state, self.capacity, self.tiles)?;
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
            Ok(())
        })
    }

    fn read_snapshot(&self, path: &Path, load_state: bool) -> Result<SnapshotData, String> {
        let mut file = File::open(path).map_err(|e| format!("open snapshot: {e}"))?;
        let mut magic = [0u8; 9];
        file.read_exact(&mut magic)
            .map_err(|e| format!("read snapshot header: {e}"))?;
        if &magic != SNAPSHOT_MAGIC {
            return Err("snapshot header differs; only CNSSTATE1 is accepted".into());
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
        let state_bytes = N * self.tiles * size_of::<[f32; 4]>();
        let state = if load_state {
            let mut arrays = [Vec::new(), Vec::new(), Vec::new()];
            for values in &mut arrays {
                values.resize(N * self.tiles, [0f32; 4]);
                file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, state_bytes)
                })
                .map_err(|e| format!("read snapshot CNS state: {e}"))?;
                if values.iter().flatten().any(|x| !x.is_finite()) {
                    return Err("snapshot CNS state is nonfinite".into());
                }
            }
            Some(arrays)
        } else {
            file.seek(SeekFrom::Current((3 * state_bytes) as i64))
                .map_err(|e| format!("inspect snapshot state length: {e}"))?;
            None
        };
        if let Some(values) = &state {
            validate_physical_state(values, self.capacity, self.tiles)?;
        }
        let expected_end = 9
            + 8
            + header_len as u64
            + (self.capacity * size_of::<f64>()) as u64
            + (3 * state_bytes) as u64;
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
            state,
        })
    }

    fn restore(&mut self, path: &Path, mask: u32) -> Result<String, String> {
        if mask == 0 || mask & !self.valid_mask() != 0 {
            return Err("restore mask must select slots within capacity".into());
        }
        let loaded = self.read_snapshot(path, true)?;
        let state = loaded.state.unwrap();
        for (buffer, values) in [&self.rate[0], &self.adapt, &self.support]
            .into_iter()
            .zip(state.iter())
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
                    selected_neuron_indices,
                }) => match engine.step(dt, active_mask, &sensory, &selected_neuron_indices) {
                    Ok((latent, selected, physiology, times, gpu_ms, phase_ms)) => {
                        ok_reply(json!({
                            "latent": latent,
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
