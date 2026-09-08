// AGPL-3.0-or-later
// Z = tanh(U @ (V @ (mask * (rate-r0))) + b), rank 64. Injected rows
// are authenticated as mask=0 and therefore cannot directly reach Z.

const NEURONS: u32 = 165122u;
const TYPES: u32 = 11752u;
const RANK: u32 = 64u;
const LATENT: u32 = 512u;
const BLOCK_SIZE: u32 = 256u;
const BLOCKS: u32 = 646u;

struct Config {
  capacity: u32,
  active_mask: u32,
  reset_mask: u32,
  selected_resident: u32,
  dt: f32,
  neuron_count: u32,
  edge_count: u32,
  _pad: u32,
};

struct NeuronState {
  rate: vec4<f32>,
  adaptation: vec4<f32>,
  support: vec4<f32>,
  release: vec4<f32>,
  m_da: vec4<f32>,
  m_oa: vec4<f32>,
  m_ht: vec4<f32>,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> state: array<NeuronState>;
@group(0) @binding(2) var<storage, read> baseline_rate: array<f32>;
@group(0) @binding(4) var<storage, read> readout_mask: array<u32>;
@group(0) @binding(5) var<storage, read> projection: array<f32>;
@group(0) @binding(6) var<storage, read_write> projection_partial: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> projected: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> output_weight: array<f32>;
@group(0) @binding(9) var<storage, read> output_bias: array<f32>;
@group(0) @binding(10) var<storage, read_write> latent: array<vec4<f32>>;

var<workgroup> reduction: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn project_partial(
  @builtin(local_invocation_index) local: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let neuron = group.x * BLOCK_SIZE + local;
  let rank = group.y;
  var value = vec4<f32>(0.0);
  if (neuron < NEURONS && readout_mask[neuron] != 0u) {
    let x = state[neuron].rate - vec4<f32>(baseline_rate[neuron]);
    value = projection[rank * NEURONS + neuron] * x;
  }
  reduction[local] = value;
  workgroupBarrier();
  var stride = BLOCK_SIZE / 2u;
  loop {
    if (local < stride) { reduction[local] += reduction[local + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (local == 0u) {
    projection_partial[rank * BLOCKS + group.x] = reduction[0];
  }
}

@compute @workgroup_size(256)
fn reduce_projection(
  @builtin(local_invocation_index) local: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let rank = group.x;
  var value = vec4<f32>(0.0);
  for (var block = local; block < BLOCKS; block += BLOCK_SIZE) {
    value += projection_partial[rank * BLOCKS + block];
  }
  reduction[local] = value;
  workgroupBarrier();
  var stride = BLOCK_SIZE / 2u;
  loop {
    if (local < stride) { reduction[local] += reduction[local + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (local == 0u) { projected[rank] = reduction[0]; }
}

@compute @workgroup_size(128)
fn output_latent(@builtin(global_invocation_id) id: vec3<u32>) {
  let output = id.x;
  if (output >= LATENT) { return; }
  var value = vec4<f32>(output_bias[output]);
  for (var rank = 0u; rank < RANK; rank++) {
    value += output_weight[output * RANK + rank] * projected[rank];
  }
  let enabled_lanes = vec4<bool>(
    (config.active_mask & 1u) != 0u,
    (config.active_mask & 2u) != 0u,
    (config.active_mask & 4u) != 0u,
    (config.active_mask & 8u) != 0u,
  );
  latent[output] = select(vec4<f32>(0.0), tanh(value), enabled_lanes);
}
