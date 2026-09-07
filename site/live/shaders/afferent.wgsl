// AGPL-3.0-or-later
// Raw optical and body-local measurements end at this adapter. Only `drive`
// enters the recurrent CNS; it is never exposed to the readout or controller.

const CAPACITY: u32 = 4u;
const OPTIC_SITES: u32 = 1771u;
const OPTIC_VALUES: u32 = 5313u;
const RECEPTOR_TYPES: u32 = 10u;
const RECEPTORS: u32 = 4107u;
const BODY_CHANNELS: u32 = 43u;
const BODY_HIDDEN: u32 = 128u;
const BODY_TARGETS: u32 = 11233u;

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

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> optic_rgb: array<f32>;
@group(0) @binding(2) var<storage, read> body_input: array<f32>;
@group(0) @binding(3) var<storage, read_write> drive: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> receptor_rows: array<u32>;
@group(0) @binding(5) var<storage, read> receptor_type: array<u32>;
@group(0) @binding(6) var<storage, read> receptor_ptr: array<u32>;
@group(0) @binding(7) var<storage, read> site_indices: array<u32>;
@group(0) @binding(8) var<storage, read> site_weight: array<f32>;
// Per type: spectral logits RGB, gain_raw, bias (stride 5).
@group(0) @binding(9) var<storage, read> optic_parameters: array<f32>;
@group(0) @binding(10) var<storage, read_write> optic_mixed: array<vec4<f32>>;
// mean[43], scale[43]
@group(0) @binding(11) var<storage, read> body_normalizer: array<f32>;
@group(0) @binding(12) var<storage, read> body_input_weight: array<f32>;
@group(0) @binding(13) var<storage, read> body_input_bias: array<f32>;
@group(0) @binding(14) var<storage, read_write> body_hidden: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read> body_output_weight: array<f32>;
@group(0) @binding(16) var<storage, read> body_output_bias: array<f32>;
@group(0) @binding(17) var<storage, read> body_rows: array<u32>;

fn active_lanes() -> vec4<bool> {
  return vec4<bool>(
    (config.active_mask & 1u) != 0u,
    (config.active_mask & 2u) != 0u,
    (config.active_mask & 4u) != 0u,
    (config.active_mask & 8u) != 0u,
  );
}

fn sigmoid4(value: vec4<f32>) -> vec4<f32> {
  return 1.0 / (1.0 + exp(-value));
}

@compute @workgroup_size(256)
fn clear_drive(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < config.neuron_count) {
    drive[id.x] = vec4<f32>(0.0);
  }
}

@compute @workgroup_size(128)
fn mix_optic(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= OPTIC_SITES * RECEPTOR_TYPES) { return; }
  let type_index = id.x / OPTIC_SITES;
  let site = id.x - type_index * OPTIC_SITES;
  let p = type_index * 5u;
  let logits = vec3<f32>(optic_parameters[p], optic_parameters[p + 1u], optic_parameters[p + 2u]);
  let spectral_exp = exp(logits - vec3<f32>(max(logits.x, max(logits.y, logits.z))));
  let spectral = spectral_exp / dot(spectral_exp, vec3<f32>(1.0));
  var mixed = vec4<f32>(0.0);
  for (var lane = 0u; lane < CAPACITY; lane++) {
    if (lane < config.capacity) {
      let first = lane * OPTIC_VALUES + site * 3u;
      let rgb = vec3<f32>(optic_rgb[first], optic_rgb[first + 1u], optic_rgb[first + 2u]);
      mixed[lane] = dot(rgb, spectral);
    }
  }
  optic_mixed[id.x] = mixed;
}

@compute @workgroup_size(128)
fn scatter_optic(@builtin(global_invocation_id) id: vec3<u32>) {
  let receptor = id.x;
  if (receptor >= RECEPTORS) { return; }
  let start = receptor_ptr[receptor];
  let stop = receptor_ptr[receptor + 1u];
  var site_mix = vec4<f32>(0.0);
  let type_index = receptor_type[receptor];
  for (var edge = start; edge < stop; edge++) {
    site_mix += site_weight[edge] * optic_mixed[type_index * OPTIC_SITES + site_indices[edge]];
  }
  // Unsupported receptors have no site support and exactly zero drive.
  var current = vec4<f32>(0.0);
  if (stop > start) {
    let p = type_index * 5u;
    let gain_raw = optic_parameters[p + 3u];
    let gain = max(gain_raw, 0.0) + log(1.0 + exp(-abs(gain_raw)));
    current = sigmoid4(vec4<f32>(optic_parameters[p + 4u]) + gain * (2.0 * site_mix - 1.0));
  }
  // Inactive lanes can carry an unused adapter value; dynamics and readout mask
  // them. Omitting Config here keeps the pass within the portable eight-storage
  // binding floor while preserving every anatomical lookup separately.
  drive[receptor_rows[receptor]] = current;
}

@compute @workgroup_size(128)
fn encode_body(@builtin(global_invocation_id) id: vec3<u32>) {
  let hidden_index = id.x;
  if (hidden_index >= BODY_HIDDEN) { return; }
  var value = vec4<f32>(body_input_bias[hidden_index]);
  for (var channel = 0u; channel < BODY_CHANNELS; channel++) {
    var standardized = vec4<f32>(0.0);
    for (var lane = 0u; lane < CAPACITY; lane++) {
      if (lane < config.capacity) {
        let raw = body_input[lane * BODY_CHANNELS + channel];
        standardized[lane] = clamp(
          (raw - body_normalizer[channel]) / body_normalizer[BODY_CHANNELS + channel],
          -8.0,
          8.0,
        );
      }
    }
    value += body_input_weight[hidden_index * BODY_CHANNELS + channel] * standardized;
  }
  body_hidden[hidden_index] = tanh(value);
}

@compute @workgroup_size(128)
fn scatter_body(@builtin(global_invocation_id) id: vec3<u32>) {
  let body_target = id.x;
  if (body_target >= BODY_TARGETS) { return; }
  var value = vec4<f32>(body_output_bias[body_target]);
  let first = body_target * BODY_HIDDEN;
  for (var hidden_index = 0u; hidden_index < BODY_HIDDEN; hidden_index++) {
    value += body_output_weight[first + hidden_index] * body_hidden[hidden_index];
  }
  drive[body_rows[body_target]] = select(vec4<f32>(0.0), sigmoid4(value), active_lanes());
}
