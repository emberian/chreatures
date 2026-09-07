// AGPL-3.0-or-later
// Exact operating-point-relative CNS Dynamics V2. The two Jacobi updates are
// dispatched separately by the host so afferent drive/adaptation/support stay
// fixed while every recurrent source comes from the previous substep.

const NEURONS: u32 = 165122u;
const TYPES: u32 = 11752u;

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
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> crow: array<u32>;
@group(0) @binding(2) var<storage, read> col: array<u32>;
@group(0) @binding(3) var<storage, read> packed_weight: array<u32>;
@group(0) @binding(4) var<storage, read> neuron_type: array<u32>;
// Five type-major arrays in order: baseline, recurrent gain, tau,
// adaptation gain, adaptation tau.
@group(0) @binding(5) var<storage, read> dynamics_raw: array<f32>;
@group(0) @binding(6) var<storage, read> state_in: array<NeuronState>;
@group(0) @binding(7) var<storage, read_write> state_out: array<NeuronState>;
@group(0) @binding(8) var<storage, read_write> deviation: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> recurrence: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> drive: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> neutral_drive: array<f32>;

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}

fn half_at(index: u32) -> f32 {
  let pair = unpack2x16float(packed_weight[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) != 0u);
}

fn lane_mask(bits: u32) -> vec4<bool> {
  return vec4<bool>(
    (bits & 1u) != 0u,
    (bits & 2u) != 0u,
    (bits & 4u) != 0u,
    (bits & 8u) != 0u,
  );
}

fn raw(field: u32, type_index: u32) -> f32 {
  return dynamics_raw[field * TYPES + type_index];
}

fn baseline(type_index: u32) -> f32 {
  return 0.05 + 0.4 * sigmoid(raw(0u, type_index));
}

fn one_minus_exp_neg(value: f32) -> f32 {
  return 1.0 - exp(-value);
}

@compute @workgroup_size(256)
fn reset_state(@builtin(global_invocation_id) id: vec3<u32>) {
  let neuron = id.x;
  if (neuron >= NEURONS) { return; }
  let old = state_out[neuron];
  let reset = lane_mask(config.reset_mask);
  let r0 = baseline(neuron_type[neuron]);
  state_out[neuron].rate = select(old.rate, vec4<f32>(r0), reset);
  state_out[neuron].adaptation = select(old.adaptation, vec4<f32>(0.0), reset);
  state_out[neuron].support = select(old.support, vec4<f32>(1.0), reset);
}

@compute @workgroup_size(256)
fn derive_deviation(@builtin(global_invocation_id) id: vec3<u32>) {
  let neuron = id.x;
  if (neuron >= NEURONS) { return; }
  deviation[neuron] = state_in[neuron].rate - vec4<f32>(baseline(neuron_type[neuron]));
}

@compute @workgroup_size(256)
fn recurrent_sum(@builtin(global_invocation_id) id: vec3<u32>) {
  let neuron = id.x;
  if (neuron >= NEURONS) { return; }
  var total = vec4<f32>(0.0);
  let start = crow[neuron];
  let stop = crow[neuron + 1u];
  for (var edge = start; edge < stop; edge++) {
    total += half_at(edge) * deviation[col[edge]];
  }
  recurrence[neuron] = total;
}

@compute @workgroup_size(256)
fn jacobi_update(@builtin(global_invocation_id) id: vec3<u32>) {
  let neuron = id.x;
  if (neuron >= NEURONS) { return; }
  let type_index = neuron_type[neuron];
  let r0 = baseline(type_index);
  let h = min(r0, 1.0 - r0);
  let gain = 0.5 + 1.5 * sigmoid(raw(1u, type_index));
  let tau = 0.02 + 0.23 * sigmoid(raw(2u, type_index));
  let adaptation_gain = 0.5 * sigmoid(raw(3u, type_index));
  let alpha = one_minus_exp_neg(config.dt / (2.0 * tau));
  let old = state_in[neuron];
  let u = drive[neuron] - vec4<f32>(neutral_drive[neuron])
    + gain * recurrence[neuron] - adaptation_gain * old.adaptation;
  let desired_rate = vec4<f32>(r0) + old.support * h * tanh(u / h);
  let next_rate = old.rate + alpha * (desired_rate - old.rate);
  state_out[neuron].rate = select(old.rate, next_rate, lane_mask(config.active_mask));
  state_out[neuron].adaptation = old.adaptation;
  state_out[neuron].support = old.support;
}

@compute @workgroup_size(256)
fn finalize_tick(@builtin(global_invocation_id) id: vec3<u32>) {
  let neuron = id.x;
  if (neuron >= NEURONS) { return; }
  let type_index = neuron_type[neuron];
  let r0 = baseline(type_index);
  let h = min(r0, 1.0 - r0);
  let adaptation_tau = 0.25 + 4.75 * sigmoid(raw(4u, type_index));
  let alpha = one_minus_exp_neg(config.dt / adaptation_tau);
  let old = state_out[neuron];
  let x = old.rate - vec4<f32>(r0);
  let next_adaptation = old.adaptation + alpha * (x - old.adaptation);
  let next_support = clamp(
    old.support + config.dt * (0.024 * (vec4<f32>(1.0) - old.support) - 0.003 * abs(x) / h),
    vec4<f32>(0.65),
    vec4<f32>(1.0),
  );
  let enabled_lanes = lane_mask(config.active_mask);
  state_out[neuron].adaptation = select(old.adaptation, next_adaptation, enabled_lanes);
  state_out[neuron].support = select(old.support, next_support, enabled_lanes);
}
