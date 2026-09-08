// AGPL-3.0-or-later -- exact V4 seven-state recurrence.
const N: u32 = 165122u;
const T: u32 = 11752u;

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

struct State {
    rate: vec4<f32>,
    adapt: vec4<f32>,
    support: vec4<f32>,
    release: vec4<f32>,
    da: vec4<f32>,
    oa: vec4<f32>,
    ht: vec4<f32>,
};

struct Recur {
    fast: vec4<f32>,
    da: vec4<f32>,
    oa: vec4<f32>,
    ht: vec4<f32>,
};

struct Activity {
    rate: vec4<f32>,
    release: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read> crow: array<u32>;
@group(0) @binding(2) var<storage, read> col: array<u32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> index: array<u32>;
@group(0) @binding(5) var<storage, read> raws: array<f32>;
@group(0) @binding(6) var<storage, read> src: array<State>;
@group(0) @binding(7) var<storage, read_write> dst: array<State>;
@group(0) @binding(8) var<storage, read_write> recur: array<Recur>;
@group(0) @binding(9) var<storage, read> drive: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> neutral: array<f32>;
@group(0) @binding(11) var<storage, read> activity_src: array<Activity>;
@group(0) @binding(12) var<storage, read_write> activity_dst: array<Activity>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn raw_value(family: u32, neuron_type: u32) -> f32 {
    return raws[family * T + neuron_type];
}

fn baseline(neuron_type: u32) -> f32 {
    return 0.05 + 0.4 * sigmoid(raw_value(0u, neuron_type));
}

fn decay_alpha(x: f32) -> f32 {
    return 1.0 - exp(-x);
}

fn lane_mask(bits: u32) -> vec4<bool> {
    return vec4<bool>(
        (bits & 1u) != 0u,
        (bits & 2u) != 0u,
        (bits & 4u) != 0u,
        (bits & 8u) != 0u,
    );
}

@compute @workgroup_size(256)
fn reset_state(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let neuron = invocation.x;
    if (neuron >= N) {
        return;
    }
    let reset = lane_mask(cfg.reset_mask);
    let old = dst[neuron];
    let resting_rate = vec4<f32>(baseline(index[neuron]));
    dst[neuron].rate = select(old.rate, resting_rate, reset);
    dst[neuron].adapt = select(old.adapt, vec4<f32>(0), reset);
    dst[neuron].support = select(old.support, vec4<f32>(1), reset);
    dst[neuron].release = select(old.release, vec4<f32>(1), reset);
    dst[neuron].da = select(old.da, vec4<f32>(0), reset);
    dst[neuron].oa = select(old.oa, vec4<f32>(0), reset);
    dst[neuron].ht = select(old.ht, vec4<f32>(0), reset);
    activity_dst[neuron].rate = dst[neuron].rate;
    activity_dst[neuron].release = dst[neuron].release;
}

@compute @workgroup_size(256)
fn sync_activity(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let neuron = invocation.x;
    if (neuron >= N) { return; }
    activity_dst[neuron].rate = src[neuron].rate;
    activity_dst[neuron].release = src[neuron].release;
}

// index=graph.channel, raws=baseline_by_neuron for this pass.
@compute @workgroup_size(256)
fn recurrent_sum(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let neuron = invocation.x;
    if (neuron >= N) {
        return;
    }
    var sum = Recur(
        vec4<f32>(0), vec4<f32>(0), vec4<f32>(0), vec4<f32>(0)
    );
    for (var edge = crow[neuron]; edge < crow[neuron + 1u]; edge++) {
        let source = col[edge];
        let activity = activity_src[source];
        let value = weight[edge]
            * (activity.rate - vec4<f32>(raws[source]));
        switch index[source] {
            case 1u: { sum.fast += value * activity.release; }
            case 2u: { sum.da += value; }
            case 3u: { sum.oa += value; }
            case 4u: { sum.ht += value; }
            default: {}
        }
    }
    recur[neuron] = sum;
}

// index=neuron_type, raws=13 type arrays followed by modulation_tau_raw[3].
@compute @workgroup_size(256)
fn jacobi_update(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let neuron = invocation.x;
    if (neuron >= N) {
        return;
    }
    let neuron_type = index[neuron];
    let resting_rate = baseline(neuron_type);
    let half_range = min(resting_rate, 1.0 - resting_rate);
    let old = src[neuron];
    let input = recur[neuron];
    let modulation_offset = 13u * T;
    let da = old.da + decay_alpha(cfg.dt / (2.0 * (0.1 + 4.9 * sigmoid(raws[modulation_offset])))) * (input.da - old.da);
    let oa = old.oa + decay_alpha(cfg.dt / (2.0 * (0.1 + 4.9 * sigmoid(raws[modulation_offset + 1u])))) * (input.oa - old.oa);
    let ht = old.ht + decay_alpha(cfg.dt / (2.0 * (0.1 + 4.9 * sigmoid(raws[modulation_offset + 2u])))) * (input.ht - old.ht);
    let gain_modulation = (0.5 * tanh(raw_value(7u, neuron_type)) * da + 0.5 * tanh(raw_value(8u, neuron_type)) * oa + 0.5 * tanh(raw_value(9u, neuron_type)) * ht) / half_range;
    let adaptation_modulation = (0.5 * tanh(raw_value(10u, neuron_type)) * da + 0.5 * tanh(raw_value(11u, neuron_type)) * oa + 0.5 * tanh(raw_value(12u, neuron_type)) * ht) / half_range;
    let gain = (0.5 + 1.5 * sigmoid(raw_value(1u, neuron_type))) * exp(0.5 * tanh(gain_modulation));
    let adaptation_gain = 0.5 * sigmoid(raw_value(3u, neuron_type)) * (1.0 + 0.5 * tanh(adaptation_modulation));
    let current = drive[neuron] - vec4<f32>(neutral[neuron]) + gain * input.fast - adaptation_gain * old.adapt;
    let desired = vec4<f32>(resting_rate) + old.support * half_range * tanh(current / half_range);
    let rate = old.rate + decay_alpha(cfg.dt / (2.0 * (0.02 + 0.23 * sigmoid(raw_value(2u, neuron_type))))) * (desired - old.rate);
    let enabled = lane_mask(cfg.active_mask);
    dst[neuron].rate = select(old.rate, rate, enabled);
    dst[neuron].adapt = old.adapt;
    dst[neuron].support = old.support;
    dst[neuron].release = old.release;
    dst[neuron].da = select(old.da, da, enabled);
    dst[neuron].oa = select(old.oa, oa, enabled);
    dst[neuron].ht = select(old.ht, ht, enabled);
    activity_dst[neuron].rate = dst[neuron].rate;
    activity_dst[neuron].release = dst[neuron].release;
}

@compute @workgroup_size(256)
fn finalize_tick(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let neuron = invocation.x;
    if (neuron >= N) {
        return;
    }
    let neuron_type = index[neuron];
    let old = dst[neuron];
    let resting_rate = baseline(neuron_type);
    let half_range = min(resting_rate, 1.0 - resting_rate);
    let deviation = old.rate - vec4<f32>(resting_rate);
    let adaptation = old.adapt + decay_alpha(cfg.dt / (0.25 + 4.75 * sigmoid(raw_value(4u, neuron_type)))) * (deviation - old.adapt);
    let support = clamp(old.support + cfg.dt * (0.024 * (vec4<f32>(1) - old.support) - 0.003 * abs(deviation) / half_range), vec4<f32>(0.65), vec4<f32>(1));
    let release_tau = 0.05 + 1.95 * sigmoid(raw_value(5u, neuron_type));
    let release_use = 0.01 + 0.49 * sigmoid(raw_value(6u, neuron_type));
    let release = clamp(old.release + cfg.dt * ((vec4<f32>(1) - old.release) / release_tau - release_use * abs(deviation) / half_range * old.release), vec4<f32>(0.2), vec4<f32>(1));
    let enabled = lane_mask(cfg.active_mask);
    dst[neuron].adapt = select(old.adapt, adaptation, enabled);
    dst[neuron].support = select(old.support, support, enabled);
    dst[neuron].release = select(old.release, release, enabled);
    activity_dst[neuron].release = dst[neuron].release;
}
