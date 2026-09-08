// AGPL-3.0-or-later -- private gamma1pedc cue-before-PPL plasticity.
const N: u32 = 165122u;
const E: u32 = 4184u;
const TARGETS: u32 = 2u;

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

struct Activity {
    rate: vec4<f32>,
    release: vec4<f32>,
};

struct Recur {
    fast: vec4<f32>,
    da: vec4<f32>,
    oa: vec4<f32>,
    ht: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read> source_rows: array<u32>;
@group(0) @binding(2) var<storage, read> target_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> target_rows: array<u32>;
@group(0) @binding(4) var<storage, read> dan_rows: array<u32>;
@group(0) @binding(5) var<storage, read> rule: array<f32>;
@group(0) @binding(6) var<storage, read> baseline_weight: array<f32>;
@group(0) @binding(7) var<storage, read> baseline: array<f32>;
@group(0) @binding(8) var<storage, read> activity: array<Activity>;
@group(0) @binding(9) var<storage, read_write> recurrence: array<Recur>;
@group(0) @binding(10) var<storage, read_write> efficacy: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> eligibility: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> state: array<State>;

fn lane_mask(bits: u32) -> vec4<bool> {
    return vec4<bool>(
        (bits & 1u) != 0u,
        (bits & 2u) != 0u,
        (bits & 4u) != 0u,
        (bits & 8u) != 0u,
    );
}

@compute @workgroup_size(256)
fn reset_plasticity(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let edge = invocation.x;
    if (edge >= E) {
        return;
    }
    let reset = lane_mask(cfg.reset_mask);
    efficacy[edge] = select(efficacy[edge], vec4<f32>(0), reset);
    eligibility[edge] = select(eligibility[edge], vec4<f32>(0), reset);
}

// One invocation owns one target and visits its selected canonical graph edge
// positions in ascending order. This is deliberately not an atomic reduction.
@compute @workgroup_size(1)
fn apply_correction(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let target_index = invocation.x;
    if (target_index >= TARGETS) {
        return;
    }
    var correction = vec4<f32>(0);
    for (var edge = target_ptr[target_index]; edge < target_ptr[target_index + 1u]; edge++) {
        let source = source_rows[edge];
        let source_activity = activity[source];
        correction += baseline_weight[edge] * efficacy[edge]
            * (source_activity.rate - vec4<f32>(baseline[source]))
            * source_activity.release;
    }
    recurrence[target_rows[target_index]].fast += correction;
}

@compute @workgroup_size(256)
fn learn_edges(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let edge = invocation.x;
    if (edge >= E) {
        return;
    }
    var target_index = 1u;
    if (edge < target_ptr[1u]) {
        target_index = 0u;
    }
    let source = source_rows[edge];
    let source_state = state[source];
    let dan_state = state[dan_rows[target_index]];
    let source_baseline = baseline[source];
    let dan_baseline = baseline[dan_rows[target_index]];
    let source_half_range = min(source_baseline, 1.0 - source_baseline);
    let dan_half_range = min(dan_baseline, 1.0 - dan_baseline);
    let cue = clamp(
        max(vec4<f32>(0), (source_state.rate - vec4<f32>(source_baseline)) / source_half_range)
            * source_state.release,
        vec4<f32>(0), vec4<f32>(1),
    );
    let gate = clamp(
        max(vec4<f32>(0), (dan_state.rate - vec4<f32>(dan_baseline)) / dan_half_range)
            * dan_state.release,
        vec4<f32>(0), vec4<f32>(1),
    );
    let tau = rule[target_index * 3u];
    let depression = rule[target_index * 3u + 1u];
    let maximum = rule[target_index * 3u + 2u];
    let old_efficacy = efficacy[edge];
    let old_eligibility = eligibility[edge];
    let decayed = old_eligibility * exp(-cfg.dt / tau);
    let next_efficacy = clamp(
        old_efficacy - depression * cfg.dt * decayed * gate,
        vec4<f32>(-maximum), vec4<f32>(0),
    );
    let next_eligibility = max(decayed, cue);
    let enabled = lane_mask(cfg.active_mask);
    efficacy[edge] = select(old_efficacy, next_efficacy, enabled);
    eligibility[edge] = select(old_eligibility, next_eligibility, enabled);
}
