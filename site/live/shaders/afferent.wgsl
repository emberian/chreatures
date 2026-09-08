// AGPL-3.0-or-later -- V5 adapters; raw values terminate in drive.
const CAP: u32 = 4u;
const OS: u32 = 1771u;
const OV: u32 = 5313u;
const RT: u32 = 10u;
const R: u32 = 4107u;
const BC: u32 = 807u;
const BT: u32 = 11798u;
const CC: u32 = 12u;
const CT: u32 = 1314u;

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

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> drive: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rows: array<u32>;
@group(0) @binding(4) var<storage, read> param: array<f32>;
@group(0) @binding(5) var<storage, read> aux0: array<u32>;
@group(0) @binding(6) var<storage, read> aux1: array<u32>;
@group(0) @binding(7) var<storage, read_write> scratch: array<vec4<f32>>;

fn enabled() -> vec4<bool> {
    return vec4<bool>(
        (cfg.active_mask & 1u) != 0u,
        (cfg.active_mask & 2u) != 0u,
        (cfg.active_mask & 4u) != 0u,
        (cfg.active_mask & 8u) != 0u,
    );
}

fn lanes(channel: u32, width: u32) -> vec4<f32> {
    var value = vec4<f32>(0);
    for (var lane = 0u; lane < CAP; lane++) {
        if (lane < cfg.capacity) {
            value[lane] = input[lane * width + channel];
        }
    }
    return value;
}

@compute @workgroup_size(256)
fn clear_drive(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if (invocation.x < cfg.neuron_count) {
        drive[invocation.x] = vec4<f32>(0);
    }
}

@compute @workgroup_size(128)
fn mix_optic(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if (invocation.x >= OS * RT) {
        return;
    }
    let receptor_type = invocation.x / OS;
    let site = invocation.x - receptor_type * OS;
    let parameter_offset = receptor_type * 5u;
    let logits = vec3<f32>(
        param[parameter_offset],
        param[parameter_offset + 1u],
        param[parameter_offset + 2u],
    );
    let exponentials = exp(logits - vec3<f32>(max(logits.x, max(logits.y, logits.z))));
    let spectral_weight = exponentials / dot(exponentials, vec3<f32>(1));
    var mixture = vec4<f32>(0);
    for (var lane = 0u; lane < CAP; lane++) {
        if (lane < cfg.capacity) {
            let input_offset = lane * OV + site * 3u;
            mixture[lane] = dot(
                vec3<f32>(
                    input[input_offset],
                    input[input_offset + 1u],
                    input[input_offset + 2u],
                ),
                spectral_weight,
            );
        }
    }
    scratch[invocation.x] = mixture;
}

// aux0 receptor_ptr; aux1 layout receptor_type[R], site_index[edges], f32 site_weight[edges].
@compute @workgroup_size(128)
fn scatter_optic(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let receptor = invocation.x;
    if (receptor >= R) {
        return;
    }
    let start = aux0[receptor];
    let stop = aux0[receptor + 1u];
    let receptor_type = aux1[receptor];
    var mixture = vec4<f32>(0);
    for (var edge = start; edge < stop; edge++) {
        let site_weight = bitcast<f32>(aux1[R + 4669u + edge]);
        mixture += site_weight * scratch[receptor_type * OS + aux1[R + edge]];
    }
    var current = vec4<f32>(0);
    if (stop > start) {
        let parameter_offset = receptor_type * 5u;
        let gain_raw = param[parameter_offset + 3u];
        let gain = max(gain_raw, 0.0) + log(1.0 + exp(-abs(gain_raw)));
        current = 1.0 / (1.0 + exp(-(
            vec4<f32>(param[parameter_offset + 4u]) + gain * (2.0 * mixture - 1.0)
        )));
    }
    drive[rows[receptor]] = current;
}

// param mean,scale,weight,bias; aux0 structural mask.
@compute @workgroup_size(128)
fn encode_body(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let target_index = invocation.x;
    if (target_index >= BT) {
        return;
    }
    var current = vec4<f32>(param[2u * BC + BT * BC + target_index]);
    for (var channel = 0u; channel < BC; channel++) {
        if (aux0[target_index * BC + channel] != 0u) {
            let standardized = clamp(
                (lanes(channel, BC) - vec4<f32>(param[channel]))
                    / vec4<f32>(param[BC + channel]),
                vec4<f32>(-8),
                vec4<f32>(8),
            );
            current += param[2u * BC + target_index * BC + channel] * standardized;
        }
    }
    drive[rows[target_index]] = select(
        vec4<f32>(0), 1.0 / (1.0 + exp(-current)), enabled()
    );
}

// param context.weight then context.bias; zero context produces exactly zero current.
@compute @workgroup_size(128)
fn inject_context(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let target_index = invocation.x;
    if (target_index >= CT) {
        return;
    }
    let bias = param[CT * CC + target_index];
    var current = vec4<f32>(bias);
    for (var channel = 0u; channel < CC; channel++) {
        current += param[target_index * CC + channel] * lanes(channel, CC);
    }
    drive[rows[target_index]] = select(
        vec4<f32>(0),
        0.15 * (tanh(current) - vec4<f32>(tanh(bias))),
        enabled(),
    );
}
