// AGPL-3.0-or-later -- MOTOR92 from annotated motor-cell rate only.
struct State {
    rate: vec4<f32>,
    adapt: vec4<f32>,
    support: vec4<f32>,
    release: vec4<f32>,
    da: vec4<f32>,
    oa: vec4<f32>,
    ht: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> state: array<State>;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> structural_mask: array<f32>;
@group(0) @binding(4) var<storage, read> intercept: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(6) var<storage, read> reference_rate: array<f32>;
@group(0) @binding(7) var<storage, read> rate_scale: array<f32>;

@compute @workgroup_size(64)
fn output_motor(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let output_index = invocation.x;
    if (output_index >= 92u) {
        return;
    }
    var preactivation = vec4<f32>(intercept[output_index]);
    for (var cell_index = 0u; cell_index < 815u; cell_index++) {
        let parameter_index = output_index * 815u + cell_index;
        preactivation += weight[parameter_index]
            * structural_mask[parameter_index]
            * ((state[rows[cell_index]].rate - vec4<f32>(reference_rate[cell_index])) / rate_scale[cell_index]);
    }
    var activation = 1.0 / (1.0 + exp(-preactivation));
    if (output_index < 84u) {
        activation = tanh(preactivation);
    }
    output[output_index] = activation;
}
