#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    float dt;
    uint final_step;
    uint active_mask;
    uint capacity;
    uint tiles;
};

inline bool resident_active(constant Params &p, uint tile, uint lane) {
    uint resident = tile * 4 + lane;
    return resident < p.capacity && (p.active_mask & (1u << resident));
}

inline float4 mask_inactive(float4 value, constant Params &p, uint tile) {
    for (uint lane = 0; lane < 4; ++lane) {
        if (!resident_active(p, tile, lane)) value[lane] = 0.0f;
    }
    return value;
}

inline float4 hold_inactive(float4 next, float4 old, constant Params &p, uint tile) {
    for (uint lane = 0; lane < 4; ++lane) {
        if (!resident_active(p, tile, lane)) next[lane] = old[lane];
    }
    return next;
}

// Metal Shading Language does not expose expm1. These exponents are negative
// and bounded away from zero at resident tick scales, so this is the direct
// single-precision spelling of -expm1(x).
inline float negative_expm1(float x) {
    return 1.0f - exp(x);
}

inline float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float z = exp(x);
    return z / (1.0f + z);
}

// Embodiment-driven CNS V4 entry points use only the current CHCNS4 contract.
kernel void v4_project_body_masked(
    device const float4 *body [[buffer(0)]],
    device const float *mean [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device const float *weights [[buffer(3)]],
    device const float *mask [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device const uint *body_rows [[buffer(6)]],
    device float4 *drive [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint body_channels = 807;
    constexpr uint body_afferents = 11798;
    if (p.tiles == 0) return;
    uint afferent = gid / p.tiles, tile = gid % p.tiles;
    if (afferent >= body_afferents || tile >= p.tiles) return;
    uint target = body_rows[afferent];
    if (target >= p.n) return;
    float4 u = bias[afferent];
    uint base = afferent * body_channels;
    for (uint channel = 0; channel < body_channels; ++channel) {
        float4 z = clamp((body[channel * p.tiles + tile] - mean[channel]) /
                         scale[channel], -8.0f, 8.0f);
        u += weights[base + channel] * mask[base + channel] * z;
    }
    float4 current;
    for (uint lane = 0; lane < 4; ++lane) current[lane] = stable_sigmoid(u[lane]);
    drive[target * p.tiles + tile] = mask_inactive(current, p, tile);
}

kernel void v4_project_context_zero_neutral(
    device const float4 *context [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const uint *context_rows [[buffer(3)]],
    device float4 *drive [[buffer(4)]],
    constant Params &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint context_channels = 12;
    constexpr uint context_afferents = 1314;
    if (p.tiles == 0) return;
    uint afferent = gid / p.tiles, tile = gid % p.tiles;
    if (afferent >= context_afferents || tile >= p.tiles) return;
    uint target = context_rows[afferent];
    if (target >= p.n) return;
    float4 u = bias[afferent];
    uint base = afferent * context_channels;
    for (uint channel = 0; channel < context_channels; ++channel) {
        u += weights[base + channel] * context[channel * p.tiles + tile];
    }
    float4 current = 0.15f * (tanh(u) - tanh(bias[afferent]));
    drive[target * p.tiles + tile] = mask_inactive(current, p, tile);
}

kernel void v4_csr_dynamics_f16(
    device const uint *rowptr [[buffer(0)]],
    device const uint *columns [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device const uint *channel [[buffer(3)]],
    device const float4 *rate_in [[buffer(4)]],
    device float4 *rate_out [[buffer(5)]],
    device const float4 *modulation_in [[buffer(6)]],
    device float4 *modulation_out [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    device const float4 *adaptation [[buffer(9)]],
    device const float4 *support [[buffer(10)]],
    device const float4 *release [[buffer(11)]],
    device const float4 *drive [[buffer(12)]],
    device const float *baseline [[buffer(13)]],
    device const float *recurrent_gain [[buffer(14)]],
    device const float *tau [[buffer(15)]],
    device const float *adaptation_gain [[buffer(16)]],
    device const uint *neuron_type [[buffer(17)]],
    device const float *mod_gain_raw [[buffer(18)]],
    device const float *mod_adaptation_raw [[buffer(19)]],
    device const float *modulation_tau [[buffer(20)]],
    device const float *neutral_drive [[buffer(21)]],
    uint gid [[thread_position_in_grid]]) {
    if (p.tiles == 0) return;
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= p.n || tile >= p.tiles) return;
    uint index = row * p.tiles + tile;
    float4 fast = 0.0f;
    float4 mod_input[3] = { float4(0.0f), float4(0.0f), float4(0.0f) };
    uint begin = rowptr[row], end = rowptr[row + 1];
    for (uint edge = begin; edge < end; ++edge) {
        uint source = columns[edge];
        if (source >= p.n) continue;
        float4 x = rate_in[source * p.tiles + tile] - baseline[source];
        uint family = channel[source];
        if (family == 1u) {
            fast += weights[edge] * x * release[source * p.tiles + tile];
        } else if (family >= 2u && family <= 4u) {
            mod_input[family - 2u] += weights[edge] * x;
        }
    }
    float r0 = baseline[row];
    float h = min(r0, 1.0f - r0);
    constexpr uint neuron_types = 11752;
    uint type = min(neuron_type[row], neuron_types - 1);
    float4 mg = 0.0f, ma = 0.0f;
    for (uint family = 0; family < 3; ++family) {
        uint mi = (row * 3 + family) * p.tiles + tile;
        float4 old_m = modulation_in[mi];
        float alpha = negative_expm1(-p.dt / (2.0f * modulation_tau[family]));
        float4 next_m = old_m + alpha * (mod_input[family] - old_m);
        next_m = hold_inactive(next_m, old_m, p, tile);
        modulation_out[mi] = next_m;
        mg += 0.5f * tanh(mod_gain_raw[type * 3 + family]) * next_m / h;
        ma += 0.5f * tanh(mod_adaptation_raw[type * 3 + family]) * next_m / h;
    }
    float4 old_rate = rate_in[index];
    float4 u = drive[index] - neutral_drive[row]
        + recurrent_gain[row] * exp(0.5f * tanh(mg)) * fast
        - adaptation_gain[row] * (1.0f + 0.5f * tanh(ma)) * adaptation[index];
    float4 target = r0 + support[index] * h * tanh(u / h);
    float alpha = negative_expm1(-p.dt / (2.0f * tau[row]));
    rate_out[index] = hold_inactive(old_rate + alpha * (target - old_rate),
                                    old_rate, p, tile);
}

kernel void v4_finalize_private_state(
    device const float4 *rate [[buffer(0)]],
    device float4 *adaptation [[buffer(1)]],
    device float4 *support [[buffer(2)]],
    device float4 *release [[buffer(3)]],
    device const float *baseline [[buffer(4)]],
    device const float *adaptation_tau [[buffer(5)]],
    device const float *release_tau [[buffer(6)]],
    device const float *release_use [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (p.tiles == 0) return;
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= p.n || tile >= p.tiles) return;
    uint index = row * p.tiles + tile;
    float4 old_a = adaptation[index], old_s = support[index], old_q = release[index];
    float r0 = baseline[row], h = min(r0, 1.0f - r0);
    float4 x = rate[index] - r0;
    float aa = negative_expm1(-p.dt / adaptation_tau[row]);
    float4 next_a = old_a + aa * (x - old_a);
    float4 next_s = clamp(old_s + p.dt *
        (0.024f * (1.0f - old_s) - 0.003f * abs(x) / h), 0.65f, 1.0f);
    float4 next_q = clamp(old_q + p.dt *
        ((1.0f - old_q) / release_tau[row]
         - release_use[row] * abs(x) / h * old_q), 0.2f, 1.0f);
    adaptation[index] = hold_inactive(next_a, old_a, p, tile);
    support[index] = hold_inactive(next_s, old_s, p, tile);
    release[index] = hold_inactive(next_q, old_q, p, tile);
}

kernel void v4_motor92_centered(
    device const float4 *rates [[buffer(0)]],
    device const uint *motor_rows [[buffer(1)]],
    device const float *reference_rate [[buffer(2)]],
    device const float *rate_scale [[buffer(3)]],
    device const float *weight [[buffer(4)]],
    device const float *mask [[buffer(5)]],
    device const float *intercept [[buffer(6)]],
    device float4 *motor [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint motor_outputs = 92;
    constexpr uint motor_neurons = 815;
    if (p.tiles == 0) return;
    uint output = gid / p.tiles, tile = gid % p.tiles;
    if (output >= motor_outputs || tile >= p.tiles) return;
    float4 u = intercept[output];
    uint base = output * motor_neurons;
    for (uint j = 0; j < motor_neurons; ++j) {
        uint source = motor_rows[j];
        if (source >= p.n) continue;
        float4 centered = (rates[source * p.tiles + tile] - reference_rate[j]) /
                          rate_scale[j];
        u += weight[base + j] * mask[base + j] * centered;
    }
    float4 value;
    for (uint lane = 0; lane < 4; ++lane) {
        value[lane] = output < 84u ? tanh(u[lane]) : stable_sigmoid(u[lane]);
    }
    motor[gid] = mask_inactive(value, p, tile);
}

kernel void clear_drive(device float4 *drive [[buffer(0)]],
                        constant Params &p [[buffer(8)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < p.n * p.tiles) drive[gid] = 0.0f;
}

kernel void project_optic(
    device const uint *receptor_rows [[buffer(0)]],
    device const uint *receptor_types [[buffer(1)]],
    device const uint *ptr [[buffer(2)]],
    device const uint *sites [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    device const float4 *sensory [[buffer(5)]],
    device const float *spectral [[buffer(6)]],
    device const float *gain [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    device const float *bias [[buffer(9)]],
    device float4 *drive [[buffer(10)]],
    device const float *neutral_drive [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
    uint receptor = gid / p.tiles, tile = gid % p.tiles;
    if (receptor >= 4107) return;
    uint begin = ptr[receptor], end = ptr[receptor + 1];
    if (begin == end) {
        drive[receptor_rows[receptor] * p.tiles + tile] = 0.0f;
        return;
    }
    uint type = receptor_types[receptor];
    float4 mixture = 0.0f;
    for (uint edge = begin; edge < end; ++edge) {
        uint site = sites[edge];
        float w = weights[edge];
        mixture += w * (
            spectral[type * 3] * sensory[(site * 3) * p.tiles + tile] +
            spectral[type * 3 + 1] * sensory[(site * 3 + 1) * p.tiles + tile] +
            spectral[type * 3 + 2] * sensory[(site * 3 + 2) * p.tiles + tile]);
    }
    float4 current = 1.0f / (1.0f + exp(-(bias[type] + gain[type] * (2.0f * mixture - 1.0f))));
    for (uint lane = 0; lane < 4; ++lane) {
        bool neutral = true;
        for (uint edge = begin; edge < end && neutral; ++edge) {
            uint site = sites[edge];
            for (uint color = 0; color < 3; ++color) {
                neutral = neutral && sensory[(site * 3 + color) * p.tiles + tile][lane] == 0.5f;
            }
        }
        if (neutral) current[lane] = neutral_drive[receptor_rows[receptor]];
    }
    drive[receptor_rows[receptor] * p.tiles + tile] = mask_inactive(current, p, tile);
}

kernel void gather_rates(device const float4 *rates [[buffer(0)]],
                         device const uint *indices [[buffer(1)]],
                         device float4 *selected [[buffer(2)]],
                         constant Params &p [[buffer(8)]],
                         uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    selected[gid] = mask_inactive(rates[indices[row] * p.tiles + tile], p, tile);
}

kernel void dense_projection(device const float *weights [[buffer(0)]],
                          device const float4 *rates [[buffer(1)]],
                          device const float *baseline [[buffer(2)]],
                          device float4 *output [[buffer(3)]],
                          constant Params &p [[buffer(8)]],
                          constant uint &weight_stride [[buffer(9)]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {
    uint projection_row = group / p.tiles, tile = group % p.tiles;
    threadgroup float4 partial[256];
    float4 sum = 0.0f;
    if (projection_row < 64) {
        device const float *row = weights + projection_row * weight_stride;
        for (uint neuron = lane; neuron < p.n; neuron += 256) {
            sum += row[neuron] * (rates[neuron * p.tiles + tile] - baseline[neuron]);
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0 && projection_row < 64) {
        output[projection_row * p.tiles + tile] = mask_inactive(partial[0], p, tile);
    }
}

kernel void dense_readout_output(device const float *weights [[buffer(0)]],
                                 device const float4 *hidden [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device float4 *output [[buffer(3)]],
                                 constant Params &p [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= 512) return;
    float4 sum = bias[row];
    for (uint feature = 0; feature < 64; ++feature) {
        sum += weights[row * 64 + feature] * hidden[feature * p.tiles + tile];
    }
    output[gid] = mask_inactive(tanh(sum), p, tile);
}

kernel void physiology_partials(device const float4 *rates [[buffer(0)]],
                                device const float4 *support [[buffer(1)]],
                                device float4 *partial [[buffer(2)]],
                                constant Params &p [[buffer(8)]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]]) {
    uint tile = group % p.tiles, neuron_group = group / p.tiles;
    uint row = neuron_group * 256 + lane;
    threadgroup float4 sums[256], peaks[256], supports[256];
    float4 rate = row < p.n ? rates[row * p.tiles + tile] : 0.0f;
    sums[lane] = rate;
    peaks[lane] = rate;
    supports[lane] = row < p.n ? support[row * p.tiles + tile] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            peaks[lane] = max(peaks[lane], peaks[lane + stride]);
            supports[lane] += supports[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        partial[group * 3] = sums[0];
        partial[group * 3 + 1] = peaks[0];
        partial[group * 3 + 2] = supports[0];
    }
}

kernel void physiology_final(device const float4 *partial [[buffer(0)]],
                             device float4 *output [[buffer(1)]],
                             constant Params &p [[buffer(8)]],
                             uint tile [[thread_position_in_grid]]) {
    if (tile >= p.tiles) return;
    uint groups = (p.n + 255) / 256;
    float4 sum = 0.0f, peak = 0.0f, support = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        uint base = (group * p.tiles + tile) * 3;
        sum += partial[base];
        peak = max(peak, partial[base + 1]);
        support += partial[base + 2];
    }
    output[tile] = mask_inactive(sum / p.n, p, tile);
    output[p.tiles + tile] = mask_inactive(peak, p, tile);
    output[2 * p.tiles + tile] = mask_inactive(support / p.n, p, tile);
}
