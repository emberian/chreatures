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

// Metal Shading Language does not expose expm1. All V2 exponents are negative
// and bounded away from zero at resident tick scales, so this is the direct
// single-precision spelling of -expm1(x).
inline float negative_expm1(float x) {
    return 1.0f - exp(x);
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

kernel void normalize_body(device const float4 *sensory [[buffer(0)]],
                           device const float *mean [[buffer(1)]],
                           device const float *scale [[buffer(2)]],
                           device float4 *normalized [[buffer(3)]],
                           constant Params &p [[buffer(8)]],
                           uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= 43) return;
    float4 x = (sensory[(5313 + row) * p.tiles + tile] - mean[row]) / scale[row];
    normalized[gid] = mask_inactive(clamp(x, -8.0f, 8.0f), p, tile);
}

kernel void tanh_bias(device const float4 *input [[buffer(0)]],
                      device const float *bias [[buffer(1)]],
                      device float4 *output [[buffer(2)]],
                      constant Params &p [[buffer(8)]],
                      constant uint &rows [[buffer(9)]],
                      uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= rows) return;
    output[gid] = mask_inactive(tanh(input[gid] + bias[row]), p, tile);
}

kernel void sigmoid_bias_scatter_body(device const float4 *input [[buffer(0)]],
                                      device const float *bias [[buffer(1)]],
                                      device const uint *body_rows [[buffer(2)]],
                                      device float4 *drive [[buffer(3)]],
                                      device const float4 *normalized [[buffer(4)]],
                                      device const float *neutral_drive [[buffer(5)]],
                                      constant Params &p [[buffer(8)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= 11233) return;
    float4 current = 1.0f / (1.0f + exp(-(input[gid] + bias[row])));
    for (uint lane = 0; lane < 4; ++lane) {
        bool neutral = true;
        for (uint channel = 0; channel < 43; ++channel) {
            neutral = neutral && normalized[channel * p.tiles + tile][lane] == 0.0f;
        }
        if (neutral) current[lane] = neutral_drive[body_rows[row]];
    }
    drive[body_rows[row] * p.tiles + tile] = mask_inactive(current, p, tile);
}

inline void rate_update(uint row,
                        uint tile,
                        float4 recurrent,
                        device const float4 *rate_in,
                        device float4 *rate_out,
                        device float4 *adapt,
                        device float4 *support,
                        device const float4 *drive,
                        device const float *baseline,
                        device const float *recurrent_gain,
                        device const float *tau,
                        device const float *adaptation_gain,
                        device const float *adaptation_tau,
                        device const float *neutral_drive,
                        constant Params &p) {
    uint index = row * p.tiles + tile;
    float4 old = rate_in[index], a = adapt[index], s = support[index];
    float r0 = baseline[row], h = min(r0, 1.0f - r0);
    float alpha = negative_expm1(-p.dt / (2.0f * tau[row]));
    float4 u = (drive[index] - neutral_drive[row])
        + recurrent_gain[row] * recurrent - adaptation_gain[row] * a;
    float4 target = r0 + s * h * tanh(u / h);
    float4 next = hold_inactive(old + alpha * (target - old), old, p, tile);
    rate_out[index] = next;
    if (p.final_step) {
        float4 x = next - r0;
        float adaptation_alpha = negative_expm1(-p.dt / adaptation_tau[row]);
        float4 na = a + adaptation_alpha * (x - a);
        float4 ns = clamp(s + p.dt * (0.024f * (1.0f - s) - 0.003f * abs(x) / h), 0.65f, 1.0f);
        adapt[index] = hold_inactive(na, a, p, tile);
        support[index] = hold_inactive(ns, s, p, tile);
    }
}

kernel void csr_rate(device const uint *rowptr [[buffer(0)]],
                     device const uint *columns [[buffer(1)]],
                     device const float *weights [[buffer(2)]],
                     device const float4 *rate_in [[buffer(3)]],
                     device float4 *rate_out [[buffer(4)]],
                     device float4 *adapt [[buffer(5)]],
                     device float4 *support [[buffer(6)]],
                     device const float4 *drive [[buffer(7)]],
                     constant Params &p [[buffer(8)]],
                     device const float *baseline [[buffer(9)]],
                     device const float *recurrent_gain [[buffer(10)]],
                     device const float *tau [[buffer(11)]],
                     device const float *adaptation_gain [[buffer(12)]],
                     device const float *adaptation_tau [[buffer(13)]],
                     device const float *neutral_drive [[buffer(14)]],
                     uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= p.n) return;
    float4 recurrent = 0.0f;
    for (uint edge = rowptr[row]; edge < rowptr[row + 1]; ++edge) {
        uint source = columns[edge];
        recurrent += weights[edge] * (rate_in[source * p.tiles + tile] - baseline[source]);
    }
    rate_update(row, tile, recurrent, rate_in, rate_out, adapt, support, drive,
                baseline, recurrent_gain, tau, adaptation_gain, adaptation_tau,
                neutral_drive, p);
}

kernel void csr_rate_simd(device const uint *rowptr [[buffer(0)]],
                          device const uint *columns [[buffer(1)]],
                          device const float *weights [[buffer(2)]],
                          device const float4 *rate_in [[buffer(3)]],
                          device float4 *rate_out [[buffer(4)]],
                          device float4 *adapt [[buffer(5)]],
                          device float4 *support [[buffer(6)]],
                          device const float4 *drive [[buffer(7)]],
                          constant Params &p [[buffer(8)]],
                          device const float *baseline [[buffer(9)]],
                          device const float *recurrent_gain [[buffer(10)]],
                          device const float *tau [[buffer(11)]],
                          device const float *adaptation_gain [[buffer(12)]],
                          device const float *adaptation_tau [[buffer(13)]],
                          device const float *neutral_drive [[buffer(14)]],
                          uint gid [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint row_tile = gid >> 5, row = row_tile / p.tiles, tile = row_tile % p.tiles;
    if (row >= p.n) return;
    float4 recurrent = 0.0f;
    for (uint edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        uint source = columns[edge];
        recurrent += weights[edge] * (rate_in[source * p.tiles + tile] - baseline[source]);
    }
    recurrent = float4(simd_sum(recurrent.x), simd_sum(recurrent.y),
                       simd_sum(recurrent.z), simd_sum(recurrent.w));
    if (lane) return;
    rate_update(row, tile, recurrent, rate_in, rate_out, adapt, support, drive,
                baseline, recurrent_gain, tau, adaptation_gain, adaptation_tau,
                neutral_drive, p);
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
