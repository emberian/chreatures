#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    float alpha;
    float gain;
    float dt;
    float recovery;
    uint final_step;
    uint active_mask;
    uint capacity;
    uint tiles;
};

inline bool resident_active(constant Params &p, uint tile, uint lane) {
    uint resident = tile * 4 + lane;
    return resident < p.capacity && (p.active_mask & (1u << resident));
}

inline float4 hold_inactive(float4 next, float4 old, constant Params &p, uint tile) {
    for (uint lane = 0; lane < 4; ++lane) {
        if (!resident_active(p, tile, lane)) next[lane] = old[lane];
    }
    return next;
}

kernel void project_inputs(device const uint *ptr [[buffer(0)]], device const uint *cols [[buffer(1)]], device const float *weights [[buffer(2)]], device const float4 *channels [[buffer(3)]], device float4 *drive [[buffer(4)]], device const float4 *input_gain [[buffer(5)]], constant Params &p [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= p.n) return;
    float4 x = 0.0f;
    for (uint edge = ptr[row]; edge < ptr[row + 1]; ++edge) {
        uint channel = cols[edge];
        x += weights[edge] * input_gain[channel * p.tiles + tile] * channels[channel * p.tiles + tile];
    }
    drive[gid] = x;
}

inline void rate_update(uint row, uint tile, float4 recurrent, device const float4 *rate_in, device float4 *rate_out, device float4 *adapt, device float4 *support, device const float4 *drive, device const float4 *target_gain, device const float4 *excitability, constant Params &p) {
    uint index = row * p.tiles + tile;
    float4 old = rate_in[index], a = adapt[index], s = support[index];
    float4 regulated_drive = drive[index] + p.gain * target_gain[index] * recurrent;
    float4 target = max(tanh(0.005f + excitability[index] * regulated_drive - 0.10f * a), 0.0f);
    float4 next = hold_inactive(old + p.alpha * (target * s - old), old, p, tile);
    rate_out[index] = next;
    if (p.final_step) {
        float4 na = a + p.dt / 5.0f * (next - a);
        float4 ns = clamp(s + p.dt * (p.recovery * (1.0f - s) - 0.003f * next), 0.65f, 1.0f);
        adapt[index] = hold_inactive(na, a, p, tile);
        support[index] = hold_inactive(ns, s, p, tile);
    }
}

kernel void csr_rate(device const uint *rowptr [[buffer(0)]], device const uint *columns [[buffer(1)]], device const float *weights [[buffer(2)]], device const float4 *rate_in [[buffer(3)]], device float4 *rate_out [[buffer(4)]], device float4 *adapt [[buffer(5)]], device float4 *support [[buffer(6)]], device const float4 *drive [[buffer(7)]], constant Params &p [[buffer(8)]], device const float4 *source_gain [[buffer(11)]], device const float4 *target_gain [[buffer(12)]], device const float4 *excitability [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= p.n) return;
    float4 recurrent = 0.0f;
    for (uint edge = rowptr[row]; edge < rowptr[row + 1]; ++edge) {
        uint source = columns[edge];
        recurrent += weights[edge] * source_gain[source * p.tiles + tile] * rate_in[source * p.tiles + tile];
    }
    rate_update(row, tile, recurrent, rate_in, rate_out, adapt, support, drive, target_gain, excitability, p);
}

kernel void csr_rate_simd(device const uint *rowptr [[buffer(0)]], device const uint *columns [[buffer(1)]], device const float *weights [[buffer(2)]], device const float4 *rate_in [[buffer(3)]], device float4 *rate_out [[buffer(4)]], device float4 *adapt [[buffer(5)]], device float4 *support [[buffer(6)]], device const float4 *drive [[buffer(7)]], constant Params &p [[buffer(8)]], device const float4 *source_gain [[buffer(11)]], device const float4 *target_gain [[buffer(12)]], device const float4 *excitability [[buffer(13)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row_tile = gid >> 5, row = row_tile / p.tiles, tile = row_tile % p.tiles;
    if (row >= p.n) return;
    float4 recurrent = 0.0f;
    for (uint edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        uint source = columns[edge];
        recurrent += weights[edge] * source_gain[source * p.tiles + tile] * rate_in[source * p.tiles + tile];
    }
    recurrent = float4(simd_sum(recurrent.x), simd_sum(recurrent.y), simd_sum(recurrent.z), simd_sum(recurrent.w));
    if (lane) return;
    rate_update(row, tile, recurrent, rate_in, rate_out, adapt, support, drive, target_gain, excitability, p);
}

inline float4 correction_for(uint row, uint tile, device const uint *targets, device const float4 *corrections, constant Params &p) {
    if (row == targets[0]) return corrections[tile];
    if (row == targets[1]) return corrections[p.tiles + tile];
    return 0.0f;
}

kernel void csr_rate_corrected(device const uint *rowptr [[buffer(0)]], device const uint *columns [[buffer(1)]], device const float *weights [[buffer(2)]], device const float4 *rate_in [[buffer(3)]], device float4 *rate_out [[buffer(4)]], device float4 *adapt [[buffer(5)]], device float4 *support [[buffer(6)]], device const float4 *drive [[buffer(7)]], constant Params &p [[buffer(8)]], device const uint *targets [[buffer(9)]], device const float4 *corrections [[buffer(10)]], device const float4 *source_gain [[buffer(11)]], device const float4 *target_gain [[buffer(12)]], device const float4 *excitability [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= p.n) return;
    float4 recurrent = 0.0f;
    for (uint edge = rowptr[row]; edge < rowptr[row + 1]; ++edge) {
        uint source = columns[edge];
        recurrent += weights[edge] * source_gain[source * p.tiles + tile] * rate_in[source * p.tiles + tile];
    }
    recurrent += correction_for(row, tile, targets, corrections, p);
    rate_update(row, tile, recurrent, rate_in, rate_out, adapt, support, drive, target_gain, excitability, p);
}

kernel void csr_rate_simd_corrected(device const uint *rowptr [[buffer(0)]], device const uint *columns [[buffer(1)]], device const float *weights [[buffer(2)]], device const float4 *rate_in [[buffer(3)]], device float4 *rate_out [[buffer(4)]], device float4 *adapt [[buffer(5)]], device float4 *support [[buffer(6)]], device const float4 *drive [[buffer(7)]], constant Params &p [[buffer(8)]], device const uint *targets [[buffer(9)]], device const float4 *corrections [[buffer(10)]], device const float4 *source_gain [[buffer(11)]], device const float4 *target_gain [[buffer(12)]], device const float4 *excitability [[buffer(13)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row_tile = gid >> 5, row = row_tile / p.tiles, tile = row_tile % p.tiles;
    if (row >= p.n) return;
    float4 recurrent = 0.0f;
    for (uint edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        uint source = columns[edge];
        recurrent += weights[edge] * source_gain[source * p.tiles + tile] * rate_in[source * p.tiles + tile];
    }
    recurrent = float4(simd_sum(recurrent.x), simd_sum(recurrent.y), simd_sum(recurrent.z), simd_sum(recurrent.w));
    if (lane) return;
    recurrent += correction_for(row, tile, targets, corrections, p);
    rate_update(row, tile, recurrent, rate_in, rate_out, adapt, support, drive, target_gain, excitability, p);
}

kernel void gather_rates(device const float4 *rates [[buffer(0)]], device const uint *indices [[buffer(1)]], device float4 *selected [[buffer(2)]], constant Params &p [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    selected[gid] = rates[indices[row] * p.tiles + tile];
}

kernel void project_readouts(device const uint *ptr [[buffer(0)]], device const uint *cols [[buffer(1)]], device const float *weights [[buffer(2)]], device const float4 *rates [[buffer(3)]], device float4 *output [[buffer(4)]], device const float4 *readout_gain [[buffer(5)]], constant Params &p [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    uint row = gid / p.tiles, tile = gid % p.tiles;
    if (row >= 384) return;
    float4 x = 0.0f;
    for (uint edge = ptr[row]; edge < ptr[row + 1]; ++edge) x += weights[edge] * rates[cols[edge] * p.tiles + tile];
    output[gid] = readout_gain[gid] * x;
}

kernel void physiology_partials(device const float4 *rates [[buffer(0)]], device const float4 *support [[buffer(1)]], device float4 *partial [[buffer(2)]], constant Params &p [[buffer(8)]], uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
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

kernel void physiology_final(device const float4 *partial [[buffer(0)]], device float4 *output [[buffer(1)]], constant Params &p [[buffer(8)]], uint tile [[thread_position_in_grid]]) {
    if (tile >= p.tiles) return;
    uint groups = (p.n + 255) / 256;
    float4 sum = 0.0f, peak = 0.0f, support = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        uint base = (group * p.tiles + tile) * 3;
        sum += partial[base];
        peak = max(peak, partial[base + 1]);
        support += partial[base + 2];
    }
    output[384 * p.tiles + tile] = sum / p.n;
    output[385 * p.tiles + tile] = peak;
    output[386 * p.tiles + tile] = support / p.n;
}
