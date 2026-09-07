//! CNS service adapter reference. Raw inputs end here, before actual recurrence.
//! Production Metal serving keeps all rates on-device and returns only latent512.
use crate::{gemm_into, linear, Linear};
const N: usize = 165_122;
const SITES: usize = 1771;
const RECEPTORS: usize = 4107;
const TYPES: usize = 10;
const NEURON_TYPES: usize = 11752;
const SITE_EDGES: usize = 4669;
const BODY: usize = 43;
const BODY_HIDDEN: usize = 128;
const BODY_TARGETS: usize = 11233;
const INPUT: usize = SITES * 3 + BODY;
const LATENT: usize = 512;
const READOUT_RANK: usize = 64;

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn fail(message: &str) -> String {
    message.to_owned()
}
fn take(flat: &[f32], cursor: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let stop = cursor
        .checked_add(n)
        .ok_or_else(|| fail("adapter tensor overflow"))?;
    let value = flat
        .get(*cursor..stop)
        .ok_or_else(|| fail("truncated CNS adapter parameters"))?;
    *cursor = stop;
    Ok(value.to_vec())
}
fn matrix(flat: &[f32], cursor: &mut usize, out: usize, input: usize) -> Result<Linear, String> {
    let count = out
        .checked_mul(input)
        .and_then(|n| n.checked_add(out))
        .ok_or_else(|| fail("adapter matrix overflow"))?;
    if flat.len().saturating_sub(*cursor) < count {
        return Err(fail("truncated CNS adapter matrix"));
    }
    Ok(linear(flat, cursor, out, input)?)
}

#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct CnsAdapter {
    receptor_rows: Vec<usize>,
    receptor_types: Vec<usize>,
    site_ptr: Vec<usize>,
    site_indices: Vec<usize>,
    site_weights: Vec<f32>,
    body_rows: Vec<usize>,
    neuron_types: Vec<usize>,
    spectral: Vec<f32>,
    optic_gain: Vec<f32>,
    optic_bias: Vec<f32>,
    body_mean: Vec<f32>,
    body_scale: Vec<f32>,
    body_input: Linear,
    body_output: Linear,
    dynamics: [Vec<f32>; 5],
    neutral_drive: Vec<f32>,
    readout_projection: Linear,
    readout_output: Linear,
}

impl CnsAdapter {
    fn afferents_inner(&self, sensory: &[f32], batch: usize) -> Vec<f32> {
        let mut body = vec![0.0; batch * BODY];
        for row in 0..batch {
            for i in 0..BODY {
                body[row * BODY + i] = ((sensory[row * INPUT + SITES * 3 + i] - self.body_mean[i])
                    / self.body_scale[i])
                    .clamp(-8.0, 8.0);
            }
        }
        let mut hidden = Vec::new();
        gemm_into(&body, batch, BODY, &self.body_input, &mut hidden);
        for x in &mut hidden {
            *x = x.tanh();
        }
        let mut body_current = Vec::new();
        gemm_into(
            &hidden,
            batch,
            BODY_HIDDEN,
            &self.body_output,
            &mut body_current,
        );
        let mut drive = vec![0.0; batch * N];
        for row in 0..batch {
            let body_is_neutral = (0..BODY).all(|i| {
                sensory[row * INPUT + SITES * 3 + i].to_bits() == self.body_mean[i].to_bits()
            });
            for (i, &graph_row) in self.body_rows.iter().enumerate() {
                drive[row * N + graph_row] = if body_is_neutral {
                    self.neutral_drive[graph_row]
                } else {
                    sigmoid(body_current[row * BODY_TARGETS + i])
                };
            }
            for receptor in 0..RECEPTORS {
                let start = self.site_ptr[receptor];
                let stop = self.site_ptr[receptor + 1];
                // Unsupported receptors do not acquire fabricated tonic input.
                if start == stop {
                    continue;
                }
                let kind = self.receptor_types[receptor];
                let mut mixture = 0.0;
                let mut neutral = true;
                for edge in start..stop {
                    let site = self.site_indices[edge];
                    for color in 0..3 {
                        neutral &=
                            sensory[row * INPUT + site * 3 + color].to_bits() == 0.5f32.to_bits();
                        mixture += self.site_weights[edge]
                            * self.spectral[kind * 3 + color]
                            * sensory[row * INPUT + site * 3 + color];
                    }
                }
                let graph_row = self.receptor_rows[receptor];
                drive[row * N + graph_row] = if neutral {
                    self.neutral_drive[graph_row]
                } else {
                    sigmoid(self.optic_bias[kind] + self.optic_gain[kind] * (2.0 * mixture - 1.0))
                };
            }
        }
        drive
    }
}

#[cfg(feature = "python")]
mod python;
impl CnsAdapter {
    pub fn from_packed(
        parameters: &[f32],
        neutral_drive: &[f32],
        receptor_rows: &[u32],
        receptor_types: &[u32],
        site_ptr: &[u32],
        site_indices: &[u32],
        site_weights: &[f32],
        body_rows: &[u32],
        neuron_types: &[u32],
    ) -> Result<Self, String> {
        let flat = parameters;
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS adapter parameters"));
        }
        let indices = |a: &[u32], count: usize, bound: usize| -> Result<Vec<usize>, String> {
            if a.len() != count || a.iter().any(|&x| x as usize >= bound) {
                return Err(fail("CNS adapter index shape/range differs"));
            }
            Ok(a.iter().map(|&x| x as usize).collect())
        };
        let receptor_rows = indices(receptor_rows, RECEPTORS, N)?;
        let receptor_types = indices(receptor_types, RECEPTORS, TYPES)?;
        let body_rows = indices(body_rows, BODY_TARGETS, N)?;
        let neuron_types = indices(neuron_types, N, NEURON_TYPES)?;
        let site_ptr = indices(site_ptr, RECEPTORS + 1, SITE_EDGES + 1)?;
        let site_indices = indices(site_indices, SITE_EDGES, SITES)?;
        let site_weights = site_weights.to_vec();
        if receptor_rows.windows(2).any(|x| x[0] >= x[1])
            || body_rows.windows(2).any(|x| x[0] >= x[1])
            || site_ptr[0] != 0
            || site_ptr[RECEPTORS] != SITE_EDGES
            || site_ptr.windows(2).any(|x| x[0] > x[1])
            || site_weights.len() != SITE_EDGES
            || site_weights.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(fail("invalid anatomical CNS afferent topology"));
        }
        let mut injected = vec![false; N];
        for &row in receptor_rows.iter().chain(body_rows.iter()) {
            if injected[row] {
                return Err(fail("CNS afferent rows overlap"));
            }
            injected[row] = true;
        }
        for span in site_ptr.windows(2) {
            if span[0] != span[1] {
                let sum: f32 = site_weights[span[0]..span[1]].iter().sum();
                if (sum - 1.0).abs() > 2e-6 {
                    return Err(fail("site mixture must sum to one"));
                }
            }
        }
        let mut cursor = 0;
        let mut spectral = take(flat, &mut cursor, TYPES * 3)?;
        for row in spectral.chunks_exact_mut(3) {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in row.iter_mut() {
                *x = (*x - maximum).exp();
                sum += *x;
            }
            for x in row {
                *x /= sum;
            }
        }
        let optic_gain = take(flat, &mut cursor, TYPES)?
            .into_iter()
            .map(softplus)
            .collect();
        let optic_bias = take(flat, &mut cursor, TYPES)?;
        let body_mean = take(flat, &mut cursor, BODY)?;
        let body_scale = take(flat, &mut cursor, BODY)?;
        if body_scale.iter().any(|&x| x <= 0.0) {
            return Err(fail("body scale must be positive"));
        }
        let body_input = matrix(flat, &mut cursor, BODY_HIDDEN, BODY)?;
        let body_output = matrix(flat, &mut cursor, BODY_TARGETS, BODY_HIDDEN)?;
        let mut dynamics: [Vec<f32>; 5] = std::array::from_fn(|_| Vec::new());
        for (i, array) in dynamics.iter_mut().enumerate() {
            *array = take(flat, &mut cursor, NEURON_TYPES)?
                .into_iter()
                .map(|x| match i {
                    0 => 0.05 + 0.4 * sigmoid(x),
                    1 => 0.5 + 1.5 * sigmoid(x),
                    2 => 0.02 + 0.23 * sigmoid(x),
                    3 => 0.5 * sigmoid(x),
                    4 => 0.25 + 4.75 * sigmoid(x),
                    _ => unreachable!(),
                })
                .collect();
        }
        let projection_weight = take(flat, &mut cursor, READOUT_RANK * N)?;
        let readout_output = matrix(flat, &mut cursor, LATENT, READOUT_RANK)?;
        if cursor != flat.len() {
            return Err(fail("trailing CNS adapter parameters"));
        }
        if neutral_drive.len() != N
            || neutral_drive
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(fail(
                "afferent neutral drive must be finite [165122] in [0,1]",
            ));
        }
        let mut readout_projection = Linear {
            out: READOUT_RANK,
            input: N,
            weight: projection_weight,
            bias: vec![0.0; READOUT_RANK],
        };
        // Enforce the graph-edge boundary even if unused learned columns contain
        // nonzero weights. Never rely on the trainer voluntarily zeroing them.
        for row in readout_projection.weight.chunks_exact_mut(N) {
            for &index in receptor_rows.iter().chain(body_rows.iter()) {
                row[index] = 0.0;
            }
        }
        let mut afferent = vec![false; N];
        for &row in receptor_rows.iter().chain(body_rows.iter()) {
            afferent[row] = true;
        }
        if neutral_drive
            .iter()
            .zip(&afferent)
            .any(|(&value, &is_afferent)| !is_afferent && value != 0.0)
        {
            return Err(fail("neutral drive must be zero on non-afferent rows"));
        }
        if site_ptr.windows(2).enumerate().any(|(receptor, span)| {
            span[0] == span[1] && neutral_drive[receptor_rows[receptor]] != 0.0
        }) {
            return Err(fail("unsupported receptors must have zero neutral drive"));
        }
        Ok(Self {
            receptor_rows,
            receptor_types,
            site_ptr,
            site_indices,
            site_weights,
            body_rows,
            neuron_types,
            spectral,
            optic_gain,
            optic_bias,
            body_mean,
            body_scale,
            body_input,
            body_output,
            dynamics,
            neutral_drive: neutral_drive.to_vec(),
            readout_projection,
            readout_output,
        })
    }
    /// Raw observations are allowed solely at this afferent side of the CNS.
    pub fn encode_afferents_flat(&self, values: &[f32], batch: usize) -> Result<Vec<f32>, String> {
        if !(1..=32).contains(&batch) || values.len() != batch * INPUT {
            return Err(fail("CNS sensory input must be [B,5356], B in1..32"));
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS sensory input"));
        }
        if values
            .chunks_exact(INPUT)
            .any(|row| row[..SITES * 3].iter().any(|x| !(0.0..=1.0).contains(x)))
        {
            return Err(fail("optic RGB must be in[0,1]"));
        }
        let result = self.afferents_inner(values, batch);
        if result.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS afferent result"));
        }
        Ok(result)
    }
    /// Only baseline-relative actual graph rates enter the rank-64 readout.
    pub fn project_rates_flat(&self, values: &[f32], batch: usize) -> Result<Vec<f32>, String> {
        if !(1..=32).contains(&batch) || values.len() != batch * N {
            return Err(fail("CNS rates must be [B,165122], B in1..32"));
        }
        if values
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(fail("CNS rate outside finite[0,1]"));
        }
        let baseline = self.expanded_dynamics(0);
        let mut centered = values.to_vec();
        for row in centered.chunks_exact_mut(N) {
            for (x, &r0) in row.iter_mut().zip(&baseline) {
                *x -= r0;
            }
        }
        let mut hidden = Vec::new();
        gemm_into(&centered, batch, N, &self.readout_projection, &mut hidden);
        let mut output = Vec::new();
        gemm_into(
            &hidden,
            batch,
            READOUT_RANK,
            &self.readout_output,
            &mut output,
        );
        for x in &mut output {
            *x = x.tanh();
        }
        if output.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS readout result"));
        }
        Ok(output)
    }
    fn expanded_dynamics(&self, index: usize) -> Vec<f32> {
        self.neuron_types
            .iter()
            .map(|&t| self.dynamics[index][t])
            .collect()
    }

    /// Baseline, recurrent gain, tau, adaptation gain and adaptation tau.
    pub fn effective_dynamics_flat(&self) -> [Vec<f32>; 5] {
        std::array::from_fn(|i| self.expanded_dynamics(i))
    }

    pub fn neutral_drive(&self) -> &[f32] {
        &self.neutral_drive
    }

    /// Deterministic CPU reference for the current full-graph V2 recurrence.
    /// Arrays are resident-major `[batch, 165122]`; CSR rows are postsynaptic.
    #[allow(clippy::too_many_arguments)]
    pub fn step_rates_flat(
        &self,
        graph_crow: &[u32],
        graph_col: &[u32],
        graph_weight: &[f32],
        drive: &[f32],
        rate: &[f32],
        adaptation: &[f32],
        support: &[f32],
        dt: f32,
        batch: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if !(1..=32).contains(&batch)
            || graph_crow.len() != N + 1
            || graph_col.len() != graph_weight.len()
            || graph_crow.first() != Some(&0)
            || graph_crow.last().copied() != Some(graph_col.len() as u32)
            || graph_crow.windows(2).any(|x| x[0] > x[1])
            || graph_col.iter().any(|&x| x as usize >= N)
            || graph_weight.iter().any(|x| !x.is_finite())
        {
            return Err(fail("invalid full-CNS CSR or batch"));
        }
        let state_len = batch * N;
        if drive.len() != state_len
            || rate.len() != state_len
            || adaptation.len() != state_len
            || support.len() != state_len
            || !dt.is_finite()
            || !(0.0..=0.2).contains(&dt)
            || dt == 0.0
            || drive
                .iter()
                .chain(rate)
                .chain(adaptation)
                .chain(support)
                .any(|x| !x.is_finite())
            || rate.iter().any(|x| !(0.0..=1.0).contains(x))
            || adaptation.iter().any(|x| !(-1.0..=1.0).contains(x))
            || support.iter().any(|x| !(0.65..=1.0).contains(x))
        {
            return Err(fail("invalid V2 CNS state, drive or dt"));
        }
        let [baseline, gain, tau, adaptation_gain, adaptation_tau] = self.effective_dynamics_flat();
        let mut current = rate.to_vec();
        let mut next = vec![0.0; state_len];
        for _ in 0..2 {
            for resident in 0..batch {
                let base = resident * N;
                for row in 0..N {
                    let mut recurrent = 0.0f32;
                    for edge in graph_crow[row] as usize..graph_crow[row + 1] as usize {
                        let source = graph_col[edge] as usize;
                        recurrent +=
                            graph_weight[edge] * (current[base + source] - baseline[source]);
                    }
                    let h = baseline[row].min(1.0 - baseline[row]);
                    let u = drive[base + row] - self.neutral_drive[row] + gain[row] * recurrent
                        - adaptation_gain[row] * adaptation[base + row];
                    let target = baseline[row] + support[base + row] * h * (u / h).tanh();
                    let alpha = -(-dt / (2.0 * tau[row])).exp_m1();
                    next[base + row] = current[base + row] + alpha * (target - current[base + row]);
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
        let mut next_adaptation = adaptation.to_vec();
        let mut next_support = support.to_vec();
        for resident in 0..batch {
            let base = resident * N;
            for row in 0..N {
                let x = current[base + row] - baseline[row];
                let adaptation_alpha = -(-dt / adaptation_tau[row]).exp_m1();
                next_adaptation[base + row] += adaptation_alpha * (x - next_adaptation[base + row]);
                let h = baseline[row].min(1.0 - baseline[row]);
                next_support[base + row] = (next_support[base + row]
                    + dt * (0.024 * (1.0 - next_support[base + row]) - 0.003 * x.abs() / h))
                    .clamp(0.65, 1.0);
            }
        }
        if current
            .iter()
            .chain(&next_adaptation)
            .chain(&next_support)
            .any(|x| !x.is_finite())
        {
            return Err(fail("nonfinite V2 CNS update"));
        }
        Ok((current, next_adaptation, next_support))
    }
}
