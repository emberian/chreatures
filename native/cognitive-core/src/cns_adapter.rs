//! CNS service adapter reference. Raw inputs end here, before actual recurrence.
//! Production Metal serving keeps all rates on-device and returns only latent512.
use crate::{gemm_into, linear, Linear};
use numpy::{
    ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

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
fn fail(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}
fn take(flat: &[f32], cursor: &mut usize, n: usize) -> PyResult<Vec<f32>> {
    let stop = cursor
        .checked_add(n)
        .ok_or_else(|| fail("adapter tensor overflow"))?;
    let value = flat
        .get(*cursor..stop)
        .ok_or_else(|| fail("truncated CNS adapter parameters"))?;
    *cursor = stop;
    Ok(value.to_vec())
}
fn matrix(flat: &[f32], cursor: &mut usize, out: usize, input: usize) -> PyResult<Linear> {
    let count = out
        .checked_mul(input)
        .and_then(|n| n.checked_add(out))
        .ok_or_else(|| fail("adapter matrix overflow"))?;
    if flat.len().saturating_sub(*cursor) < count {
        return Err(fail("truncated CNS adapter matrix"));
    }
    linear(flat, cursor, out, input)
}

#[pyclass]
pub(crate) struct CnsAdapter {
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
    readout: Linear,
}

#[pymethods]
impl CnsAdapter {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parameters: PyReadonlyArray1<'_, f32>,
        receptor_rows: PyReadonlyArray1<'_, u32>,
        receptor_types: PyReadonlyArray1<'_, u32>,
        site_ptr: PyReadonlyArray1<'_, u32>,
        site_indices: PyReadonlyArray1<'_, u32>,
        site_weights: PyReadonlyArray1<'_, f32>,
        body_rows: PyReadonlyArray1<'_, u32>,
        neuron_types: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Self> {
        let flat = parameters.as_slice()?;
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS adapter parameters"));
        }
        let indices = |a: &[u32], count: usize, bound: usize| -> PyResult<Vec<usize>> {
            if a.len() != count || a.iter().any(|&x| x as usize >= bound) {
                return Err(fail("CNS adapter index shape/range differs"));
            }
            Ok(a.iter().map(|&x| x as usize).collect())
        };
        let receptor_rows = indices(receptor_rows.as_slice()?, RECEPTORS, N)?;
        let receptor_types = indices(receptor_types.as_slice()?, RECEPTORS, TYPES)?;
        let body_rows = indices(body_rows.as_slice()?, BODY_TARGETS, N)?;
        let neuron_types = indices(neuron_types.as_slice()?, N, NEURON_TYPES)?;
        let site_ptr = indices(site_ptr.as_slice()?, RECEPTORS + 1, SITE_EDGES + 1)?;
        let site_indices = indices(site_indices.as_slice()?, SITE_EDGES, SITES)?;
        let site_weights = site_weights.as_slice()?.to_vec();
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
                    0 => 0.5 * x.tanh(),
                    1 => 0.025 + 0.475 * sigmoid(x),
                    _ => 0.5 + sigmoid(x),
                })
                .collect();
        }
        let mut readout = matrix(flat, &mut cursor, LATENT, N)?;
        if cursor != flat.len() {
            return Err(fail("trailing CNS adapter parameters"));
        }
        // Enforce the graph-edge boundary even if unused learned columns contain
        // nonzero weights. Never rely on the trainer voluntarily zeroing them.
        for row in readout.weight.chunks_exact_mut(N) {
            for &index in receptor_rows.iter().chain(body_rows.iter()) {
                row[index] = 0.0;
            }
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
            readout,
        })
    }

    fn encode_afferents<'py>(
        &self,
        py: Python<'py>,
        sensory: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = sensory.shape();
        if shape.len() != 2 || shape[1] != INPUT || !(1..=32).contains(&shape[0]) {
            return Err(fail("CNS sensory input must be [B,5356], B in1..32"));
        }
        let batch = shape[0];
        let values = sensory.as_slice()?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS sensory input"));
        }
        if values
            .chunks_exact(INPUT)
            .any(|row| row[..SITES * 3].iter().any(|x| !(0.0..=1.0).contains(x)))
        {
            return Err(fail("optic RGB must be in[0,1]"));
        }
        let result = py.detach(|| self.afferents_inner(values, batch));
        if result.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS afferent result"));
        }
        Ok(Array2::from_shape_vec((batch, N), result)
            .unwrap()
            .into_pyarray(py))
    }

    fn project_rates<'py>(
        &self,
        py: Python<'py>,
        rates: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = rates.shape();
        if shape.len() != 2 || shape[1] != N || !(1..=32).contains(&shape[0]) {
            return Err(fail("CNS rates must be [B,165122], B in1..32"));
        }
        let values = rates.as_slice()?;
        if values
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(fail("CNS rate outside finite[0,1]"));
        }
        let batch = shape[0];
        let result = py.detach(|| {
            let mut output = Vec::new();
            gemm_into(values, batch, N, &self.readout, &mut output);
            for x in &mut output {
                *x = x.tanh();
            }
            output
        });
        if result.iter().any(|x| !x.is_finite()) {
            return Err(fail("nonfinite CNS readout result"));
        }
        Ok(Array2::from_shape_vec((batch, LATENT), result)
            .unwrap()
            .into_pyarray(py))
    }

    fn effective_dynamics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (name, values) in ["bias", "tau", "source", "target", "excitability"]
            .iter()
            .zip(&self.dynamics)
        {
            let expanded: Vec<f32> = self.neuron_types.iter().map(|&t| values[t]).collect();
            result.set_item(
                *name,
                numpy::ndarray::Array1::from_vec(expanded).into_pyarray(py),
            )?;
        }
        Ok(result)
    }
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
            for (i, &graph_row) in self.body_rows.iter().enumerate() {
                drive[row * N + graph_row] = sigmoid(body_current[row * BODY_TARGETS + i]);
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
                for edge in start..stop {
                    let site = self.site_indices[edge];
                    for color in 0..3 {
                        mixture += self.site_weights[edge]
                            * self.spectral[kind * 3 + color]
                            * sensory[row * INPUT + site * 3 + color];
                    }
                }
                drive[row * N + self.receptor_rows[receptor]] =
                    sigmoid(self.optic_bias[kind] + self.optic_gain[kind] * (2.0 * mixture - 1.0));
            }
        }
        drive
    }
}
