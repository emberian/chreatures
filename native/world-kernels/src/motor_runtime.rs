//! Batched Torch-free inference for immutable inherited motor parameters.

use numpy::{
    ndarray::{Array1, Array2},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ACTIONS: usize = 8;
const MAX_BATCH: usize = 256;
#[cfg(target_os = "macos")]
const RUNTIME_IDENTITY: &str = "chreatures-motor-runtime-v1:f32-accelerate-sgemm";
#[cfg(not(target_os = "macos"))]
const RUNTIME_IDENTITY: &str = "chreatures-motor-runtime-v1:f32-ordered-scalar";

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[derive(Clone, Copy)]
struct Linear {
    weight: usize,
    bias: usize,
    input: usize,
    output: usize,
}

#[pyclass]
pub struct MotorRuntime {
    feature: usize,
    physiology: usize,
    context: usize,
    hidden: usize,
    projection: usize,
    gated: bool,
    state_std: bool,
    packed: Vec<f32>,
    projection_weight: usize,
    context_feature: usize,
    context_action: usize,
    context_recur: usize,
    gate_feature: usize,
    gate_action: usize,
    gate_recur: usize,
    gate_bias: usize,
    encoder: Linear,
    trunk0: Linear,
    trunk2: Linear,
    policy: Linear,
    value: Linear,
    predictor0: Linear,
    predictor2: Linear,
    log_std_offset: usize,
    std_offset: Linear,
}

fn take(cursor: &mut usize, count: usize) -> usize {
    let start = *cursor;
    *cursor += count;
    start
}
fn layer(cursor: &mut usize, input: usize, output: usize) -> Linear {
    let weight = take(cursor, input * output);
    let bias = take(cursor, output);
    Linear {
        weight,
        bias,
        input,
        output,
    }
}
fn valid(values: &[f32]) -> bool {
    values.iter().all(|x| x.is_finite())
}

impl MotorRuntime {
    fn multiply(
        &self,
        input: &[f32],
        batch: usize,
        width: usize,
        weight: usize,
        output: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; batch * output];
        #[cfg(target_os = "macos")]
        unsafe {
            // C = row-major input[B,width] * packed_weight[output,width]^T.
            cblas_sgemm(
                101,
                111,
                112,
                batch as i32,
                output as i32,
                width as i32,
                1.0,
                input.as_ptr(),
                width as i32,
                self.packed.as_ptr().add(weight),
                width as i32,
                0.0,
                out.as_mut_ptr(),
                output as i32,
            );
        }
        #[cfg(not(target_os = "macos"))]
        for row in 0..batch {
            for o in 0..output {
                let mut sum = 0.0f32;
                for i in 0..width {
                    sum += input[row * width + i] * self.packed[weight + o * width + i];
                }
                out[row * output + o] = sum;
            }
        }
        out
    }

    fn linear(&self, input: &[f32], batch: usize, layer: Linear) -> Vec<f32> {
        let mut out = self.multiply(input, batch, layer.input, layer.weight, layer.output);
        for row in 0..batch {
            for o in 0..layer.output {
                out[row * layer.output + o] += self.packed[layer.bias + o];
            }
        }
        out
    }
    fn matrix(
        &self,
        input: &[f32],
        batch: usize,
        input_width: usize,
        weight: usize,
        output: usize,
    ) -> Vec<f32> {
        self.multiply(input, batch, input_width, weight, output)
    }
    fn project_values(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let mut out = self.matrix(
            input,
            batch,
            self.feature,
            self.projection_weight,
            self.projection,
        );
        for x in &mut out {
            *x = x.tanh();
        }
        out
    }
    fn batch(shape: &[usize], width: usize) -> PyResult<usize> {
        if shape.len() != 2 || shape[1] != width || shape[0] == 0 || shape[0] > MAX_BATCH {
            Err(PyValueError::new_err("motor runtime batch shape differs"))
        } else {
            Ok(shape[0])
        }
    }
}

#[pymethods]
impl MotorRuntime {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        feature_dim: usize,
        physiology_dim: usize,
        context_dim: usize,
        hidden_dim: usize,
        projection_dim: usize,
        gated: bool,
        state_conditioned_std: bool,
        packed: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        if feature_dim == 0
            || physiology_dim == 0
            || context_dim == 0
            || hidden_dim == 0
            || projection_dim == 0
            || feature_dim > 8192
            || physiology_dim > 64
            || context_dim > 1024
            || hidden_dim > 2048
            || projection_dim > 2048
        {
            return Err(PyValueError::new_err("invalid motor runtime dimensions"));
        }
        let mut c = 0;
        let projection_weight = take(&mut c, projection_dim * feature_dim);
        let context_feature = take(&mut c, context_dim * projection_dim);
        let context_action = take(&mut c, context_dim * ACTIONS);
        let context_recur = take(&mut c, context_dim * context_dim);
        let (gate_feature, gate_action, gate_recur, gate_bias) = if gated {
            (
                take(&mut c, context_dim * projection_dim),
                take(&mut c, context_dim * ACTIONS),
                take(&mut c, context_dim * context_dim),
                take(&mut c, context_dim),
            )
        } else {
            (0, 0, 0, 0)
        };
        let encoder = layer(&mut c, feature_dim, hidden_dim);
        let trunk0 = layer(
            &mut c,
            hidden_dim + physiology_dim + context_dim,
            hidden_dim,
        );
        let trunk2 = layer(&mut c, hidden_dim, hidden_dim);
        let policy = layer(&mut c, hidden_dim, ACTIONS);
        let value = layer(&mut c, hidden_dim, 1);
        let predictor0 = layer(&mut c, hidden_dim + ACTIONS, hidden_dim);
        let predictor2 = layer(&mut c, hidden_dim, projection_dim);
        let log_std_offset = take(&mut c, ACTIONS);
        let std_offset = if state_conditioned_std {
            layer(&mut c, hidden_dim, ACTIONS)
        } else {
            Linear {
                weight: 0,
                bias: 0,
                input: 0,
                output: 0,
            }
        };
        let values = packed.as_slice()?;
        if values.len() != c || !valid(values) {
            return Err(PyValueError::new_err(
                "motor parameter pack length or values differ",
            ));
        }
        Ok(Self {
            feature: feature_dim,
            physiology: physiology_dim,
            context: context_dim,
            hidden: hidden_dim,
            projection: projection_dim,
            gated,
            state_std: state_conditioned_std,
            packed: values.to_vec(),
            projection_weight,
            context_feature,
            context_action,
            context_recur,
            gate_feature,
            gate_action,
            gate_recur,
            gate_bias,
            encoder,
            trunk0,
            trunk2,
            policy,
            value,
            predictor0,
            predictor2,
            log_std_offset,
            std_offset,
        })
    }

    #[getter]
    fn identity(&self) -> &'static str {
        RUNTIME_IDENTITY
    }

    #[allow(clippy::type_complexity)]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        normalized: PyReadonlyArray2<'_, f32>,
        physiology: PyReadonlyArray2<'_, f32>,
        context: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let batch = Self::batch(normalized.shape(), self.feature)?;
        if physiology.shape() != [batch, self.physiology]
            || context.shape() != [batch, self.context]
        {
            return Err(PyValueError::new_err("motor forward shapes differ"));
        }
        let n = normalized.as_slice()?;
        let p = physiology.as_slice()?;
        let x = context.as_slice()?;
        if !valid(n) || !valid(p) || !valid(x) {
            return Err(PyValueError::new_err("motor forward inputs must be finite"));
        }
        let mut encoded = self.linear(n, batch, self.encoder);
        for v in &mut encoded {
            *v = v.tanh();
        }
        let width = self.hidden + self.physiology + self.context;
        let mut joined = vec![0.0; batch * width];
        for b in 0..batch {
            joined[b * width..b * width + self.hidden]
                .copy_from_slice(&encoded[b * self.hidden..(b + 1) * self.hidden]);
            joined[b * width + self.hidden..b * width + self.hidden + self.physiology]
                .copy_from_slice(&p[b * self.physiology..(b + 1) * self.physiology]);
            joined[b * width + self.hidden + self.physiology..(b + 1) * width]
                .copy_from_slice(&x[b * self.context..(b + 1) * self.context]);
        }
        let mut hidden = self.linear(&joined, batch, self.trunk0);
        for v in &mut hidden {
            *v = v.tanh();
        }
        hidden = self.linear(&hidden, batch, self.trunk2);
        for v in &mut hidden {
            *v = v.tanh();
        }
        let mean = self.linear(&hidden, batch, self.policy);
        let value = self.linear(&hidden, batch, self.value);
        Ok((
            Array2::from_shape_vec((batch, ACTIONS), mean)
                .unwrap()
                .into_pyarray(py),
            Array1::from_vec(value).into_pyarray(py),
            Array2::from_shape_vec((batch, self.hidden), hidden)
                .unwrap()
                .into_pyarray(py),
        ))
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        normalized: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let batch = Self::batch(normalized.shape(), self.feature)?;
        let input = normalized.as_slice()?;
        if !valid(input) {
            return Err(PyValueError::new_err("projection input must be finite"));
        }
        Ok(
            Array2::from_shape_vec((batch, self.projection), self.project_values(input, batch))
                .unwrap()
                .into_pyarray(py),
        )
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        hidden: PyReadonlyArray2<'_, f32>,
        action: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let batch = Self::batch(hidden.shape(), self.hidden)?;
        if action.shape() != [batch, ACTIONS] {
            return Err(PyValueError::new_err("predictor shapes differ"));
        }
        let h = hidden.as_slice()?;
        let a = action.as_slice()?;
        if !valid(h) || !valid(a) {
            return Err(PyValueError::new_err("predictor inputs must be finite"));
        }
        let width = self.hidden + ACTIONS;
        let mut joined = vec![0.0; batch * width];
        for b in 0..batch {
            joined[b * width..b * width + self.hidden]
                .copy_from_slice(&h[b * self.hidden..(b + 1) * self.hidden]);
            joined[b * width + self.hidden..(b + 1) * width]
                .copy_from_slice(&a[b * ACTIONS..(b + 1) * ACTIONS]);
        }
        let mut middle = self.linear(&joined, batch, self.predictor0);
        for v in &mut middle {
            *v = v.tanh();
        }
        let out = self.linear(&middle, batch, self.predictor2);
        Ok(Array2::from_shape_vec((batch, self.projection), out)
            .unwrap()
            .into_pyarray(py))
    }

    fn log_std<'py>(
        &self,
        py: Python<'py>,
        hidden: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let batch = Self::batch(hidden.shape(), self.hidden)?;
        let h = hidden.as_slice()?;
        if !valid(h) {
            return Err(PyValueError::new_err("hidden values must be finite"));
        }
        let offsets = if self.state_std {
            self.linear(h, batch, self.std_offset)
        } else {
            vec![0.0; batch * ACTIONS]
        };
        let mut out = vec![0.0; batch * ACTIONS];
        for b in 0..batch {
            for a in 0..ACTIONS {
                let value = self.packed[self.log_std_offset + a]
                    + if self.state_std {
                        2.0 * (offsets[b * ACTIONS + a] / 2.0).tanh()
                    } else {
                        0.0
                    };
                out[b * ACTIONS + a] = value.clamp(-3.5, 0.3);
            }
        }
        Ok(Array2::from_shape_vec((batch, ACTIONS), out)
            .unwrap()
            .into_pyarray(py))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn update_context<'py>(
        &self,
        py: Python<'py>,
        next_features: PyReadonlyArray2<'_, f32>,
        action: PyReadonlyArray2<'_, f32>,
        context: PyReadonlyArray2<'_, f32>,
        previous_features: PyReadonlyArray2<'_, f32>,
        previous_prediction: PyReadonlyArray2<'_, f32>,
        has_previous: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let batch = Self::batch(next_features.shape(), self.feature)?;
        if action.shape() != [batch, ACTIONS]
            || context.shape() != [batch, self.context]
            || previous_features.shape() != [batch, self.feature]
            || previous_prediction.shape() != [batch, self.projection]
            || has_previous.shape() != [batch]
        {
            return Err(PyValueError::new_err("context update shapes differ"));
        }
        let next = next_features.as_slice()?;
        let a = action.as_slice()?;
        let old = context.as_slice()?;
        let previous = previous_features.as_slice()?;
        let predicted = previous_prediction.as_slice()?;
        let mask = has_previous.as_slice()?;
        if !valid(next) || !valid(a) || !valid(old) || !valid(previous) || !valid(predicted) {
            return Err(PyValueError::new_err(
                "context update inputs must be finite",
            ));
        }
        let projected = self.project_values(next, batch);
        let previous_projected = self.project_values(previous, batch);
        let cf = self.matrix(
            &projected,
            batch,
            self.projection,
            self.context_feature,
            self.context,
        );
        let ca = self.matrix(a, batch, ACTIONS, self.context_action, self.context);
        let cr = self.matrix(old, batch, self.context, self.context_recur, self.context);
        let mut updated = vec![0.0; batch * self.context];
        for i in 0..updated.len() {
            updated[i] = (cf[i] + ca[i] + cr[i]).tanh();
        }
        if self.gated {
            let gf = self.matrix(
                &projected,
                batch,
                self.projection,
                self.gate_feature,
                self.context,
            );
            let ga = self.matrix(a, batch, ACTIONS, self.gate_action, self.context);
            let gr = self.matrix(old, batch, self.context, self.gate_recur, self.context);
            for b in 0..batch {
                for j in 0..self.context {
                    let i = b * self.context + j;
                    let logit = gf[i] + ga[i] + gr[i] + self.packed[self.gate_bias + j];
                    let gate = 1.0 / (1.0 + (-logit).exp());
                    updated[i] = old[i] + gate * (updated[i] - old[i]);
                }
            }
        }
        let mut error = vec![f32::NAN; batch];
        for b in 0..batch {
            if mask[b] {
                let mut sum = 0.0f32;
                for q in 0..self.projection {
                    let delta = projected[b * self.projection + q]
                        - previous_projected[b * self.projection + q]
                        - predicted[b * self.projection + q];
                    sum += delta * delta;
                }
                error[b] = sum / self.projection as f32;
            }
        }
        Ok((
            Array2::from_shape_vec((batch, self.context), updated)
                .unwrap()
                .into_pyarray(py),
            Array2::from_shape_vec((batch, self.projection), projected)
                .unwrap()
                .into_pyarray(py),
            Array1::from_vec(error).into_pyarray(py),
        ))
    }
}
