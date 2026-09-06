use numpy::{
    ndarray::{Array2, Array3},
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

mod contextual_episodic;
mod developmental;
pub mod gam_law;
mod goal_memory;
pub mod personal_consequences;
pub mod personal_goals;
pub mod population_response;
mod predictive_sensory;
mod sequence_memory;

#[derive(Clone)]
pub(crate) struct Linear {
    pub(crate) out: usize,
    pub(crate) input: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}
#[derive(Clone)]
pub(crate) struct Gru {
    pub(crate) hidden: usize,
    pub(crate) input: usize,
    pub(crate) w_ih: Vec<f32>,
    pub(crate) w_hh: Vec<f32>,
    pub(crate) b_ih: Vec<f32>,
    pub(crate) b_hh: Vec<f32>,
}

#[cfg(target_os = "macos")]
fn dense_sgemm(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    out: usize,
    output: &mut [f32],
) {
    use std::ffi::c_int;

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            transpose_a: c_int,
            transpose_b: c_int,
            rows: c_int,
            out: c_int,
            cols: c_int,
            alpha: f32,
            input: *const f32,
            input_stride: c_int,
            weight: *const f32,
            weight_stride: c_int,
            beta: f32,
            output: *mut f32,
            output_stride: c_int,
        );
    }

    let rows = c_int::try_from(rows).expect("dense GEMM row count exceeds c_int");
    let cols = c_int::try_from(cols).expect("dense GEMM column count exceeds c_int");
    let out = c_int::try_from(out).expect("dense GEMM output count exceeds c_int");
    unsafe {
        // input is [rows, cols], weight is stored [out, cols], and output is
        // [rows, out], all contiguous row-major. Transposing weight therefore
        // computes input * weight^T without copying either operand.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows,
            out,
            cols,
            1.0,
            input.as_ptr(),
            cols,
            weight.as_ptr(),
            cols,
            0.0,
            output.as_mut_ptr(),
            out,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn dense_sgemm(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    out: usize,
    output: &mut [f32],
) {
    unsafe {
        matrixmultiply::sgemm(
            rows,
            cols,
            out,
            1.0,
            input.as_ptr(),
            cols as isize,
            1,
            weight.as_ptr(),
            1,
            cols as isize,
            0.0,
            output.as_mut_ptr(),
            out as isize,
            1,
        );
    }
}

fn gemm_bias_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    out: usize,
    weight: &[f32],
    bias: &[f32],
    output: &mut Vec<f32>,
) {
    assert_eq!(
        input.len(),
        rows.checked_mul(cols)
            .expect("dense GEMM input size overflow")
    );
    assert_eq!(
        weight.len(),
        out.checked_mul(cols)
            .expect("dense GEMM weight size overflow")
    );
    assert_eq!(bias.len(), out);
    let output_len = rows
        .checked_mul(out)
        .expect("dense GEMM output size overflow");
    output.resize(output_len, 0.0);
    if out == 0 {
        return;
    }
    if rows != 0 && cols != 0 {
        dense_sgemm(input, rows, cols, weight, out, output);
    } else {
        output.fill(0.0);
    }
    for row in output.chunks_exact_mut(out) {
        for (x, b) in row.iter_mut().zip(bias) {
            *x += *b;
        }
    }
}

pub(crate) fn gemm_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    layer: &Linear,
    output: &mut Vec<f32>,
) {
    assert_eq!(cols, layer.input);
    gemm_bias_into(
        input,
        rows,
        cols,
        layer.out,
        &layer.weight,
        &layer.bias,
        output,
    );
}
fn gemm_parts_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    out: usize,
    weight: &[f32],
    bias: &[f32],
    output: &mut Vec<f32>,
) {
    gemm_bias_into(input, rows, cols, out, weight, bias, output);
}
pub(crate) fn tanh_all(value: &mut [f32]) {
    for x in value {
        *x = x.tanh();
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Gru {
    pub(crate) fn step_into(
        &self,
        input: &[f32],
        state: &[f32],
        rows: usize,
        gx: &mut Vec<f32>,
        gh: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        gemm_parts_into(
            input,
            rows,
            self.input,
            3 * self.hidden,
            &self.w_ih,
            &self.b_ih,
            gx,
        );
        gemm_parts_into(
            state,
            rows,
            self.hidden,
            3 * self.hidden,
            &self.w_hh,
            &self.b_hh,
            gh,
        );
        out.resize(rows * self.hidden, 0.0);
        let h = self.hidden;
        for b in 0..rows {
            for j in 0..h {
                let r = sigmoid(gx[b * 3 * h + j] + gh[b * 3 * h + j]);
                let z = sigmoid(gx[b * 3 * h + h + j] + gh[b * 3 * h + h + j]);
                let n = (gx[b * 3 * h + 2 * h + j] + r * gh[b * 3 * h + 2 * h + j]).tanh();
                out[b * h + j] = (1.0 - z) * n + z * state[b * h + j];
            }
        }
    }
}

pub(crate) fn take(flat: &[f32], cursor: &mut usize, count: usize) -> PyResult<Vec<f32>> {
    if *cursor + count > flat.len() {
        return Err(PyValueError::new_err("packed weights are truncated"));
    }
    let v = flat[*cursor..*cursor + count].to_vec();
    *cursor += count;
    Ok(v)
}
pub(crate) fn linear(flat: &[f32], c: &mut usize, out: usize, input: usize) -> PyResult<Linear> {
    Ok(Linear {
        out,
        input,
        weight: take(flat, c, out * input)?,
        bias: take(flat, c, out)?,
    })
}
pub(crate) fn gru(flat: &[f32], c: &mut usize, h: usize, input: usize) -> PyResult<Gru> {
    Ok(Gru {
        hidden: h,
        input,
        w_ih: take(flat, c, 3 * h * input)?,
        w_hh: take(flat, c, 3 * h * h)?,
        b_ih: take(flat, c, 3 * h)?,
        b_hh: take(flat, c, 3 * h)?,
    })
}

#[pyclass]
struct PredictiveCohort {
    batch: usize,
    features: usize,
    physiology: usize,
    actions: usize,
    latent: usize,
    encoder: usize,
    max_horizon: f32,
    observation_encoder: Linear,
    action_encoder: Linear,
    observe_cell: Gru,
    transition_cell: Gru,
    feature_mean: Linear,
    feature_log_std: Linear,
    physiology_mean: Linear,
    physiology_log_std: Linear,
    input_mean: Vec<f32>,
    input_scale: Vec<f32>,
    delta_mean: Vec<f32>,
    delta_scale: Vec<f32>,
    state: Vec<f32>,
    previous_action: Vec<f32>,
    physiology_anchor: Vec<f32>,
    scratch_observation: Vec<f32>,
    scratch_encoded: Vec<f32>,
    scratch_combined: Vec<f32>,
    scratch_gx: Vec<f32>,
    scratch_gh: Vec<f32>,
    scratch_next: Vec<f32>,
    scratch_decode: Vec<f32>,
    imagine_state: Vec<f32>,
}

#[pymethods]
impl PredictiveCohort {
    #[new]
    fn new(
        batch: usize,
        features: usize,
        physiology: usize,
        actions: usize,
        latent: usize,
        encoder: usize,
        max_horizon: usize,
        packed: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        if batch == 0
            || features == 0
            || physiology == 0
            || actions == 0
            || latent == 0
            || encoder < 2
            || max_horizon == 0
        {
            return Err(PyValueError::new_err("invalid cognitive dimensions"));
        }
        let flat = packed.as_slice()?;
        let mut c = 0;
        let obs = features + physiology;
        let action_enc = encoder / 2;
        // Packing order is part of predictive_native.py's v1 contract.
        let input_mean = take(flat, &mut c, obs)?;
        let input_scale = take(flat, &mut c, obs)?;
        let delta_mean = take(flat, &mut c, max_horizon * physiology)?;
        let delta_scale = take(flat, &mut c, max_horizon * physiology)?;
        let observation_encoder = linear(flat, &mut c, encoder, obs)?;
        let action_encoder = linear(flat, &mut c, action_enc, actions)?;
        let observe_cell = gru(flat, &mut c, latent, encoder + actions)?;
        let transition_cell = gru(flat, &mut c, latent, action_enc)?;
        let feature_mean = linear(flat, &mut c, features, latent)?;
        let feature_log_std = linear(flat, &mut c, features, latent)?;
        let physiology_mean = linear(flat, &mut c, physiology, latent)?;
        let physiology_log_std = linear(flat, &mut c, physiology, latent)?;
        if c != flat.len() {
            return Err(PyValueError::new_err("packed weights have trailing values"));
        }
        Ok(Self {
            batch,
            features,
            physiology,
            actions,
            latent,
            encoder,
            max_horizon: max_horizon as f32,
            observation_encoder,
            action_encoder,
            observe_cell,
            transition_cell,
            feature_mean,
            feature_log_std,
            physiology_mean,
            physiology_log_std,
            input_mean,
            input_scale,
            delta_mean,
            delta_scale,
            state: vec![0.0; batch * latent],
            previous_action: vec![0.0; batch * actions],
            physiology_anchor: vec![0.0; batch * physiology],
            scratch_observation: vec![0.0; batch * obs],
            scratch_encoded: vec![0.0; batch * encoder],
            scratch_combined: vec![0.0; batch * (encoder + actions)],
            scratch_gx: vec![0.0; batch * 3 * latent],
            scratch_gh: vec![0.0; batch * 3 * latent],
            scratch_next: vec![0.0; batch * latent],
            scratch_decode: vec![0.0; batch * obs],
            imagine_state: vec![0.0; batch * latent],
        })
    }

    fn observe<'py>(
        &mut self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, f32>,
        physiology: PyReadonlyArray2<'_, f32>,
        previous_action: PyReadonlyArray2<'_, f32>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if features.shape() != [self.batch, self.features]
            || physiology.shape() != [self.batch, self.physiology]
            || previous_action.shape() != [self.batch, self.actions]
            || reset.shape() != [self.batch]
        {
            return Err(PyValueError::new_err("observe shapes differ"));
        }
        let f = features.as_slice()?;
        let p = physiology.as_slice()?;
        let a = previous_action.as_slice()?;
        let rst = reset.as_slice()?;
        if f.iter().chain(p).chain(a).any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("observe inputs must be finite"));
        }
        for b in 0..self.batch {
            if rst[b] {
                self.state[b * self.latent..(b + 1) * self.latent].fill(0.0);
            }
        }
        let obs_width = self.features + self.physiology;
        for b in 0..self.batch {
            let o = b * obs_width;
            for j in 0..self.features {
                self.scratch_observation[o + j] =
                    (f[b * self.features + j] - self.input_mean[j]) / self.input_scale[j];
            }
            for j in 0..self.physiology {
                self.scratch_observation[o + self.features + j] = (p[b * self.physiology + j]
                    - self.input_mean[self.features + j])
                    / self.input_scale[self.features + j];
            }
        }
        gemm_into(
            &self.scratch_observation,
            self.batch,
            obs_width,
            &self.observation_encoder,
            &mut self.scratch_encoded,
        );
        tanh_all(&mut self.scratch_encoded);
        let combined_width = self.encoder + self.actions;
        for b in 0..self.batch {
            let o = b * combined_width;
            self.scratch_combined[o..o + self.encoder]
                .copy_from_slice(&self.scratch_encoded[b * self.encoder..(b + 1) * self.encoder]);
            self.scratch_combined[o + self.encoder..o + combined_width]
                .copy_from_slice(&a[b * self.actions..(b + 1) * self.actions]);
        }
        self.observe_cell.step_into(
            &self.scratch_combined,
            &self.state,
            self.batch,
            &mut self.scratch_gx,
            &mut self.scratch_gh,
            &mut self.scratch_next,
        );
        std::mem::swap(&mut self.state, &mut self.scratch_next);
        self.previous_action.copy_from_slice(a);
        self.physiology_anchor.copy_from_slice(p);
        Ok(
            Array2::from_shape_vec((self.batch, self.latent), self.state.clone())
                .unwrap()
                .into_pyarray(py),
        )
    }

    fn imagine<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<(
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let shape = actions.shape();
        if shape.len() != 3 || shape[1] != self.batch || shape[2] != self.actions {
            return Err(PyValueError::new_err(
                "imagine actions must be [time,batch,actions]",
            ));
        }
        let a = actions.as_slice()?;
        if a.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("actions must be finite"));
        }
        let time = shape[0];
        if time == 0 || time > self.max_horizon as usize {
            return Err(PyValueError::new_err(
                "imagine horizon is outside trained range",
            ));
        }
        self.imagine_state.copy_from_slice(&self.state);
        let mut feature_means = Vec::with_capacity(time * self.batch * self.features);
        let mut feature_scales = Vec::with_capacity(time * self.batch * self.features);
        let mut physiology_means = Vec::with_capacity(time * self.batch * self.physiology);
        let mut physiology_scales = Vec::with_capacity(time * self.batch * self.physiology);
        let mut valid = Vec::with_capacity(time * self.batch);
        let mut support = Vec::with_capacity(time * self.batch);
        for t in 0..time {
            let slice = &a[t * self.batch * self.actions..(t + 1) * self.batch * self.actions];
            gemm_into(
                slice,
                self.batch,
                self.actions,
                &self.action_encoder,
                &mut self.scratch_encoded,
            );
            tanh_all(&mut self.scratch_encoded);
            self.transition_cell.step_into(
                &self.scratch_encoded,
                &self.imagine_state,
                self.batch,
                &mut self.scratch_gx,
                &mut self.scratch_gh,
                &mut self.scratch_next,
            );
            std::mem::swap(&mut self.imagine_state, &mut self.scratch_next);
            gemm_into(
                &self.imagine_state,
                self.batch,
                self.latent,
                &self.feature_mean,
                &mut self.scratch_decode,
            );
            for row in self.scratch_decode.chunks_exact(self.features) {
                for (j, x) in row.iter().enumerate() {
                    feature_means.push(self.input_mean[j] + self.input_scale[j] * x);
                }
            }
            gemm_into(
                &self.imagine_state,
                self.batch,
                self.latent,
                &self.feature_log_std,
                &mut self.scratch_decode,
            );
            for row in self.scratch_decode.chunks_exact(self.features) {
                for (j, x) in row.iter().enumerate() {
                    feature_scales
                        .push(self.input_scale[j] * (-0.5 + x.tanh()).clamp(-1.5, 0.5).exp());
                }
            }
            gemm_into(
                &self.imagine_state,
                self.batch,
                self.latent,
                &self.physiology_mean,
                &mut self.scratch_decode,
            );
            for (b, row) in self
                .scratch_decode
                .chunks_exact(self.physiology)
                .enumerate()
            {
                for (j, x) in row.iter().enumerate() {
                    physiology_means.push(
                        self.physiology_anchor[b * self.physiology + j]
                            + self.delta_mean[t * self.physiology + j]
                            + self.delta_scale[t * self.physiology + j] * x,
                    );
                }
            }
            gemm_into(
                &self.imagine_state,
                self.batch,
                self.latent,
                &self.physiology_log_std,
                &mut self.scratch_decode,
            );
            for row in self.scratch_decode.chunks_exact(self.physiology) {
                for (j, x) in row.iter().enumerate() {
                    physiology_scales.push(
                        self.delta_scale[t * self.physiology + j]
                            * (-0.5 + x.tanh()).clamp(-1.5, 0.5).exp(),
                    );
                }
            }
            let feature_start = t * self.batch * self.features;
            let physiology_start = t * self.batch * self.physiology;
            for b in 0..self.batch {
                valid.push(
                    feature_means[feature_start + b * self.features
                        ..feature_start + (b + 1) * self.features]
                        .iter()
                        .chain(
                            &physiology_means[physiology_start + b * self.physiology
                                ..physiology_start + (b + 1) * self.physiology],
                        )
                        .all(|x| x.is_finite()),
                );
            }
            let s = (-((t + 1) as f32) / self.max_horizon).exp();
            support.extend(std::iter::repeat_n(s, self.batch));
        }
        Ok((
            Array3::from_shape_vec((time, self.batch, self.features), feature_means)
                .unwrap()
                .into_pyarray(py),
            Array3::from_shape_vec((time, self.batch, self.features), feature_scales)
                .unwrap()
                .into_pyarray(py),
            Array3::from_shape_vec((time, self.batch, self.physiology), physiology_means)
                .unwrap()
                .into_pyarray(py),
            Array3::from_shape_vec((time, self.batch, self.physiology), physiology_scales)
                .unwrap()
                .into_pyarray(py),
            Array2::from_shape_vec((time, self.batch), valid)
                .unwrap()
                .into_pyarray(py),
            Array2::from_shape_vec((time, self.batch), support)
                .unwrap()
                .into_pyarray(py),
        ))
    }

    fn snapshot<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    ) {
        (
            Array2::from_shape_vec((self.batch, self.latent), self.state.clone())
                .unwrap()
                .into_pyarray(py),
            Array2::from_shape_vec((self.batch, self.actions), self.previous_action.clone())
                .unwrap()
                .into_pyarray(py),
            Array2::from_shape_vec(
                (self.batch, self.physiology),
                self.physiology_anchor.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )
    }
    fn restore(
        &mut self,
        state: PyReadonlyArray2<'_, f32>,
        previous_action: PyReadonlyArray2<'_, f32>,
        physiology: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        if state.shape() != [self.batch, self.latent]
            || previous_action.shape() != [self.batch, self.actions]
            || physiology.shape() != [self.batch, self.physiology]
        {
            return Err(PyValueError::new_err("snapshot shapes differ"));
        }
        let s = state.as_slice()?;
        let a = previous_action.as_slice()?;
        let p = physiology.as_slice()?;
        if s.iter().chain(a).chain(p).any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("snapshot must be finite"));
        }
        self.state.copy_from_slice(s);
        self.previous_action.copy_from_slice(a);
        self.physiology_anchor.copy_from_slice(p);
        Ok(())
    }
}

#[pymodule]
fn _cognitive_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PredictiveCohort>()?;
    module.add_class::<developmental::DevelopmentalResidentCohort>()?;
    module.add_class::<goal_memory::AchievedGoalMemoryCohort>()?;
    module.add_class::<predictive_sensory::PredictiveSensoryEnsemble>()?;
    Ok(())
}
