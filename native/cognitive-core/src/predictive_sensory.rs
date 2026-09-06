// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only ensemble prediction of one-tick rich sensory/body consequences.

use crate::{gemm_into, linear, tanh_all, Linear};
use numpy::{
    ndarray::{Array2, Array3, Array4},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub(crate) const INPUT: usize = 1426;
pub(crate) const OUTPUT: usize = 262;
const HIDDEN: usize = 256;
pub(crate) const MEMBERS: usize = 3;
#[derive(Clone)]
struct Member {
    first: Linear,
    second: Linear,
    output: Linear,
}
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PredictiveSensoryEnsemble {
    input_mean: Vec<f32>,
    input_scale: Vec<f32>,
    target_mean: Vec<f32>,
    target_scale: Vec<f32>,
    residual_scale: Vec<f32>,
    members: Vec<Member>,
    normalized: Vec<f32>,
    middle0: Vec<f32>,
    middle1: Vec<f32>,
    raw: Vec<f32>,
}

impl PredictiveSensoryEnsemble {
    pub(crate) fn from_flat(flat: &[f32]) -> PyResult<Self> {
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("predictor parameters must be finite"));
        }
        let mut c = 0;
        let take_vec = |c: &mut usize, n: usize| -> PyResult<Vec<f32>> {
            if *c + n > flat.len() {
                return Err(PyValueError::new_err("predictor pack truncated"));
            }
            let x = flat[*c..*c + n].to_vec();
            *c += n;
            Ok(x)
        };
        let input_mean = take_vec(&mut c, INPUT)?;
        let input_scale = take_vec(&mut c, INPUT)?;
        let target_mean = take_vec(&mut c, OUTPUT)?;
        let target_scale = take_vec(&mut c, OUTPUT)?;
        let residual_scale = take_vec(&mut c, OUTPUT)?;
        if input_scale.iter().any(|x| *x < 0.02)
            || target_scale[..256].iter().any(|x| *x < 0.001)
            || target_scale[256..].iter().any(|x| *x < 0.0001)
            || residual_scale.iter().any(|x| *x < 1e-8)
        {
            return Err(PyValueError::new_err(
                "predictor normalization floors differ",
            ));
        }
        let mut members = Vec::with_capacity(MEMBERS);
        for _ in 0..MEMBERS {
            members.push(Member {
                first: linear(flat, &mut c, HIDDEN, INPUT)?,
                second: linear(flat, &mut c, HIDDEN, HIDDEN)?,
                output: linear(flat, &mut c, OUTPUT, HIDDEN)?,
            });
        }
        if c != flat.len() {
            return Err(PyValueError::new_err("predictor pack has trailing values"));
        }
        Ok(Self {
            input_mean,
            input_scale,
            target_mean,
            target_scale,
            residual_scale,
            members,
            normalized: Vec::new(),
            middle0: Vec::new(),
            middle1: Vec::new(),
            raw: Vec::new(),
        })
    }
    pub(crate) fn forecast_into(
        &mut self,
        input: &[f32],
        rows: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
        self.normalized.resize(rows * INPUT, 0.0);
        let mut clipped = vec![false; rows];
        for row in 0..rows {
            for j in 0..INPUT {
                let z = (input[row * INPUT + j] - self.input_mean[j]) / self.input_scale[j];
                clipped[row] |= z.abs() > 8.0;
                self.normalized[row * INPUT + j] = z.clamp(-8.0, 8.0);
            }
        }
        let mut all = vec![0.0; rows * MEMBERS * OUTPUT];
        for (m, member) in self.members.iter().enumerate() {
            gemm_into(
                &self.normalized,
                rows,
                INPUT,
                &member.first,
                &mut self.middle0,
            );
            tanh_all(&mut self.middle0);
            gemm_into(
                &self.middle0,
                rows,
                HIDDEN,
                &member.second,
                &mut self.middle1,
            );
            tanh_all(&mut self.middle1);
            gemm_into(&self.middle1, rows, HIDDEN, &member.output, &mut self.raw);
            for row in 0..rows {
                for j in 0..OUTPUT {
                    all[(row * MEMBERS + m) * OUTPUT + j] =
                        self.raw[row * OUTPUT + j] * self.target_scale[j] + self.target_mean[j];
                }
            }
        }
        let mut mean = vec![0.0; rows * OUTPUT];
        let mut disagreement = vec![0.0; rows * OUTPUT];
        for row in 0..rows {
            for j in 0..OUTPUT {
                let mu = (0..MEMBERS)
                    .map(|m| all[(row * MEMBERS + m) * OUTPUT + j])
                    .sum::<f32>()
                    / MEMBERS as f32;
                mean[row * OUTPUT + j] = mu;
                disagreement[row * OUTPUT + j] = ((0..MEMBERS)
                    .map(|m| (all[(row * MEMBERS + m) * OUTPUT + j] - mu).powi(2))
                    .sum::<f32>()
                    / MEMBERS as f32)
                    .sqrt();
            }
        }
        (all, mean, disagreement, clipped)
    }
}
#[pymethods]
impl PredictiveSensoryEnsemble {
    #[new]
    fn new(packed: PyReadonlyArray1<'_, f32>) -> PyResult<Self> {
        Self::from_flat(packed.as_slice()?)
    }
    fn forecast<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = input.shape();
        if shape.len() != 3 || shape[2] != INPUT || shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("predictor input must be [B,K,1426]"));
        }
        let x = input.as_slice()?;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("predictor input must be finite"));
        }
        let b = shape[0];
        let k = shape[1];
        let (members, mean, disagreement, clipped) = py.detach(|| self.forecast_into(x, b * k));
        let out = PyDict::new(py);
        out.set_item(
            "member_delta",
            Array4::from_shape_vec((b, k, MEMBERS, OUTPUT), members)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "mean_delta",
            Array3::from_shape_vec((b, k, OUTPUT), mean)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "disagreement",
            Array3::from_shape_vec((b, k, OUTPUT), disagreement)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "input_clipped",
            Array2::from_shape_vec((b, k), clipped)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "residual_scale",
            numpy::ndarray::Array1::from_vec(self.residual_scale.clone()).into_pyarray(py),
        )?;
        Ok(out)
    }
}
