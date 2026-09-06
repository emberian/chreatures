// SPDX-License-Identifier: AGPL-3.0-or-later
//! Recurrent, action-conditioned H1..H8 sensory and full-body consequences.
use crate::{gemm_into, gru, linear, tanh_all, Gru, Linear};
use numpy::{
    ndarray::{Array1, ArrayD, IxDyn},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
pub(crate) const CONTEXT: usize = 1560;
pub(crate) const ACTION: usize = 12;
pub(crate) const OUTPUT: usize = 268;
pub(crate) const PHYSIOLOGY: usize = 12;
pub(crate) const MEMBERS: usize = 3;
pub(crate) const MAX_HORIZON: usize = 8;
const LATENT: usize = 256;
#[derive(Clone)]
struct Member {
    context: Linear,
    transition: Gru,
    output: Linear,
}
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PredictiveSensoryEnsemble {
    context_mean: Vec<f32>,
    context_scale: Vec<f32>,
    action_mean: Vec<f32>,
    action_scale: Vec<f32>,
    target_mean: Vec<f32>,
    target_scale: Vec<f32>,
    members: Vec<Member>,
    normalized_context: Vec<f32>,
    normalized_action: Vec<f32>,
    initial: Vec<f32>,
    state: Vec<f32>,
    gx: Vec<f32>,
    gh: Vec<f32>,
    next: Vec<f32>,
    raw: Vec<f32>,
    step_action: Vec<f32>,
}
impl PredictiveSensoryEnsemble {
    pub(crate) fn from_flat(flat: &[f32]) -> PyResult<Self> {
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("predictor parameters must be finite"));
        }
        let mut c = 0;
        let take = |c: &mut usize, n: usize| -> PyResult<Vec<f32>> {
            if *c + n > flat.len() {
                return Err(PyValueError::new_err("predictor pack truncated"));
            }
            let v = flat[*c..*c + n].to_vec();
            *c += n;
            Ok(v)
        };
        let context_mean = take(&mut c, CONTEXT)?;
        let context_scale = take(&mut c, CONTEXT)?;
        let action_mean = take(&mut c, ACTION)?;
        let action_scale = take(&mut c, ACTION)?;
        let target_mean = take(&mut c, OUTPUT)?;
        let target_scale = take(&mut c, OUTPUT)?;
        if context_scale.iter().any(|x| *x < 0.02)
            || action_scale.iter().any(|x| *x < 0.02)
            || target_scale[..256].iter().any(|x| *x < 0.001)
            || target_scale[256..].iter().any(|x| *x < 0.0001)
        {
            return Err(PyValueError::new_err(
                "predictor normalization floors differ",
            ));
        }
        let mut members = Vec::with_capacity(MEMBERS);
        for _ in 0..MEMBERS {
            members.push(Member {
                context: linear(flat, &mut c, LATENT, CONTEXT)?,
                transition: gru(flat, &mut c, LATENT, ACTION)?,
                output: linear(flat, &mut c, OUTPUT, LATENT)?,
            });
        }
        if c != flat.len() {
            return Err(PyValueError::new_err("predictor pack has trailing values"));
        }
        Ok(Self {
            context_mean,
            context_scale,
            action_mean,
            action_scale,
            target_mean,
            target_scale,
            members,
            normalized_context: vec![],
            normalized_action: vec![],
            initial: vec![],
            state: vec![],
            gx: vec![],
            gh: vec![],
            next: vec![],
            raw: vec![],
            step_action: vec![],
        })
    }
    /// Allocation-reusing pure Rust hot path. Layouts: context[B,C], actions[B,K,H,A].
    /// Outputs are member/mean/spread per-tick deltas and cumulatively anchored physiology.
    pub(crate) fn forecast_sequences_into(
        &mut self,
        context: &[f32],
        actions: &[f32],
        phys_anchor: &[f32],
        b: usize,
        k: usize,
        h: usize,
        member_delta: &mut Vec<f32>,
        mean_delta: &mut Vec<f32>,
        disagreement: &mut Vec<f32>,
        absolute_code: &mut Vec<f32>,
        absolute_physiology: &mut Vec<f32>,
        valid: &mut Vec<bool>,
    ) -> PyResult<()> {
        if b == 0
            || k == 0
            || h == 0
            || h > MAX_HORIZON
            || context.len() != b * CONTEXT
            || actions.len() != b * k * h * ACTION
            || phys_anchor.len() != b * PHYSIOLOGY
            || context
                .iter()
                .chain(actions)
                .chain(phys_anchor)
                .any(|x| !x.is_finite())
        {
            return Err(PyValueError::new_err(
                "predictor input shape or finiteness differs",
            ));
        }
        let rows = b * k;
        self.normalized_context.resize(rows * CONTEXT, 0.);
        self.normalized_action.resize(rows * h * ACTION, 0.);
        valid.resize(rows * h, true);
        valid.fill(true);
        for bi in 0..b {
            for ki in 0..k {
                let r = bi * k + ki;
                for j in 0..CONTEXT {
                    let z =
                        (context[bi * CONTEXT + j] - self.context_mean[j]) / self.context_scale[j];
                    if z.abs() > 8. {
                        for q in 0..h {
                            valid[r * h + q] = false;
                        }
                    }
                    self.normalized_context[r * CONTEXT + j] = z.clamp(-8., 8.);
                }
                for q in 0..h {
                    for j in 0..ACTION {
                        let z = (actions[((bi * k + ki) * h + q) * ACTION + j]
                            - self.action_mean[j])
                            / self.action_scale[j];
                        if z.abs() > 8. {
                            for future in q..h {
                                valid[r * h + future] = false;
                            }
                        }
                        self.normalized_action[(r * h + q) * ACTION + j] = z.clamp(-8., 8.);
                    }
                }
            }
        }
        member_delta.resize(rows * MEMBERS * h * OUTPUT, 0.);
        mean_delta.resize(rows * h * OUTPUT, 0.);
        disagreement.resize(rows * h * OUTPUT, 0.);
        absolute_code.resize(rows * MEMBERS * h * 256, 0.);
        absolute_physiology.resize(rows * MEMBERS * h * PHYSIOLOGY, 0.);
        for m in 0..MEMBERS {
            let member = &self.members[m];
            gemm_into(
                &self.normalized_context,
                rows,
                CONTEXT,
                &member.context,
                &mut self.initial,
            );
            tanh_all(&mut self.initial);
            self.state.clone_from(&self.initial);
            for q in 0..h {
                self.step_action.resize(rows * ACTION, 0.);
                for r in 0..rows {
                    self.step_action[r * ACTION..(r + 1) * ACTION].copy_from_slice(
                        &self.normalized_action[(r * h + q) * ACTION..(r * h + q + 1) * ACTION],
                    );
                }
                member.transition.step_into(
                    &self.step_action,
                    &self.state,
                    rows,
                    &mut self.gx,
                    &mut self.gh,
                    &mut self.next,
                );
                std::mem::swap(&mut self.state, &mut self.next);
                gemm_into(&self.state, rows, LATENT, &member.output, &mut self.raw);
                for r in 0..rows {
                    for j in 0..OUTPUT {
                        let value =
                            self.raw[r * OUTPUT + j] * self.target_scale[j] + self.target_mean[j];
                        member_delta[((r * MEMBERS + m) * h + q) * OUTPUT + j] = value;
                    }
                    for j in 0..256 {
                        let prior = if q == 0 {
                            context[(r / k) * CONTEXT + 3 * 256 + j]
                        } else {
                            absolute_code[((r * MEMBERS + m) * h + q - 1) * 256 + j]
                        };
                        let value = prior + member_delta[((r * MEMBERS + m) * h + q) * OUTPUT + j];
                        absolute_code[((r * MEMBERS + m) * h + q) * 256 + j] = value;
                        if !value.is_finite() {
                            for future in q..h {
                                valid[r * h + future] = false;
                            }
                        }
                    }
                    for j in 0..PHYSIOLOGY {
                        let prior = if q == 0 {
                            phys_anchor[(r / k) * PHYSIOLOGY + j]
                        } else {
                            absolute_physiology[((r * MEMBERS + m) * h + q - 1) * PHYSIOLOGY + j]
                        };
                        let output_index = ((r * MEMBERS + m) * h + q) * OUTPUT + 256 + j;
                        let proposal = member_delta[output_index];
                        let lower = if j == 3 || j == 4 { -1.0 } else { 0.0 };
                        let absolute = proposal.abs();
                        let root = (proposal * proposal + 1e-8).sqrt();
                        let small = 1e-8 / (2.0 * (root + absolute));
                        let (positive, negative) = if proposal >= 0.0 {
                            (absolute + small, small)
                        } else {
                            (small, absolute + small)
                        };
                        let upward = 1.0 - prior;
                        let downward = prior - lower;
                        let delta = upward * (positive / upward.max(1e-4)).tanh()
                            - downward * (negative / downward.max(1e-4)).tanh();
                        member_delta[output_index] = delta;
                        let value = prior + delta;
                        absolute_physiology[((r * MEMBERS + m) * h + q) * PHYSIOLOGY + j] = value;
                        if !value.is_finite() || value < lower || value > 1.0 {
                            for future in q..h {
                                valid[r * h + future] = false;
                            }
                        }
                    }
                }
            }
        }
        for r in 0..rows {
            for q in 0..h {
                // A later step depends on the complete proposed prefix. An
                // out-of-domain prefix cannot regain validity by returning
                // to the physiological range at a later horizon.
                if q > 0 && !valid[r * h + q - 1] {
                    valid[r * h + q] = false;
                }
                for j in 0..OUTPUT {
                    let mu = (0..MEMBERS)
                        .map(|m| member_delta[((r * MEMBERS + m) * h + q) * OUTPUT + j])
                        .sum::<f32>()
                        / MEMBERS as f32;
                    mean_delta[(r * h + q) * OUTPUT + j] = mu;
                    disagreement[(r * h + q) * OUTPUT + j] = ((0..MEMBERS)
                        .map(|m| {
                            (member_delta[((r * MEMBERS + m) * h + q) * OUTPUT + j] - mu).powi(2)
                        })
                        .sum::<f32>()
                        / MEMBERS as f32)
                        .sqrt();
                }
            }
        }
        Ok(())
    }
}
#[pymethods]
impl PredictiveSensoryEnsemble {
    #[new]
    fn new(packed: PyReadonlyArray1<'_, f32>) -> PyResult<Self> {
        Self::from_flat(packed.as_slice()?)
    }
    fn forecast_sequences<'py>(
        &mut self,
        py: Python<'py>,
        context: PyReadonlyArray2<'_, f32>,
        actions: PyReadonlyArray4<'_, f32>,
        physiology_anchor: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cs = context.shape();
        let shape = actions.shape();
        let ps = physiology_anchor.shape();
        if cs.len() != 2
            || shape.len() != 4
            || ps.len() != 2
            || cs[0] != shape[0]
            || ps[0] != shape[0]
            || cs[1] != CONTEXT
            || shape[3] != ACTION
            || ps[1] != PHYSIOLOGY
        {
            return Err(PyValueError::new_err(
                "expected context[B,1560], actions[B,K,H,12], physiology[B,12]",
            ));
        }
        let (b, k, h) = (shape[0], shape[1], shape[2]);
        let (mut members, mut mean, mut spread, mut code, mut physical, mut valid) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        self.forecast_sequences_into(
            context.as_slice()?,
            actions.as_slice()?,
            physiology_anchor.as_slice()?,
            b,
            k,
            h,
            &mut members,
            &mut mean,
            &mut spread,
            &mut code,
            &mut physical,
            &mut valid,
        )?;
        let out = PyDict::new(py);
        out.set_item(
            "member_delta",
            ArrayD::from_shape_vec(IxDyn(&[b, k, MEMBERS, h, OUTPUT]), members)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "mean_delta",
            ArrayD::from_shape_vec(IxDyn(&[b, k, h, OUTPUT]), mean)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "disagreement",
            ArrayD::from_shape_vec(IxDyn(&[b, k, h, OUTPUT]), spread)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "absolute_code",
            ArrayD::from_shape_vec(IxDyn(&[b, k, MEMBERS, h, 256]), code)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "absolute_physiology",
            ArrayD::from_shape_vec(IxDyn(&[b, k, MEMBERS, h, PHYSIOLOGY]), physical)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "valid",
            ArrayD::from_shape_vec(IxDyn(&[b, k, h]), valid)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "horizon_seconds",
            Array1::from_iter((1..=h).map(|x| x as f32 * 0.05)).into_pyarray(py),
        )?;
        Ok(out)
    }
}
