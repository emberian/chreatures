use crate::gam_law::{
    desired_goal_consequences, transition_targets, CandidateAction, DecisionContext, LawBank,
};
use crate::goal_memory::{
    random_u64, splitmix64, unit_f64, AchievedGoalMemoryCohort, SelectionArrays,
};
use crate::personal_consequences::{ConsequenceConfig, ConsequenceTarget, PersonalConsequences};
use crate::{gemm_into, gru, linear, take, tanh_all, Gru, Linear};
use numpy::{
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

const OBS: usize = 4453;
const RICH: usize = 4096;
const BODY: usize = 357;
const PREVIOUS: usize = 9;
const ACTIONS: usize = 8;
const WINDOW: usize = 4;
const GOAL: usize = 64;
const HIDDEN: usize = 128;
const POLICY: usize = 256;
const NEURAL: usize = 384;
const PHYSIOLOGY: usize = 6;
const RESERVOIR: usize = 128;
const SIGNED: [usize; 4] = [0, 1, 2, 7];
const POSITIVE: [usize; 4] = [3, 4, 5, 6];
const FORMAT: &str = "chreatures-developmental-resident-native-rich-v1";
const CANDIDATES: usize = 4;
const TILT: f64 = 0.5;

struct Core {
    mean: Vec<f32>,
    scale: Vec<f32>,
    peripheral_first: Linear,
    peripheral_second: Linear,
    foveal_first: Linear,
    foveal_second: Linear,
    peripheral_projection: Linear,
    foveal_projection: Linear,
    body: Linear,
    goal0: Linear,
    goal2: Linear,
    observation: Linear,
    history: Gru,
    policy: Linear,
    signed: Linear,
    active: Linear,
    positive: Linear,
    manager0: Linear,
    manager2: Linear,
    query_gain: f32,
}

#[pyclass]
pub(crate) struct DevelopmentalResidentCohort {
    batch: usize,
    conditioned: bool,
    sample: bool,
    core: Core,
    memory: AchievedGoalMemoryCohort,
    state: Vec<f32>,
    previous_action: Vec<f32>,
    action_rng: Vec<u64>,
    normalized: Vec<f32>,
    peripheral_input: Vec<f32>,
    foveal_input: Vec<f32>,
    body_input: Vec<f32>,
    conv_columns: Vec<f32>,
    conv_spatial: Vec<f32>,
    conv_first: Vec<f32>,
    conv_second: Vec<f32>,
    peripheral_flat: Vec<f32>,
    foveal_flat: Vec<f32>,
    peripheral_code: Vec<f32>,
    foveal_code: Vec<f32>,
    body_code: Vec<f32>,
    frame_code: Vec<f32>,
    observation_input: Vec<f32>,
    encoded: Vec<f32>,
    gx: Vec<f32>,
    gh: Vec<f32>,
    next_state: Vec<f32>,
    goal_observations: Vec<f32>,
    goal_flat: Vec<f32>,
    goal_middle: Vec<f32>,
    manager_input: Vec<f32>,
    manager_hidden: Vec<f32>,
    query: Vec<f32>,
    logits: Vec<f32>,
    policy_input: Vec<f32>,
    policy_hidden: Vec<f32>,
    signed_logits: Vec<f32>,
    active_logits: Vec<f32>,
    positive_logits: Vec<f32>,
    law: LawBank,
    consequences: PersonalConsequences,
    pending_action: Vec<f32>,
    pending_physiology: Vec<f32>,
    pending_tick: Vec<Option<u64>>,
    candidate_scores: Vec<f32>,
    candidate_ood: Vec<bool>,
    selected_candidate: Vec<i32>,
    selected_correction: Vec<f32>,
    personal_updates: Vec<u64>,
}

fn categorical(logits: &[f32], state: &mut [u64]) -> usize {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits.iter().map(|x| ((*x - maximum) as f64).exp()).sum();
    let threshold = unit_f64(state) * total;
    let mut cumulative = 0.0;
    for (index, value) in logits.iter().enumerate() {
        cumulative += ((*value - maximum) as f64).exp();
        if threshold < cumulative {
            return index;
        }
    }
    logits.len() - 1
}

impl DevelopmentalResidentCohort {
    #[allow(clippy::too_many_arguments)]
    fn conv_stage(
        input: &[f32],
        rows: usize,
        channels: usize,
        height: usize,
        width: usize,
        layer: &Linear,
        stride_y: usize,
        stride_x: usize,
        columns: &mut Vec<f32>,
        spatial: &mut Vec<f32>,
        output: &mut Vec<f32>,
    ) -> (usize, usize) {
        let oh = (height - 1) / stride_y + 1;
        let ow = (width - 1) / stride_x + 1;
        let kernel = channels * 9;
        columns.resize(rows * oh * ow * kernel, 0.0);
        for row in 0..rows {
            for oy in 0..oh {
                for ox in 0..ow {
                    let patch = ((row * oh + oy) * ow + ox) * kernel;
                    for channel in 0..channels {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = (oy * stride_y + ky).saturating_sub(1).min(height - 1);
                                let ix = (ox * stride_x + kx).saturating_sub(1).min(width - 1);
                                columns[patch + channel * 9 + ky * 3 + kx] =
                                    input[((row * channels + channel) * height + iy) * width + ix];
                            }
                        }
                    }
                }
            }
        }
        gemm_into(columns, rows * oh * ow, kernel, layer, spatial);
        tanh_all(spatial);
        output.resize(rows * layer.out * oh * ow, 0.0);
        for row in 0..rows {
            for channel in 0..layer.out {
                for y in 0..oh {
                    for x in 0..ow {
                        output[((row * layer.out + channel) * oh + y) * ow + x] =
                            spatial[((row * oh + y) * ow + x) * layer.out + channel];
                    }
                }
            }
        }
        (oh, ow)
    }

    fn encode_frames(&mut self, normalized: &[f32], rows: usize) {
        self.peripheral_input.resize(rows * 4 * 8 * 32, 0.0);
        self.foveal_input.resize(rows * 4 * 24 * 32, 0.0);
        self.body_input.resize(rows * BODY, 0.0);
        for row in 0..rows {
            for ray in 0..1024 {
                let (target, local, height) = if ray < 256 {
                    (&mut self.peripheral_input, ray, 8)
                } else {
                    (&mut self.foveal_input, ray - 256, 24)
                };
                let y = local / 32;
                let x = local % 32;
                for channel in 0..4 {
                    target[((row * 4 + channel) * height + y) * 32 + x] =
                        normalized[row * OBS + ray * 4 + channel];
                }
            }
        }
        let (ph, pw) = Self::conv_stage(
            &self.peripheral_input,
            rows,
            4,
            8,
            32,
            &self.core.peripheral_first,
            1,
            2,
            &mut self.conv_columns,
            &mut self.conv_spatial,
            &mut self.conv_first,
        );
        let (ph2, pw2) = Self::conv_stage(
            &self.conv_first,
            rows,
            16,
            ph,
            pw,
            &self.core.peripheral_second,
            2,
            2,
            &mut self.conv_columns,
            &mut self.conv_spatial,
            &mut self.conv_second,
        );
        self.peripheral_flat.resize(rows * 24 * ph2 * pw2, 0.0);
        self.peripheral_flat.copy_from_slice(&self.conv_second);
        gemm_into(
            &self.peripheral_flat,
            rows,
            24 * ph2 * pw2,
            &self.core.peripheral_projection,
            &mut self.peripheral_code,
        );
        tanh_all(&mut self.peripheral_code);
        let (fh, fw) = Self::conv_stage(
            &self.foveal_input,
            rows,
            4,
            24,
            32,
            &self.core.foveal_first,
            1,
            2,
            &mut self.conv_columns,
            &mut self.conv_spatial,
            &mut self.conv_first,
        );
        let (fh2, fw2) = Self::conv_stage(
            &self.conv_first,
            rows,
            16,
            fh,
            fw,
            &self.core.foveal_second,
            2,
            2,
            &mut self.conv_columns,
            &mut self.conv_spatial,
            &mut self.conv_second,
        );
        self.foveal_flat.resize(rows * 24 * fh2 * fw2, 0.0);
        self.foveal_flat.copy_from_slice(&self.conv_second);
        gemm_into(
            &self.foveal_flat,
            rows,
            24 * fh2 * fw2,
            &self.core.foveal_projection,
            &mut self.foveal_code,
        );
        tanh_all(&mut self.foveal_code);
        for row in 0..rows {
            self.body_input[row * BODY..(row + 1) * BODY]
                .copy_from_slice(&normalized[row * OBS + RICH..(row + 1) * OBS]);
        }
        gemm_into(
            &self.body_input,
            rows,
            BODY,
            &self.core.body,
            &mut self.body_code,
        );
        tanh_all(&mut self.body_code);
        self.frame_code.resize(rows * 256, 0.0);
        for row in 0..rows {
            self.frame_code[row * 256..row * 256 + 64]
                .copy_from_slice(&self.peripheral_code[row * 64..(row + 1) * 64]);
            self.frame_code[row * 256 + 64..row * 256 + 128]
                .copy_from_slice(&self.foveal_code[row * 64..(row + 1) * 64]);
            self.frame_code[row * 256 + 128..(row + 1) * 256]
                .copy_from_slice(&self.body_code[row * 128..(row + 1) * 128]);
        }
    }

    fn encode_windows(&mut self, windows: &[f32], valid: &[bool]) -> Vec<f32> {
        if !valid.iter().any(|include| *include) {
            return vec![0.0; self.batch * GOAL];
        }
        self.goal_observations
            .resize(self.batch * WINDOW * OBS, 0.0);
        self.goal_observations.fill(0.0);
        for row in 0..self.batch {
            if !valid[row] {
                continue;
            }
            for frame in 0..WINDOW {
                let source =
                    &windows[(row * WINDOW + frame) * OBS..(row * WINDOW + frame + 1) * OBS];
                let target = &mut self.goal_observations
                    [(row * WINDOW + frame) * OBS..(row * WINDOW + frame + 1) * OBS];
                for j in 0..OBS {
                    target[j] =
                        ((source[j] - self.core.mean[j]) / self.core.scale[j]).clamp(-8.0, 8.0);
                }
            }
        }
        let normalized = std::mem::take(&mut self.goal_observations);
        self.encode_frames(&normalized, self.batch * WINDOW);
        self.goal_observations = normalized;
        self.goal_flat.resize(self.batch * WINDOW * 256, 0.0);
        self.goal_flat.copy_from_slice(&self.frame_code);
        gemm_into(
            &self.goal_flat,
            self.batch,
            WINDOW * 256,
            &self.core.goal0,
            &mut self.goal_middle,
        );
        tanh_all(&mut self.goal_middle);
        let mut keys = Vec::new();
        gemm_into(
            &self.goal_middle,
            self.batch,
            POLICY,
            &self.core.goal2,
            &mut keys,
        );
        for (row, include) in valid.iter().enumerate() {
            if !include {
                keys[row * GOAL..(row + 1) * GOAL].fill(0.0);
            }
        }
        keys
    }

    fn observe(&mut self, observations: &[f32], previous: &[f32], reset: &[bool]) {
        for row in 0..self.batch {
            if reset[row] {
                self.state[row * HIDDEN..(row + 1) * HIDDEN].fill(0.0);
            }
            for j in 0..OBS {
                self.normalized[row * OBS + j] =
                    ((observations[row * OBS + j] - self.core.mean[j]) / self.core.scale[j])
                        .clamp(-8.0, 8.0);
            }
            self.previous_action[row * ACTIONS..(row + 1) * ACTIONS]
                .copy_from_slice(&previous[row * PREVIOUS..row * PREVIOUS + ACTIONS]);
        }
        let normalized = std::mem::take(&mut self.normalized);
        self.encode_frames(&normalized, self.batch);
        self.normalized = normalized;
        for row in 0..self.batch {
            let offset = row * (256 + PREVIOUS);
            self.observation_input[offset..offset + 256]
                .copy_from_slice(&self.frame_code[row * 256..(row + 1) * 256]);
            self.observation_input[offset + 256..offset + 256 + PREVIOUS]
                .copy_from_slice(&previous[row * PREVIOUS..(row + 1) * PREVIOUS]);
        }
        gemm_into(
            &self.observation_input,
            self.batch,
            256 + PREVIOUS,
            &self.core.observation,
            &mut self.encoded,
        );
        tanh_all(&mut self.encoded);
        self.core.history.step_into(
            &self.encoded,
            &self.state,
            self.batch,
            &mut self.gx,
            &mut self.gh,
            &mut self.next_state,
        );
        std::mem::swap(&mut self.state, &mut self.next_state);
    }

    fn manager_selection(
        &mut self,
        neural: &[f32],
        physiology: &[f32],
        ticks: &[u64],
    ) -> PyResult<SelectionArrays> {
        for row in 0..self.batch {
            let offset = row * (HIDDEN + NEURAL + PHYSIOLOGY);
            self.manager_input[offset..offset + HIDDEN]
                .copy_from_slice(&self.state[row * HIDDEN..(row + 1) * HIDDEN]);
            self.manager_input[offset + HIDDEN..offset + HIDDEN + NEURAL]
                .copy_from_slice(&neural[row * NEURAL..(row + 1) * NEURAL]);
            self.manager_input
                [offset + HIDDEN + NEURAL..(row + 1) * (HIDDEN + NEURAL + PHYSIOLOGY)]
                .copy_from_slice(&physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]);
        }
        gemm_into(
            &self.manager_input,
            self.batch,
            HIDDEN + NEURAL + PHYSIOLOGY,
            &self.core.manager0,
            &mut self.manager_hidden,
        );
        tanh_all(&mut self.manager_hidden);
        gemm_into(
            &self.manager_hidden,
            self.batch,
            HIDDEN,
            &self.core.manager2,
            &mut self.query,
        );
        self.logits.fill(0.0);
        let (keys, counts) = self.memory.key_rows();
        for row in 0..self.batch {
            for slot in 0..counts[row] as usize {
                let mut dot = 0.0f32;
                for j in 0..GOAL {
                    dot += self.query[row * GOAL + j] * keys[(row * RESERVOIR + slot) * GOAL + j];
                }
                self.logits[row * RESERVOIR + slot] = dot * self.core.query_gain / 8.0;
            }
        }
        let boundary = ticks.iter().all(|tick| tick % 10 == 0);
        if boundary {
            self.memory.choose_inner(&self.logits, 1.0, ticks)
        } else {
            Ok(self.memory.current_selection(ticks))
        }
    }

    fn policy_actions(&mut self, goal: &[f32], remaining: &[u64]) -> Vec<f32> {
        let width = HIDDEN + GOAL + 1 + ACTIONS;
        for row in 0..self.batch {
            let offset = row * width;
            self.policy_input[offset..offset + HIDDEN]
                .copy_from_slice(&self.state[row * HIDDEN..(row + 1) * HIDDEN]);
            if self.conditioned {
                self.policy_input[offset + HIDDEN..offset + HIDDEN + GOAL]
                    .copy_from_slice(&goal[row * GOAL..(row + 1) * GOAL]);
            } else {
                self.policy_input[offset + HIDDEN..offset + HIDDEN + GOAL].fill(0.0);
            }
            self.policy_input[offset + HIDDEN + GOAL] = if remaining[row] == 0 {
                0.0
            } else {
                ((remaining[row].max(1) as f32) + 1.0).ln() / 41.0f32.ln()
            };
            self.policy_input[offset + HIDDEN + GOAL + 1..(row + 1) * width]
                .copy_from_slice(&self.previous_action[row * ACTIONS..(row + 1) * ACTIONS]);
        }
        gemm_into(
            &self.policy_input,
            self.batch,
            width,
            &self.core.policy,
            &mut self.policy_hidden,
        );
        tanh_all(&mut self.policy_hidden);
        gemm_into(
            &self.policy_hidden,
            self.batch,
            POLICY,
            &self.core.signed,
            &mut self.signed_logits,
        );
        gemm_into(
            &self.policy_hidden,
            self.batch,
            POLICY,
            &self.core.active,
            &mut self.active_logits,
        );
        gemm_into(
            &self.policy_hidden,
            self.batch,
            POLICY,
            &self.core.positive,
            &mut self.positive_logits,
        );
        let mut result = vec![0.0; self.batch * CANDIDATES * ACTIONS];
        for row in 0..self.batch {
            let rng = &mut self.action_rng[row * 4..(row + 1) * 4];
            for candidate in 0..CANDIDATES {
                let base = (row * CANDIDATES + candidate) * ACTIONS;
                for (head, axis) in SIGNED.iter().enumerate() {
                    let values =
                        &self.signed_logits[(row * 4 + head) * 65..(row * 4 + head + 1) * 65];
                    let index = if self.sample {
                        categorical(values, rng)
                    } else {
                        values
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0
                    };
                    result[base + axis] = index as f32 / 32.0 - 1.0;
                }
                for (head, axis) in POSITIVE.iter().enumerate() {
                    let active = if self.sample {
                        unit_f64(rng)
                            < (1.0 / (1.0 + (-self.active_logits[row * 4 + head]).exp())) as f64
                    } else {
                        let values =
                            &self.positive_logits[(row * 4 + head) * 32..(row * 4 + head + 1) * 32];
                        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        self.active_logits[row * 4 + head]
                            - values
                                .iter()
                                .map(|x| (*x - maximum).exp())
                                .sum::<f32>()
                                .ln()
                            > 0.0
                    };
                    if active {
                        let values =
                            &self.positive_logits[(row * 4 + head) * 32..(row * 4 + head + 1) * 32];
                        let index = if self.sample {
                            categorical(values, rng)
                        } else {
                            values
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.total_cmp(b.1))
                                .unwrap()
                                .0
                        };
                        result[base + axis] = (index + 1) as f32 / 32.0;
                    } else if self.sample {
                        // Keep RNG position independent of the active hurdle outcome.
                        let _ = random_u64(rng);
                    }
                }
            }
        }
        result
    }

    fn refine_candidates(
        &mut self,
        candidates: &[f32],
        selection: &SelectionArrays,
        physiology: &[f32],
        neural: &[f32],
        ticks: &[u64],
        reset: &[bool],
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let mut actions = vec![0.0; self.batch * ACTIONS];
        let mut oral = vec![0.0; self.batch];
        for row in 0..self.batch {
            if reset[row] {
                self.consequences
                    .cancel_pending(row)
                    .map_err(PyValueError::new_err)?;
                self.pending_tick[row] = None;
            }
            let current: [f32; 6] = physiology[row * 6..(row + 1) * 6].try_into().unwrap();
            let neural_row: &[f32; 384] =
                neural[row * NEURAL..(row + 1) * NEURAL].try_into().unwrap();
            oral[row] = ((1.0 - current[1]) * (1.1 - current[0])).clamp(0.0, 1.0);
            let desired = if selection.valid[row] && selection.remaining[row] > 0 {
                let end = (row * WINDOW + WINDOW - 1) * OBS;
                let goal: [f32; 6] = selection.window[end + OBS - 6..end + OBS]
                    .try_into()
                    .unwrap();
                Some(
                    desired_goal_consequences(&current, &goal, selection.remaining[row] as u32)
                        .map_err(PyValueError::new_err)?,
                )
            } else {
                None
            };
            let mut inherited_all = Vec::with_capacity(CANDIDATES);
            let mut features_all = Vec::with_capacity(CANDIDATES);
            let mut corrections_all = Vec::with_capacity(CANDIDATES);
            let mut scores = [0.0f64; CANDIDATES];
            for k in 0..CANDIDATES {
                let action: &[f32; 8] = candidates
                    [(row * CANDIDATES + k) * ACTIONS..(row * CANDIDATES + k + 1) * ACTIONS]
                    .try_into()
                    .unwrap();
                let raw = self
                    .law
                    .fitted_features(
                        &DecisionContext {
                            physiology: &current,
                            neural: neural_row,
                        },
                        &CandidateAction {
                            action,
                            oral: oral[row],
                        },
                    )
                    .map_err(PyValueError::new_err)?;
                let estimates = self.law.evaluate(&raw).map_err(PyValueError::new_err)?;
                let ood = estimates.iter().any(|x| x.out_of_domain);
                let (features, _) = self
                    .law
                    .normalize_private(&raw)
                    .map_err(PyValueError::new_err)?;
                let inherited: Vec<f64> = estimates.iter().map(|x| x.expected).collect();
                let personal = self
                    .consequences
                    .estimate(
                        row,
                        &features.iter().map(|x| f64::from(*x)).collect::<Vec<_>>(),
                        &inherited,
                        !ood,
                    )
                    .map_err(PyValueError::new_err)?;
                if let Some(target) = desired {
                    if !ood {
                        scores[k] = -personal
                            .expected
                            .iter()
                            .zip(target)
                            .zip(&self.law.laws)
                            .map(|((p, t), law)| {
                                ((p - t) / law.target_scale).clamp(-4.0, 4.0).powi(2)
                            })
                            .sum::<f64>()
                            / 3.0;
                    }
                }
                self.personal_updates[row] = personal.updates;
                corrections_all.push(personal.correction);
                self.candidate_ood[row * CANDIDATES + k] = ood;
                inherited_all.push(inherited);
                features_all.push(features);
            }
            let in_domain = (0..CANDIDATES)
                .filter(|k| !self.candidate_ood[row * CANDIDATES + *k])
                .collect::<Vec<_>>();
            let mean = if in_domain.is_empty() {
                0.0
            } else {
                in_domain.iter().map(|k| scores[*k]).sum::<f64>() / in_domain.len() as f64
            };
            for k in 0..CANDIDATES {
                self.candidate_scores[row * CANDIDATES + k] =
                    if self.candidate_ood[row * CANDIDATES + k] {
                        0.0
                    } else {
                        (TILT * (scores[k] - mean).tanh()) as f32
                    };
            }
            let chosen = if self.sample {
                categorical(
                    &self.candidate_scores[row * CANDIDATES..(row + 1) * CANDIDATES],
                    &mut self.action_rng[row * 4..(row + 1) * 4],
                )
            } else {
                0
            };
            self.selected_candidate[row] = chosen as i32;
            for j in 0..3 {
                self.selected_correction[row * 3 + j] = corrections_all[chosen][j] as f32;
            }
            actions[row * ACTIONS..(row + 1) * ACTIONS].copy_from_slice(
                &candidates[(row * CANDIDATES + chosen) * ACTIONS
                    ..(row * CANDIDATES + chosen + 1) * ACTIONS],
            );
            self.pending_action[row * PREVIOUS..row * PREVIOUS + ACTIONS]
                .copy_from_slice(&actions[row * ACTIONS..(row + 1) * ACTIONS]);
            self.pending_action[row * PREVIOUS + ACTIONS] = oral[row];
            self.pending_physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                .copy_from_slice(&current);
            self.pending_tick[row] = Some(ticks[row]);
            let f: Vec<f64> = features_all[chosen].iter().map(|x| f64::from(*x)).collect();
            self.consequences
                .record_executed(
                    row,
                    ticks[row],
                    &f,
                    &inherited_all[chosen],
                    !self.candidate_ood[row * CANDIDATES + chosen],
                )
                .map_err(PyValueError::new_err)?;
        }
        Ok((actions, oral))
    }
}

#[pymethods]
impl DevelopmentalResidentCohort {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        batch: usize,
        artifact_mode: &str,
        action_mode: &str,
        goal_seed: u64,
        action_seed: u64,
        packed: PyReadonlyArray1<'_, f32>,
        law_json: &str,
        law_identity: &str,
        learning_rate: f64,
        error_decay: f64,
        innovation_limit: f64,
    ) -> PyResult<Self> {
        if batch == 0
            || batch > 256
            || artifact_mode != "rich-achieved-goal"
            || !matches!(action_mode, "sample" | "map")
        {
            return Err(PyValueError::new_err(
                "invalid developmental cohort configuration",
            ));
        }
        let flat = packed.as_slice()?;
        let law: LawBank =
            serde_json::from_str(law_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        law.validate().map_err(PyValueError::new_err)?;
        if law.laws.iter().map(|x| x.name.as_str()).collect::<Vec<_>>()
            != ["movement_response", "energy_cost", "fatigue_recovery"]
        {
            return Err(PyValueError::new_err("GAM target order differs"));
        }
        let config = ConsequenceConfig {
            inherited_identity: law_identity.into(),
            feature_names: law.features.iter().map(|x| x.name.clone()).collect(),
            targets: law
                .laws
                .iter()
                .map(|x| ConsequenceTarget {
                    name: x.name.clone(),
                    unit: x.unit.clone(),
                    scale: x.target_scale,
                    correction_limit: x.conservative_residual_bound,
                })
                .collect(),
            learning_rate,
            error_decay,
            innovation_limit,
        };
        let consequences =
            PersonalConsequences::new(batch, config).map_err(PyValueError::new_err)?;
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "developmental weights must be finite",
            ));
        }
        let mut c = 0;
        let mean = take(flat, &mut c, OBS)?;
        let scale = take(flat, &mut c, OBS)?;
        if scale.iter().any(|x| *x <= 0.0) {
            return Err(PyValueError::new_err("developmental normalizer differs"));
        }
        let core = Core {
            mean,
            scale,
            peripheral_first: linear(flat, &mut c, 16, 4 * 3 * 3)?,
            peripheral_second: linear(flat, &mut c, 24, 16 * 3 * 3)?,
            foveal_first: linear(flat, &mut c, 16, 4 * 3 * 3)?,
            foveal_second: linear(flat, &mut c, 24, 16 * 3 * 3)?,
            peripheral_projection: linear(flat, &mut c, 64, 24 * 4 * 8)?,
            foveal_projection: linear(flat, &mut c, 64, 24 * 12 * 8)?,
            body: linear(flat, &mut c, 128, BODY)?,
            goal0: linear(flat, &mut c, POLICY, WINDOW * 256)?,
            goal2: linear(flat, &mut c, GOAL, POLICY)?,
            observation: linear(flat, &mut c, HIDDEN, 256 + PREVIOUS)?,
            history: gru(flat, &mut c, HIDDEN, HIDDEN)?,
            policy: linear(flat, &mut c, POLICY, HIDDEN + GOAL + 1 + ACTIONS)?,
            signed: linear(flat, &mut c, 4 * 65, POLICY)?,
            active: linear(flat, &mut c, 4, POLICY)?,
            positive: linear(flat, &mut c, 4 * 32, POLICY)?,
            manager0: linear(flat, &mut c, HIDDEN, HIDDEN + NEURAL + PHYSIOLOGY)?,
            manager2: linear(flat, &mut c, GOAL, HIDDEN)?,
            query_gain: *flat
                .get(c)
                .ok_or_else(|| PyValueError::new_err("manager gain missing"))?,
        };
        c += 1;
        if c != flat.len() {
            return Err(PyValueError::new_err(
                "developmental weight pack has trailing values",
            ));
        }
        let mut action_rng = vec![0; batch * 4];
        for row in 0..batch {
            let mut seed = action_seed ^ (row as u64).wrapping_mul(0xa076_1d64_78bd_642f);
            for word in &mut action_rng[row * 4..(row + 1) * 4] {
                *word = splitmix64(&mut seed);
            }
        }
        Ok(Self {
            batch,
            conditioned: true,
            sample: action_mode == "sample",
            core,
            memory: AchievedGoalMemoryCohort::new(batch, OBS, goal_seed)?,
            state: vec![0.0; batch * HIDDEN],
            previous_action: vec![0.0; batch * ACTIONS],
            action_rng,
            normalized: vec![0.0; batch * OBS],
            peripheral_input: Vec::new(),
            foveal_input: Vec::new(),
            body_input: Vec::new(),
            conv_columns: Vec::new(),
            conv_spatial: Vec::new(),
            conv_first: Vec::new(),
            conv_second: Vec::new(),
            peripheral_flat: Vec::new(),
            foveal_flat: Vec::new(),
            peripheral_code: Vec::new(),
            foveal_code: Vec::new(),
            body_code: Vec::new(),
            frame_code: Vec::new(),
            observation_input: vec![0.0; batch * (256 + PREVIOUS)],
            encoded: vec![0.0; batch * HIDDEN],
            gx: vec![0.0; batch * 3 * HIDDEN],
            gh: vec![0.0; batch * 3 * HIDDEN],
            next_state: vec![0.0; batch * HIDDEN],
            goal_observations: vec![0.0; batch * WINDOW * OBS],
            goal_flat: vec![0.0; batch * WINDOW * 256],
            goal_middle: vec![0.0; batch * POLICY],
            manager_input: vec![0.0; batch * (HIDDEN + NEURAL + PHYSIOLOGY)],
            manager_hidden: vec![0.0; batch * HIDDEN],
            query: vec![0.0; batch * GOAL],
            logits: vec![0.0; batch * RESERVOIR],
            policy_input: vec![0.0; batch * (HIDDEN + GOAL + 1 + ACTIONS)],
            policy_hidden: vec![0.0; batch * POLICY],
            signed_logits: vec![0.0; batch * 4 * 65],
            active_logits: vec![0.0; batch * 4],
            positive_logits: vec![0.0; batch * 4 * 32],
            law,
            consequences,
            pending_action: vec![0.0; batch * PREVIOUS],
            pending_physiology: vec![0.0; batch * PHYSIOLOGY],
            pending_tick: vec![None; batch],
            candidate_scores: vec![0.0; batch * CANDIDATES],
            candidate_ood: vec![false; batch * CANDIDATES],
            selected_candidate: vec![-1; batch],
            selected_correction: vec![0.0; batch * 3],
            personal_updates: vec![0; batch],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'_, f32>,
        neural: PyReadonlyArray2<'_, f32>,
        physiology: PyReadonlyArray2<'_, f32>,
        previous: PyReadonlyArray2<'_, f32>,
        ticks: PyReadonlyArray1<'_, u64>,
        times: PyReadonlyArray1<'_, f64>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if observations.shape() != [self.batch, OBS]
            || neural.shape() != [self.batch, NEURAL]
            || physiology.shape() != [self.batch, PHYSIOLOGY]
            || previous.shape() != [self.batch, PREVIOUS]
            || ticks.shape() != [self.batch]
            || times.shape() != [self.batch]
            || reset.shape() != [self.batch]
        {
            return Err(PyValueError::new_err("developmental step shapes differ"));
        }
        let o = observations.as_slice()?;
        let n = neural.as_slice()?;
        let p = physiology.as_slice()?;
        let a = previous.as_slice()?;
        let t = ticks.as_slice()?;
        let time = times.as_slice()?;
        let rst = reset.as_slice()?;
        if o.iter().chain(n).chain(p).chain(a).any(|x| !x.is_finite())
            || time.iter().any(|x| !x.is_finite())
        {
            return Err(PyValueError::new_err(
                "developmental step inputs must be finite",
            ));
        }
        if t.iter().any(|tick| *tick != t[0]) {
            return Err(PyValueError::new_err(
                "developmental cohort requires one shared model tick",
            ));
        }
        let (inserted, selection, actions, oral) = py.detach(|| -> PyResult<_> {
            self.observe(o, a, rst);
            let (windows, valid) = self.memory.push_inner(o, t, time, rst)?;
            let keys = self.encode_windows(&windows, &valid);
            let inserted = self.memory.remember_inner(&keys, &valid)?;
            let selection = self.manager_selection(n, p, t)?;
            let candidates = self.policy_actions(&selection.key, &selection.remaining);
            let (actions, oral) = self.refine_candidates(&candidates, &selection, p, n, t, rst)?;
            Ok((inserted, selection, actions, oral))
        })?;
        let out = PyDict::new(py);
        out.set_item(
            "proposed_action",
            Array2::from_shape_vec((self.batch, ACTIONS), actions)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("oral_command", Array1::from_vec(oral).into_pyarray(py))?;
        out.set_item(
            "candidate_scores",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_scores.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_out_of_domain",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_ood.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "selected_candidate",
            Array1::from_vec(self.selected_candidate.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "selected_consequence_correction",
            Array2::from_shape_vec((self.batch, 3), self.selected_correction.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "personal_consequence_updates",
            Array1::from_vec(self.personal_updates.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "actual_previous_action",
            Array2::from_shape_vec((self.batch, ACTIONS), self.previous_action.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "hidden",
            Array2::from_shape_vec((self.batch, HIDDEN), self.state.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "physiology",
            Array2::from_shape_vec((self.batch, PHYSIOLOGY), p.to_vec())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "memory_inserted_slot",
            Array1::from_vec(inserted).into_pyarray(py),
        )?;
        out.set_item(
            "memory_count",
            Array1::from_vec(self.memory.counts().to_vec()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_slot",
            Array1::from_vec(selection.slot).into_pyarray(py),
        )?;
        out.set_item(
            "goal_valid",
            Array1::from_vec(selection.valid).into_pyarray(py),
        )?;
        out.set_item(
            "goal_changed",
            Array1::from_vec(selection.changed).into_pyarray(py),
        )?;
        out.set_item(
            "goal_recorded_tick",
            Array1::from_vec(selection.recorded_tick).into_pyarray(py),
        )?;
        out.set_item(
            "goal_recorded_time",
            Array1::from_vec(selection.recorded_time).into_pyarray(py),
        )?;
        out.set_item(
            "goal_remaining_ticks",
            Array1::from_vec(selection.remaining).into_pyarray(py),
        )?;
        out.set_item(
            "goal_key",
            Array2::from_shape_vec((self.batch, GOAL), selection.key)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "goal_window",
            Array3::from_shape_vec((self.batch, WINDOW, OBS), selection.window)
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok(out)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item("version", 2)?;
        out.set_item("batch", self.batch)?;
        out.set_item("conditioned", self.conditioned)?;
        out.set_item("sample", self.sample)?;
        out.set_item(
            "hidden",
            Array2::from_shape_vec((self.batch, HIDDEN), self.state.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "actual_previous_action",
            Array2::from_shape_vec((self.batch, ACTIONS), self.previous_action.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "action_rng",
            Array2::from_shape_vec((self.batch, 4), self.action_rng.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("goal_memory", self.memory.snapshot(py)?)?;
        out.set_item(
            "personal_consequences",
            self.consequences
                .snapshot()
                .map_err(PyValueError::new_err)?,
        )?;
        out.set_item(
            "pending_action",
            Array2::from_shape_vec((self.batch, PREVIOUS), self.pending_action.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "pending_physiology",
            Array2::from_shape_vec((self.batch, PHYSIOLOGY), self.pending_physiology.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("pending_tick", self.pending_tick.clone())?;
        Ok(out)
    }

    fn observe_consequences(
        &mut self,
        ticks: PyReadonlyArray1<'_, u64>,
        before: PyReadonlyArray2<'_, f32>,
        after: PyReadonlyArray2<'_, f32>,
        executed: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        if ticks.shape() != [self.batch]
            || before.shape() != [self.batch, 6]
            || after.shape() != [self.batch, 6]
            || executed.shape() != [self.batch, PREVIOUS]
        {
            return Err(PyValueError::new_err("consequence outcome shapes differ"));
        }
        let t = ticks.as_slice()?;
        let b = before.as_slice()?;
        let a = after.as_slice()?;
        let x = executed.as_slice()?;
        if b.iter().chain(a).chain(x).any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("consequence receipt must be finite"));
        }
        let mut targets = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            if self.pending_tick[row] != Some(t[row])
                || self.pending_action[row * PREVIOUS..(row + 1) * PREVIOUS]
                    != x[row * PREVIOUS..(row + 1) * PREVIOUS]
                || self.pending_physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                    != b[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
            {
                return Err(PyValueError::new_err(
                    "executed consequence receipt differs",
                ));
            }
            let bb: [f32; 6] = b[row * 6..(row + 1) * 6].try_into().unwrap();
            let aa: [f32; 6] = a[row * 6..(row + 1) * 6].try_into().unwrap();
            let target = transition_targets(&bb, &aa);
            if target.iter().any(|v| !v.is_finite()) {
                return Err(PyValueError::new_err("consequence target is nonfinite"));
            }
            targets.push(target);
        }
        for row in 0..self.batch {
            self.consequences
                .observe(row, t[row], &targets[row])
                .map_err(PyValueError::new_err)?;
            self.pending_tick[row] = None;
        }
        Ok(())
    }

    fn restore(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        let get = |name: &str| {
            value.get_item(name)?.ok_or_else(|| {
                PyValueError::new_err(format!("developmental snapshot lacks {name}"))
            })
        };
        if get("format")?.extract::<String>()? != FORMAT
            || get("version")?.extract::<u8>()? != 2
            || get("batch")?.extract::<usize>()? != self.batch
            || get("conditioned")?.extract::<bool>()? != self.conditioned
            || get("sample")?.extract::<bool>()? != self.sample
        {
            return Err(PyValueError::new_err(
                "developmental snapshot identity differs",
            ));
        }
        let hidden: PyReadonlyArray2<'_, f32> = get("hidden")?.extract()?;
        let previous: PyReadonlyArray2<'_, f32> = get("actual_previous_action")?.extract()?;
        let rng: PyReadonlyArray2<'_, u64> = get("action_rng")?.extract()?;
        if hidden.shape() != [self.batch, HIDDEN]
            || previous.shape() != [self.batch, ACTIONS]
            || rng.shape() != [self.batch, 4]
        {
            return Err(PyValueError::new_err(
                "developmental snapshot shapes differ",
            ));
        }
        let h = hidden.as_slice()?;
        let p = previous.as_slice()?;
        let r = rng.as_slice()?;
        if h.iter().chain(p).any(|x| !x.is_finite())
            || r.chunks_exact(4).any(|x| x.iter().all(|v| *v == 0))
        {
            return Err(PyValueError::new_err(
                "developmental snapshot values differ",
            ));
        }
        let memory: Bound<'_, PyDict> = get("goal_memory")?.extract()?;
        self.memory.restore(&memory)?;
        let personal = get("personal_consequences")?.extract::<String>()?;
        let config = self.consequences.config().clone();
        self.consequences = PersonalConsequences::restore(&personal, &config, self.batch)
            .map_err(PyValueError::new_err)?;
        let pending: PyReadonlyArray2<'_, f32> = get("pending_action")?.extract()?;
        if pending.shape() != [self.batch, PREVIOUS] {
            return Err(PyValueError::new_err("pending action shape differs"));
        }
        self.pending_action.copy_from_slice(pending.as_slice()?);
        let pending_physiology: PyReadonlyArray2<'_, f32> = get("pending_physiology")?.extract()?;
        if pending_physiology.shape() != [self.batch, PHYSIOLOGY] {
            return Err(PyValueError::new_err("pending physiology shape differs"));
        }
        self.pending_physiology
            .copy_from_slice(pending_physiology.as_slice()?);
        self.pending_tick = get("pending_tick")?.extract()?;
        self.state.copy_from_slice(h);
        self.previous_action.copy_from_slice(p);
        self.action_rng.copy_from_slice(r);
        Ok(())
    }
}
