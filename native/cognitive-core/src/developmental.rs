use crate::contextual_episodic::{ContextualEpisodicLearner, CONTEXT};
use crate::gam_law::{
    desired_goal_consequences, transition_targets, CandidateAction, DecisionContext, LawBank,
};
use crate::goal_memory::{
    random_u64, splitmix64, unit_f64, AchievedGoalMemoryCohort, SelectionArrays,
};
use crate::motor_suffix::{MotorSuffixMemory, CONTEXT as SUFFIX_CONTEXT};
use crate::personal_consequences::{ConsequenceConfig, ConsequenceTarget, PersonalConsequences};
use crate::personal_goals::{
    GoalSlotIdentity, GoalSlotReplacement, GoalStart, GoalTransition, PersonalGoalAssociations,
    PersonalGoalConfig,
};
use crate::population_response::{
    population_response_features, PopulationHistory, PopulationResponseBank,
};
use crate::predictive_sensory::{
    PredictiveSensoryEnsemble, CONTEXT as PREDICTOR_CONTEXT, MEMBERS as PREDICTOR_MEMBERS,
};
use crate::sequence_memory::{GoalSequenceMemory, SequenceEpisode, SequenceNode};
use crate::{gemm_into, gru, linear, take, tanh_all, Gru, Linear};
use numpy::{
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

const OBS: usize = 4459;
const RICH: usize = 4096;
const BODY: usize = 363;
const PREVIOUS: usize = 12;
const ACTIONS: usize = 12;
const WINDOW: usize = 4;
const GOAL: usize = 64;
const HIDDEN: usize = 128;
const POLICY: usize = 256;
const NEURAL: usize = 384;
const PHYSIOLOGY: usize = 12;
const RESERVOIR: usize = 128;
const SIGNED: [usize; 4] = [0, 1, 2, 3];
const POSITIVE: [usize; 8] = [4, 5, 6, 7, 8, 9, 10, 11];
const FORMAT: &str = "chreatures-developmental-resident-native-population-v7";
const GOAL_ATTAINMENT_RMS: f32 = 0.35;
const SEQUENCE_CONSOLIDATION_BUDGET: usize = 4;
const LOCAL_CANDIDATES: usize = 4;
const RECALLED_CANDIDATES: usize = 4;
const CANDIDATES: usize = LOCAL_CANDIDATES + RECALLED_CANDIDATES;
const FORECAST_HORIZON: usize = 8;
const TILT: f64 = 0.5;

#[derive(Clone)]
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
    physiology_adapter: Linear,
    new_actuator_active: Linear,
    new_actuator_positive: Linear,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
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
    recent_codes: Vec<f32>,
    recent_code_cursor: Vec<usize>,
    recent_code_count: Vec<usize>,
    observation_input: Vec<f32>,
    physiology_input: Vec<f32>,
    physiology_delta: Vec<f32>,
    encoded: Vec<f32>,
    gx: Vec<f32>,
    gh: Vec<f32>,
    next_state: Vec<f32>,
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
    new_active_logits: Vec<f32>,
    new_positive_logits: Vec<f32>,
    law: LawBank,
    population_response: Option<PopulationResponseBank>,
    population_history: Option<PopulationHistory>,
    pending_population_history: Vec<f32>,
    population_in_domain: Vec<u64>,
    population_out_of_domain: Vec<u64>,
    population_last_in_domain: Vec<bool>,
    consequences: PersonalConsequences,
    pending_action: Vec<f32>,
    pending_physiology: Vec<f32>,
    pending_tick: Vec<Option<u64>>,
    candidate_scores: Vec<f32>,
    candidate_ood: Vec<bool>,
    selected_candidate: Vec<i32>,
    selected_correction: Vec<f32>,
    personal_updates: Vec<u64>,
    forecast_progress: Vec<f32>,
    forecast_disagreement: Vec<f32>,
    forecast_invalid: Vec<bool>,
    forecast_tilt: Vec<f32>,
    suffixes: MotorSuffixMemory,
    pending_suffix_context: Vec<f32>,
    pending_suffix_slot: Vec<i32>,
    pending_suffix_generation: Vec<u64>,
    candidate_recalled: Vec<bool>,
    candidate_available: Vec<bool>,
    candidate_suffix_slot: Vec<i32>,
    candidate_suffix_generation: Vec<u64>,
    candidate_suffix_length: Vec<u8>,
    candidate_suffix_support: Vec<u32>,
    candidate_suffix_empirical_score: Vec<f32>,
    candidate_suffix_recall_score: Vec<f32>,
    candidate_first_action: Vec<f32>,
    predictor: PredictiveSensoryEnsemble,
    predictor_context: Vec<f32>,
    predictor_actions: Vec<f32>,
    predictor_member_delta: Vec<f32>,
    predictor_mean_delta: Vec<f32>,
    predictor_disagreement: Vec<f32>,
    predictor_absolute_code: Vec<f32>,
    predictor_absolute_physiology: Vec<f32>,
    predictor_valid: Vec<bool>,
    predictor_goal_windows: Vec<f32>,
    predictor_goal_hidden: Vec<f32>,
    predictor_goal_keys: Vec<f32>,
    forecast_goal_rms: f32,
    personal_goals: PersonalGoalAssociations,
    goal_credit_pending: Vec<bool>,
    selected_goal_bias: Vec<f32>,
    selected_goal_prediction: Vec<f32>,
    last_goal_reward: Vec<f32>,
    last_goal_return: Vec<f32>,
    last_goal_completed: Vec<bool>,
    last_goal_attributed: Vec<bool>,
    last_goal_learned: Vec<bool>,
    contextual: ContextualEpisodicLearner,
    contextual_bias: Vec<f32>,
    sequence: GoalSequenceMemory,
    goal_measurement_valid: Vec<bool>,
    goal_measurement_slot: Vec<i32>,
    goal_measurement_recorded_tick: Vec<u64>,
    goal_measurement_generation: Vec<u64>,
    goal_measurement_start_rms: Vec<f32>,
    goal_measurement_min_rms: Vec<f32>,
    goal_measurement_latest_rms: Vec<f32>,
    goal_measurement_samples: Vec<u16>,
    goal_measurement_last_observed_tick: Vec<u64>,
    goal_measurement_context: Vec<f64>,
    last_goal_attained: Vec<bool>,
    last_goal_normalized_progress: Vec<f32>,
    sequence_selected_bias: Vec<f32>,
    sequence_experienced_path_depth: Vec<u8>,
    sequence_selected_confidence: Vec<f32>,
    candidate_sha256: Vec<String>,
    loci_sha256: Vec<String>,
    recurrent_gain: Vec<f32>,
    learning_rate_gain: Vec<f32>,
    action_gain: Vec<f32>,
    action_temperature_offset: Vec<f32>,
    recurrent_adapter: Vec<f32>,
    policy_adapter_count: usize,
    policy_adapter_rank: usize,
    policy_adapter_down: Vec<f32>,
    policy_adapter_up: Vec<f32>,
    policy_adapter_bias: Vec<f32>,
    policy_adapter_index: Vec<usize>,
}

fn categorical(logits: &[f32], inverse_temperature: f32, state: &mut [u64]) -> usize {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits
        .iter()
        .map(|x| (((*x - maximum) * inverse_temperature) as f64).exp())
        .sum();
    let threshold = unit_f64(state) * total;
    let mut cumulative = 0.0;
    for (index, value) in logits.iter().enumerate() {
        cumulative += (((*value - maximum) * inverse_temperature) as f64).exp();
        if threshold < cumulative {
            return index;
        }
    }
    logits.len() - 1
}

fn categorical_masked(logits: &[f32], available: &[bool], state: &mut [u64]) -> usize {
    let maximum = logits
        .iter()
        .zip(available)
        .filter(|(_, valid)| **valid)
        .map(|(value, _)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits
        .iter()
        .zip(available)
        .filter(|(_, valid)| **valid)
        .map(|(value, _)| ((*value - maximum) as f64).exp())
        .sum();
    let threshold = unit_f64(state) * total;
    let mut cumulative = 0.0;
    let fallback = available.iter().position(|valid| *valid).unwrap();
    for (index, (value, valid)) in logits.iter().zip(available).enumerate() {
        if *valid {
            cumulative += ((*value - maximum) as f64).exp();
            if threshold < cumulative {
                return index;
            }
        }
    }
    fallback
}

impl DevelopmentalResidentCohort {
    #[allow(clippy::too_many_arguments)]
    fn expanded_inner(
        &self,
        goal_seed: u64,
        action_seed: u64,
        candidate_sha256: Vec<String>,
        loci_sha256: Vec<String>,
        policy_adapter_index: PyReadonlyArray1<'_, u16>,
        recurrent_gain: PyReadonlyArray1<'_, f32>,
        learning_rate_gain: PyReadonlyArray1<'_, f32>,
        action_gain: PyReadonlyArray2<'_, f32>,
        action_temperature_offset: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Self> {
        let additions = candidate_sha256.len();
        let indices = policy_adapter_index.as_slice()?;
        let recurrent = recurrent_gain.as_slice()?;
        let learning = learning_rate_gain.as_slice()?;
        let gains = action_gain.as_slice()?;
        let temperatures = action_temperature_offset.as_slice()?;
        let valid_hash = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if additions == 0
            || self.batch.checked_add(additions).is_none()
            || self.batch + additions > 4096
            || loci_sha256.len() != additions
            || indices.len() != additions
            || recurrent.len() != additions
            || learning.len() != additions
            || gains.len() != additions * ACTIONS
            || temperatures.len() != additions * ACTIONS
            || candidate_sha256.iter().any(|value| !valid_hash(value))
            || loci_sha256.iter().any(|value| !valid_hash(value))
            || indices
                .iter()
                .any(|value| *value as usize >= self.policy_adapter_count)
            || recurrent
                .iter()
                .any(|value| !value.is_finite() || !(0.8..=1.2).contains(value))
            || learning
                .iter()
                .any(|value| !value.is_finite() || !(0.5..=1.5).contains(value))
            || gains
                .iter()
                .any(|value| !value.is_finite() || !(0.75..=1.25).contains(value))
            || temperatures
                .iter()
                .any(|value| !value.is_finite() || !(-0.5..=0.5).contains(value))
        {
            return Err(PyValueError::new_err(
                "candidate cohort expansion contract differs",
            ));
        }
        let old_batch = self.batch;
        let new_batch = old_batch + additions;
        let mut result = self.clone();
        result.grow_to(new_batch, goal_seed)?;
        for index in 0..additions {
            let row = old_batch + index;
            let mut state = action_seed ^ (row as u64).wrapping_mul(0xa076_1d64_78bd_642f);
            for word in &mut result.action_rng[row * 4..(row + 1) * 4] {
                *word = splitmix64(&mut state);
            }
            result.candidate_sha256[row] = candidate_sha256[index].clone();
            result.loci_sha256[row] = loci_sha256[index].clone();
            result.policy_adapter_index[row] = indices[index] as usize;
            result.recurrent_gain[row] = recurrent[index];
            result.learning_rate_gain[row] = learning[index];
            result.action_gain[row * ACTIONS..(row + 1) * ACTIONS]
                .copy_from_slice(&gains[index * ACTIONS..(index + 1) * ACTIONS]);
            result.action_temperature_offset[row * ACTIONS..(row + 1) * ACTIONS]
                .copy_from_slice(&temperatures[index * ACTIONS..(index + 1) * ACTIONS]);
        }
        Ok(result)
    }

    fn grow_to(&mut self, new_batch: usize, goal_seed: u64) -> PyResult<()> {
        if new_batch <= self.batch || new_batch > 4096 {
            return Err(PyValueError::new_err(
                "developmental growth must append within 4096 rows",
            ));
        }
        self.memory.grow(new_batch, goal_seed)?;
        self.consequences
            .grow(new_batch)
            .map_err(PyValueError::new_err)?;
        self.personal_goals
            .grow(new_batch)
            .map_err(PyValueError::new_err)?;
        self.contextual
            .grow(new_batch)
            .map_err(PyValueError::new_err)?;
        self.sequence
            .grow(new_batch)
            .map_err(PyValueError::new_err)?;
        self.suffixes
            .grow(new_batch, goal_seed ^ 0x4d4f_544f_525f_5637)
            .map_err(PyValueError::new_err)?;
        if let Some(history) = &mut self.population_history {
            history.grow(new_batch).map_err(PyValueError::new_err)?;
        }

        macro_rules! resize_f32 {
            ($field:ident, $stride:expr) => {
                self.$field.resize(new_batch * $stride, 0.0)
            };
        }
        resize_f32!(state, HIDDEN);
        resize_f32!(previous_action, ACTIONS);
        self.action_rng.resize(new_batch * 4, 0);
        resize_f32!(normalized, OBS);
        resize_f32!(recent_codes, WINDOW * 256);
        self.recent_code_cursor.resize(new_batch, 0);
        self.recent_code_count.resize(new_batch, 0);
        resize_f32!(observation_input, 256 + PREVIOUS);
        resize_f32!(physiology_input, PHYSIOLOGY);
        resize_f32!(physiology_delta, HIDDEN);
        resize_f32!(encoded, HIDDEN);
        resize_f32!(gx, 3 * HIDDEN);
        resize_f32!(gh, 3 * HIDDEN);
        resize_f32!(next_state, HIDDEN);
        resize_f32!(goal_flat, WINDOW * 256);
        resize_f32!(goal_middle, POLICY);
        resize_f32!(manager_input, HIDDEN + NEURAL + PHYSIOLOGY);
        resize_f32!(manager_hidden, HIDDEN);
        resize_f32!(query, GOAL);
        resize_f32!(logits, RESERVOIR);
        resize_f32!(policy_input, HIDDEN + GOAL + 1 + ACTIONS);
        resize_f32!(policy_hidden, POLICY);
        resize_f32!(signed_logits, 4 * 65);
        resize_f32!(active_logits, 8);
        resize_f32!(positive_logits, 8 * 32);
        resize_f32!(new_active_logits, 4);
        resize_f32!(new_positive_logits, 4 * 32);
        resize_f32!(pending_action, PREVIOUS);
        resize_f32!(pending_physiology, PHYSIOLOGY);
        resize_f32!(pending_population_history, 4);
        self.pending_tick.resize(new_batch, None);
        self.population_in_domain.resize(new_batch, 0);
        self.population_out_of_domain.resize(new_batch, 0);
        self.population_last_in_domain.resize(new_batch, false);
        resize_f32!(candidate_scores, CANDIDATES);
        self.candidate_ood.resize(new_batch * CANDIDATES, false);
        self.selected_candidate.resize(new_batch, -1);
        resize_f32!(selected_correction, 3);
        self.personal_updates.resize(new_batch, 0);
        resize_f32!(forecast_progress, CANDIDATES);
        resize_f32!(forecast_disagreement, CANDIDATES);
        self.forecast_invalid.resize(new_batch * CANDIDATES, false);
        resize_f32!(forecast_tilt, CANDIDATES);
        resize_f32!(pending_suffix_context, SUFFIX_CONTEXT);
        self.pending_suffix_slot.resize(new_batch, -1);
        self.pending_suffix_generation.resize(new_batch, 0);
        self.candidate_recalled
            .resize(new_batch * CANDIDATES, false);
        self.candidate_available
            .resize(new_batch * CANDIDATES, false);
        self.candidate_suffix_slot
            .resize(new_batch * CANDIDATES, -1);
        self.candidate_suffix_generation
            .resize(new_batch * CANDIDATES, 0);
        self.candidate_suffix_length
            .resize(new_batch * CANDIDATES, 0);
        self.candidate_suffix_support
            .resize(new_batch * CANDIDATES, 0);
        resize_f32!(candidate_suffix_empirical_score, CANDIDATES);
        resize_f32!(candidate_suffix_recall_score, CANDIDATES);
        resize_f32!(candidate_first_action, CANDIDATES * ACTIONS);
        self.goal_credit_pending.resize(new_batch, false);
        resize_f32!(selected_goal_bias, 1);
        resize_f32!(selected_goal_prediction, 1);
        resize_f32!(last_goal_reward, 1);
        resize_f32!(last_goal_return, 1);
        self.last_goal_completed.resize(new_batch, false);
        self.last_goal_attributed.resize(new_batch, false);
        self.last_goal_learned.resize(new_batch, false);
        resize_f32!(contextual_bias, 1);
        self.goal_measurement_valid.resize(new_batch, false);
        self.goal_measurement_slot.resize(new_batch, -1);
        self.goal_measurement_recorded_tick.resize(new_batch, 0);
        self.goal_measurement_generation.resize(new_batch, 0);
        resize_f32!(goal_measurement_start_rms, 1);
        resize_f32!(goal_measurement_min_rms, 1);
        resize_f32!(goal_measurement_latest_rms, 1);
        self.goal_measurement_samples.resize(new_batch, 0);
        self.goal_measurement_last_observed_tick
            .resize(new_batch, 0);
        self.goal_measurement_context
            .resize(new_batch * CONTEXT, 0.0);
        self.last_goal_attained.resize(new_batch, false);
        resize_f32!(last_goal_normalized_progress, 1);
        resize_f32!(sequence_selected_bias, 1);
        self.sequence_experienced_path_depth.resize(new_batch, 0);
        resize_f32!(sequence_selected_confidence, 1);
        self.candidate_sha256.resize(new_batch, String::new());
        self.loci_sha256.resize(new_batch, String::new());
        resize_f32!(recurrent_gain, 1);
        resize_f32!(learning_rate_gain, 1);
        resize_f32!(action_gain, ACTIONS);
        resize_f32!(action_temperature_offset, ACTIONS);
        resize_f32!(recurrent_adapter, HIDDEN);
        self.policy_adapter_index.resize(new_batch, 0);
        self.batch = new_batch;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hatch_slots_inner(
        &mut self,
        rows: PyReadonlyArray1<'_, u16>,
        goal_seeds: PyReadonlyArray1<'_, u64>,
        action_seeds: PyReadonlyArray1<'_, u64>,
        candidate_sha256: Vec<String>,
        loci_sha256: Vec<String>,
        policy_adapter_index: PyReadonlyArray1<'_, u16>,
        recurrent_gain: PyReadonlyArray1<'_, f32>,
        learning_rate_gain: PyReadonlyArray1<'_, f32>,
        action_gain: PyReadonlyArray2<'_, f32>,
        action_temperature_offset: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        let rows = rows.as_slice()?;
        let goal_seeds = goal_seeds.as_slice()?;
        let action_seeds = action_seeds.as_slice()?;
        let adapter_index = policy_adapter_index.as_slice()?;
        let recurrent_gain = recurrent_gain.as_slice()?;
        let learning_rate_gain = learning_rate_gain.as_slice()?;
        let action_gain = action_gain.as_slice()?;
        let temperature = action_temperature_offset.as_slice()?;
        let count = rows.len();
        if count == 0
            || goal_seeds.len() != count
            || action_seeds.len() != count
            || candidate_sha256.len() != count
            || loci_sha256.len() != count
            || adapter_index.len() != count
            || recurrent_gain.len() != count
            || learning_rate_gain.len() != count
            || action_gain.len() != count * ACTIONS
            || temperature.len() != count * ACTIONS
        {
            return Err(PyValueError::new_err("hatch slot shapes differ"));
        }
        let mut seen = vec![false; self.batch];
        for index in 0..count {
            let row = rows[index] as usize;
            let digest_ok = |value: &str| {
                value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            };
            if row >= self.batch
                || std::mem::replace(&mut seen[row], true)
                || !digest_ok(&candidate_sha256[index])
                || !digest_ok(&loci_sha256[index])
                || adapter_index[index] as usize >= self.policy_adapter_count
                || !recurrent_gain[index].is_finite()
                || !(0.8..=1.2).contains(&recurrent_gain[index])
                || !learning_rate_gain[index].is_finite()
                || !(0.5..=1.5).contains(&learning_rate_gain[index])
                || action_gain[index * ACTIONS..(index + 1) * ACTIONS]
                    .iter()
                    .any(|value| !value.is_finite() || !(0.75..=1.25).contains(value))
                || temperature[index * ACTIONS..(index + 1) * ACTIONS]
                    .iter()
                    .any(|value| !value.is_finite() || !(-0.5..=0.5).contains(value))
            {
                return Err(PyValueError::new_err("invalid hatch slot request"));
            }
        }
        for index in 0..count {
            let row = rows[index] as usize;
            self.memory.clear_resident(row, goal_seeds[index])?;
            self.personal_goals
                .clear_resident(row)
                .map_err(PyValueError::new_err)?;
            self.consequences
                .clear_resident(row)
                .map_err(PyValueError::new_err)?;
            self.contextual
                .clear_resident(row)
                .map_err(PyValueError::new_err)?;
            self.sequence
                .clear_resident(row)
                .map_err(PyValueError::new_err)?;
            self.suffixes
                .clear_resident(row, goal_seeds[index] ^ 0x4d4f_544f_525f_5637)
                .map_err(PyValueError::new_err)?;

            self.state[row * HIDDEN..(row + 1) * HIDDEN].fill(0.0);
            self.previous_action[row * ACTIONS..(row + 1) * ACTIONS].fill(0.0);
            self.recurrent_adapter[row * HIDDEN..(row + 1) * HIDDEN].fill(0.0);
            self.recent_codes[row * WINDOW * 256..(row + 1) * WINDOW * 256].fill(0.0);
            self.recent_code_cursor[row] = 0;
            self.recent_code_count[row] = 0;
            self.pending_action[row * PREVIOUS..(row + 1) * PREVIOUS].fill(0.0);
            self.pending_physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY].fill(0.0);
            self.pending_tick[row] = None;
            self.goal_credit_pending[row] = false;
            self.selected_goal_bias[row] = 0.0;
            self.selected_goal_prediction[row] = 0.0;
            self.last_goal_reward[row] = 0.0;
            self.last_goal_return[row] = 0.0;
            self.last_goal_completed[row] = false;
            self.last_goal_attributed[row] = false;
            self.last_goal_learned[row] = false;
            self.contextual_bias[row] = 0.0;
            self.clear_goal_measurement(row);
            self.last_goal_attained[row] = false;
            self.last_goal_normalized_progress[row] = 0.0;
            self.sequence_selected_bias[row] = 0.0;
            self.sequence_experienced_path_depth[row] = 0;
            self.sequence_selected_confidence[row] = 0.0;
            self.candidate_scores[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0.0);
            self.candidate_ood[row * CANDIDATES..(row + 1) * CANDIDATES].fill(false);
            self.forecast_progress[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0.0);
            self.forecast_disagreement[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0.0);
            self.forecast_invalid[row * CANDIDATES..(row + 1) * CANDIDATES].fill(false);
            self.forecast_tilt[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0.0);
            self.pending_suffix_context[row * SUFFIX_CONTEXT..(row + 1) * SUFFIX_CONTEXT].fill(0.0);
            self.pending_suffix_slot[row] = -1;
            self.pending_suffix_generation[row] = 0;
            self.candidate_recalled[row * CANDIDATES..(row + 1) * CANDIDATES].fill(false);
            self.candidate_available[row * CANDIDATES..(row + 1) * CANDIDATES].fill(false);
            self.candidate_suffix_slot[row * CANDIDATES..(row + 1) * CANDIDATES].fill(-1);
            self.candidate_suffix_generation[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0);
            self.candidate_suffix_length[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0);
            self.candidate_suffix_support[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0);
            self.candidate_suffix_empirical_score[row * CANDIDATES..(row + 1) * CANDIDATES]
                .fill(0.0);
            self.candidate_suffix_recall_score[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0.0);
            self.candidate_first_action
                [row * CANDIDATES * ACTIONS..(row + 1) * CANDIDATES * ACTIONS]
                .fill(0.0);
            self.selected_candidate[row] = -1;
            self.selected_correction[row * 3..(row + 1) * 3].fill(0.0);
            self.personal_updates[row] = 0;
            if let Some(history) = &mut self.population_history {
                history.clear(row).map_err(PyValueError::new_err)?;
            }
            self.pending_population_history[row * 4..(row + 1) * 4].fill(0.0);
            self.population_in_domain[row] = 0;
            self.population_out_of_domain[row] = 0;
            self.population_last_in_domain[row] = false;

            let mut state = action_seeds[index] ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            for word in &mut self.action_rng[row * 4..(row + 1) * 4] {
                *word = splitmix64(&mut state);
            }
            self.candidate_sha256[row] = candidate_sha256[index].clone();
            self.loci_sha256[row] = loci_sha256[index].clone();
            self.policy_adapter_index[row] = adapter_index[index] as usize;
            self.recurrent_gain[row] = recurrent_gain[index];
            self.learning_rate_gain[row] = learning_rate_gain[index];
            self.action_gain[row * ACTIONS..(row + 1) * ACTIONS]
                .copy_from_slice(&action_gain[index * ACTIONS..(index + 1) * ACTIONS]);
            self.action_temperature_offset[row * ACTIONS..(row + 1) * ACTIONS]
                .copy_from_slice(&temperature[index * ACTIONS..(index + 1) * ACTIONS]);
        }
        Ok(())
    }

    fn episodic_context(&self, row: usize, physiology: &[f32]) -> [f64; CONTEXT] {
        let mut context = [0.0; CONTEXT];
        for index in 0..PHYSIOLOGY {
            context[index] = physiology[row * PHYSIOLOGY + index] as f64;
        }
        for index in 0..(CONTEXT - PHYSIOLOGY) {
            context[PHYSIOLOGY + index] = self.state[row * HIDDEN + index * 31] as f64;
        }
        context
    }

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

    fn encode_windows(&mut self, valid: &[bool]) -> Vec<f32> {
        if !valid.iter().any(|include| *include) {
            return vec![0.0; self.batch * GOAL];
        }
        self.goal_flat.fill(0.0);
        for row in 0..self.batch {
            if !valid[row] {
                continue;
            }
            for frame in 0..WINDOW {
                let slot = (self.recent_code_cursor[row] + frame) % WINDOW;
                let source = (row * WINDOW + slot) * 256;
                let target = (row * WINDOW + frame) * 256;
                self.goal_flat[target..target + 256]
                    .copy_from_slice(&self.recent_codes[source..source + 256]);
            }
        }
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

    fn remember_frame_codes(&mut self, reset: &[bool]) {
        for row in 0..self.batch {
            if reset[row] {
                self.recent_code_cursor[row] = 0;
                self.recent_code_count[row] = 0;
            }
            let slot = self.recent_code_cursor[row];
            let target = (row * WINDOW + slot) * 256;
            self.recent_codes[target..target + 256]
                .copy_from_slice(&self.frame_code[row * 256..(row + 1) * 256]);
            self.recent_code_cursor[row] = (slot + 1) % WINDOW;
            self.recent_code_count[row] = (self.recent_code_count[row] + 1).min(WINDOW);
        }
    }

    fn observe(&mut self, observations: &[f32], previous: &[f32], reset: &[bool]) {
        for row in 0..self.batch {
            if reset[row] {
                self.state[row * HIDDEN..(row + 1) * HIDDEN].fill(0.0);
                self.recurrent_adapter[row * HIDDEN..(row + 1) * HIDDEN].fill(0.0);
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
            self.physiology_input[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY].copy_from_slice(
                &self.normalized[row * OBS + RICH + 351..row * OBS + RICH + 351 + PHYSIOLOGY],
            );
        }
        gemm_into(
            &self.observation_input,
            self.batch,
            256 + PREVIOUS,
            &self.core.observation,
            &mut self.encoded,
        );
        gemm_into(
            &self.physiology_input,
            self.batch,
            PHYSIOLOGY,
            &self.core.physiology_adapter,
            &mut self.physiology_delta,
        );
        for (encoded, delta) in self.encoded.iter_mut().zip(&self.physiology_delta) {
            *encoded += *delta;
        }
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

    fn clear_goal_measurement(&mut self, row: usize) {
        self.goal_measurement_valid[row] = false;
        self.goal_measurement_slot[row] = -1;
        self.goal_measurement_recorded_tick[row] = 0;
        self.goal_measurement_generation[row] = 0;
        self.goal_measurement_start_rms[row] = 0.0;
        self.goal_measurement_min_rms[row] = 0.0;
        self.goal_measurement_latest_rms[row] = 0.0;
        self.goal_measurement_samples[row] = 0;
        self.goal_measurement_last_observed_tick[row] = 0;
        self.goal_measurement_context[row * CONTEXT..(row + 1) * CONTEXT].fill(0.0);
    }

    fn goal_rms(current: &[f32], target: &[f32]) -> f32 {
        (current
            .iter()
            .zip(target)
            .map(|(a, b)| (*a - *b) * (*a - *b))
            .sum::<f32>()
            / GOAL as f32)
            .sqrt()
    }

    fn sample_goal_codes(
        &mut self,
        selection: &SelectionArrays,
        keys: &[f32],
        physiology: &[f32],
        ticks: &[u64],
    ) {
        for row in 0..self.batch {
            if !selection.valid[row] {
                continue;
            }
            let same = self.goal_measurement_valid[row]
                && self.goal_measurement_slot[row] == selection.slot[row]
                && self.goal_measurement_recorded_tick[row] == selection.recorded_tick[row]
                && self.goal_measurement_generation[row] == selection.generation[row];
            if !same {
                self.clear_goal_measurement(row);
                self.goal_measurement_valid[row] = true;
                self.goal_measurement_slot[row] = selection.slot[row];
                self.goal_measurement_recorded_tick[row] = selection.recorded_tick[row];
                self.goal_measurement_generation[row] = selection.generation[row];
            }
            let rms = Self::goal_rms(
                &keys[row * GOAL..(row + 1) * GOAL],
                &selection.key[row * GOAL..(row + 1) * GOAL],
            );
            if self.goal_measurement_samples[row] == 0 {
                self.goal_measurement_start_rms[row] = rms;
                self.goal_measurement_min_rms[row] = rms;
            } else {
                self.goal_measurement_min_rms[row] = self.goal_measurement_min_rms[row].min(rms);
            }
            self.goal_measurement_latest_rms[row] = rms;
            self.goal_measurement_samples[row] =
                self.goal_measurement_samples[row].saturating_add(1);
            self.goal_measurement_last_observed_tick[row] = ticks[row];
            let context = self.episodic_context(row, physiology);
            self.goal_measurement_context[row * CONTEXT..(row + 1) * CONTEXT]
                .copy_from_slice(&context);
        }
    }

    fn manager_selection(
        &mut self,
        neural: &[f32],
        physiology: &[f32],
        ticks: &[u64],
    ) -> PyResult<SelectionArrays> {
        if !ticks.iter().all(|tick| tick % 10 == 0) {
            return Ok(self.memory.current_selection(ticks));
        }
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
        let (recorded_ticks, generations) = self.memory.slot_identities();
        let recorded_ticks = recorded_ticks.to_vec();
        let generations = generations.to_vec();
        let mut sequence_plans = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            for slot in 0..counts[row] as usize {
                let mut dot = 0.0f32;
                for j in 0..GOAL {
                    dot += self.query[row * GOAL + j] * keys[(row * RESERVOIR + slot) * GOAL + j];
                }
                self.logits[row * RESERVOIR + slot] = dot * self.core.query_gain / 8.0;
            }
            let biases = self
                .personal_goals
                .selection_biases(
                    row,
                    &recorded_ticks[row * RESERVOIR..(row + 1) * RESERVOIR],
                    &generations[row * RESERVOIR..(row + 1) * RESERVOIR],
                    [
                        physiology[row * PHYSIOLOGY] as f64,
                        physiology[row * PHYSIOLOGY + 1] as f64,
                        physiology[row * PHYSIOLOGY + 2] as f64,
                    ],
                )
                .map_err(PyValueError::new_err)?;
            let episodic = self
                .contextual
                .biases(
                    row,
                    &generations[row * RESERVOIR..(row + 1) * RESERVOIR],
                    &self.episodic_context(row, physiology),
                )
                .map_err(PyValueError::new_err)?;
            let sequence = self
                .sequence
                .selection_biases(
                    row,
                    &recorded_ticks[row * RESERVOIR..(row + 1) * RESERVOIR],
                    &generations[row * RESERVOIR..(row + 1) * RESERVOIR],
                    &self.episodic_context(row, physiology),
                )
                .map_err(PyValueError::new_err)?;
            for slot in 0..counts[row] as usize {
                self.logits[row * RESERVOIR + slot] +=
                    (biases[slot] + episodic[slot] + sequence.biases[slot]) as f32;
            }
            sequence_plans.push(sequence);
        }
        let selection = self.memory.choose_inner(&self.logits, 1.0, ticks)?;
        let mut starts = Vec::new();
        let mut start_rows = Vec::new();
        for row in 0..self.batch {
            if selection.valid[row] {
                let slot = selection.slot[row] as usize;
                self.sequence_selected_bias[row] = sequence_plans[row].biases[slot] as f32;
                self.sequence_experienced_path_depth[row] =
                    sequence_plans[row].experienced_path_depth[slot];
                self.sequence_selected_confidence[row] =
                    sequence_plans[row].confidence[slot] as f32;
            } else {
                self.sequence_selected_bias[row] = 0.0;
                self.sequence_experienced_path_depth[row] = 0;
                self.sequence_selected_confidence[row] = 0.0;
            }
            if selection.changed[row] && selection.valid[row] {
                starts.push(GoalStart {
                    resident: row,
                    slot: selection.slot[row] as usize,
                    identity: GoalSlotIdentity {
                        recorded_tick: selection.recorded_tick[row],
                        generation: selection.generation[row],
                    },
                    selected_at_tick: ticks[row],
                    physiology: [
                        physiology[row * PHYSIOLOGY] as f64,
                        physiology[row * PHYSIOLOGY + 1] as f64,
                        physiology[row * PHYSIOLOGY + 2] as f64,
                    ],
                });
                start_rows.push(row);
            }
        }
        let estimates = self
            .personal_goals
            .begin_goals(&starts)
            .map_err(PyValueError::new_err)?;
        for (row, estimate) in start_rows.into_iter().zip(estimates) {
            self.goal_credit_pending[row] = true;
            self.selected_goal_bias[row] = estimate.logit_bias as f32;
            self.selected_goal_prediction[row] = estimate.predicted_normalized_return as f32;
            let slot = selection.slot[row] as usize;
            let context = self.episodic_context(row, physiology);
            let episodic = self
                .contextual
                .biases(
                    row,
                    &generations[row * RESERVOIR..(row + 1) * RESERVOIR],
                    &context,
                )
                .map_err(PyValueError::new_err)?;
            self.contextual_bias[row] = episodic[slot] as f32;
            self.contextual
                .begin(row, slot, selection.generation[row], context)
                .map_err(PyValueError::new_err)?;
        }
        Ok(selection)
    }

    fn policy_actions(&mut self, goal: &[f32], remaining: &[u64]) -> Vec<f32> {
        let width = HIDDEN + GOAL + 1 + ACTIONS;
        for row in 0..self.batch {
            let offset = row * width;
            self.policy_input[offset..offset + HIDDEN]
                .copy_from_slice(&self.state[row * HIDDEN..(row + 1) * HIDDEN]);
            for j in 0..HIDDEN {
                let state = self.state[row * HIDDEN + j];
                let adapter = &mut self.recurrent_adapter[row * HIDDEN + j];
                *adapter = 0.95 * *adapter + 0.05 * (self.recurrent_gain[row] - 1.0) * state;
                self.policy_input[offset + j] += *adapter;
            }
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
        let mut low_rank = vec![0.0f32; self.policy_adapter_rank];
        for row in 0..self.batch {
            let adapter = self.policy_adapter_index[row];
            low_rank.fill(0.0);
            for (rank, inner) in low_rank.iter_mut().enumerate() {
                let down = (adapter * self.policy_adapter_rank + rank) * POLICY;
                *inner = (0..POLICY)
                    .map(|j| {
                        self.policy_hidden[row * POLICY + j] * self.policy_adapter_down[down + j]
                    })
                    .sum();
            }
            for j in 0..POLICY {
                let up = (adapter * POLICY + j) * self.policy_adapter_rank;
                let delta = self.policy_adapter_bias[adapter * POLICY + j]
                    + (0..self.policy_adapter_rank)
                        .map(|rank| self.policy_adapter_up[up + rank] * low_rank[rank])
                        .sum::<f32>();
                self.policy_hidden[row * POLICY + j] += delta;
            }
        }
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
        gemm_into(
            &self.policy_hidden,
            self.batch,
            POLICY,
            &self.core.new_actuator_active,
            &mut self.new_active_logits,
        );
        gemm_into(
            &self.policy_hidden,
            self.batch,
            POLICY,
            &self.core.new_actuator_positive,
            &mut self.new_positive_logits,
        );
        for row in 0..self.batch {
            for head in 0..4 {
                self.active_logits[row * 8 + 4 + head] += self.new_active_logits[row * 4 + head];
                let destination = (row * 8 + 4 + head) * 32;
                let source = (row * 4 + head) * 32;
                for bin in 0..32 {
                    self.positive_logits[destination + bin] +=
                        self.new_positive_logits[source + bin];
                }
            }
        }
        let mut result = vec![0.0; self.batch * LOCAL_CANDIDATES * ACTIONS];
        for row in 0..self.batch {
            let rng = &mut self.action_rng[row * 4..(row + 1) * 4];
            for candidate in 0..LOCAL_CANDIDATES {
                let base = (row * LOCAL_CANDIDATES + candidate) * ACTIONS;
                for (head, axis) in SIGNED.iter().enumerate() {
                    let values =
                        &self.signed_logits[(row * 4 + head) * 65..(row * 4 + head + 1) * 65];
                    let index = if self.sample {
                        categorical(
                            values,
                            (-self.action_temperature_offset[row * ACTIONS + axis]).exp(),
                            rng,
                        )
                    } else {
                        values
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0
                    };
                    result[base + axis] = ((index as f32 / 32.0 - 1.0)
                        * self.action_gain[row * ACTIONS + axis])
                        .clamp(-1.0, 1.0);
                }
                for (head, axis) in POSITIVE.iter().enumerate() {
                    let inverse_temperature =
                        (-self.action_temperature_offset[row * ACTIONS + axis]).exp();
                    let active = if self.sample {
                        unit_f64(rng)
                            < (1.0
                                / (1.0
                                    + (-self.active_logits[row * 8 + head] * inverse_temperature)
                                        .exp())) as f64
                    } else {
                        let values =
                            &self.positive_logits[(row * 8 + head) * 32..(row * 8 + head + 1) * 32];
                        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        self.active_logits[row * 8 + head]
                            - values
                                .iter()
                                .map(|x| (*x - maximum).exp())
                                .sum::<f32>()
                                .ln()
                            > 0.0
                    };
                    if active {
                        let values =
                            &self.positive_logits[(row * 8 + head) * 32..(row * 8 + head + 1) * 32];
                        let index = if self.sample {
                            categorical(values, inverse_temperature, rng)
                        } else {
                            values
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.total_cmp(b.1))
                                .unwrap()
                                .0
                        };
                        result[base + axis] = (((index + 1) as f32 / 32.0)
                            * self.action_gain[row * ACTIONS + axis])
                            .clamp(0.0, 1.0);
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
        previous: &[f32],
        current_key: &[f32],
        ticks: &[u64],
        reset: &[bool],
    ) -> PyResult<Vec<f32>> {
        let mut actions = vec![0.0; self.batch * ACTIONS];
        let rows = self.batch * CANDIDATES;
        if candidates.len() != self.batch * LOCAL_CANDIDATES * ACTIONS {
            return Err(PyValueError::new_err("local candidate actions differ"));
        }
        let mut candidate_actions = vec![0.0; rows * ACTIONS];
        let mut candidate_horizons = vec![FORECAST_HORIZON; rows];
        self.predictor_actions
            .resize(rows * FORECAST_HORIZON * ACTIONS, 0.0);
        self.predictor_actions.fill(0.0);
        for row in 0..self.batch {
            self.candidate_recalled[row * CANDIDATES..(row + 1) * CANDIDATES].fill(false);
            self.candidate_available[row * CANDIDATES..(row + 1) * CANDIDATES].fill(false);
            self.candidate_suffix_slot[row * CANDIDATES..(row + 1) * CANDIDATES].fill(-1);
            self.candidate_suffix_generation[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0);
            self.candidate_suffix_length[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0);
            self.candidate_suffix_support[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0);
            self.candidate_suffix_empirical_score[row * CANDIDATES..(row + 1) * CANDIDATES]
                .fill(0.0);
            self.candidate_suffix_recall_score[row * CANDIDATES..(row + 1) * CANDIDATES].fill(0.0);
            for k in 0..LOCAL_CANDIDATES {
                self.candidate_available[row * CANDIDATES + k] = true;
                let src = (row * LOCAL_CANDIDATES + k) * ACTIONS;
                let first = (row * CANDIDATES + k) * ACTIONS;
                candidate_actions[first..first + ACTIONS]
                    .copy_from_slice(&candidates[src..src + ACTIONS]);
                for h in 0..FORECAST_HORIZON {
                    let dst = ((row * CANDIDATES + k) * FORECAST_HORIZON + h) * ACTIONS;
                    self.predictor_actions[dst..dst + ACTIONS]
                        .copy_from_slice(&candidates[src..src + ACTIONS]);
                }
            }
            let recalled = self.suffixes.recall(
                row,
                &current_key[row * SUFFIX_CONTEXT..(row + 1) * SUFFIX_CONTEXT],
                RECALLED_CANDIDATES,
            );
            for extra in 0..RECALLED_CANDIDATES {
                let k = LOCAL_CANDIDATES + extra;
                if let Some(suffix) = recalled.get(extra) {
                    let first = (row * CANDIDATES + k) * ACTIONS;
                    candidate_actions[first..first + ACTIONS]
                        .copy_from_slice(&suffix.actions[..ACTIONS]);
                    candidate_horizons[row * CANDIDATES + k] = suffix.length;
                    let last = (suffix.length - 1) * ACTIONS;
                    for h in 0..FORECAST_HORIZON {
                        let src = h.min(suffix.length - 1) * ACTIONS;
                        let dst = ((row * CANDIDATES + k) * FORECAST_HORIZON + h) * ACTIONS;
                        self.predictor_actions[dst..dst + ACTIONS]
                            .copy_from_slice(&suffix.actions[src..src + ACTIONS]);
                    }
                    debug_assert!(last < suffix.actions.len());
                    self.candidate_recalled[row * CANDIDATES + k] = true;
                    self.candidate_available[row * CANDIDATES + k] = true;
                    self.candidate_suffix_slot[row * CANDIDATES + k] = suffix.slot as i32;
                    self.candidate_suffix_generation[row * CANDIDATES + k] = suffix.generation;
                    self.candidate_suffix_length[row * CANDIDATES + k] = suffix.length as u8;
                    self.candidate_suffix_support[row * CANDIDATES + k] = suffix.support;
                    self.candidate_suffix_empirical_score[row * CANDIDATES + k] =
                        suffix.empirical_utility;
                    self.candidate_suffix_recall_score[row * CANDIDATES + k] =
                        suffix.recall_score.clamp(-8.0, 1.0);
                } else {
                    let local = (row * LOCAL_CANDIDATES) * ACTIONS;
                    let first = (row * CANDIDATES + k) * ACTIONS;
                    candidate_actions[first..first + ACTIONS]
                        .copy_from_slice(&candidates[local..local + ACTIONS]);
                    for h in 0..FORECAST_HORIZON {
                        let dst = ((row * CANDIDATES + k) * FORECAST_HORIZON + h) * ACTIONS;
                        self.predictor_actions[dst..dst + ACTIONS]
                            .copy_from_slice(&candidates[local..local + ACTIONS]);
                    }
                }
            }
        }
        self.candidate_first_action
            .copy_from_slice(&candidate_actions);
        self.predictor_context
            .resize(self.batch * PREDICTOR_CONTEXT, 0.0);
        for row in 0..self.batch {
            let dst = row * PREDICTOR_CONTEXT;
            for frame in 0..WINDOW {
                let slot = (self.recent_code_cursor[row] + frame) % WINDOW;
                let src = (row * WINDOW + slot) * 256;
                self.predictor_context[dst + frame * 256..dst + (frame + 1) * 256]
                    .copy_from_slice(&self.recent_codes[src..src + 256]);
            }
            for channel in 0..HIDDEN {
                self.predictor_context[dst + 1024 + channel] = self.state[row * HIDDEN + channel]
                    + self.recurrent_adapter[row * HIDDEN + channel];
            }
            self.predictor_context[dst + 1152..dst + 1536]
                .copy_from_slice(&neural[row * NEURAL..(row + 1) * NEURAL]);
            self.predictor_context[dst + 1536..dst + 1548]
                .copy_from_slice(&physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]);
            self.predictor_context[dst + 1548..dst + 1560]
                .copy_from_slice(&previous[row * PREVIOUS..(row + 1) * PREVIOUS]);
        }
        // Counterfactual hold, not a commitment: only the first action is
        // delivered, and the next physical observation triggers a new decision.
        self.predictor.forecast_sequences_into(
            &self.predictor_context,
            &self.predictor_actions,
            physiology,
            self.batch,
            CANDIDATES,
            FORECAST_HORIZON,
            &mut self.predictor_member_delta,
            &mut self.predictor_mean_delta,
            &mut self.predictor_disagreement,
            &mut self.predictor_absolute_code,
            &mut self.predictor_absolute_physiology,
            &mut self.predictor_valid,
        )?;
        let forecast_rows = rows * PREDICTOR_MEMBERS;
        self.predictor_goal_windows
            .resize(forecast_rows * 1024, 0.0);
        for outrow in 0..forecast_rows {
            let start = candidate_horizons[outrow / PREDICTOR_MEMBERS] - WINDOW;
            // The goal window ends at the candidate's actually observed suffix
            // length. Local proposals use the full H8 counterfactual hold.
            for frame in 0..WINDOW {
                let src = (outrow * FORECAST_HORIZON + start + frame) * 256;
                let dst = outrow * 1024 + frame * 256;
                self.predictor_goal_windows[dst..dst + 256]
                    .copy_from_slice(&self.predictor_absolute_code[src..src + 256]);
            }
        }
        gemm_into(
            &self.predictor_goal_windows,
            forecast_rows,
            1024,
            &self.core.goal0,
            &mut self.predictor_goal_hidden,
        );
        tanh_all(&mut self.predictor_goal_hidden);
        gemm_into(
            &self.predictor_goal_hidden,
            forecast_rows,
            256,
            &self.core.goal2,
            &mut self.predictor_goal_keys,
        );
        let forecast_keys = &self.predictor_goal_keys;
        for row in 0..self.batch {
            if reset[row] {
                self.suffixes
                    .reset_episode(row)
                    .map_err(PyValueError::new_err)?;
                self.pending_suffix_slot[row] = -1;
                self.pending_suffix_generation[row] = 0;
                self.consequences
                    .cancel_pending(row)
                    .map_err(PyValueError::new_err)?;
                self.pending_tick[row] = None;
                if let Some(history) = &mut self.population_history {
                    history.clear(row).map_err(PyValueError::new_err)?;
                }
                self.population_in_domain[row] = 0;
                self.population_out_of_domain[row] = 0;
                self.population_last_in_domain[row] = false;
            }
            let current: [f32; 6] = physiology[row * PHYSIOLOGY..row * PHYSIOLOGY + 6]
                .try_into()
                .unwrap();
            let current12: [f32; 12] = physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                .try_into()
                .unwrap();
            let population_history = self
                .population_history
                .as_ref()
                .map(|history| history.summary(row))
                .transpose()
                .map_err(PyValueError::new_err)?;
            let neural_row: &[f32; 384] =
                neural[row * NEURAL..(row + 1) * NEURAL].try_into().unwrap();
            let desired = if selection.valid[row] && selection.remaining[row] > 0 {
                let end = (row * WINDOW + WINDOW - 1) * OBS;
                let goal: [f32; 6] = selection.window
                    [end + OBS - PHYSIOLOGY..end + OBS - PHYSIOLOGY + 6]
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
            let mut population_tilts = [0.0f32; CANDIDATES];
            let mut population_evaluations = Vec::with_capacity(CANDIDATES);
            let mut population_indices = Vec::with_capacity(CANDIDATES);
            let mut forecast_progress = [0.0f32; CANDIDATES];
            let mut forecast_disagreement = [0.0f32; CANDIDATES];
            let mut forecast_valid = [false; CANDIDATES];
            for k in 0..CANDIDATES {
                let candidate = &candidate_actions
                    [(row * CANDIDATES + k) * ACTIONS..(row * CANDIDATES + k + 1) * ACTIONS];
                if self.candidate_available[row * CANDIDATES + k] {
                    if let (Some(bank), Some(history)) =
                        (&self.population_response, population_history)
                    {
                        let candidate12: [f32; 12] = candidate.try_into().unwrap();
                        let features =
                            population_response_features(&current12, &history, &candidate12)
                                .map_err(PyValueError::new_err)?;
                        population_evaluations
                            .push(bank.evaluate(&features).map_err(PyValueError::new_err)?);
                        population_indices.push(k);
                    }
                }
                let action = [
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[4],
                    candidate[5],
                    candidate[6],
                    candidate[7],
                    candidate[3],
                ];
                let raw = self
                    .law
                    .fitted_features(
                        &DecisionContext {
                            physiology: &current,
                            neural: neural_row,
                        },
                        &CandidateAction {
                            action: &action,
                            oral: candidate[8],
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
            if let Some(bank) = &self.population_response {
                let tilts = bank
                    .candidate_score_tilts(&population_evaluations)
                    .map_err(PyValueError::new_err)?;
                for (k, tilt) in population_indices.into_iter().zip(tilts) {
                    population_tilts[k] = tilt;
                }
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
                self.candidate_scores[row * CANDIDATES + k] += population_tilts[k];
                if self.candidate_recalled[row * CANDIDATES + k] {
                    self.candidate_scores[row * CANDIDATES + k] +=
                        0.10 * self.candidate_suffix_empirical_score[row * CANDIDATES + k];
                }
                if !self.candidate_available[row * CANDIDATES + k] {
                    self.candidate_scores[row * CANDIDATES + k] = -32.0;
                }
                if selection.valid[row]
                    && self.recent_code_count[row] == WINDOW
                    && selection.remaining[row] >= candidate_horizons[row * CANDIDATES + k] as u64
                    && self.predictor_valid[(row * CANDIDATES + k) * FORECAST_HORIZON
                        + candidate_horizons[row * CANDIDATES + k]
                        - 1]
                {
                    let target = &selection.key[row * 64..(row + 1) * 64];
                    let current = (current_key[row * 64..(row + 1) * 64]
                        .iter()
                        .zip(target)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        / 64.0)
                        .sqrt();
                    let mut distances = [0.0f32; PREDICTOR_MEMBERS];
                    for m in 0..PREDICTOR_MEMBERS {
                        let fk = &forecast_keys[((row * CANDIDATES + k) * PREDICTOR_MEMBERS + m)
                            * 64
                            ..((row * CANDIDATES + k) * PREDICTOR_MEMBERS + m + 1) * 64];
                        distances[m] = (fk
                            .iter()
                            .zip(target)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            / 64.0)
                            .sqrt();
                    }
                    let progress =
                        current - distances.iter().sum::<f32>() / PREDICTOR_MEMBERS as f32;
                    let mean_key = (0..64)
                        .map(|j| {
                            (0..PREDICTOR_MEMBERS)
                                .map(|m| {
                                    forecast_keys
                                        [((row * CANDIDATES + k) * PREDICTOR_MEMBERS + m) * 64 + j]
                                })
                                .sum::<f32>()
                                / PREDICTOR_MEMBERS as f32
                        })
                        .collect::<Vec<_>>();
                    let mut spread = 0.0f32;
                    for m in 0..PREDICTOR_MEMBERS {
                        for j in 0..64 {
                            let v = forecast_keys
                                [((row * CANDIDATES + k) * PREDICTOR_MEMBERS + m) * 64 + j]
                                - mean_key[j];
                            spread += v * v;
                        }
                    }
                    let disagreement = (spread / (PREDICTOR_MEMBERS * 64) as f32).sqrt();
                    forecast_progress[k] = progress;
                    forecast_disagreement[k] = disagreement;
                    forecast_valid[k] = true;
                }
            }
            let valid_count = forecast_valid.iter().filter(|x| **x).count();
            if valid_count > 0 {
                let center = forecast_progress
                    .iter()
                    .zip(forecast_valid)
                    .filter(|(_, v)| *v)
                    .map(|(p, _)| *p)
                    .sum::<f32>()
                    / valid_count as f32;
                for k in 0..CANDIDATES {
                    if forecast_valid[k] {
                        let tilt = 0.25
                            * ((forecast_progress[k] - center) / self.forecast_goal_rms).tanh()
                            / (1.0 + forecast_disagreement[k] / self.forecast_goal_rms);
                        self.candidate_scores[row * CANDIDATES + k] += tilt;
                        self.forecast_tilt[row * CANDIDATES + k] = tilt;
                    }
                }
            }
            for k in 0..CANDIDATES {
                self.forecast_progress[row * CANDIDATES + k] = forecast_progress[k];
                self.forecast_disagreement[row * CANDIDATES + k] = forecast_disagreement[k];
                self.forecast_invalid[row * CANDIDATES + k] =
                    !self.predictor_valid[(row * CANDIDATES + k) * FORECAST_HORIZON
                        + candidate_horizons[row * CANDIDATES + k]
                        - 1];
                if !forecast_valid[k] {
                    self.forecast_tilt[row * CANDIDATES + k] = 0.0;
                }
            }
            let chosen = if self.sample {
                categorical_masked(
                    &self.candidate_scores[row * CANDIDATES..(row + 1) * CANDIDATES],
                    &self.candidate_available[row * CANDIDATES..(row + 1) * CANDIDATES],
                    &mut self.action_rng[row * 4..(row + 1) * 4],
                )
            } else {
                self.candidate_scores[row * CANDIDATES..(row + 1) * CANDIDATES]
                    .iter()
                    .zip(&self.candidate_available[row * CANDIDATES..(row + 1) * CANDIDATES])
                    .enumerate()
                    .filter(|(_, (_, available))| **available)
                    .max_by(|a, b| a.1 .0.total_cmp(b.1 .0))
                    .unwrap()
                    .0
            };
            self.selected_candidate[row] = chosen as i32;
            for j in 0..3 {
                self.selected_correction[row * 3 + j] = corrections_all[chosen][j] as f32;
            }
            actions[row * ACTIONS..(row + 1) * ACTIONS].copy_from_slice(
                &candidate_actions[(row * CANDIDATES + chosen) * ACTIONS
                    ..(row * CANDIDATES + chosen + 1) * ACTIONS],
            );
            self.pending_suffix_context[row * SUFFIX_CONTEXT..(row + 1) * SUFFIX_CONTEXT]
                .copy_from_slice(&current_key[row * SUFFIX_CONTEXT..(row + 1) * SUFFIX_CONTEXT]);
            self.pending_suffix_slot[row] = self.candidate_suffix_slot[row * CANDIDATES + chosen];
            self.pending_suffix_generation[row] =
                self.candidate_suffix_generation[row * CANDIDATES + chosen];
            self.pending_action[row * PREVIOUS..row * PREVIOUS + ACTIONS]
                .copy_from_slice(&actions[row * ACTIONS..(row + 1) * ACTIONS]);
            self.pending_physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                .copy_from_slice(&physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]);
            self.pending_tick[row] = Some(ticks[row]);
            if let Some(history) = population_history {
                self.pending_population_history[row * 4..(row + 1) * 4].copy_from_slice(&history);
            }
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
        Ok(actions)
    }
}

#[pymethods]
impl DevelopmentalResidentCohort {
    #[allow(clippy::too_many_arguments)]
    fn expanded(
        &self,
        goal_seed: u64,
        action_seed: u64,
        candidate_sha256: Vec<String>,
        loci_sha256: Vec<String>,
        policy_adapter_index: PyReadonlyArray1<'_, u16>,
        recurrent_gain: PyReadonlyArray1<'_, f32>,
        learning_rate_gain: PyReadonlyArray1<'_, f32>,
        action_gain: PyReadonlyArray2<'_, f32>,
        action_temperature_offset: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Self> {
        self.expanded_inner(
            goal_seed,
            action_seed,
            candidate_sha256,
            loci_sha256,
            policy_adapter_index,
            recurrent_gain,
            learning_rate_gain,
            action_gain,
            action_temperature_offset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn hatch_slots(
        &mut self,
        rows: PyReadonlyArray1<'_, u16>,
        goal_seeds: PyReadonlyArray1<'_, u64>,
        action_seeds: PyReadonlyArray1<'_, u64>,
        candidate_sha256: Vec<String>,
        loci_sha256: Vec<String>,
        policy_adapter_index: PyReadonlyArray1<'_, u16>,
        recurrent_gain: PyReadonlyArray1<'_, f32>,
        learning_rate_gain: PyReadonlyArray1<'_, f32>,
        action_gain: PyReadonlyArray2<'_, f32>,
        action_temperature_offset: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        self.hatch_slots_inner(
            rows,
            goal_seeds,
            action_seeds,
            candidate_sha256,
            loci_sha256,
            policy_adapter_index,
            recurrent_gain,
            learning_rate_gain,
            action_gain,
            action_temperature_offset,
        )
    }

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
        population_response_json: Option<&str>,
        population_response_identity: Option<&str>,
        population_feature_contract_identity: Option<&str>,
        predictor_packed: PyReadonlyArray1<'_, f32>,
        forecast_goal_rms: f32,
        physiology_adapter_packed: PyReadonlyArray1<'_, f32>,
        policy_adapter_packed: PyReadonlyArray1<'_, f32>,
        policy_adapter_count: usize,
        policy_adapter_rank: usize,
        new_actuator_packed: PyReadonlyArray1<'_, f32>,
        policy_adapter_index: PyReadonlyArray1<'_, u16>,
        candidate_sha256: Vec<String>,
        loci_sha256: Vec<String>,
        recurrent_gain: PyReadonlyArray1<'_, f32>,
        learning_rate_gain: PyReadonlyArray1<'_, f32>,
        action_gain: PyReadonlyArray2<'_, f32>,
        action_temperature_offset: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Self> {
        if batch == 0
            || batch > 4096
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
        let population_response = match (
            population_response_json,
            population_response_identity,
            population_feature_contract_identity,
        ) {
            (None, None, None) => None,
            (Some(json), Some(identity), Some(contract)) => Some(
                PopulationResponseBank::from_authenticated_json(json, identity, contract)
                    .map_err(PyValueError::new_err)?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "population response identities are incomplete",
                ))
            }
        };
        let population_history = population_response
            .as_ref()
            .map(|_| PopulationHistory::new(batch).map_err(PyValueError::new_err))
            .transpose()?;
        if !forecast_goal_rms.is_finite() || forecast_goal_rms < 1e-4 {
            return Err(PyValueError::new_err("forecast goal RMS differs"));
        }
        let predictor = PredictiveSensoryEnsemble::from_flat(predictor_packed.as_slice()?)?;
        let physiology_adapter = physiology_adapter_packed.as_slice()?;
        if physiology_adapter.len() != HIDDEN * PHYSIOLOGY
            || physiology_adapter.iter().any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err("physiology adapter weights differ"));
        }
        if policy_adapter_count == 0 || policy_adapter_rank == 0 || policy_adapter_rank > 256 {
            return Err(PyValueError::new_err(
                "population policy adapter dimensions differ",
            ));
        }
        let policy_flat = policy_adapter_packed.as_slice()?;
        let expected_policy = policy_adapter_count
            .checked_mul(policy_adapter_rank * POLICY + POLICY * policy_adapter_rank + POLICY)
            .ok_or_else(|| PyValueError::new_err("population policy adapter size overflow"))?;
        if policy_flat.len() != expected_policy
            || policy_flat.iter().any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err(
                "population policy adapter weights differ",
            ));
        }
        let down_end = policy_adapter_count * policy_adapter_rank * POLICY;
        let up_end = down_end + policy_adapter_count * POLICY * policy_adapter_rank;
        let policy_adapter_down = policy_flat[..down_end].to_vec();
        let policy_adapter_up = policy_flat[down_end..up_end].to_vec();
        let policy_adapter_bias = policy_flat[up_end..].to_vec();
        let policy_adapter_index: Vec<usize> = policy_adapter_index
            .as_slice()?
            .iter()
            .map(|value| *value as usize)
            .collect();
        let recurrent_gain = recurrent_gain.as_slice()?.to_vec();
        let learning_rate_gain = learning_rate_gain.as_slice()?.to_vec();
        let action_gain = action_gain.as_slice()?.to_vec();
        let action_temperature_offset = action_temperature_offset.as_slice()?.to_vec();
        let valid_hash =
            |value: &str| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
        if candidate_sha256.len() != batch
            || loci_sha256.len() != batch
            || candidate_sha256.iter().any(|value| !valid_hash(value))
            || loci_sha256.iter().any(|value| !valid_hash(value))
            || recurrent_gain.len() != batch
            || learning_rate_gain.len() != batch
            || action_gain.len() != batch * ACTIONS
            || action_temperature_offset.len() != batch * ACTIONS
            || policy_adapter_index.len() != batch
            || policy_adapter_index
                .iter()
                .any(|index| *index >= policy_adapter_count)
            || recurrent_gain
                .iter()
                .any(|value| !value.is_finite() || !(0.8..=1.2).contains(value))
            || learning_rate_gain
                .iter()
                .any(|value| !value.is_finite() || !(0.5..=1.5).contains(value))
            || action_gain
                .iter()
                .any(|value| !value.is_finite() || !(0.75..=1.25).contains(value))
            || action_temperature_offset
                .iter()
                .any(|value| !value.is_finite() || !(-0.5..=0.5).contains(value))
        {
            return Err(PyValueError::new_err(
                "candidate controller adapter contract differs",
            ));
        }
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
        let mut actuator_cursor = 0;
        let actuator_flat = new_actuator_packed.as_slice()?;
        let new_actuator_active = linear(actuator_flat, &mut actuator_cursor, 4, POLICY)?;
        let new_actuator_positive = linear(actuator_flat, &mut actuator_cursor, 4 * 32, POLICY)?;
        if actuator_cursor != actuator_flat.len() {
            return Err(PyValueError::new_err(
                "new actuator weight pack has trailing values",
            ));
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
            active: linear(flat, &mut c, 8, POLICY)?,
            positive: linear(flat, &mut c, 8 * 32, POLICY)?,
            manager0: linear(flat, &mut c, HIDDEN, HIDDEN + NEURAL + PHYSIOLOGY)?,
            manager2: linear(flat, &mut c, GOAL, HIDDEN)?,
            query_gain: *flat
                .get(c)
                .ok_or_else(|| PyValueError::new_err("manager gain missing"))?,
            physiology_adapter: Linear {
                out: HIDDEN,
                input: PHYSIOLOGY,
                weight: physiology_adapter.to_vec(),
                bias: vec![0.0; HIDDEN],
            },
            new_actuator_active,
            new_actuator_positive,
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
            recent_codes: vec![0.0; batch * WINDOW * 256],
            recent_code_cursor: vec![0; batch],
            recent_code_count: vec![0; batch],
            observation_input: vec![0.0; batch * (256 + PREVIOUS)],
            physiology_input: vec![0.0; batch * PHYSIOLOGY],
            physiology_delta: vec![0.0; batch * HIDDEN],
            encoded: vec![0.0; batch * HIDDEN],
            gx: vec![0.0; batch * 3 * HIDDEN],
            gh: vec![0.0; batch * 3 * HIDDEN],
            next_state: vec![0.0; batch * HIDDEN],
            goal_flat: vec![0.0; batch * WINDOW * 256],
            goal_middle: vec![0.0; batch * POLICY],
            manager_input: vec![0.0; batch * (HIDDEN + NEURAL + PHYSIOLOGY)],
            manager_hidden: vec![0.0; batch * HIDDEN],
            query: vec![0.0; batch * GOAL],
            logits: vec![0.0; batch * RESERVOIR],
            policy_input: vec![0.0; batch * (HIDDEN + GOAL + 1 + ACTIONS)],
            policy_hidden: vec![0.0; batch * POLICY],
            signed_logits: vec![0.0; batch * 4 * 65],
            active_logits: vec![0.0; batch * 8],
            positive_logits: vec![0.0; batch * 8 * 32],
            new_active_logits: vec![0.0; batch * 4],
            new_positive_logits: vec![0.0; batch * 4 * 32],
            law,
            population_response,
            population_history,
            pending_population_history: vec![0.0; batch * 4],
            population_in_domain: vec![0; batch],
            population_out_of_domain: vec![0; batch],
            population_last_in_domain: vec![false; batch],
            consequences,
            pending_action: vec![0.0; batch * PREVIOUS],
            pending_physiology: vec![0.0; batch * PHYSIOLOGY],
            pending_tick: vec![None; batch],
            candidate_scores: vec![0.0; batch * CANDIDATES],
            candidate_ood: vec![false; batch * CANDIDATES],
            selected_candidate: vec![-1; batch],
            selected_correction: vec![0.0; batch * 3],
            personal_updates: vec![0; batch],
            forecast_progress: vec![0.0; batch * CANDIDATES],
            forecast_disagreement: vec![0.0; batch * CANDIDATES],
            forecast_invalid: vec![false; batch * CANDIDATES],
            forecast_tilt: vec![0.0; batch * CANDIDATES],
            suffixes: MotorSuffixMemory::new(batch, goal_seed ^ 0x4d4f_544f_525f_5637)
                .map_err(PyValueError::new_err)?,
            pending_suffix_context: vec![0.0; batch * SUFFIX_CONTEXT],
            pending_suffix_slot: vec![-1; batch],
            pending_suffix_generation: vec![0; batch],
            candidate_recalled: vec![false; batch * CANDIDATES],
            candidate_available: vec![false; batch * CANDIDATES],
            candidate_suffix_slot: vec![-1; batch * CANDIDATES],
            candidate_suffix_generation: vec![0; batch * CANDIDATES],
            candidate_suffix_length: vec![0; batch * CANDIDATES],
            candidate_suffix_support: vec![0; batch * CANDIDATES],
            candidate_suffix_empirical_score: vec![0.0; batch * CANDIDATES],
            candidate_suffix_recall_score: vec![0.0; batch * CANDIDATES],
            candidate_first_action: vec![0.0; batch * CANDIDATES * ACTIONS],
            predictor,
            predictor_context: Vec::new(),
            predictor_actions: Vec::new(),
            predictor_member_delta: Vec::new(),
            predictor_mean_delta: Vec::new(),
            predictor_disagreement: Vec::new(),
            predictor_absolute_code: Vec::new(),
            predictor_absolute_physiology: Vec::new(),
            predictor_valid: Vec::new(),
            predictor_goal_windows: Vec::new(),
            predictor_goal_hidden: Vec::new(),
            predictor_goal_keys: Vec::new(),
            forecast_goal_rms,
            personal_goals: PersonalGoalAssociations::new(batch, PersonalGoalConfig::current(true))
                .map_err(PyValueError::new_err)?,
            goal_credit_pending: vec![false; batch],
            selected_goal_bias: vec![0.0; batch],
            selected_goal_prediction: vec![0.0; batch],
            last_goal_reward: vec![0.0; batch],
            last_goal_return: vec![0.0; batch],
            last_goal_completed: vec![false; batch],
            last_goal_attributed: vec![false; batch],
            last_goal_learned: vec![false; batch],
            contextual: ContextualEpisodicLearner::new(batch, RESERVOIR)
                .map_err(PyValueError::new_err)?,
            contextual_bias: vec![0.0; batch],
            sequence: GoalSequenceMemory::new(batch, RESERVOIR).map_err(PyValueError::new_err)?,
            goal_measurement_valid: vec![false; batch],
            goal_measurement_slot: vec![-1; batch],
            goal_measurement_recorded_tick: vec![0; batch],
            goal_measurement_generation: vec![0; batch],
            goal_measurement_start_rms: vec![0.0; batch],
            goal_measurement_min_rms: vec![0.0; batch],
            goal_measurement_latest_rms: vec![0.0; batch],
            goal_measurement_samples: vec![0; batch],
            goal_measurement_last_observed_tick: vec![0; batch],
            goal_measurement_context: vec![0.0; batch * CONTEXT],
            last_goal_attained: vec![false; batch],
            last_goal_normalized_progress: vec![0.0; batch],
            sequence_selected_bias: vec![0.0; batch],
            sequence_experienced_path_depth: vec![0; batch],
            sequence_selected_confidence: vec![0.0; batch],
            candidate_sha256,
            loci_sha256,
            recurrent_gain,
            learning_rate_gain,
            action_gain,
            action_temperature_offset,
            recurrent_adapter: vec![0.0; batch * HIDDEN],
            policy_adapter_count,
            policy_adapter_rank,
            policy_adapter_down,
            policy_adapter_up,
            policy_adapter_bias,
            policy_adapter_index,
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
        let (inserted, selection, actions) = py.detach(|| -> PyResult<_> {
            for (row, should_reset) in rst.iter().enumerate() {
                if *should_reset {
                    self.personal_goals
                        .cancel_pending(row)
                        .map_err(PyValueError::new_err)?;
                    self.contextual.cancel(row).map_err(PyValueError::new_err)?;
                    self.sequence
                        .cancel_chain(row)
                        .map_err(PyValueError::new_err)?;
                    self.clear_goal_measurement(row);
                    self.goal_credit_pending[row] = false;
                    self.selected_goal_bias[row] = 0.0;
                    self.selected_goal_prediction[row] = 0.0;
                    self.last_goal_reward[row] = 0.0;
                    self.last_goal_return[row] = 0.0;
                    self.last_goal_completed[row] = false;
                    self.last_goal_attributed[row] = false;
                    self.last_goal_learned[row] = false;
                }
            }
            self.observe(o, a, rst);
            self.remember_frame_codes(rst);
            let (_windows, valid) = self.memory.push_inner(o, t, time, rst)?;
            if valid
                .iter()
                .enumerate()
                .any(|(row, v)| *v != (self.recent_code_count[row] == WINDOW))
            {
                return Err(PyValueError::new_err("raw/code goal rings diverged"));
            }
            let keys = self.encode_windows(&valid);
            let remembered = self.memory.remember_with_changes_inner(&keys, &valid)?;
            let replacements: Vec<_> = remembered
                .slots
                .iter()
                .zip(&remembered.generations)
                .enumerate()
                .filter_map(|(row, (slot, generation))| {
                    (*slot >= 0).then_some(GoalSlotReplacement {
                        resident: row,
                        slot: *slot as usize,
                        identity: GoalSlotIdentity {
                            recorded_tick: t[row],
                            generation: *generation,
                        },
                    })
                })
                .collect();
            self.personal_goals
                .replace_slots(&replacements)
                .map_err(PyValueError::new_err)?;
            self.sequence
                .replace_slots(&replacements)
                .map_err(PyValueError::new_err)?;
            for replacement in &replacements {
                self.contextual
                    .replace(
                        replacement.resident,
                        replacement.slot,
                        replacement.identity.generation,
                        self.episodic_context(replacement.resident, p),
                    )
                    .map_err(PyValueError::new_err)?;
            }
            let selection = self.manager_selection(n, p, t)?;
            self.sample_goal_codes(&selection, &keys, p, t);
            let candidates = self.policy_actions(&selection.key, &selection.remaining);
            let actions =
                self.refine_candidates(&candidates, &selection, p, n, a, &keys, t, rst)?;
            Ok((remembered.slots, selection, actions))
        })?;
        let out = PyDict::new(py);
        out.set_item(
            "proposed_action",
            Array2::from_shape_vec((self.batch, ACTIONS), actions)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "worker_recurrent_context",
            Array2::from_shape_vec(
                (self.batch, HIDDEN),
                self.state
                    .iter()
                    .zip(&self.recurrent_adapter)
                    .map(|(state, adapter)| state + adapter)
                    .collect(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_scores",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_scores.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_is_recalled_suffix",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_recalled.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_available",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_available.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_slot",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_suffix_slot.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_generation",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.candidate_suffix_generation.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_length",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.candidate_suffix_length.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_support",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.candidate_suffix_support.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_empirical_score",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.candidate_suffix_empirical_score.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_recall_score",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.candidate_suffix_recall_score.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        let suffix_counts: Vec<_> = (0..self.batch)
            .map(|row| self.suffixes.counts(row))
            .collect();
        out.set_item(
            "motor_suffix_slots",
            suffix_counts
                .iter()
                .map(|value| value.0)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "motor_suffix_learned_total",
            suffix_counts
                .iter()
                .map(|value| value.1)
                .collect::<Vec<_>>(),
        )?;
        out.set_item("candidate_count", CANDIDATES)?;
        out.set_item(
            "candidate_first_action",
            Array3::from_shape_vec(
                (self.batch, CANDIDATES, ACTIONS),
                self.candidate_first_action.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "motor_suffix_empirical_components",
            ["movement_response", "energy_cost", "fatigue_recovery"],
        )?;
        out.set_item("motor_suffix_empirical_tilt_limit", 0.10f32)?;
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
            "forecast_progress",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.forecast_progress.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "forecast_disagreement",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.forecast_disagreement.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "forecast_invalid",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.forecast_invalid.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "forecast_tilt",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.forecast_tilt.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("forecast_goal_rms", self.forecast_goal_rms)?;
        out.set_item("forecast_horizon_ticks", FORECAST_HORIZON)?;
        let mut predicted_body = vec![0.0f32; self.batch * CANDIDATES * PHYSIOLOGY];
        for row in 0..self.batch * CANDIDATES {
            let horizon = if self.candidate_recalled[row] {
                usize::from(self.candidate_suffix_length[row])
            } else {
                FORECAST_HORIZON
            };
            for member in 0..PREDICTOR_MEMBERS {
                let src = ((row * PREDICTOR_MEMBERS + member) * FORECAST_HORIZON + horizon - 1)
                    * PHYSIOLOGY;
                for channel in 0..PHYSIOLOGY {
                    predicted_body[row * PHYSIOLOGY + channel] += self
                        .predictor_absolute_physiology[src + channel]
                        / PREDICTOR_MEMBERS as f32;
                }
            }
        }
        out.set_item(
            "forecast_physiology",
            Array3::from_shape_vec((self.batch, CANDIDATES, PHYSIOLOGY), predicted_body)
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
            "goal_generation",
            Array1::from_vec(selection.generation).into_pyarray(py),
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
        out.set_item(
            "personal_goal_selected_bias",
            Array1::from_vec(self.selected_goal_bias.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_prediction",
            Array1::from_vec(self.selected_goal_prediction.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_last_reward",
            Array1::from_vec(self.last_goal_reward.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_last_return",
            Array1::from_vec(self.last_goal_return.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_completed",
            Array1::from_vec(self.last_goal_completed.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_attributed",
            Array1::from_vec(self.last_goal_attributed.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_learned",
            Array1::from_vec(self.last_goal_learned.clone()).into_pyarray(py),
        )?;
        let stats: Vec<_> = (0..self.batch)
            .map(|row| self.personal_goals.stats(row))
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;
        out.set_item(
            "personal_goal_completed_total",
            Array1::from_vec(stats.iter().map(|value| value.completed_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_learned_total",
            Array1::from_vec(stats.iter().map(|value| value.learned_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_frozen_total",
            Array1::from_vec(stats.iter().map(|value| value.frozen_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_skipped_total",
            Array1::from_vec(
                stats
                    .iter()
                    .map(|value| value.skipped_replaced_goals)
                    .collect(),
            )
            .into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_cancelled_total",
            Array1::from_vec(stats.iter().map(|value| value.cancelled_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "personal_goal_learning_enabled",
            self.personal_goals.config().learning_enabled,
        )?;
        out.set_item(
            "contextual_retrieval_bias",
            Array1::from_vec(self.contextual_bias.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "contextual_episodic_updates",
            Array1::from_vec(self.contextual.updates().to_vec()).into_pyarray(py),
        )?;
        out.set_item("goal_attainment_rms_threshold", GOAL_ATTAINMENT_RMS)?;
        out.set_item(
            "goal_sequence_selected_bias",
            Array1::from_vec(self.sequence_selected_bias.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_sequence_experienced_path_depth",
            Array1::from_vec(self.sequence_experienced_path_depth.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_sequence_selected_confidence",
            Array1::from_vec(self.sequence_selected_confidence.clone()).into_pyarray(py),
        )?;
        let sequence_stats: Vec<_> = (0..self.batch)
            .map(|row| self.sequence.stats(row))
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;
        out.set_item(
            "goal_sequence_learned_transitions_total",
            sequence_stats
                .iter()
                .map(|value| value.learned_transitions)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "goal_sequence_failed_attempts_total",
            sequence_stats
                .iter()
                .map(|value| value.failed_attempts)
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item("version", 7)?;
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
            "recent_frame_codes",
            Array3::from_shape_vec((self.batch, WINDOW, 256), self.recent_codes.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("recent_code_cursor", self.recent_code_cursor.clone())?;
        out.set_item("recent_code_count", self.recent_code_count.clone())?;
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
        out.set_item(
            "personal_goals",
            self.personal_goals
                .snapshot()
                .map_err(PyValueError::new_err)?,
        )?;
        out.set_item("goal_credit_pending", self.goal_credit_pending.clone())?;
        macro_rules! goal_f32 {
            ($name:literal, $field:expr) => {
                out.set_item($name, Array1::from_vec($field.clone()).into_pyarray(py))?;
            };
        }
        goal_f32!("selected_goal_bias", self.selected_goal_bias);
        goal_f32!("selected_goal_prediction", self.selected_goal_prediction);
        goal_f32!("last_goal_reward", self.last_goal_reward);
        goal_f32!("last_goal_return", self.last_goal_return);
        out.set_item("last_goal_completed", self.last_goal_completed.clone())?;
        out.set_item("last_goal_attributed", self.last_goal_attributed.clone())?;
        out.set_item("last_goal_learned", self.last_goal_learned.clone())?;
        out.set_item("candidate_sha256", self.candidate_sha256.clone())?;
        out.set_item("loci_sha256", self.loci_sha256.clone())?;
        out.set_item("policy_adapter_count", self.policy_adapter_count)?;
        out.set_item("policy_adapter_rank", self.policy_adapter_rank)?;
        out.set_item("policy_adapter_index", self.policy_adapter_index.clone())?;
        out.set_item(
            "recurrent_gain",
            Array1::from_vec(self.recurrent_gain.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "learning_rate_gain",
            Array1::from_vec(self.learning_rate_gain.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "action_gain",
            Array2::from_shape_vec((self.batch, ACTIONS), self.action_gain.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "action_temperature_offset",
            Array2::from_shape_vec(
                (self.batch, ACTIONS),
                self.action_temperature_offset.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "recurrent_adapter",
            Array2::from_shape_vec((self.batch, HIDDEN), self.recurrent_adapter.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "contextual_episodic",
            self.contextual.snapshot().map_err(PyValueError::new_err)?,
        )?;
        out.set_item(
            "contextual_bias",
            Array1::from_vec(self.contextual_bias.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_sequence",
            self.sequence.snapshot().map_err(PyValueError::new_err)?,
        )?;
        out.set_item(
            "goal_measurement_valid",
            self.goal_measurement_valid.clone(),
        )?;
        out.set_item("goal_measurement_slot", self.goal_measurement_slot.clone())?;
        out.set_item(
            "goal_measurement_recorded_tick",
            self.goal_measurement_recorded_tick.clone(),
        )?;
        out.set_item(
            "goal_measurement_generation",
            self.goal_measurement_generation.clone(),
        )?;
        goal_f32!(
            "goal_measurement_start_rms",
            self.goal_measurement_start_rms
        );
        goal_f32!("goal_measurement_min_rms", self.goal_measurement_min_rms);
        goal_f32!(
            "goal_measurement_latest_rms",
            self.goal_measurement_latest_rms
        );
        out.set_item(
            "goal_measurement_samples",
            self.goal_measurement_samples.clone(),
        )?;
        out.set_item(
            "goal_measurement_last_observed_tick",
            self.goal_measurement_last_observed_tick.clone(),
        )?;
        out.set_item(
            "goal_measurement_context",
            self.goal_measurement_context.clone(),
        )?;
        out.set_item("last_goal_attained", self.last_goal_attained.clone())?;
        goal_f32!(
            "last_goal_normalized_progress",
            self.last_goal_normalized_progress
        );
        goal_f32!("sequence_selected_bias", self.sequence_selected_bias);
        out.set_item(
            "sequence_experienced_path_depth",
            self.sequence_experienced_path_depth.clone(),
        )?;
        goal_f32!(
            "sequence_selected_confidence",
            self.sequence_selected_confidence
        );
        out.set_item(
            "population_response_identity",
            self.population_response
                .as_ref()
                .map(|bank| bank.artifact_sha256.clone()),
        )?;
        out.set_item(
            "population_feature_contract_identity",
            self.population_response
                .as_ref()
                .map(|bank| bank.feature_contract_sha256.clone()),
        )?;
        out.set_item(
            "population_history",
            self.population_history
                .as_ref()
                .map(|history| serde_json::to_string(&history.snapshot()).unwrap()),
        )?;
        out.set_item("population_in_domain", self.population_in_domain.clone())?;
        out.set_item(
            "population_out_of_domain",
            self.population_out_of_domain.clone(),
        )?;
        out.set_item(
            "population_last_in_domain",
            self.population_last_in_domain.clone(),
        )?;
        out.set_item(
            "motor_suffix_memory",
            self.suffixes
                .snapshot_json()
                .map_err(PyValueError::new_err)?,
        )?;
        out.set_item(
            "pending_suffix_context",
            Array2::from_shape_vec(
                (self.batch, SUFFIX_CONTEXT),
                self.pending_suffix_context.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item("pending_suffix_slot", self.pending_suffix_slot.clone())?;
        out.set_item(
            "pending_suffix_generation",
            self.pending_suffix_generation.clone(),
        )?;
        Ok(out)
    }

    fn observe_consequences<'py>(
        &mut self,
        py: Python<'py>,
        ticks: PyReadonlyArray1<'_, u64>,
        before: PyReadonlyArray2<'_, f32>,
        after: PyReadonlyArray2<'_, f32>,
        executed: PyReadonlyArray2<'_, f32>,
        effort: PyReadonlyArray1<'_, f32>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if ticks.shape() != [self.batch]
            || before.shape() != [self.batch, PHYSIOLOGY]
            || after.shape() != [self.batch, PHYSIOLOGY]
            || executed.shape() != [self.batch, PREVIOUS]
            || effort.shape() != [self.batch]
        {
            return Err(PyValueError::new_err("consequence outcome shapes differ"));
        }
        let t = ticks.as_slice()?;
        let b = before.as_slice()?;
        let a = after.as_slice()?;
        let x = executed.as_slice()?;
        let e = effort.as_slice()?;
        if b.iter().chain(a).chain(x).chain(e).any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("consequence receipt must be finite"));
        }
        let action_discontinuity: Vec<bool> = (0..self.batch)
            .map(|row| {
                self.pending_action[row * PREVIOUS..(row + 1) * PREVIOUS]
                    != x[row * PREVIOUS..(row + 1) * PREVIOUS]
            })
            .collect();
        let mut targets = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            if self.pending_tick[row] != Some(t[row])
                || self.pending_physiology[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                    != b[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
            {
                return Err(PyValueError::new_err(
                    "executed consequence receipt differs",
                ));
            }
            let bb: [f32; 6] = b[row * PHYSIOLOGY..row * PHYSIOLOGY + 6]
                .try_into()
                .unwrap();
            let aa: [f32; 6] = a[row * PHYSIOLOGY..row * PHYSIOLOGY + 6]
                .try_into()
                .unwrap();
            let target = transition_targets(&bb, &aa);
            if target.iter().any(|v| !v.is_finite()) {
                return Err(PyValueError::new_err("consequence target is nonfinite"));
            }
            targets.push(target);
        }
        for row in 0..self.batch {
            if action_discontinuity[row] {
                self.personal_goals
                    .cancel_pending(row)
                    .map_err(PyValueError::new_err)?;
                self.contextual.cancel(row).map_err(PyValueError::new_err)?;
                self.sequence
                    .cancel_chain(row)
                    .map_err(PyValueError::new_err)?;
                self.goal_credit_pending[row] = false;
                self.clear_goal_measurement(row);
            }
            let outcome = [
                targets[row][0] as f32,
                targets[row][1] as f32,
                targets[row][2] as f32,
            ];
            self.suffixes
                .record_executed(
                    row,
                    t[row],
                    &self.pending_suffix_context[row * SUFFIX_CONTEXT..(row + 1) * SUFFIX_CONTEXT],
                    &x[row * ACTIONS..(row + 1) * ACTIONS],
                    &outcome,
                )
                .map_err(PyValueError::new_err)?;
            if !action_discontinuity[row] && self.pending_suffix_slot[row] >= 0 {
                self.suffixes.note_executed(
                    row,
                    self.pending_suffix_slot[row] as usize,
                    self.pending_suffix_generation[row],
                );
            }
            self.pending_suffix_slot[row] = -1;
            self.pending_suffix_generation[row] = 0;
        }
        let goal_transitions: Vec<_> = (0..self.batch)
            .filter(|row| self.goal_credit_pending[*row])
            .map(|row| GoalTransition {
                resident: row,
                transition_tick: t[row],
                before: [
                    b[row * PHYSIOLOGY] as f64,
                    b[row * PHYSIOLOGY + 1] as f64,
                    b[row * PHYSIOLOGY + 2] as f64,
                ],
                after: [
                    a[row * PHYSIOLOGY] as f64,
                    a[row * PHYSIOLOGY + 1] as f64,
                    a[row * PHYSIOLOGY + 2] as f64,
                ],
                effort: e[row] as f64,
                dt,
            })
            .collect();
        let goal_outcomes = self
            .personal_goals
            .observe_transitions(&goal_transitions)
            .map_err(PyValueError::new_err)?;
        self.last_goal_completed.fill(false);
        self.last_goal_attributed.fill(false);
        self.last_goal_learned.fill(false);
        self.last_goal_attained.fill(false);
        self.last_goal_normalized_progress.fill(0.0);
        for (transition, outcome) in goal_transitions.iter().zip(goal_outcomes) {
            let row = transition.resident;
            self.last_goal_reward[row] = outcome.reward;
            if let Some(receipt) = outcome.receipt {
                self.contextual
                    .observe(
                        row,
                        receipt.normalized_target,
                        receipt.attributed,
                        self.learning_rate_gain[row] as f64,
                    )
                    .map_err(PyValueError::new_err)?;
                self.goal_credit_pending[row] = false;
                self.last_goal_return[row] = receipt.summed_objective_return as f32;
                self.last_goal_completed[row] = true;
                self.last_goal_attributed[row] = receipt.attributed;
                self.last_goal_learned[row] = receipt.learned;
                let measurement_matches = self.goal_measurement_valid[row]
                    && self.goal_measurement_slot[row] == receipt.slot as i32
                    && self.goal_measurement_recorded_tick[row] == receipt.identity.recorded_tick
                    && self.goal_measurement_generation[row] == receipt.identity.generation
                    && self.goal_measurement_samples[row] > 0;
                let attained = measurement_matches
                    && self.goal_measurement_min_rms[row] <= GOAL_ATTAINMENT_RMS;
                let normalized_progress = if measurement_matches {
                    let denominator = self.goal_measurement_start_rms[row]
                        .max(GOAL_ATTAINMENT_RMS)
                        .max(f32::EPSILON);
                    ((self.goal_measurement_start_rms[row] - self.goal_measurement_latest_rms[row])
                        / denominator)
                        .clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                self.last_goal_attained[row] = attained;
                self.last_goal_normalized_progress[row] = normalized_progress;
                // A cached target can remain measurable after reservoir eviction,
                // but its receipt must never be attached to the replacement slot.
                // Let the pending ten-tick receipt finish (unattributed) normally.
                if measurement_matches
                    && receipt.attributed
                    && self.personal_goals.config().learning_enabled
                {
                    let context: [f64; CONTEXT] = self.goal_measurement_context
                        [row * CONTEXT..(row + 1) * CONTEXT]
                        .try_into()
                        .unwrap();
                    self.sequence
                        .observe_episode(SequenceEpisode {
                            resident: row,
                            node: SequenceNode {
                                slot: receipt.slot,
                                identity: receipt.identity,
                            },
                            completed_tick: receipt.completed_at_tick,
                            context,
                            normalized_return: receipt.normalized_target,
                            normalized_progress: normalized_progress as f64,
                            attained,
                            attributed: receipt.attributed,
                        })
                        .map_err(PyValueError::new_err)?;
                    self.sequence
                        .consolidate(row, SEQUENCE_CONSOLIDATION_BUDGET)
                        .map_err(PyValueError::new_err)?;
                }
                self.goal_measurement_valid[row] = false;
                self.goal_measurement_slot[row] = -1;
                self.goal_measurement_recorded_tick[row] = 0;
                self.goal_measurement_generation[row] = 0;
                self.goal_measurement_context[row * CONTEXT..(row + 1) * CONTEXT].fill(0.0);
            }
        }
        for row in 0..self.batch {
            if let Some(bank) = &self.population_response {
                let before12: [f32; 12] = b[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                    .try_into()
                    .unwrap();
                let executed12: [f32; 12] = x[row * PREVIOUS..row * PREVIOUS + ACTIONS]
                    .try_into()
                    .unwrap();
                let history: [f32; 4] = self.pending_population_history[row * 4..(row + 1) * 4]
                    .try_into()
                    .unwrap();
                let features = population_response_features(&before12, &history, &executed12)
                    .map_err(PyValueError::new_err)?;
                let evaluation = bank.evaluate(&features).map_err(PyValueError::new_err)?;
                self.population_last_in_domain[row] = !evaluation.out_of_domain;
                if evaluation.out_of_domain {
                    self.population_out_of_domain[row] += 1;
                } else {
                    self.population_in_domain[row] += 1;
                }
            }
            if let Some(history) = &mut self.population_history {
                let after12: [f32; 12] = a[row * PHYSIOLOGY..(row + 1) * PHYSIOLOGY]
                    .try_into()
                    .unwrap();
                history
                    .record(row, &after12)
                    .map_err(PyValueError::new_err)?;
            }
            self.consequences
                .observe(row, t[row], &targets[row])
                .map_err(PyValueError::new_err)?;
            self.pending_tick[row] = None;
        }
        let out = PyDict::new(py);
        if self.population_response.is_some() {
            let bank = self.population_response.as_ref().unwrap();
            out.set_item("population_response_identity", bank.artifact_sha256.clone())?;
            out.set_item(
                "population_feature_contract_identity",
                bank.feature_contract_sha256.clone(),
            )?;
            out.set_item(
                "population_response_in_domain",
                self.population_last_in_domain.clone(),
            )?;
            out.set_item(
                "population_response_in_domain_total",
                self.population_in_domain.clone(),
            )?;
            out.set_item(
                "population_response_out_of_domain_total",
                self.population_out_of_domain.clone(),
            )?;
        }
        out.set_item(
            "reward",
            Array1::from_vec(self.last_goal_reward.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "completed",
            Array1::from_vec(self.last_goal_completed.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "summed_return",
            Array1::from_vec(self.last_goal_return.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "attributed",
            Array1::from_vec(self.last_goal_attributed.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "learned",
            Array1::from_vec(self.last_goal_learned.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "actual_attained",
            Array1::from_vec(self.last_goal_attained.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "observed_normalized_progress",
            Array1::from_vec(self.last_goal_normalized_progress.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "measurement_start_rms",
            Array1::from_vec(self.goal_measurement_start_rms.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "measurement_min_rms",
            Array1::from_vec(self.goal_measurement_min_rms.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "measurement_latest_rms",
            Array1::from_vec(self.goal_measurement_latest_rms.clone()).into_pyarray(py),
        )?;
        out.set_item("measurement_samples", self.goal_measurement_samples.clone())?;
        out.set_item(
            "measurement_window_ending_last_observed_tick",
            self.goal_measurement_last_observed_tick.clone(),
        )?;
        let stats: Vec<_> = (0..self.batch)
            .map(|row| self.personal_goals.stats(row))
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;
        out.set_item(
            "completed_total",
            Array1::from_vec(stats.iter().map(|value| value.completed_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "learned_total",
            Array1::from_vec(stats.iter().map(|value| value.learned_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "frozen_total",
            Array1::from_vec(stats.iter().map(|value| value.frozen_goals).collect())
                .into_pyarray(py),
        )?;
        out.set_item(
            "skipped_total",
            Array1::from_vec(
                stats
                    .iter()
                    .map(|value| value.skipped_replaced_goals)
                    .collect(),
            )
            .into_pyarray(py),
        )?;
        out.set_item(
            "cancelled_total",
            Array1::from_vec(stats.iter().map(|value| value.cancelled_goals).collect())
                .into_pyarray(py),
        )?;
        Ok(out)
    }

    fn set_personal_goal_learning(&mut self, enabled: bool) {
        self.personal_goals.set_learning_enabled(enabled);
        if !enabled {
            for row in 0..self.batch {
                let _ = self.sequence.cancel_chain(row);
            }
        }
    }

    fn restore(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        let get = |name: &str| {
            value.get_item(name)?.ok_or_else(|| {
                PyValueError::new_err(format!("developmental snapshot lacks {name}"))
            })
        };
        if get("format")?.extract::<String>()? != FORMAT
            || get("version")?.extract::<u8>()? != 7
            || get("batch")?.extract::<usize>()? != self.batch
            || get("conditioned")?.extract::<bool>()? != self.conditioned
            || get("sample")?.extract::<bool>()? != self.sample
        {
            return Err(PyValueError::new_err(
                "developmental snapshot identity differs",
            ));
        }
        let expected_bank = self
            .population_response
            .as_ref()
            .map(|bank| bank.artifact_sha256.clone());
        let expected_contract = self
            .population_response
            .as_ref()
            .map(|bank| bank.feature_contract_sha256.clone());
        if get("population_response_identity")?.extract::<Option<String>>()? != expected_bank
            || get("population_feature_contract_identity")?.extract::<Option<String>>()?
                != expected_contract
        {
            return Err(PyValueError::new_err(
                "population response snapshot identity differs",
            ));
        }
        let history_json = get("population_history")?.extract::<Option<String>>()?;
        self.population_history = match history_json {
            Some(json) if self.population_response.is_some() => Some(
                PopulationHistory::restore(
                    serde_json::from_str(&json)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                )
                .map_err(PyValueError::new_err)?,
            ),
            None if self.population_response.is_none() => None,
            _ => {
                return Err(PyValueError::new_err(
                    "population history snapshot identity differs",
                ))
            }
        };
        self.population_in_domain = get("population_in_domain")?.extract()?;
        self.population_out_of_domain = get("population_out_of_domain")?.extract()?;
        self.population_last_in_domain = get("population_last_in_domain")?.extract()?;
        if self.population_in_domain.len() != self.batch
            || self.population_out_of_domain.len() != self.batch
            || self.population_last_in_domain.len() != self.batch
        {
            return Err(PyValueError::new_err(
                "population response receipt dimensions differ",
            ));
        }
        self.suffixes = MotorSuffixMemory::restore_json(
            &get("motor_suffix_memory")?.extract::<String>()?,
            self.batch,
        )
        .map_err(PyValueError::new_err)?;
        let pending_suffix_context: PyReadonlyArray2<'_, f32> =
            get("pending_suffix_context")?.extract()?;
        self.pending_suffix_slot = get("pending_suffix_slot")?.extract()?;
        self.pending_suffix_generation = get("pending_suffix_generation")?.extract()?;
        if pending_suffix_context.shape() != [self.batch, SUFFIX_CONTEXT]
            || pending_suffix_context
                .as_slice()?
                .iter()
                .any(|x| !x.is_finite())
            || self.pending_suffix_slot.len() != self.batch
            || self.pending_suffix_generation.len() != self.batch
            || self
                .pending_suffix_slot
                .iter()
                .any(|slot| *slot < -1 || *slot >= 32)
        {
            return Err(PyValueError::new_err(
                "pending motor suffix snapshot dimensions differ",
            ));
        }
        self.pending_suffix_context = pending_suffix_context.as_slice()?.to_vec();
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
        let codes: PyReadonlyArray3<'_, f32> = get("recent_frame_codes")?.extract()?;
        let cursor: Vec<usize> = get("recent_code_cursor")?.extract()?;
        let count: Vec<usize> = get("recent_code_count")?.extract()?;
        if codes.shape() != [self.batch, WINDOW, 256]
            || cursor.len() != self.batch
            || count.len() != self.batch
            || cursor.iter().any(|x| *x >= WINDOW)
            || count.iter().any(|x| *x > WINDOW)
            || codes.as_slice()?.iter().any(|x| !x.is_finite())
        {
            return Err(PyValueError::new_err("cached goal-code state differs"));
        }
        self.recent_codes.copy_from_slice(codes.as_slice()?);
        self.recent_code_cursor = cursor;
        self.recent_code_count = count;
        let (raw_count, raw_cursor) = self.memory.recent_ring_state();
        if (0..self.batch).any(|row| {
            self.recent_code_count[row] != raw_count[row] as usize
                || self.recent_code_cursor[row] != raw_cursor[row] as usize
        }) {
            return Err(PyValueError::new_err(
                "raw and encoded recent rings diverged",
            ));
        }
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
        let personal_goals = get("personal_goals")?.extract::<String>()?;
        self.personal_goals = PersonalGoalAssociations::restore(
            &personal_goals,
            &PersonalGoalConfig::current(true),
            self.batch,
        )
        .map_err(PyValueError::new_err)?;
        self.goal_credit_pending = get("goal_credit_pending")?.extract()?;
        macro_rules! restore_goal_f32 {
            ($name:literal, $field:expr) => {{
                let values: PyReadonlyArray1<'_, f32> = get($name)?.extract()?;
                if values.shape() != [self.batch]
                    || values.as_slice()?.iter().any(|value| !value.is_finite())
                {
                    return Err(PyValueError::new_err(concat!(
                        "invalid developmental snapshot field: ",
                        $name
                    )));
                }
                $field.copy_from_slice(values.as_slice()?);
            }};
        }
        restore_goal_f32!("selected_goal_bias", self.selected_goal_bias);
        restore_goal_f32!("selected_goal_prediction", self.selected_goal_prediction);
        restore_goal_f32!("last_goal_reward", self.last_goal_reward);
        restore_goal_f32!("last_goal_return", self.last_goal_return);
        self.last_goal_completed = get("last_goal_completed")?.extract()?;
        self.last_goal_attributed = get("last_goal_attributed")?.extract()?;
        self.last_goal_learned = get("last_goal_learned")?.extract()?;
        if self.goal_credit_pending.len() != self.batch
            || self.last_goal_completed.len() != self.batch
            || self.last_goal_attributed.len() != self.batch
            || self.last_goal_learned.len() != self.batch
        {
            return Err(PyValueError::new_err("private goal snapshot shapes differ"));
        }
        if get("candidate_sha256")?.extract::<Vec<String>>()? != self.candidate_sha256
            || get("loci_sha256")?.extract::<Vec<String>>()? != self.loci_sha256
            || get("policy_adapter_count")?.extract::<usize>()? != self.policy_adapter_count
            || get("policy_adapter_rank")?.extract::<usize>()? != self.policy_adapter_rank
            || get("policy_adapter_index")?.extract::<Vec<usize>>()? != self.policy_adapter_index
        {
            return Err(PyValueError::new_err(
                "candidate adapter snapshot identity differs",
            ));
        }
        macro_rules! require_adapter {
            ($name:literal, $expected:expr, $shape:expr) => {{
                let values: PyReadonlyArray2<'_, f32> = get($name)?.extract()?;
                if values.shape() != $shape || values.as_slice()? != $expected.as_slice() {
                    return Err(PyValueError::new_err(concat!(
                        "candidate adapter snapshot differs: ",
                        $name
                    )));
                }
            }};
        }
        let recurrent: PyReadonlyArray1<'_, f32> = get("recurrent_gain")?.extract()?;
        let learning: PyReadonlyArray1<'_, f32> = get("learning_rate_gain")?.extract()?;
        if recurrent.shape() != [self.batch]
            || learning.shape() != [self.batch]
            || recurrent.as_slice()? != self.recurrent_gain
            || learning.as_slice()? != self.learning_rate_gain
        {
            return Err(PyValueError::new_err(
                "candidate scalar adapter snapshot differs",
            ));
        }
        require_adapter!("action_gain", self.action_gain, [self.batch, ACTIONS]);
        require_adapter!(
            "action_temperature_offset",
            self.action_temperature_offset,
            [self.batch, ACTIONS]
        );
        let adapter: PyReadonlyArray2<'_, f32> = get("recurrent_adapter")?.extract()?;
        if adapter.shape() != [self.batch, HIDDEN]
            || adapter.as_slice()?.iter().any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err(
                "private recurrent adapter snapshot differs",
            ));
        }
        self.recurrent_adapter.copy_from_slice(adapter.as_slice()?);
        self.contextual = ContextualEpisodicLearner::restore(
            &get("contextual_episodic")?.extract::<String>()?,
            self.batch,
            RESERVOIR,
        )
        .map_err(PyValueError::new_err)?;
        let contextual_bias: PyReadonlyArray1<'_, f32> = get("contextual_bias")?.extract()?;
        if contextual_bias.shape() != [self.batch]
            || contextual_bias
                .as_slice()?
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err("contextual bias snapshot differs"));
        }
        self.contextual_bias
            .copy_from_slice(contextual_bias.as_slice()?);
        self.sequence = GoalSequenceMemory::restore(
            &get("goal_sequence")?.extract::<String>()?,
            self.batch,
            RESERVOIR,
        )
        .map_err(PyValueError::new_err)?;
        self.goal_measurement_valid = get("goal_measurement_valid")?.extract()?;
        self.goal_measurement_slot = get("goal_measurement_slot")?.extract()?;
        self.goal_measurement_recorded_tick = get("goal_measurement_recorded_tick")?.extract()?;
        self.goal_measurement_generation = get("goal_measurement_generation")?.extract()?;
        restore_goal_f32!(
            "goal_measurement_start_rms",
            self.goal_measurement_start_rms
        );
        restore_goal_f32!("goal_measurement_min_rms", self.goal_measurement_min_rms);
        restore_goal_f32!(
            "goal_measurement_latest_rms",
            self.goal_measurement_latest_rms
        );
        self.goal_measurement_samples = get("goal_measurement_samples")?.extract()?;
        self.goal_measurement_last_observed_tick =
            get("goal_measurement_last_observed_tick")?.extract()?;
        self.goal_measurement_context = get("goal_measurement_context")?.extract()?;
        self.last_goal_attained = get("last_goal_attained")?.extract()?;
        restore_goal_f32!(
            "last_goal_normalized_progress",
            self.last_goal_normalized_progress
        );
        restore_goal_f32!("sequence_selected_bias", self.sequence_selected_bias);
        self.sequence_experienced_path_depth = get("sequence_experienced_path_depth")?.extract()?;
        restore_goal_f32!(
            "sequence_selected_confidence",
            self.sequence_selected_confidence
        );
        if self.goal_measurement_valid.len() != self.batch
            || self.goal_measurement_slot.len() != self.batch
            || self.goal_measurement_recorded_tick.len() != self.batch
            || self.goal_measurement_generation.len() != self.batch
            || self.goal_measurement_samples.len() != self.batch
            || self.goal_measurement_last_observed_tick.len() != self.batch
            || self.goal_measurement_context.len() != self.batch * CONTEXT
            || self
                .goal_measurement_context
                .iter()
                .any(|value| !value.is_finite())
            || self.last_goal_attained.len() != self.batch
            || self.sequence_experienced_path_depth.len() != self.batch
            || (0..self.batch).any(|row| {
                self.goal_measurement_valid[row]
                    && (self.goal_measurement_slot[row] < 0
                        || self.goal_measurement_slot[row] as usize >= RESERVOIR
                        || self.goal_measurement_generation[row] == 0
                        || self.goal_measurement_samples[row] == 0)
            })
        {
            return Err(PyValueError::new_err(
                "goal sequence snapshot dimensions differ",
            ));
        }
        self.state.copy_from_slice(h);
        self.previous_action.copy_from_slice(p);
        self.action_rng.copy_from_slice(r);
        Ok(())
    }
}
