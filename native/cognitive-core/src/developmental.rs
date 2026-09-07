// SPDX-License-Identifier: AGPL-3.0-or-later
//! CNS-only resident controller. The physical world never enters this module as features.

use crate::learned_sequence_control::{
    ControlDecision, LearnedSequenceControl, CANDIDATES, CHOICE, STATE as CONTROL_STATE,
};
use crate::motor_suffix::{CancellationReason, MotorSuffixMemory, ACTIONS, CONTEXT, MAX_HORIZON};
use crate::{gemm_into, gru, linear, tanh_all, Gru, Linear};
use numpy::{
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::{Deserialize, Serialize};

const FORMAT: &str = "chreatures-cns-resident-native-v10";
const Z: usize = 512;
const HIDDEN: usize = 256;
const GOAL: usize = 128;
const CORE_INPUT: usize = Z + ACTIONS;
const GOAL_INPUT: usize = Z + HIDDEN;
const PROPOSAL_INPUT: usize = HIDDEN + GOAL + ACTIONS;
const PROPOSAL_HIDDEN: usize = 256;
const LOCAL: usize = 4;
const RECALLED: usize = CANDIDATES - LOCAL;
const CANONICAL_CONTEXT: usize = Z + HIDDEN + GOAL + ACTIONS;
const PREDICTOR_MEMBERS: usize = 3;
const GOAL_SLOTS: usize = 128;
const GOAL_HORIZON: u64 = MAX_HORIZON as u64;
const GOAL_MEMORY_FORMAT: &str = "chreatures-private-cns-goal-memory-v1";

#[derive(Clone)]
struct Core {
    recurrent: Gru,
    goal_encoder: Linear,
    proposal_hidden: Linear,
    proposal_out: Linear,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CnsGoalMemory {
    format: String,
    batch: usize,
    valid: Vec<bool>,
    keys: Vec<f32>,
    recorded_tick: Vec<u64>,
    generation: Vec<u64>,
    count: Vec<usize>,
    seen: Vec<u64>,
    rng: Vec<u64>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl CnsGoalMemory {
    fn new(batch: usize, seed: u64) -> Self {
        let mut rng = vec![0; batch];
        for (row, value) in rng.iter_mut().enumerate() {
            let mut s = seed ^ (row as u64).wrapping_mul(0xa076_1d64_78bd_642f);
            *value = splitmix64(&mut s);
        }
        Self {
            format: GOAL_MEMORY_FORMAT.into(),
            batch,
            valid: vec![false; batch * GOAL_SLOTS],
            keys: vec![0.0; batch * GOAL_SLOTS * GOAL],
            recorded_tick: vec![0; batch * GOAL_SLOTS],
            generation: vec![0; batch * GOAL_SLOTS],
            count: vec![0; batch],
            seen: vec![0; batch],
            rng,
        }
    }
    fn select(
        &mut self,
        row: usize,
        query: &[f32],
        sample: bool,
    ) -> Option<(usize, u64, u64, Vec<f32>)> {
        let available = (0..GOAL_SLOTS)
            .filter(|slot| self.valid[row * GOAL_SLOTS + *slot])
            .collect::<Vec<_>>();
        if available.is_empty() {
            return None;
        }
        let scores = available
            .iter()
            .map(|slot| {
                let i = row * GOAL_SLOTS + *slot;
                self.keys[i * GOAL..(i + 1) * GOAL]
                    .iter()
                    .zip(query)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let chosen = if sample {
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total = scores.iter().map(|x| (*x - max).exp() as f64).sum::<f64>();
            let draw = ((splitmix64(&mut self.rng[row]) >> 11) as f64)
                * (1.0 / ((1u64 << 53) as f64))
                * total;
            let mut sum = 0.0;
            let mut chosen = 0;
            for (i, score) in scores.iter().enumerate() {
                sum += (*score - max).exp() as f64;
                if draw < sum {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            let mut chosen = 0;
            for i in 1..scores.len() {
                if scores[i] > scores[chosen] {
                    chosen = i
                }
            }
            chosen
        };
        let slot = available[chosen];
        let index = row * GOAL_SLOTS + slot;
        Some((
            slot,
            self.generation[index],
            self.recorded_tick[index],
            self.keys[index * GOAL..(index + 1) * GOAL].to_vec(),
        ))
    }
    fn grow(&mut self, new_batch: usize, seed: u64) -> Result<(), String> {
        if new_batch <= self.batch || new_batch > 4096 {
            return Err("CNS goal-memory growth differs".into());
        }
        let old_batch = self.batch;
        self.valid.resize(new_batch * GOAL_SLOTS, false);
        self.keys.resize(new_batch * GOAL_SLOTS * GOAL, 0.0);
        self.recorded_tick.resize(new_batch * GOAL_SLOTS, 0);
        self.generation.resize(new_batch * GOAL_SLOTS, 0);
        self.count.resize(new_batch, 0);
        self.seen.resize(new_batch, 0);
        self.rng.resize(new_batch, 0);
        for row in old_batch..new_batch {
            let mut state = seed ^ (row as u64).wrapping_mul(0xa076_1d64_78bd_642f);
            self.rng[row] = splitmix64(&mut state);
        }
        self.batch = new_batch;
        Ok(())
    }
    fn insert(&mut self, row: usize, key: &[f32], tick: u64, protected: i32) -> i32 {
        self.seen[row] = self.seen[row].saturating_add(1);
        let slot = if self.count[row] < GOAL_SLOTS {
            let s = self.count[row];
            self.count[row] += 1;
            s
        } else {
            let draw = splitmix64(&mut self.rng[row]) % self.seen[row];
            if draw >= GOAL_SLOTS as u64 || draw as i32 == protected {
                return -1;
            }
            draw as usize
        };
        let index = row * GOAL_SLOTS + slot;
        self.valid[index] = true;
        self.generation[index] = self.generation[index].wrapping_add(1).max(1);
        self.recorded_tick[index] = tick;
        self.keys[index * GOAL..(index + 1) * GOAL].copy_from_slice(key);
        slot as i32
    }
    fn validate(&self, batch: usize) -> bool {
        self.format == GOAL_MEMORY_FORMAT
            && self.batch == batch
            && self.valid.len() == batch * GOAL_SLOTS
            && self.keys.len() == batch * GOAL_SLOTS * GOAL
            && self.recorded_tick.len() == batch * GOAL_SLOTS
            && self.generation.len() == batch * GOAL_SLOTS
            && self.count.len() == batch
            && self.seen.len() == batch
            && self.rng.len() == batch
            && self.count.iter().all(|x| *x <= GOAL_SLOTS)
            && (0..batch).all(|row| {
                self.valid[row * GOAL_SLOTS..(row + 1) * GOAL_SLOTS]
                    .iter()
                    .filter(|valid| **valid)
                    .count()
                    == self.count[row]
                    && self.seen[row] >= self.count[row] as u64
            })
            && self.keys.iter().all(|x| x.is_finite())
    }
}

#[derive(Clone)]
struct PredictorMember {
    context_encoder: Linear,
    transition: Gru,
    latent_delta: Linear,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivateSnapshot {
    format: String,
    batch: usize,
    sample: bool,
    research_training: bool,
    core_sha256: String,
    predictor_sha256: String,
    state: Vec<f32>,
    previous_command: Vec<f32>,
    current_key: Vec<f32>,
    goal: Vec<f32>,
    goal_origin_slot: Vec<i32>,
    goal_origin_generation: Vec<u64>,
    goal_origin_tick: Vec<u64>,
    goal_selected_tick: Vec<u64>,
    memory_inserted_slot: Vec<i32>,
    command_pending: Vec<bool>,
    pending_tick: Vec<u64>,
    pending_command: Vec<f32>,
    pending_context: Vec<f32>,
    pending_goal: Vec<f32>,
    acknowledged_valid: Vec<bool>,
    acknowledged_suffix: Vec<bool>,
    acknowledged_tick: Vec<u64>,
    acknowledged_command: Vec<f32>,
    acknowledged_context: Vec<f32>,
    acknowledged_goal: Vec<f32>,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct DevelopmentalResidentCohort {
    batch: usize,
    sample: bool,
    research_training: bool,
    core_sha256: String,
    predictor_sha256: String,
    core: Core,
    predictor: Vec<PredictorMember>,
    sequence_control: LearnedSequenceControl,
    suffixes: MotorSuffixMemory,
    goal_memory: CnsGoalMemory,
    state: Vec<f32>,
    previous_command: Vec<f32>,
    current_key: Vec<f32>,
    goal: Vec<f32>,
    goal_origin_slot: Vec<i32>,
    goal_origin_generation: Vec<u64>,
    goal_origin_tick: Vec<u64>,
    goal_selected_tick: Vec<u64>,
    memory_inserted_slot: Vec<i32>,
    canonical_context: Vec<f32>,
    command_pending: Vec<bool>,
    pending_tick: Vec<u64>,
    pending_command: Vec<f32>,
    pending_context: Vec<f32>,
    pending_goal: Vec<f32>,
    acknowledged_valid: Vec<bool>,
    acknowledged_suffix: Vec<bool>,
    acknowledged_tick: Vec<u64>,
    acknowledged_command: Vec<f32>,
    acknowledged_context: Vec<f32>,
    acknowledged_goal: Vec<f32>,
    candidate_actions: Vec<f32>,
    candidate_sequences: Vec<f32>,
    candidate_mask: Vec<bool>,
    candidate_recalled: Vec<bool>,
    candidate_slot: Vec<i32>,
    candidate_generation: Vec<u64>,
    candidate_length: Vec<u8>,
    candidate_support: Vec<u32>,
    candidate_recall: Vec<f32>,
    candidate_empirical: Vec<f32>,
    control_state: Vec<f32>,
    control_proposal: Vec<f32>,
    control_active: Vec<f32>,
    control_active_mask: Vec<bool>,
    active_source_slot: Vec<i32>,
    active_source_generation: Vec<u64>,
    active_phase: Vec<u8>,
    active_remaining: Vec<u8>,
    decision: ControlDecision,
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}

fn commands_valid(values: &[f32]) -> bool {
    values.chunks_exact(ACTIONS).all(|command| {
        command[..4]
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
            && command[4..]
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    })
}

fn normalize_rows(values: &mut [f32], width: usize) {
    for row in values.chunks_exact_mut(width) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in row {
            *x /= norm;
        }
    }
}

fn decision_empty(batch: usize) -> ControlDecision {
    ControlDecision {
        hazard_logits: vec![0.0; batch],
        selector_logits: vec![0.0; batch * CANDIDATES],
        values: vec![0.0; batch],
        terminate: vec![false; batch],
        selected: vec![-1; batch],
        hazard_mask: vec![false; batch],
        selector_mask: vec![false; batch],
        hazard_logp: vec![0.0; batch],
        selector_logp: vec![0.0; batch],
        logp: vec![0.0; batch],
    }
}

impl DevelopmentalResidentCohort {
    fn expanded_inner(
        &self,
        additions: usize,
        action_seed: u64,
        suffix_seed: u64,
    ) -> Result<Self, String> {
        let new_batch = self
            .batch
            .checked_add(additions)
            .filter(|batch| additions > 0 && *batch <= 4096)
            .ok_or("CNS resident expansion differs")?;
        if self.command_pending.iter().any(|pending| *pending) {
            return Err("CNS resident expansion requires acknowledged commands".into());
        }
        let mut result = self.clone();
        result
            .sequence_control
            .grow(new_batch, action_seed ^ 0x5345_515f_4354_524c)?;
        result.suffixes.grow(new_batch, suffix_seed)?;
        result
            .goal_memory
            .grow(new_batch, suffix_seed ^ 0x474f_414c_4d45_4d31)?;
        macro_rules! grow {
            ($field:ident, $stride:expr, $value:expr) => {
                result.$field.resize(new_batch * $stride, $value)
            };
        }
        grow!(state, HIDDEN, 0.0);
        grow!(previous_command, ACTIONS, 0.0);
        grow!(current_key, GOAL, 0.0);
        grow!(goal, GOAL, 0.0);
        grow!(goal_origin_slot, 1, -1);
        grow!(goal_origin_generation, 1, 0);
        grow!(goal_origin_tick, 1, 0);
        grow!(goal_selected_tick, 1, 0);
        grow!(memory_inserted_slot, 1, -1);
        grow!(canonical_context, CANONICAL_CONTEXT, 0.0);
        grow!(command_pending, 1, false);
        grow!(pending_tick, 1, 0);
        grow!(pending_command, ACTIONS, 0.0);
        grow!(pending_context, CONTEXT, 0.0);
        grow!(pending_goal, GOAL, 0.0);
        grow!(acknowledged_valid, 1, false);
        grow!(acknowledged_suffix, 1, false);
        grow!(acknowledged_tick, 1, 0);
        grow!(acknowledged_command, ACTIONS, 0.0);
        grow!(acknowledged_context, CONTEXT, 0.0);
        grow!(acknowledged_goal, GOAL, 0.0);
        grow!(candidate_actions, CANDIDATES * ACTIONS, 0.0);
        grow!(candidate_sequences, CANDIDATES * MAX_HORIZON * ACTIONS, 0.0);
        grow!(candidate_mask, CANDIDATES, false);
        grow!(candidate_recalled, CANDIDATES, false);
        grow!(candidate_slot, CANDIDATES, -1);
        grow!(candidate_generation, CANDIDATES, 0);
        grow!(candidate_length, CANDIDATES, 0);
        grow!(candidate_support, CANDIDATES, 0);
        grow!(candidate_recall, CANDIDATES, 0.0);
        grow!(candidate_empirical, CANDIDATES, 0.0);
        grow!(control_state, CONTROL_STATE, 0.0);
        grow!(control_proposal, CANDIDATES * CHOICE, 0.0);
        grow!(control_active, CHOICE, 0.0);
        grow!(control_active_mask, 1, false);
        grow!(active_source_slot, 1, -1);
        grow!(active_source_generation, 1, 0);
        grow!(active_phase, 1, 0);
        grow!(active_remaining, 1, 0);
        result.decision.hazard_logits.resize(new_batch, 0.0);
        result
            .decision
            .selector_logits
            .resize(new_batch * CANDIDATES, 0.0);
        result.decision.values.resize(new_batch, 0.0);
        result.decision.terminate.resize(new_batch, false);
        result.decision.selected.resize(new_batch, -1);
        result.decision.hazard_mask.resize(new_batch, false);
        result.decision.selector_mask.resize(new_batch, false);
        result.decision.hazard_logp.resize(new_batch, 0.0);
        result.decision.selector_logp.resize(new_batch, 0.0);
        result.decision.logp.resize(new_batch, 0.0);
        result.batch = new_batch;
        Ok(result)
    }

    fn encode_goal(&self, z: &[f32], hidden: &[f32]) -> Vec<f32> {
        let rows = z.len() / Z;
        let mut input = vec![0.0; rows * GOAL_INPUT];
        for row in 0..rows {
            input[row * GOAL_INPUT..row * GOAL_INPUT + Z]
                .copy_from_slice(&z[row * Z..(row + 1) * Z]);
            input[row * GOAL_INPUT + Z..(row + 1) * GOAL_INPUT]
                .copy_from_slice(&hidden[row * HIDDEN..(row + 1) * HIDDEN]);
        }
        let mut goal = Vec::new();
        gemm_into(&input, rows, GOAL_INPUT, &self.core.goal_encoder, &mut goal);
        tanh_all(&mut goal);
        normalize_rows(&mut goal, GOAL);
        goal
    }

    fn core_observe(&mut self, z: &[f32], previous: &[f32], ticks: &[u64], reset: &[bool]) {
        for row in 0..self.batch {
            if reset[row] {
                self.state[row * HIDDEN..(row + 1) * HIDDEN].fill(0.0);
                self.previous_command[row * ACTIONS..(row + 1) * ACTIONS].fill(0.0);
            }
        }
        let mut input = vec![0.0; self.batch * CORE_INPUT];
        for row in 0..self.batch {
            input[row * CORE_INPUT..row * CORE_INPUT + Z]
                .copy_from_slice(&z[row * Z..(row + 1) * Z]);
            if !reset[row] {
                input[row * CORE_INPUT + Z..(row + 1) * CORE_INPUT]
                    .copy_from_slice(&previous[row * ACTIONS..(row + 1) * ACTIONS]);
            }
        }
        let mut gx = Vec::new();
        let mut gh = Vec::new();
        let mut next = Vec::new();
        self.core
            .recurrent
            .step_into(&input, &self.state, self.batch, &mut gx, &mut gh, &mut next);
        self.state = next;
        for row in 0..self.batch {
            if reset[row] {
                self.previous_command[row * ACTIONS..(row + 1) * ACTIONS].fill(0.0);
            } else {
                self.previous_command[row * ACTIONS..(row + 1) * ACTIONS]
                    .copy_from_slice(&previous[row * ACTIONS..(row + 1) * ACTIONS]);
            }
        }
        self.current_key = self.encode_goal(z, &self.state);
        for row in 0..self.batch {
            let elapsed = ticks[row].checked_sub(self.goal_selected_tick[row]);
            let refresh = reset[row]
                || self.goal_origin_slot[row] < 0
                || elapsed.is_none()
                || elapsed.is_some_and(|age| age >= GOAL_HORIZON);
            if refresh {
                if let Some((slot, generation, tick, key)) = self.goal_memory.select(
                    row,
                    &self.current_key[row * GOAL..(row + 1) * GOAL],
                    self.sample,
                ) {
                    self.goal[row * GOAL..(row + 1) * GOAL].copy_from_slice(&key);
                    self.goal_origin_slot[row] = slot as i32;
                    self.goal_origin_generation[row] = generation;
                    self.goal_origin_tick[row] = tick;
                } else {
                    self.goal[row * GOAL..(row + 1) * GOAL]
                        .copy_from_slice(&self.current_key[row * GOAL..(row + 1) * GOAL]);
                    self.goal_origin_slot[row] = -1;
                    self.goal_origin_generation[row] = 0;
                    self.goal_origin_tick[row] = 0;
                }
                self.goal_selected_tick[row] = ticks[row];
            }
            let dst = row * CANONICAL_CONTEXT;
            self.canonical_context[dst..dst + Z].copy_from_slice(&z[row * Z..(row + 1) * Z]);
            self.canonical_context[dst + Z..dst + Z + HIDDEN]
                .copy_from_slice(&self.state[row * HIDDEN..(row + 1) * HIDDEN]);
            self.canonical_context[dst + Z + HIDDEN..dst + Z + HIDDEN + GOAL]
                .copy_from_slice(&self.goal[row * GOAL..(row + 1) * GOAL]);
            self.canonical_context[dst + Z + HIDDEN + GOAL..(row + 1) * CANONICAL_CONTEXT]
                .copy_from_slice(&self.previous_command[row * ACTIONS..(row + 1) * ACTIONS]);
        }
    }

    fn resolve_acknowledged(&mut self, ticks: &[u64], reset: &[bool]) -> Result<(), String> {
        for row in 0..self.batch {
            if reset[row] {
                self.acknowledged_valid[row] = false;
                self.acknowledged_suffix[row] = false;
                self.suffixes.reset_episode(row)?;
                continue;
            }
            if !self.acknowledged_valid[row] {
                continue;
            }
            let contiguous = self.acknowledged_tick[row].checked_add(1) == Some(ticks[row]);
            let current = &self.current_key[row * GOAL..(row + 1) * GOAL];
            let before = &self.acknowledged_context[row * CONTEXT..(row + 1) * CONTEXT];
            let target = &self.acknowledged_goal[row * GOAL..(row + 1) * GOAL];
            let outcome = current.iter().zip(target).map(|(a, b)| a * b).sum::<f32>()
                - before.iter().zip(target).map(|(a, b)| a * b).sum::<f32>();
            let action = &self.acknowledged_command[row * ACTIONS..(row + 1) * ACTIONS];
            if self.acknowledged_suffix[row] {
                if contiguous {
                    self.suffixes.note_executed(
                        row,
                        self.acknowledged_tick[row],
                        action,
                        &[outcome],
                    );
                } else {
                    self.suffixes
                        .cancel_execution(row, CancellationReason::TickGap);
                }
            }
            self.suffixes.record_executed(
                row,
                self.acknowledged_tick[row],
                &self.acknowledged_context[row * CONTEXT..(row + 1) * CONTEXT],
                action,
                &[outcome],
            )?;
            self.acknowledged_valid[row] = false;
            self.acknowledged_suffix[row] = false;
        }
        Ok(())
    }

    fn make_local_candidates(&mut self) {
        let mut input = vec![0.0; self.batch * PROPOSAL_INPUT];
        for row in 0..self.batch {
            let dst = row * PROPOSAL_INPUT;
            input[dst..dst + HIDDEN].copy_from_slice(&self.state[row * HIDDEN..(row + 1) * HIDDEN]);
            input[dst + HIDDEN..dst + HIDDEN + GOAL]
                .copy_from_slice(&self.goal[row * GOAL..(row + 1) * GOAL]);
            input[dst + HIDDEN + GOAL..(row + 1) * PROPOSAL_INPUT]
                .copy_from_slice(&self.previous_command[row * ACTIONS..(row + 1) * ACTIONS]);
        }
        let mut hidden = Vec::new();
        gemm_into(
            &input,
            self.batch,
            PROPOSAL_INPUT,
            &self.core.proposal_hidden,
            &mut hidden,
        );
        tanh_all(&mut hidden);
        let mut local = Vec::new();
        gemm_into(
            &hidden,
            self.batch,
            PROPOSAL_HIDDEN,
            &self.core.proposal_out,
            &mut local,
        );
        for command in local.chunks_exact_mut(ACTIONS) {
            for value in &mut command[..4] {
                *value = value.tanh();
            }
            for value in &mut command[4..] {
                *value = if *value >= 0.0 {
                    1.0 / (1.0 + (-*value).exp())
                } else {
                    let exp = value.exp();
                    exp / (1.0 + exp)
                };
            }
        }
        for row in 0..self.batch {
            for k in 0..LOCAL {
                let src = (row * LOCAL + k) * ACTIONS;
                let dst = (row * CANDIDATES + k) * ACTIONS;
                self.candidate_actions[dst..dst + ACTIONS]
                    .copy_from_slice(&local[src..src + ACTIONS]);
            }
        }
    }

    fn forecast(&self, row: usize, actions: &[f32], horizon: usize) -> (Vec<f32>, f32, f32, bool) {
        let context =
            &self.canonical_context[row * CANONICAL_CONTEXT..(row + 1) * CANONICAL_CONTEXT];
        let base_z = &context[..Z];
        let base_h = &self.state[row * HIDDEN..(row + 1) * HIDDEN];
        let mut goals = Vec::with_capacity(PREDICTOR_MEMBERS * GOAL);
        for member in &self.predictor {
            let mut pred_state = Vec::new();
            gemm_into(
                context,
                1,
                CANONICAL_CONTEXT,
                &member.context_encoder,
                &mut pred_state,
            );
            tanh_all(&mut pred_state);
            let mut core_state = base_h.to_vec();
            let mut pred_z = base_z.to_vec();
            for step in 0..horizon {
                let action = &actions[step * ACTIONS..(step + 1) * ACTIONS];
                let mut gx = Vec::new();
                let mut gh = Vec::new();
                let mut next = Vec::new();
                member
                    .transition
                    .step_into(action, &pred_state, 1, &mut gx, &mut gh, &mut next);
                pred_state = next;
                let mut delta = Vec::new();
                gemm_into(&pred_state, 1, HIDDEN, &member.latent_delta, &mut delta);
                for j in 0..Z {
                    pred_z[j] = base_z[j] + delta[j];
                }
                let mut core_input = vec![0.0; CORE_INPUT];
                core_input[..Z].copy_from_slice(&pred_z);
                core_input[Z..].copy_from_slice(action);
                let mut core_next = Vec::new();
                self.core.recurrent.step_into(
                    &core_input,
                    &core_state,
                    1,
                    &mut gx,
                    &mut gh,
                    &mut core_next,
                );
                core_state = core_next;
            }
            goals.extend(self.encode_goal(&pred_z, &core_state));
        }
        if goals.iter().any(|x| !x.is_finite()) {
            return (vec![0.0; GOAL], 0.0, 0.0, false);
        }
        let mut mean = vec![0.0; GOAL];
        for member in goals.chunks_exact(GOAL) {
            for j in 0..GOAL {
                mean[j] += member[j] / PREDICTOR_MEMBERS as f32;
            }
        }
        normalize_rows(&mut mean, GOAL);
        let current = &self.goal[row * GOAL..(row + 1) * GOAL];
        let progress = mean.iter().zip(current).map(|(a, b)| a * b).sum::<f32>();
        let disagreement = (goals
            .chunks_exact(GOAL)
            .flat_map(|member| member.iter().zip(&mean).map(|(a, b)| (a - b) * (a - b)))
            .sum::<f32>()
            / (PREDICTOR_MEMBERS * GOAL) as f32)
            .sqrt();
        (mean, progress, disagreement, true)
    }

    fn prepare_decision(
        &mut self,
        z: &[f32],
        previous: &[f32],
        ticks: &[u64],
        reset: &[bool],
    ) -> Result<Vec<f32>, String> {
        if self.command_pending.iter().any(|x| *x) {
            return Err("an unacknowledged command is pending".into());
        }
        self.core_observe(z, previous, ticks, reset);
        self.resolve_acknowledged(ticks, reset)?;
        for row in 0..self.batch {
            self.memory_inserted_slot[row] = self.goal_memory.insert(
                row,
                &self.current_key[row * GOAL..(row + 1) * GOAL],
                ticks[row],
                self.goal_origin_slot[row],
            );
        }
        self.candidate_actions.fill(0.0);
        self.candidate_sequences.fill(0.0);
        self.candidate_mask.fill(false);
        self.candidate_recalled.fill(false);
        self.candidate_slot.fill(-1);
        self.candidate_generation.fill(0);
        self.candidate_length.fill(0);
        self.candidate_support.fill(0);
        self.candidate_recall.fill(0.0);
        self.candidate_empirical.fill(0.0);
        self.control_proposal.fill(0.0);
        self.control_active.fill(0.0);
        self.control_active_mask.fill(false);
        self.active_source_slot.fill(-1);
        self.active_source_generation.fill(0);
        self.active_phase.fill(0);
        self.active_remaining.fill(0);
        self.make_local_candidates();
        let mut recalled_rows = vec![Vec::new(); self.batch];
        for row in 0..self.batch {
            for k in 0..LOCAL {
                self.candidate_mask[row * CANDIDATES + k] = true;
                self.candidate_length[row * CANDIDATES + k] = 1;
                let src = (row * CANDIDATES + k) * ACTIONS;
                let seq = (row * CANDIDATES + k) * MAX_HORIZON * ACTIONS;
                self.candidate_sequences[seq..seq + ACTIONS]
                    .copy_from_slice(&self.candidate_actions[src..src + ACTIONS]);
                for h in 1..MAX_HORIZON {
                    let dst = seq + h * ACTIONS;
                    self.candidate_sequences
                        .copy_within(seq..seq + ACTIONS, dst);
                }
            }
            let recalled = self.suffixes.recall(
                row,
                &self.current_key[row * GOAL..(row + 1) * GOAL],
                RECALLED,
            );
            for (extra, suffix) in recalled.iter().enumerate() {
                let k = LOCAL + extra;
                let index = row * CANDIDATES + k;
                let seq = index * MAX_HORIZON * ACTIONS;
                self.candidate_mask[index] = true;
                self.candidate_recalled[index] = true;
                self.candidate_slot[index] = suffix.slot as i32;
                self.candidate_generation[index] = suffix.generation;
                self.candidate_length[index] = suffix.length as u8;
                self.candidate_support[index] = suffix.support;
                self.candidate_recall[index] = suffix.recall_score;
                self.candidate_empirical[index] = suffix.empirical_utility;
                self.candidate_actions[index * ACTIONS..(index + 1) * ACTIONS]
                    .copy_from_slice(&suffix.actions[..ACTIONS]);
                self.candidate_sequences[seq..seq + suffix.length * ACTIONS]
                    .copy_from_slice(&suffix.actions[..suffix.length * ACTIONS]);
                let last = seq + (suffix.length - 1) * ACTIONS;
                for h in suffix.length..MAX_HORIZON {
                    let dst = seq + h * ACTIONS;
                    self.candidate_sequences
                        .copy_within(last..last + ACTIONS, dst);
                }
            }
            recalled_rows[row] = recalled;
            if let Some(active) = self.suffixes.active(row) {
                let remaining = active.length - active.phase;
                let base = row * CHOICE;
                self.control_active_mask[row] = true;
                self.active_source_slot[row] = active.source_slot;
                self.active_source_generation[row] = active.source_generation;
                self.active_phase[row] = active.phase as u8;
                self.active_remaining[row] = remaining as u8;
                self.control_active[base..base + remaining * ACTIONS]
                    .copy_from_slice(&active.actions[..remaining * ACTIONS]);
                self.control_active[base + 96] = remaining as f32 / 8.0;
                self.control_active[base + 97] = active.phase as f32 / 8.0;
                self.control_active[base + 98] = active.length as f32 / 8.0;
                self.control_active[base + 99] = 1.0;
                self.control_active[base + 100] = (active.support as f32).ln_1p();
                self.control_active[base + 101] = active.recall_score;
                self.control_active[base + 102] = active.empirical_utility;
                let mut forecast_actions = vec![0.0; MAX_HORIZON * ACTIONS];
                forecast_actions[..remaining * ACTIONS]
                    .copy_from_slice(&active.actions[..remaining * ACTIONS]);
                let last = (remaining - 1) * ACTIONS;
                for h in remaining..MAX_HORIZON {
                    let dst = h * ACTIONS;
                    forecast_actions.copy_within(last..last + ACTIONS, dst);
                }
                let (pred, progress, disagreement, valid) =
                    self.forecast(row, &forecast_actions, remaining);
                self.control_active[base + 103] = valid as u8 as f32;
                self.control_active[base + 104] = progress;
                self.control_active[base + 105] = disagreement;
                self.control_active[base + 106..base + 234].copy_from_slice(&pred);
            }
        }
        for row in 0..self.batch {
            for k in 0..CANDIDATES {
                if !self.candidate_mask[row * CANDIDATES + k] {
                    continue;
                }
                let index = row * CANDIDATES + k;
                let base = index * CHOICE;
                let seq = index * MAX_HORIZON * ACTIONS;
                let duration = usize::from(self.candidate_length[index]);
                if self.candidate_recalled[index] {
                    self.control_proposal[base..base + duration * ACTIONS]
                        .copy_from_slice(&self.candidate_sequences[seq..seq + duration * ACTIONS]);
                } else {
                    self.control_proposal[base..base + ACTIONS].copy_from_slice(
                        &self.candidate_actions[index * ACTIONS..(index + 1) * ACTIONS],
                    );
                }
                self.control_proposal[base + 96] = duration as f32 / 8.0;
                self.control_proposal[base + 98] = duration as f32 / 8.0;
                self.control_proposal[base + 99] = self.candidate_recalled[index] as u8 as f32;
                self.control_proposal[base + 100] = (self.candidate_support[index] as f32).ln_1p();
                self.control_proposal[base + 101] = self.candidate_recall[index].clamp(-8.0, 1.0);
                self.control_proposal[base + 102] = self.candidate_empirical[index];
                let (pred, progress, disagreement, valid) = self.forecast(
                    row,
                    &self.candidate_sequences[seq..seq + MAX_HORIZON * ACTIONS],
                    if self.candidate_recalled[index] {
                        duration
                    } else {
                        MAX_HORIZON
                    },
                );
                self.control_proposal[base + 103] = valid as u8 as f32;
                self.control_proposal[base + 104] = progress;
                self.control_proposal[base + 105] = disagreement;
                self.control_proposal[base + 106..base + 234].copy_from_slice(&pred);
            }
        }
        for row in 0..self.batch {
            let dst = row * CONTROL_STATE;
            self.control_state[dst..dst + CANONICAL_CONTEXT].copy_from_slice(
                &self.canonical_context[row * CANONICAL_CONTEXT..(row + 1) * CANONICAL_CONTEXT],
            );
            self.control_state[dst + CANONICAL_CONTEXT] =
                self.control_active_mask[row] as u8 as f32;
        }
        self.decision = self.sequence_control.decide(
            &self.control_state,
            &self.control_proposal,
            &self.control_active,
            &self.candidate_mask,
            &self.control_active_mask,
            self.sample,
        )?;
        let mut proposed = vec![0.0; self.batch * ACTIONS];
        for row in 0..self.batch {
            let chosen = self.decision.selected[row];
            if chosen < 0 {
                let active = self
                    .suffixes
                    .active(row)
                    .ok_or("continued suffix is absent")?;
                proposed[row * ACTIONS..(row + 1) * ACTIONS]
                    .copy_from_slice(&active.actions[..ACTIONS]);
                self.suffixes.continue_execution(row)?;
            } else {
                if self.decision.terminate[row] {
                    self.suffixes
                        .cancel_execution(row, CancellationReason::Policy);
                }
                let k = chosen as usize;
                let index = row * CANDIDATES + k;
                proposed[row * ACTIONS..(row + 1) * ACTIONS].copy_from_slice(
                    &self.candidate_actions[index * ACTIONS..(index + 1) * ACTIONS],
                );
                if k >= LOCAL {
                    let suffix = recalled_rows[row]
                        .iter()
                        .find(|s| {
                            s.slot as i32 == self.candidate_slot[index]
                                && s.generation == self.candidate_generation[index]
                        })
                        .ok_or("selected suffix vanished")?;
                    self.suffixes.start(row, suffix)?;
                }
            }
            self.command_pending[row] = true;
            self.pending_tick[row] = ticks[row];
            self.pending_command[row * ACTIONS..(row + 1) * ACTIONS]
                .copy_from_slice(&proposed[row * ACTIONS..(row + 1) * ACTIONS]);
            self.pending_context[row * CONTEXT..(row + 1) * CONTEXT]
                .copy_from_slice(&self.current_key[row * GOAL..(row + 1) * GOAL]);
            self.pending_goal[row * GOAL..(row + 1) * GOAL]
                .copy_from_slice(&self.goal[row * GOAL..(row + 1) * GOAL]);
        }
        Ok(proposed)
    }

    fn private_snapshot(&self) -> PrivateSnapshot {
        PrivateSnapshot {
            format: FORMAT.into(),
            batch: self.batch,
            sample: self.sample,
            research_training: self.research_training,
            core_sha256: self.core_sha256.clone(),
            predictor_sha256: self.predictor_sha256.clone(),
            state: self.state.clone(),
            previous_command: self.previous_command.clone(),
            current_key: self.current_key.clone(),
            goal: self.goal.clone(),
            goal_origin_slot: self.goal_origin_slot.clone(),
            goal_origin_generation: self.goal_origin_generation.clone(),
            goal_origin_tick: self.goal_origin_tick.clone(),
            goal_selected_tick: self.goal_selected_tick.clone(),
            memory_inserted_slot: self.memory_inserted_slot.clone(),
            command_pending: self.command_pending.clone(),
            pending_tick: self.pending_tick.clone(),
            pending_command: self.pending_command.clone(),
            pending_context: self.pending_context.clone(),
            pending_goal: self.pending_goal.clone(),
            acknowledged_valid: self.acknowledged_valid.clone(),
            acknowledged_suffix: self.acknowledged_suffix.clone(),
            acknowledged_tick: self.acknowledged_tick.clone(),
            acknowledged_command: self.acknowledged_command.clone(),
            acknowledged_context: self.acknowledged_context.clone(),
            acknowledged_goal: self.acknowledged_goal.clone(),
        }
    }

    fn output<'py>(&self, py: Python<'py>, proposed: Vec<f32>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item(
            "proposed_command",
            Array2::from_shape_vec((self.batch, ACTIONS), proposed)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "cns_recurrent_state",
            Array2::from_shape_vec((self.batch, HIDDEN), self.state.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "latent_goal",
            Array2::from_shape_vec((self.batch, GOAL), self.goal.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "current_cns_key",
            Array2::from_shape_vec((self.batch, GOAL), self.current_key.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "goal_origin_slot",
            Array1::from_vec(self.goal_origin_slot.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_origin_generation",
            Array1::from_vec(self.goal_origin_generation.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_origin_tick",
            Array1::from_vec(self.goal_origin_tick.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_selected_tick",
            Array1::from_vec(self.goal_selected_tick.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "memory_inserted_slot",
            Array1::from_vec(self.memory_inserted_slot.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "memory_count",
            Array1::from_vec(
                self.goal_memory
                    .count
                    .iter()
                    .map(|count| *count as u16)
                    .collect::<Vec<_>>(),
            )
            .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_policy_version",
            self.sequence_control.policy_version,
        )?;
        out.set_item(
            "sequence_control_policy_sha256",
            self.sequence_control.policy_sha256.clone(),
        )?;
        out.set_item(
            "sequence_control_state",
            Array2::from_shape_vec((self.batch, CONTROL_STATE), self.control_state.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_proposal",
            Array3::from_shape_vec(
                (self.batch, CANDIDATES, CHOICE),
                self.control_proposal.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_active",
            Array2::from_shape_vec((self.batch, CHOICE), self.control_active.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_proposal_mask",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_mask.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_active_mask",
            Array1::from_vec(self.control_active_mask.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_logit",
            Array1::from_vec(self.decision.hazard_logits.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_logits",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.decision.selector_logits.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_value",
            Array1::from_vec(self.decision.values.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_decision",
            Array1::from_vec(self.decision.terminate.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "selected_candidate",
            Array1::from_vec(self.decision.selected.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_mask",
            Array1::from_vec(self.decision.hazard_mask.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_mask",
            Array1::from_vec(self.decision.selector_mask.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_hazard_logp",
            Array1::from_vec(self.decision.hazard_logp.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_selector_logp",
            Array1::from_vec(self.decision.selector_logp.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_logp",
            Array1::from_vec(self.decision.logp.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "candidate_is_recalled_suffix",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_recalled.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_slot",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_slot.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_generation",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_generation.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_length",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_length.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "active_source_slot",
            Array1::from_vec(self.active_source_slot.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "active_source_generation",
            Array1::from_vec(self.active_source_generation.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "active_phase",
            Array1::from_vec(self.active_phase.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "active_remaining",
            Array1::from_vec(self.active_remaining.clone()).into_pyarray(py),
        )?;
        let cancellation: Vec<_> = (0..self.batch)
            .flat_map(|row| self.suffixes.cancellation_counts(row))
            .collect();
        out.set_item(
            "motor_suffix_cancellation_totals",
            Array2::from_shape_vec((self.batch, 5), cancellation)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "motor_suffix_cancellation_reason",
            (0..self.batch)
                .map(|row| self.suffixes.cancellation_reason(row))
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "command_pending",
            Array1::from_vec(self.command_pending.clone()).into_pyarray(py),
        )?;
        Ok(out)
    }
}

#[pymethods]
impl DevelopmentalResidentCohort {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        batch: usize,
        action_mode: &str,
        action_seed: u64,
        suffix_seed: u64,
        core_packed: PyReadonlyArray1<'_, f32>,
        core_sha256: String,
        predictor_packed: PyReadonlyArray1<'_, f32>,
        predictor_sha256: String,
        sequence_control_packed: PyReadonlyArray1<'_, f32>,
        sequence_control_version: u64,
        sequence_control_sha256: String,
        research_training: bool,
    ) -> PyResult<Self> {
        if batch == 0
            || batch > 4096
            || !matches!(action_mode, "sample" | "map")
            || !valid_hash(&core_sha256)
            || !valid_hash(&predictor_sha256)
        {
            return Err(PyValueError::new_err("CNS resident configuration differs"));
        }
        let flat = core_packed.as_slice()?;
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("CNS core weights must be finite"));
        }
        let mut c = 0;
        let core = Core {
            recurrent: gru(flat, &mut c, HIDDEN, CORE_INPUT)?,
            goal_encoder: linear(flat, &mut c, GOAL, GOAL_INPUT)?,
            proposal_hidden: linear(flat, &mut c, PROPOSAL_HIDDEN, PROPOSAL_INPUT)?,
            proposal_out: linear(flat, &mut c, LOCAL * ACTIONS, PROPOSAL_HIDDEN)?,
        };
        if c != flat.len() {
            return Err(PyValueError::new_err(
                "CNS core weights have trailing values",
            ));
        }
        let pflat = predictor_packed.as_slice()?;
        if pflat.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "CNS predictor weights must be finite",
            ));
        }
        let mut pc = 0;
        let mut predictor = Vec::new();
        for _ in 0..PREDICTOR_MEMBERS {
            predictor.push(PredictorMember {
                context_encoder: linear(pflat, &mut pc, HIDDEN, CANONICAL_CONTEXT)?,
                transition: gru(pflat, &mut pc, HIDDEN, ACTIONS)?,
                latent_delta: linear(pflat, &mut pc, Z, HIDDEN)?,
            });
        }
        if pc != pflat.len() {
            return Err(PyValueError::new_err(
                "CNS predictor weights have trailing values",
            ));
        }
        let sequence_control = LearnedSequenceControl::from_flat(
            batch,
            sequence_control_packed.as_slice()?,
            sequence_control_version,
            sequence_control_sha256,
            action_seed ^ 0x5345_515f_4354_524c,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            batch,
            sample: action_mode == "sample",
            research_training,
            core_sha256,
            predictor_sha256,
            core,
            predictor,
            sequence_control,
            suffixes: MotorSuffixMemory::new(batch, suffix_seed).map_err(PyValueError::new_err)?,
            goal_memory: CnsGoalMemory::new(batch, suffix_seed ^ 0x474f_414c_4d45_4d31),
            state: vec![0.0; batch * HIDDEN],
            previous_command: vec![0.0; batch * ACTIONS],
            current_key: vec![0.0; batch * GOAL],
            goal: vec![0.0; batch * GOAL],
            goal_origin_slot: vec![-1; batch],
            goal_origin_generation: vec![0; batch],
            goal_origin_tick: vec![0; batch],
            goal_selected_tick: vec![0; batch],
            memory_inserted_slot: vec![-1; batch],
            canonical_context: vec![0.0; batch * CANONICAL_CONTEXT],
            command_pending: vec![false; batch],
            pending_tick: vec![0; batch],
            pending_command: vec![0.0; batch * ACTIONS],
            pending_context: vec![0.0; batch * CONTEXT],
            pending_goal: vec![0.0; batch * GOAL],
            acknowledged_valid: vec![false; batch],
            acknowledged_suffix: vec![false; batch],
            acknowledged_tick: vec![0; batch],
            acknowledged_command: vec![0.0; batch * ACTIONS],
            acknowledged_context: vec![0.0; batch * CONTEXT],
            acknowledged_goal: vec![0.0; batch * GOAL],
            candidate_actions: vec![0.0; batch * CANDIDATES * ACTIONS],
            candidate_sequences: vec![0.0; batch * CANDIDATES * MAX_HORIZON * ACTIONS],
            candidate_mask: vec![false; batch * CANDIDATES],
            candidate_recalled: vec![false; batch * CANDIDATES],
            candidate_slot: vec![-1; batch * CANDIDATES],
            candidate_generation: vec![0; batch * CANDIDATES],
            candidate_length: vec![0; batch * CANDIDATES],
            candidate_support: vec![0; batch * CANDIDATES],
            candidate_recall: vec![0.0; batch * CANDIDATES],
            candidate_empirical: vec![0.0; batch * CANDIDATES],
            control_state: vec![0.0; batch * CONTROL_STATE],
            control_proposal: vec![0.0; batch * CANDIDATES * CHOICE],
            control_active: vec![0.0; batch * CHOICE],
            control_active_mask: vec![false; batch],
            active_source_slot: vec![-1; batch],
            active_source_generation: vec![0; batch],
            active_phase: vec![0; batch],
            active_remaining: vec![0; batch],
            decision: decision_empty(batch),
        })
    }

    fn expanded(&self, additions: usize, action_seed: u64, suffix_seed: u64) -> PyResult<Self> {
        self.expanded_inner(additions, action_seed, suffix_seed)
            .map_err(PyValueError::new_err)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        cns_latent: PyReadonlyArray2<'_, f32>,
        previous_command: PyReadonlyArray2<'_, f32>,
        ticks: PyReadonlyArray1<'_, u64>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if cns_latent.shape() != [self.batch, Z]
            || previous_command.shape() != [self.batch, ACTIONS]
            || ticks.shape() != [self.batch]
            || reset.shape() != [self.batch]
        {
            return Err(PyValueError::new_err("CNS resident step shapes differ"));
        }
        let z = cns_latent.as_slice()?;
        let previous = previous_command.as_slice()?;
        if z.iter().any(|x| !x.is_finite()) || !commands_valid(previous) {
            return Err(PyValueError::new_err(
                "CNS latent must be finite and previous commands must obey canonical bounds",
            ));
        }
        let proposed = self
            .prepare_decision(z, previous, ticks.as_slice()?, reset.as_slice()?)
            .map_err(PyValueError::new_err)?;
        self.output(py, proposed)
    }

    fn preview_sequence_control<'py>(
        &self,
        py: Python<'py>,
        cns_latent: PyReadonlyArray2<'_, f32>,
        previous_command: PyReadonlyArray2<'_, f32>,
        ticks: PyReadonlyArray1<'_, u64>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut fork = self.clone();
        fork.step(py, cns_latent, previous_command, ticks, reset)
    }

    fn acknowledge<'py>(
        &mut self,
        py: Python<'py>,
        ticks: PyReadonlyArray1<'_, u64>,
        delivered_command: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if ticks.shape() != [self.batch] || delivered_command.shape() != [self.batch, ACTIONS] {
            return Err(PyValueError::new_err("command receipt shapes differ"));
        }
        let t = ticks.as_slice()?;
        let delivered = delivered_command.as_slice()?;
        if !commands_valid(delivered) {
            return Err(PyValueError::new_err(
                "delivered commands must obey canonical bounds",
            ));
        }
        for row in 0..self.batch {
            if !self.command_pending[row] || self.pending_tick[row] != t[row] {
                return Err(PyValueError::new_err("command receipt boundary differs"));
            }
        }
        let mut exact = vec![false; self.batch];
        for row in 0..self.batch {
            let actual = &delivered[row * ACTIONS..(row + 1) * ACTIONS];
            let expected = &self.pending_command[row * ACTIONS..(row + 1) * ACTIONS];
            exact[row] = actual == expected;
            let suffix_pending = self.suffixes.active(row).is_some();
            if !exact[row] && suffix_pending {
                self.suffixes
                    .cancel_execution(row, CancellationReason::Receipt);
            }
            self.acknowledged_valid[row] = true;
            self.acknowledged_suffix[row] = exact[row] && suffix_pending;
            self.acknowledged_tick[row] = t[row];
            self.acknowledged_command[row * ACTIONS..(row + 1) * ACTIONS].copy_from_slice(actual);
            self.acknowledged_context[row * CONTEXT..(row + 1) * CONTEXT]
                .copy_from_slice(&self.pending_context[row * CONTEXT..(row + 1) * CONTEXT]);
            self.acknowledged_goal[row * GOAL..(row + 1) * GOAL]
                .copy_from_slice(&self.pending_goal[row * GOAL..(row + 1) * GOAL]);
            self.command_pending[row] = false;
        }
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item(
            "sequence_control_policy_version",
            self.sequence_control.policy_version,
        )?;
        out.set_item(
            "sequence_control_policy_sha256",
            self.sequence_control.policy_sha256.clone(),
        )?;
        out.set_item(
            "acknowledged",
            Array1::from_vec(vec![true; self.batch]).into_pyarray(py),
        )?;
        out.set_item("command_exact", Array1::from_vec(exact).into_pyarray(py))?;
        out.set_item(
            "command_pending",
            Array1::from_vec(self.command_pending.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "cns_outcome_pending",
            Array1::from_vec(self.acknowledged_valid.clone()).into_pyarray(py),
        )?;
        let cancellation: Vec<_> = (0..self.batch)
            .flat_map(|row| self.suffixes.cancellation_counts(row))
            .collect();
        out.set_item(
            "motor_suffix_cancellation_totals",
            Array2::from_shape_vec((self.batch, 5), cancellation)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "motor_suffix_cancellation_reason",
            (0..self.batch)
                .map(|row| self.suffixes.cancellation_reason(row))
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn sequence_control_likelihood<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray2<'_, f32>,
        proposal: PyReadonlyArray3<'_, f32>,
        active: PyReadonlyArray2<'_, f32>,
        proposal_mask: PyReadonlyArray2<'_, bool>,
        active_mask: PyReadonlyArray1<'_, bool>,
        hazard_decision: PyReadonlyArray1<'_, bool>,
        selected_candidate: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if state.shape() != [self.batch, CONTROL_STATE]
            || proposal.shape() != [self.batch, CANDIDATES, CHOICE]
            || active.shape() != [self.batch, CHOICE]
            || proposal_mask.shape() != [self.batch, CANDIDATES]
            || active_mask.shape() != [self.batch]
            || hazard_decision.shape() != [self.batch]
            || selected_candidate.shape() != [self.batch]
        {
            return Err(PyValueError::new_err(
                "sequence-control likelihood shapes differ",
            ));
        }
        let decision = self
            .sequence_control
            .likelihood(
                state.as_slice()?,
                proposal.as_slice()?,
                active.as_slice()?,
                proposal_mask.as_slice()?,
                active_mask.as_slice()?,
                hazard_decision.as_slice()?,
                selected_candidate.as_slice()?,
            )
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item(
            "sequence_control_policy_version",
            self.sequence_control.policy_version,
        )?;
        out.set_item(
            "sequence_control_policy_sha256",
            self.sequence_control.policy_sha256.clone(),
        )?;
        out.set_item(
            "sequence_control_hazard_logit",
            Array1::from_vec(decision.hazard_logits).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_logits",
            Array2::from_shape_vec((self.batch, CANDIDATES), decision.selector_logits)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_value",
            Array1::from_vec(decision.values).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_decision",
            Array1::from_vec(decision.terminate).into_pyarray(py),
        )?;
        out.set_item(
            "selected_candidate",
            Array1::from_vec(decision.selected).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_mask",
            Array1::from_vec(decision.hazard_mask).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_mask",
            Array1::from_vec(decision.selector_mask).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_hazard_logp",
            Array1::from_vec(decision.hazard_logp).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_selector_logp",
            Array1::from_vec(decision.selector_logp).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_logp",
            Array1::from_vec(decision.logp).into_pyarray(py),
        )?;
        Ok(out)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item("version", 10u8)?;
        out.set_item(
            "private",
            serde_json::to_string(&self.private_snapshot())
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )?;
        out.set_item(
            "motor_suffix_memory",
            self.suffixes
                .snapshot_json()
                .map_err(PyValueError::new_err)?,
        )?;
        out.set_item(
            "goal_memory",
            serde_json::to_string(&self.goal_memory)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )?;
        out.set_item(
            "sequence_control",
            self.sequence_control
                .snapshot_json()
                .map_err(PyValueError::new_err)?,
        )?;
        Ok(out)
    }

    fn restore(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        if value.len() != 6 {
            return Err(PyValueError::new_err("CNS snapshot fields differ"));
        }
        let get = |name: &str| {
            value
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("CNS snapshot lacks {name}")))
        };
        if get("format")?.extract::<String>()? != FORMAT || get("version")?.extract::<u8>()? != 10 {
            return Err(PyValueError::new_err("CNS snapshot identity differs"));
        }
        let p: PrivateSnapshot = serde_json::from_str(&get("private")?.extract::<String>()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let expected = self.private_snapshot();
        if p.format != FORMAT
            || p.batch != self.batch
            || p.sample != self.sample
            || p.research_training != self.research_training
            || p.core_sha256 != self.core_sha256
            || p.predictor_sha256 != self.predictor_sha256
            || p.state.len() != expected.state.len()
            || p.previous_command.len() != expected.previous_command.len()
            || p.current_key.len() != expected.current_key.len()
            || p.goal.len() != expected.goal.len()
            || p.goal_origin_slot.len() != self.batch
            || p.goal_origin_generation.len() != self.batch
            || p.goal_origin_tick.len() != self.batch
            || p.goal_selected_tick.len() != self.batch
            || p.memory_inserted_slot.len() != self.batch
            || p.command_pending.len() != self.batch
            || p.pending_tick.len() != self.batch
            || p.pending_command.len() != expected.pending_command.len()
            || p.pending_context.len() != expected.pending_context.len()
            || p.pending_goal.len() != expected.pending_goal.len()
            || p.acknowledged_valid.len() != self.batch
            || p.acknowledged_suffix.len() != self.batch
            || p.acknowledged_tick.len() != self.batch
            || p.acknowledged_command.len() != expected.acknowledged_command.len()
            || p.acknowledged_context.len() != expected.acknowledged_context.len()
            || p.acknowledged_goal.len() != expected.acknowledged_goal.len()
            || !commands_valid(&p.previous_command)
            || !commands_valid(&p.pending_command)
            || !commands_valid(&p.acknowledged_command)
            || p.goal_origin_slot
                .iter()
                .chain(&p.memory_inserted_slot)
                .any(|slot| *slot < -1 || *slot >= GOAL_SLOTS as i32)
            || (0..self.batch).any(|row| p.command_pending[row] && p.acknowledged_valid[row])
            || p.state
                .iter()
                .chain(&p.previous_command)
                .chain(&p.current_key)
                .chain(&p.goal)
                .chain(&p.pending_command)
                .chain(&p.pending_context)
                .chain(&p.pending_goal)
                .chain(&p.acknowledged_command)
                .chain(&p.acknowledged_context)
                .chain(&p.acknowledged_goal)
                .any(|x| !x.is_finite())
        {
            return Err(PyValueError::new_err("CNS snapshot state differs"));
        }
        let suffixes = MotorSuffixMemory::restore_json(
            &get("motor_suffix_memory")?.extract::<String>()?,
            self.batch,
        )
        .map_err(PyValueError::new_err)?;
        let goal_memory: CnsGoalMemory =
            serde_json::from_str(&get("goal_memory")?.extract::<String>()?)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if !goal_memory.validate(self.batch) {
            return Err(PyValueError::new_err("CNS goal memory differs"));
        }
        for row in 0..self.batch {
            let slot = p.goal_origin_slot[row];
            if slot < 0 {
                if p.goal_origin_generation[row] != 0
                    || p.goal_origin_tick[row] != 0
                    || p.goal[row * GOAL..(row + 1) * GOAL]
                        != p.current_key[row * GOAL..(row + 1) * GOAL]
                {
                    return Err(PyValueError::new_err("CNS goal origin differs"));
                }
                continue;
            }
            let index = row * GOAL_SLOTS + slot as usize;
            if !goal_memory.valid[index]
                || goal_memory.generation[index] != p.goal_origin_generation[row]
                || goal_memory.recorded_tick[index] != p.goal_origin_tick[row]
                || goal_memory.keys[index * GOAL..(index + 1) * GOAL]
                    != p.goal[row * GOAL..(row + 1) * GOAL]
            {
                return Err(PyValueError::new_err("CNS goal origin differs"));
            }
        }
        let mut sequence_control = self.sequence_control.clone();
        sequence_control
            .restore_checked(&get("sequence_control")?.extract::<String>()?)
            .map_err(PyValueError::new_err)?;
        self.state = p.state;
        self.previous_command = p.previous_command;
        self.current_key = p.current_key;
        self.goal = p.goal;
        self.goal_origin_slot = p.goal_origin_slot;
        self.goal_origin_generation = p.goal_origin_generation;
        self.goal_origin_tick = p.goal_origin_tick;
        self.goal_selected_tick = p.goal_selected_tick;
        self.memory_inserted_slot = p.memory_inserted_slot;
        self.command_pending = p.command_pending;
        self.pending_tick = p.pending_tick;
        self.pending_command = p.pending_command;
        self.pending_context = p.pending_context;
        self.pending_goal = p.pending_goal;
        self.acknowledged_valid = p.acknowledged_valid;
        self.acknowledged_suffix = p.acknowledged_suffix;
        self.acknowledged_tick = p.acknowledged_tick;
        self.acknowledged_command = p.acknowledged_command;
        self.acknowledged_context = p.acknowledged_context;
        self.acknowledged_goal = p.acknowledged_goal;
        self.suffixes = suffixes;
        self.goal_memory = goal_memory;
        self.sequence_control = sequence_control;
        Ok(())
    }

    fn replace_sequence_control<'py>(
        &mut self,
        py: Python<'py>,
        packed: PyReadonlyArray1<'_, f32>,
        new_version: u64,
        new_sha256: String,
        expected_version: u64,
        expected_sha256: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !self.research_training {
            return Err(PyValueError::new_err(
                "sequence-control replacement is restricted to research cohorts",
            ));
        }
        if self.command_pending.iter().any(|x| *x) || self.acknowledged_valid.iter().any(|x| *x) {
            return Err(PyValueError::new_err(
                "sequence-control replacement requires a fully coherent boundary",
            ));
        }
        let old_version = self.sequence_control.policy_version;
        let old_sha256 = self.sequence_control.policy_sha256.clone();
        self.sequence_control
            .replace(
                packed.as_slice()?,
                new_version,
                new_sha256,
                expected_version,
                &expected_sha256,
            )
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("old_version", old_version)?;
        out.set_item("old_sha256", old_sha256)?;
        out.set_item("new_version", self.sequence_control.policy_version)?;
        out.set_item("new_sha256", self.sequence_control.policy_sha256.clone())?;
        Ok(out)
    }
}
