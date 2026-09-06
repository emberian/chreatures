// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private associations between achieved-goal slots and experienced outcomes.
//!
//! This mechanism never inspects a world. The host supplies actual body
//! physiology and effort for each committed physical transition; this module
//! computes the authenticated `FiniteEnergyObjective.transition` reward. Credit is bound to an exact reservoir
//! slot identity `(recorded_tick, generation)` for ten transitions. Replacing
//! that slot clears its four learned weights and makes pending credit
//! ineligible, even though goal memory may retain an independent copied window.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod f64_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(value.to_bits())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        Ok(f64::from_bits(u64::deserialize(deserializer)?))
    }
}

mod f64_array4_bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[f64; 4], serializer: S) -> Result<S::Ok, S::Error> {
        value.map(f64::to_bits).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f64; 4], D::Error> {
        Ok(<[u64; 4]>::deserialize(deserializer)?.map(f64::from_bits))
    }
}

pub const PERSONAL_GOAL_SLOTS: usize = 128;
pub const PERSONAL_GOAL_FEATURES: usize = 4;
pub const PERSONAL_GOAL_HORIZON_TICKS: u8 = 10;
pub const FINITE_ENERGY_OBJECTIVE_SHA256: &str =
    "01ae937a153a056c8cc5fa5be4d55cdfb38dbfcede4dbceb16ec33e19c5f4d00";

const ASSIMILATION_EFFICIENCY: f64 = 0.84;
const RESERVE_TARGET: f64 = 0.85;
const RESERVE_TEMPERATURE: f64 = 0.08;
const FATIGUE_ENERGY_WEIGHT: f64 = 0.08;
const GUT_COMFORT: f64 = 0.55;
const GUT_OVERLOAD_ENERGY_WEIGHT: f64 = 0.08;
const EFFORT_ENERGY_RATE: f64 = 0.0042;
const EFFORT_EXTRA_WEIGHT: f64 = 0.25;
const REWARD_PER_ENERGY: f64 = 12.0;
const MAX_INTERVAL_SECONDS: f64 = 2.0;
const FINITE_ENERGY_FORMAT: &str = "chreatures-finite-energy-homeostasis-v1";

const FORMAT: &str = "chreatures-private-goal-associations-v1";

#[derive(Serialize)]
struct FiniteEnergyCoefficientIdentity {
    // Field order matches Python's canonical sorted-key encoding.
    assimilation_efficiency: f64,
    effort_energy_rate: f64,
    effort_extra_weight: f64,
    fatigue_energy_weight: f64,
    gut_comfort: f64,
    gut_overload_energy_weight: f64,
    max_interval_seconds: f64,
    reserve_target: f64,
    reserve_temperature: f64,
    reward_per_energy: f64,
    version: u8,
}

#[derive(Serialize)]
struct FiniteEnergyIdentity<'a> {
    config: FiniteEnergyCoefficientIdentity,
    format: &'a str,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PersonalGoalConfig {
    pub objective_sha256: String,
    pub feature_names: [String; PERSONAL_GOAL_FEATURES],
    pub slots: usize,
    pub horizon_ticks: u8,
    /// A summed return is clipped to `[-return_scale, return_scale]` before
    /// conversion to the dimensionless NLMS target.
    #[serde(with = "f64_bits")]
    pub return_scale: f64,
    #[serde(with = "f64_bits")]
    pub learning_rate: f64,
    #[serde(with = "f64_bits")]
    pub weight_norm_limit: f64,
    #[serde(with = "f64_bits")]
    pub logit_gain: f64,
    pub learning_enabled: bool,
}

impl PersonalGoalConfig {
    pub fn current(learning_enabled: bool) -> Self {
        Self {
            objective_sha256: FINITE_ENERGY_OBJECTIVE_SHA256.into(),
            feature_names: [
                "bias".into(),
                "two_energy_minus_one".into(),
                "two_gut_minus_one".into(),
                "two_fatigue_minus_one".into(),
            ],
            slots: PERSONAL_GOAL_SLOTS,
            horizon_ticks: PERSONAL_GOAL_HORIZON_TICKS,
            return_scale: 0.01,
            learning_rate: 0.05,
            weight_norm_limit: 4.0,
            logit_gain: 0.35,
            learning_enabled,
        }
    }

    fn validate(&self) -> Result<(), String> {
        let expected = Self::current(self.learning_enabled);
        if self != &expected
            || finite_energy_objective_identity()? != FINITE_ENERGY_OBJECTIVE_SHA256
        {
            return Err("private goal association configuration differs".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        // `learning_enabled` is a runtime intervention, not part of the
        // anatomical/rule identity. Snapshots still persist its exact value.
        let immutable = (
            &self.objective_sha256,
            &self.feature_names,
            self.slots,
            self.horizon_ticks,
            self.return_scale.to_bits(),
            self.learning_rate.to_bits(),
            self.weight_norm_limit.to_bits(),
            self.logit_gain.to_bits(),
        );
        let encoded = serde_json::to_vec(&immutable).map_err(|error| error.to_string())?;
        Ok(hex_sha256(&encoded))
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalSlotIdentity {
    pub recorded_tick: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalSlotReplacement {
    pub resident: usize,
    pub slot: usize,
    pub identity: GoalSlotIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalStart {
    pub resident: usize,
    pub slot: usize,
    pub identity: GoalSlotIdentity,
    /// Tick of the first physical transition attributed to this goal.
    pub selected_at_tick: u64,
    /// Ordered `[energy, gut, fatigue]`, each in `[0,1]`.
    pub physiology: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalReward {
    pub resident: usize,
    /// Tick of the committed physical transition whose outcome was measured.
    pub transition_tick: u64,
    pub objective_transition_reward: f64,
}

/// Actual body measurements for one committed physical transition. Nutrition
/// is intentionally absent: ingested energy is already represented by the
/// measured after-state gut load and is never paid again here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalTransition {
    pub resident: usize,
    pub transition_tick: u64,
    pub before: [f64; 3],
    pub after: [f64; 3],
    pub effort: f64,
    pub dt: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteEnergyComponents {
    pub reserve_before: f64,
    pub reserve_after: f64,
    pub reserve_shortfall_before: f64,
    pub reserve_shortfall_after: f64,
    pub fatigue_cost_before: f64,
    pub fatigue_cost_after: f64,
    pub gut_overload_cost_before: f64,
    pub gut_overload_cost_after: f64,
    pub potential_before: f64,
    pub potential_after: f64,
    pub potential_delta_energy: f64,
    pub effort_cost_energy: f64,
    pub hunger_gate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GoalTransitionOutcome {
    /// The current objective returns float32 rewards; storing this as `f32`
    /// preserves the exact value accumulated by the prior Python boundary.
    pub reward: f32,
    pub components: FiniteEnergyComponents,
    pub receipt: Option<GoalOutcomeReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersonalGoalStats {
    pub completed_goals: u64,
    pub learned_goals: u64,
    pub frozen_goals: u64,
    pub skipped_replaced_goals: u64,
    pub cancelled_goals: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoalSelectionEstimate {
    pub predicted_normalized_return: f64,
    pub logit_bias: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoalOutcomeReceipt {
    pub slot: usize,
    pub identity: GoalSlotIdentity,
    pub selected_at_tick: u64,
    pub completed_at_tick: u64,
    pub transition_count: u8,
    #[serde(with = "f64_bits")]
    pub summed_objective_return: f64,
    #[serde(with = "f64_bits")]
    pub normalized_target: f64,
    #[serde(with = "f64_bits")]
    pub prediction_before_update: f64,
    pub attributed: bool,
    pub learned: bool,
    pub slot_was_replaced: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SlotModel {
    valid: bool,
    identity: GoalSlotIdentity,
    #[serde(with = "f64_array4_bits")]
    weights: [f64; PERSONAL_GOAL_FEATURES],
}

impl SlotModel {
    fn empty() -> Self {
        Self {
            valid: false,
            identity: GoalSlotIdentity {
                recorded_tick: 0,
                generation: 0,
            },
            weights: [0.0; PERSONAL_GOAL_FEATURES],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct PendingGoal {
    slot: usize,
    identity: GoalSlotIdentity,
    selected_at_tick: u64,
    #[serde(with = "f64_array4_bits")]
    features: [f64; PERSONAL_GOAL_FEATURES],
    #[serde(with = "f64_bits")]
    prediction_before_update: f64,
    observed_ticks: u8,
    #[serde(with = "f64_bits")]
    summed_return: f64,
    slot_was_replaced: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Individual {
    slots: Vec<SlotModel>,
    pending: Option<PendingGoal>,
    completed_goals: u64,
    learned_goals: u64,
    skipped_replaced_goals: u64,
    frozen_goals: u64,
    cancelled_goals: u64,
    last_receipt: Option<GoalOutcomeReceipt>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PersonalGoalAssociations {
    schema: String,
    config: PersonalGoalConfig,
    config_sha256: String,
    residents: usize,
    individuals: Vec<Individual>,
}

fn hex_sha256(value: &[u8]) -> String {
    Sha256::digest(value)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn finite_energy_objective_identity() -> Result<String, String> {
    let identity = FiniteEnergyIdentity {
        config: FiniteEnergyCoefficientIdentity {
            assimilation_efficiency: ASSIMILATION_EFFICIENCY,
            effort_energy_rate: EFFORT_ENERGY_RATE,
            effort_extra_weight: EFFORT_EXTRA_WEIGHT,
            fatigue_energy_weight: FATIGUE_ENERGY_WEIGHT,
            gut_comfort: GUT_COMFORT,
            gut_overload_energy_weight: GUT_OVERLOAD_ENERGY_WEIGHT,
            max_interval_seconds: MAX_INTERVAL_SECONDS,
            reserve_target: RESERVE_TARGET,
            reserve_temperature: RESERVE_TEMPERATURE,
            reward_per_energy: REWARD_PER_ENERGY,
            version: 1,
        },
        format: FINITE_ENERGY_FORMAT,
    };
    serde_json::to_vec(&identity)
        .map(|encoded| hex_sha256(&encoded))
        .map_err(|error| error.to_string())
}

fn features(physiology: [f64; 3]) -> Result<[f64; PERSONAL_GOAL_FEATURES], String> {
    if physiology
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("goal-selection physiology must be finite and in [0,1]".into());
    }
    Ok([
        1.0,
        2.0 * physiology[0] - 1.0,
        2.0 * physiology[1] - 1.0,
        2.0 * physiology[2] - 1.0,
    ])
}

fn dot(weights: &[f64; PERSONAL_GOAL_FEATURES], x: &[f64; PERSONAL_GOAL_FEATURES]) -> f64 {
    weights
        .iter()
        .zip(x)
        .map(|(weight, value)| weight * value)
        .sum()
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

#[derive(Clone, Copy)]
struct FiniteEnergyState {
    reserve: f64,
    reserve_shortfall: f64,
    fatigue_cost: f64,
    gut_overload_cost: f64,
    potential: f64,
}

fn finite_energy_potential(physiology: [f64; 3]) -> Result<FiniteEnergyState, String> {
    features(physiology)?;
    let reserve = physiology[0] + ASSIMILATION_EFFICIENCY * physiology[1];
    let shortfall =
        RESERVE_TEMPERATURE * softplus((RESERVE_TARGET - reserve) / RESERVE_TEMPERATURE);
    let fatigue_cost = FATIGUE_ENERGY_WEIGHT * physiology[2] * physiology[2];
    let gut_excess = (physiology[1] - GUT_COMFORT).max(0.0);
    let gut_overload_cost = GUT_OVERLOAD_ENERGY_WEIGHT * gut_excess * gut_excess;
    Ok(FiniteEnergyState {
        reserve,
        reserve_shortfall: shortfall,
        fatigue_cost,
        gut_overload_cost,
        potential: -shortfall - fatigue_cost - gut_overload_cost,
    })
}

/// Native port of the current `FiniteEnergyObjective.transition` arithmetic.
/// Its coefficients are authenticated by `FINITE_ENERGY_OBJECTIVE_SHA256`.
pub fn finite_energy_transition(
    transition: &GoalTransition,
) -> Result<(f32, FiniteEnergyComponents), String> {
    if !transition.dt.is_finite()
        || !(0.0..=MAX_INTERVAL_SECONDS).contains(&transition.dt)
        || transition.dt == 0.0
    {
        return Err("dt must be finite and in (0,2] seconds".into());
    }
    if !transition.effort.is_finite() || !(0.0..=1.0).contains(&transition.effort) {
        return Err("effort must be finite and in [0,1]".into());
    }
    let before = finite_energy_potential(transition.before)?;
    let after = finite_energy_potential(transition.after)?;
    let potential_delta_energy = after.potential - before.potential;
    let effort_cost_energy =
        EFFORT_EXTRA_WEIGHT * EFFORT_ENERGY_RATE * transition.effort * transition.dt;
    let reward = (REWARD_PER_ENERGY * (potential_delta_energy - effort_cost_energy)) as f32;
    let hunger_argument =
        ((before.reserve - RESERVE_TARGET) / RESERVE_TEMPERATURE).clamp(-60.0, 60.0);
    let hunger_gate = 1.0 / (1.0 + hunger_argument.exp());
    Ok((
        reward,
        FiniteEnergyComponents {
            reserve_before: before.reserve,
            reserve_after: after.reserve,
            reserve_shortfall_before: before.reserve_shortfall,
            reserve_shortfall_after: after.reserve_shortfall,
            fatigue_cost_before: before.fatigue_cost,
            fatigue_cost_after: after.fatigue_cost,
            gut_overload_cost_before: before.gut_overload_cost,
            gut_overload_cost_after: after.gut_overload_cost,
            potential_before: before.potential,
            potential_after: after.potential,
            potential_delta_energy,
            effort_cost_energy,
            hunger_gate,
        },
    ))
}

impl PersonalGoalAssociations {
    pub fn new(residents: usize, config: PersonalGoalConfig) -> Result<Self, String> {
        config.validate()?;
        if !(1..=4096).contains(&residents) {
            return Err("invalid private goal association cohort size".into());
        }
        let individual = Individual {
            slots: vec![SlotModel::empty(); config.slots],
            pending: None,
            completed_goals: 0,
            learned_goals: 0,
            skipped_replaced_goals: 0,
            frozen_goals: 0,
            cancelled_goals: 0,
            last_receipt: None,
        };
        Ok(Self {
            schema: FORMAT.into(),
            config_sha256: config.identity()?,
            config,
            residents,
            individuals: vec![individual; residents],
        })
    }

    pub fn config(&self) -> &PersonalGoalConfig {
        &self.config
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn residents(&self) -> usize {
        self.residents
    }

    /// Enable or freeze NLMS updates without changing the rule identity.
    /// Pending goals continue to consume their full outcome horizon and emit
    /// receipts while frozen.
    pub fn set_learning_enabled(&mut self, enabled: bool) {
        self.config.learning_enabled = enabled;
    }

    pub fn stats(&self, resident: usize) -> Result<PersonalGoalStats, String> {
        let individual = self.individual(resident)?;
        Ok(PersonalGoalStats {
            completed_goals: individual.completed_goals,
            learned_goals: individual.learned_goals,
            frozen_goals: individual.frozen_goals,
            skipped_replaced_goals: individual.skipped_replaced_goals,
            cancelled_goals: individual.cancelled_goals,
        })
    }

    fn individual(&self, resident: usize) -> Result<&Individual, String> {
        self.individuals
            .get(resident)
            .ok_or("unknown resident".into())
    }

    fn individual_mut(&mut self, resident: usize) -> Result<&mut Individual, String> {
        self.individuals
            .get_mut(resident)
            .ok_or("unknown resident".into())
    }

    fn validate_identity(identity: GoalSlotIdentity) -> Result<(), String> {
        if identity.generation == 0 {
            return Err("goal slot generation must be positive".into());
        }
        Ok(())
    }

    fn check_replacement(&self, replacement: &GoalSlotReplacement) -> Result<(), String> {
        Self::validate_identity(replacement.identity)?;
        let individual = self.individual(replacement.resident)?;
        if replacement.slot >= individual.slots.len() {
            return Err("goal slot is outside the configured reservoir".into());
        }
        Ok(())
    }

    fn apply_replacement(&mut self, replacement: GoalSlotReplacement) -> bool {
        let individual = &mut self.individuals[replacement.resident];
        let slot = &mut individual.slots[replacement.slot];
        if slot.valid && slot.identity == replacement.identity {
            return false;
        }
        if let Some(pending) = individual.pending.as_mut() {
            if pending.slot == replacement.slot && pending.identity != replacement.identity {
                pending.slot_was_replaced = true;
            }
        }
        *slot = SlotModel {
            valid: true,
            identity: replacement.identity,
            weights: [0.0; PERSONAL_GOAL_FEATURES],
        };
        true
    }

    /// Register reservoir insertions/replacements. Changed identities clear
    /// that slot's learned values. A batch may contain at most one row per
    /// resident/slot and is validated before any mutation.
    pub fn replace_slots(
        &mut self,
        replacements: &[GoalSlotReplacement],
    ) -> Result<Vec<bool>, String> {
        if replacements.len() > self.residents * self.config.slots {
            return Err("goal slot replacement batch exceeds cohort capacity".into());
        }
        let mut keys = Vec::with_capacity(replacements.len());
        for replacement in replacements {
            self.check_replacement(replacement)?;
            keys.push((replacement.resident, replacement.slot));
        }
        keys.sort_unstable();
        if keys.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("goal slot replacement batch contains duplicates".into());
        }
        Ok(replacements
            .iter()
            .map(|replacement| self.apply_replacement(*replacement))
            .collect())
    }

    fn check_start(&self, request: &GoalStart) -> Result<[f64; PERSONAL_GOAL_FEATURES], String> {
        Self::validate_identity(request.identity)?;
        let x = features(request.physiology)?;
        let individual = self.individual(request.resident)?;
        request
            .selected_at_tick
            .checked_add(self.config.horizon_ticks as u64 - 1)
            .ok_or("selected goal horizon exceeds the tick range")?;
        if individual.pending.is_some() {
            return Err("a selected goal already has pending outcome credit".into());
        }
        let slot = individual
            .slots
            .get(request.slot)
            .ok_or("goal slot is outside the configured reservoir")?;
        if !slot.valid || slot.identity != request.identity {
            return Err("selected goal does not match the current slot identity".into());
        }
        Ok(x)
    }

    fn apply_start(
        &mut self,
        request: GoalStart,
        x: [f64; PERSONAL_GOAL_FEATURES],
    ) -> GoalSelectionEstimate {
        let individual = &mut self.individuals[request.resident];
        let predicted = dot(&individual.slots[request.slot].weights, &x);
        let estimate = GoalSelectionEstimate {
            predicted_normalized_return: predicted,
            logit_bias: self.config.logit_gain * predicted.tanh(),
        };
        individual.pending = Some(PendingGoal {
            slot: request.slot,
            identity: request.identity,
            selected_at_tick: request.selected_at_tick,
            features: x,
            prediction_before_update: predicted,
            observed_ticks: 0,
            summed_return: 0.0,
            slot_was_replaced: false,
        });
        estimate
    }

    /// Begin exact ten-transition credit for newly selected goals.
    pub fn begin_goals(
        &mut self,
        requests: &[GoalStart],
    ) -> Result<Vec<GoalSelectionEstimate>, String> {
        if requests.len() > self.residents {
            return Err("goal-start batch exceeds resident cohort".into());
        }
        let mut seen = vec![false; self.residents];
        let mut checked = Vec::with_capacity(requests.len());
        for request in requests {
            if request.resident >= self.residents {
                return Err("unknown resident".into());
            }
            if seen[request.resident] {
                return Err("goal-start batch contains a duplicate resident".into());
            }
            seen[request.resident] = true;
            checked.push(self.check_start(request)?);
        }
        Ok(requests
            .iter()
            .copied()
            .zip(checked)
            .map(|(request, x)| self.apply_start(request, x))
            .collect())
    }

    /// Return one bounded bias per reservoir slot. A model contributes only
    /// when the caller's current memory identity still matches it.
    pub fn selection_biases(
        &self,
        resident: usize,
        recorded_ticks: &[u64],
        generations: &[u64],
        physiology: [f64; 3],
    ) -> Result<Vec<f64>, String> {
        let x = features(physiology)?;
        let individual = self.individual(resident)?;
        if recorded_ticks.len() != self.config.slots || generations.len() != self.config.slots {
            return Err("goal selection slot identity arrays have the wrong length".into());
        }
        Ok(individual
            .slots
            .iter()
            .zip(recorded_ticks.iter().zip(generations))
            .map(|(slot, (recorded_tick, generation))| {
                if slot.valid
                    && slot.identity.recorded_tick == *recorded_tick
                    && slot.identity.generation == *generation
                {
                    self.config.logit_gain * dot(&slot.weights, &x).tanh()
                } else {
                    0.0
                }
            })
            .collect())
    }

    fn check_reward(&self, observation: &GoalReward) -> Result<(), String> {
        if !observation.objective_transition_reward.is_finite() {
            return Err("homeostatic transition reward must be finite".into());
        }
        let pending = self
            .individual(observation.resident)?
            .pending
            .as_ref()
            .ok_or("no selected goal has pending outcome credit")?;
        let expected = pending
            .selected_at_tick
            .checked_add(pending.observed_ticks as u64)
            .ok_or("goal outcome tick overflow")?;
        if observation.transition_tick != expected {
            return Err("goal outcome tick is not the next selected physical transition".into());
        }
        if !(pending.summed_return + observation.objective_transition_reward).is_finite() {
            return Err("summed homeostatic return exceeds the numerical range".into());
        }
        Ok(())
    }

    fn apply_reward(&mut self, observation: GoalReward) -> Option<GoalOutcomeReceipt> {
        let individual = &mut self.individuals[observation.resident];
        let pending = individual.pending.as_mut().unwrap();
        pending.summed_return += observation.objective_transition_reward;
        pending.observed_ticks += 1;
        if pending.observed_ticks != self.config.horizon_ticks {
            return None;
        }

        let pending = individual.pending.take().unwrap();
        let slot_matches = individual.slots[pending.slot].valid
            && individual.slots[pending.slot].identity == pending.identity;
        let attributed = slot_matches && !pending.slot_was_replaced;
        let normalized_target = (pending.summed_return / self.config.return_scale).clamp(-1.0, 1.0);
        let learned = attributed && self.config.learning_enabled;
        if learned {
            let slot = &mut individual.slots[pending.slot];
            let prediction = dot(&slot.weights, &pending.features);
            let error = normalized_target - prediction;
            let norm_squared = pending
                .features
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let amount = self.config.learning_rate * error / (1.0 + norm_squared);
            for (weight, value) in slot.weights.iter_mut().zip(pending.features) {
                *weight += amount * value;
            }
            let norm = slot
                .weights
                .iter()
                .map(|weight| weight * weight)
                .sum::<f64>()
                .sqrt();
            if norm > self.config.weight_norm_limit {
                let scale = self.config.weight_norm_limit / norm;
                slot.weights.iter_mut().for_each(|weight| *weight *= scale);
            }
            individual.learned_goals = individual.learned_goals.saturating_add(1);
        } else if attributed {
            individual.frozen_goals = individual.frozen_goals.saturating_add(1);
        } else {
            individual.skipped_replaced_goals = individual.skipped_replaced_goals.saturating_add(1);
        }
        individual.completed_goals = individual.completed_goals.saturating_add(1);
        let receipt = GoalOutcomeReceipt {
            slot: pending.slot,
            identity: pending.identity,
            selected_at_tick: pending.selected_at_tick,
            completed_at_tick: observation.transition_tick,
            transition_count: pending.observed_ticks,
            summed_objective_return: pending.summed_return,
            normalized_target,
            prediction_before_update: pending.prediction_before_update,
            attributed,
            learned,
            slot_was_replaced: pending.slot_was_replaced || !slot_matches,
        };
        individual.last_receipt = Some(receipt.clone());
        Some(receipt)
    }

    /// Consume actual objective rewards for committed transitions. The batch
    /// is validated before mutation and contains at most one row per resident.
    pub fn observe_rewards(
        &mut self,
        observations: &[GoalReward],
    ) -> Result<Vec<Option<GoalOutcomeReceipt>>, String> {
        if observations.len() > self.residents {
            return Err("goal reward batch exceeds resident cohort".into());
        }
        let mut seen = vec![false; self.residents];
        for observation in observations {
            if observation.resident >= self.residents {
                return Err("unknown resident".into());
            }
            if seen[observation.resident] {
                return Err("goal reward batch contains a duplicate resident".into());
            }
            seen[observation.resident] = true;
            self.check_reward(observation)?;
        }
        Ok(observations
            .iter()
            .copied()
            .map(|observation| self.apply_reward(observation))
            .collect())
    }

    /// Compute the authenticated finite-energy reward from actual body
    /// measurements, then apply exact pending-goal credit. All transitions and
    /// pending ticks are validated before any association state is mutated.
    pub fn observe_transitions(
        &mut self,
        transitions: &[GoalTransition],
    ) -> Result<Vec<GoalTransitionOutcome>, String> {
        if transitions.len() > self.residents {
            return Err("goal transition batch exceeds resident cohort".into());
        }
        let mut rewards = Vec::with_capacity(transitions.len());
        let mut components = Vec::with_capacity(transitions.len());
        for transition in transitions {
            let (reward, terms) = finite_energy_transition(transition)?;
            rewards.push(GoalReward {
                resident: transition.resident,
                transition_tick: transition.transition_tick,
                objective_transition_reward: reward as f64,
            });
            components.push((reward, terms));
        }
        // `observe_rewards` performs cohort bounds, duplicate-resident, and
        // pending tick validation across the complete batch before mutation.
        let receipts = self.observe_rewards(&rewards)?;
        Ok(components
            .into_iter()
            .zip(receipts)
            .map(|((reward, components), receipt)| GoalTransitionOutcome {
                reward,
                components,
                receipt,
            })
            .collect())
    }

    /// Cancel credit on a reset or discontinuity without changing learned slot
    /// models. Returns whether a pending goal was cancelled.
    pub fn cancel_pending(&mut self, resident: usize) -> Result<bool, String> {
        let individual = self.individual_mut(resident)?;
        let cancelled = individual.pending.take().is_some();
        if cancelled {
            individual.cancelled_goals = individual.cancelled_goals.saturating_add(1);
        }
        Ok(cancelled)
    }

    pub fn last_receipt(&self, resident: usize) -> Result<Option<&GoalOutcomeReceipt>, String> {
        Ok(self.individual(resident)?.last_receipt.as_ref())
    }

    pub fn snapshot(&self) -> Result<String, String> {
        self.validate_state()?;
        serde_json::to_string(self).map_err(|error| error.to_string())
    }

    pub fn restore(
        value: &str,
        config: &PersonalGoalConfig,
        residents: usize,
    ) -> Result<Self, String> {
        config.validate()?;
        let state: Self = serde_json::from_str(value).map_err(|error| error.to_string())?;
        if state.schema != FORMAT
            || state.config.identity()? != config.identity()?
            || state.config_sha256 != config.identity()?
            || state.residents != residents
        {
            return Err("private goal snapshot belongs to another contract".into());
        }
        state.validate_state()?;
        Ok(state)
    }

    fn validate_state(&self) -> Result<(), String> {
        self.config.validate()?;
        if self.schema != FORMAT
            || self.config_sha256 != self.config.identity()?
            || self.individuals.len() != self.residents
            || !(1..=4096).contains(&self.residents)
        {
            return Err("private goal association state header differs".into());
        }
        for individual in &self.individuals {
            if individual.slots.len() != self.config.slots {
                return Err("private goal association slot count differs".into());
            }
            for slot in &individual.slots {
                let norm = slot
                    .weights
                    .iter()
                    .map(|weight| weight * weight)
                    .sum::<f64>()
                    .sqrt();
                if slot.weights.iter().any(|weight| !weight.is_finite())
                    || norm > self.config.weight_norm_limit * (1.0 + 1e-12)
                    || (!slot.valid
                        && (slot.identity.generation != 0
                            || slot.weights.iter().any(|weight| *weight != 0.0)))
                    || (slot.valid && slot.identity.generation == 0)
                {
                    return Err("private goal slot state is invalid".into());
                }
            }
            if let Some(pending) = &individual.pending {
                if pending.slot >= self.config.slots
                    || pending.identity.generation == 0
                    || pending.observed_ticks >= self.config.horizon_ticks
                    || !pending.summed_return.is_finite()
                    || !pending.prediction_before_update.is_finite()
                    || pending.features.iter().any(|value| !value.is_finite())
                    || (!pending.slot_was_replaced
                        && (!individual.slots[pending.slot].valid
                            || individual.slots[pending.slot].identity != pending.identity))
                {
                    return Err("pending private goal outcome state is invalid".into());
                }
            }
            if let Some(receipt) = &individual.last_receipt {
                if receipt.transition_count != self.config.horizon_ticks
                    || receipt.identity.generation == 0
                    || !receipt.summed_objective_return.is_finite()
                    || !receipt.normalized_target.is_finite()
                    || !receipt.prediction_before_update.is_finite()
                    || receipt.normalized_target.abs() > 1.0
                {
                    return Err("private goal outcome receipt is invalid".into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These float32 values were produced by the repository's current
    // FiniteEnergyObjective.transition. The mechanism consumes them without
    // reconstructing or augmenting the objective.
    const RESTORATIVE: [f64; 10] = [
        0.03414672613143921,
        0.034137967973947525,
        0.03412896767258644,
        0.03411971405148506,
        0.03411019593477249,
        0.03410040959715843,
        0.03409034013748169,
        0.034079983830451965,
        0.034069325774908066,
        0.034058354794979095,
    ];
    const COSTLY: [f64; 10] = [
        -0.025145791471004486,
        -0.025156665593385696,
        -0.02516746148467064,
        -0.025178181007504463,
        -0.025188827887177467,
        -0.025199400261044502,
        -0.025209901854395866,
        -0.025220336392521858,
        -0.02523070201277733,
        -0.025241000577807426,
    ];

    #[test]
    fn two_histories_shift_retrieval_and_replacement_blocks_credit() {
        let config = PersonalGoalConfig::current(true);
        let mut model = PersonalGoalAssociations::new(1, config.clone()).unwrap();
        let positive = GoalSlotIdentity {
            recorded_tick: 20,
            generation: 4,
        };
        let negative = GoalSlotIdentity {
            recorded_tick: 40,
            generation: 8,
        };
        model
            .replace_slots(&[
                GoalSlotReplacement {
                    resident: 0,
                    slot: 0,
                    identity: positive,
                },
                GoalSlotReplacement {
                    resident: 0,
                    slot: 1,
                    identity: negative,
                },
            ])
            .unwrap();
        for (slot, identity, start, rewards) in [
            (0, positive, 100, RESTORATIVE.as_slice()),
            (1, negative, 200, COSTLY.as_slice()),
        ] {
            model
                .begin_goals(&[GoalStart {
                    resident: 0,
                    slot,
                    identity,
                    selected_at_tick: start,
                    physiology: [0.35, 0.10, 0.20],
                }])
                .unwrap();
            let mut final_receipt = None;
            for (offset, reward) in rewards.iter().enumerate() {
                final_receipt = model
                    .observe_rewards(&[GoalReward {
                        resident: 0,
                        transition_tick: start + offset as u64,
                        objective_transition_reward: *reward,
                    }])
                    .unwrap()[0]
                    .clone();
            }
            let receipt = final_receipt.unwrap();
            let expected: f64 = rewards.iter().sum();
            assert!((receipt.summed_objective_return - expected).abs() < 1e-15);
            assert_eq!(
                receipt.normalized_target,
                if slot == 0 { 1.0 } else { -1.0 }
            );
            assert!(receipt.attributed && receipt.learned);
        }
        let ticks = [20, 40]
            .into_iter()
            .chain(std::iter::repeat(0).take(PERSONAL_GOAL_SLOTS - 2))
            .collect::<Vec<_>>();
        let generations = [4, 8]
            .into_iter()
            .chain(std::iter::repeat(0).take(PERSONAL_GOAL_SLOTS - 2))
            .collect::<Vec<_>>();
        let biases = model
            .selection_biases(0, &ticks, &generations, [0.35, 0.10, 0.20])
            .unwrap();
        assert!(biases[0] > 0.0 && biases[1] < 0.0);
        assert!(biases.iter().all(|bias| bias.abs() <= 0.35));

        model
            .begin_goals(&[GoalStart {
                resident: 0,
                slot: 0,
                identity: positive,
                selected_at_tick: 300,
                physiology: [0.35, 0.10, 0.20],
            }])
            .unwrap();
        for offset in 0..5 {
            assert!(model
                .observe_rewards(&[GoalReward {
                    resident: 0,
                    transition_tick: 300 + offset,
                    objective_transition_reward: RESTORATIVE[offset as usize],
                }])
                .unwrap()[0]
                .is_none());
        }
        let pending_snapshot = model.snapshot().unwrap();
        let mut replay = PersonalGoalAssociations::restore(&pending_snapshot, &config, 1).unwrap();
        let replacement = GoalSlotIdentity {
            recorded_tick: 301,
            generation: 9,
        };
        let mut skipped = [None, None];
        for (branch, result) in [&mut model, &mut replay].into_iter().enumerate() {
            result
                .replace_slots(&[GoalSlotReplacement {
                    resident: 0,
                    slot: 0,
                    identity: replacement,
                }])
                .unwrap();
            for offset in 5..10 {
                skipped[branch] = result
                    .observe_rewards(&[GoalReward {
                        resident: 0,
                        transition_tick: 300 + offset,
                        objective_transition_reward: RESTORATIVE[offset as usize],
                    }])
                    .unwrap()[0]
                    .clone();
            }
        }
        assert_eq!(model, replay);
        let receipt = skipped[0].as_ref().unwrap();
        assert!(!receipt.attributed && !receipt.learned && receipt.slot_was_replaced);

        let snapshot = model.snapshot().unwrap();
        let restored = PersonalGoalAssociations::restore(&snapshot, &config, 1).unwrap();
        assert_eq!(model, restored);

        let frozen_config = PersonalGoalConfig::current(false);
        let mut frozen = PersonalGoalAssociations::new(1, frozen_config).unwrap();
        frozen
            .replace_slots(&[GoalSlotReplacement {
                resident: 0,
                slot: 0,
                identity: positive,
            }])
            .unwrap();
        frozen
            .begin_goals(&[GoalStart {
                resident: 0,
                slot: 0,
                identity: positive,
                selected_at_tick: 400,
                physiology: [0.35, 0.10, 0.20],
            }])
            .unwrap();
        let mut frozen_receipt = None;
        for offset in 0..10 {
            frozen_receipt = frozen
                .observe_rewards(&[GoalReward {
                    resident: 0,
                    transition_tick: 400 + offset,
                    objective_transition_reward: RESTORATIVE[offset as usize],
                }])
                .unwrap()[0]
                .clone();
        }
        let frozen_receipt = frozen_receipt.unwrap();
        assert!(frozen_receipt.attributed && !frozen_receipt.learned);
        assert_eq!(frozen.last_receipt(0).unwrap(), Some(&frozen_receipt));
    }
}
