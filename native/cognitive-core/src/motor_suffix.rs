// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded private memory of action suffixes learned only from physical execution.

use serde::{Deserialize, Serialize};

pub(crate) const ACTIONS: usize = 12;
pub(crate) const CONTEXT: usize = 64;
pub(crate) const OUTCOMES: usize = 3;
pub(crate) const MAX_HORIZON: usize = 8;
pub(crate) const SLOTS: usize = 32;
const FORMAT: &str = "chreatures-private-motor-suffix-v1";

#[derive(Clone, Debug)]
pub(crate) struct RecalledSuffix {
    pub slot: usize,
    pub generation: u64,
    pub length: usize,
    pub recall_score: f32,
    pub empirical_utility: f32,
    pub support: u32,
    pub actions: [f32; MAX_HORIZON * ACTIONS],
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MotorSuffixMemory {
    format: String,
    batch: usize,
    valid: Vec<bool>,
    generation: Vec<u64>,
    length: Vec<u8>,
    context: Vec<f32>,
    actions: Vec<f32>,
    outcomes: Vec<f32>,
    recorded_tick: Vec<u64>,
    support: Vec<u32>,
    proposals: Vec<u32>,
    executions: Vec<u32>,
    capture_context: Vec<f32>,
    capture_actions: Vec<f32>,
    capture_outcomes: Vec<f32>,
    capture_ticks: Vec<u64>,
    capture_cursor: Vec<usize>,
    capture_count: Vec<usize>,
    seen_sequences: Vec<u64>,
    rng: Vec<u64>,
    learned_total: Vec<u64>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl MotorSuffixMemory {
    pub(crate) fn new(batch: usize, seed: u64) -> Result<Self, String> {
        if batch == 0 || batch > 4096 {
            return Err("motor suffix batch differs".into());
        }
        let mut rng = vec![0; batch];
        for (row, value) in rng.iter_mut().enumerate() {
            let mut state = seed ^ (row as u64).wrapping_mul(0xd134_2543_de82_ef95);
            *value = splitmix64(&mut state);
        }
        Ok(Self {
            format: FORMAT.into(),
            batch,
            valid: vec![false; batch * SLOTS],
            generation: vec![0; batch * SLOTS],
            length: vec![0; batch * SLOTS],
            context: vec![0.0; batch * SLOTS * CONTEXT],
            actions: vec![0.0; batch * SLOTS * MAX_HORIZON * ACTIONS],
            outcomes: vec![0.0; batch * SLOTS * OUTCOMES],
            recorded_tick: vec![0; batch * SLOTS],
            support: vec![0; batch * SLOTS],
            proposals: vec![0; batch * SLOTS],
            executions: vec![0; batch * SLOTS],
            capture_context: vec![0.0; batch * MAX_HORIZON * CONTEXT],
            capture_actions: vec![0.0; batch * MAX_HORIZON * ACTIONS],
            capture_outcomes: vec![0.0; batch * MAX_HORIZON * OUTCOMES],
            capture_ticks: vec![0; batch * MAX_HORIZON],
            capture_cursor: vec![0; batch],
            capture_count: vec![0; batch],
            seen_sequences: vec![0; batch],
            rng,
            learned_total: vec![0; batch],
        })
    }

    pub(crate) fn grow(&mut self, new_batch: usize, seed: u64) -> Result<(), String> {
        if new_batch <= self.batch || new_batch > 4096 {
            return Err("motor suffix growth differs".into());
        }
        let old = self.batch;
        macro_rules! grow {
            ($field:ident, $stride:expr, $value:expr) => {
                self.$field.resize(new_batch * $stride, $value)
            };
        }
        grow!(valid, SLOTS, false);
        grow!(generation, SLOTS, 0);
        grow!(length, SLOTS, 0);
        grow!(context, SLOTS * CONTEXT, 0.0);
        grow!(actions, SLOTS * MAX_HORIZON * ACTIONS, 0.0);
        grow!(outcomes, SLOTS * OUTCOMES, 0.0);
        grow!(recorded_tick, SLOTS, 0);
        grow!(support, SLOTS, 0);
        grow!(proposals, SLOTS, 0);
        grow!(executions, SLOTS, 0);
        grow!(capture_context, MAX_HORIZON * CONTEXT, 0.0);
        grow!(capture_actions, MAX_HORIZON * ACTIONS, 0.0);
        grow!(capture_outcomes, MAX_HORIZON * OUTCOMES, 0.0);
        grow!(capture_ticks, MAX_HORIZON, 0);
        self.capture_cursor.resize(new_batch, 0);
        self.capture_count.resize(new_batch, 0);
        self.seen_sequences.resize(new_batch, 0);
        self.rng.resize(new_batch, 0);
        self.learned_total.resize(new_batch, 0);
        for row in old..new_batch {
            let mut state = seed ^ (row as u64).wrapping_mul(0xd134_2543_de82_ef95);
            self.rng[row] = splitmix64(&mut state);
        }
        self.batch = new_batch;
        Ok(())
    }

    pub(crate) fn clear_resident(&mut self, row: usize, seed: u64) -> Result<(), String> {
        if row >= self.batch {
            return Err("motor suffix resident differs".into());
        }
        let slots = row * SLOTS..(row + 1) * SLOTS;
        self.valid[slots.clone()].fill(false);
        self.generation[slots.clone()].fill(0);
        self.length[slots.clone()].fill(0);
        self.recorded_tick[slots.clone()].fill(0);
        self.support[slots.clone()].fill(0);
        self.proposals[slots.clone()].fill(0);
        self.executions[slots].fill(0);
        self.context[row * SLOTS * CONTEXT..(row + 1) * SLOTS * CONTEXT].fill(0.0);
        self.actions
            [row * SLOTS * MAX_HORIZON * ACTIONS..(row + 1) * SLOTS * MAX_HORIZON * ACTIONS]
            .fill(0.0);
        self.outcomes[row * SLOTS * OUTCOMES..(row + 1) * SLOTS * OUTCOMES].fill(0.0);
        self.capture_context[row * MAX_HORIZON * CONTEXT..(row + 1) * MAX_HORIZON * CONTEXT]
            .fill(0.0);
        self.capture_actions[row * MAX_HORIZON * ACTIONS..(row + 1) * MAX_HORIZON * ACTIONS]
            .fill(0.0);
        self.capture_outcomes[row * MAX_HORIZON * OUTCOMES..(row + 1) * MAX_HORIZON * OUTCOMES]
            .fill(0.0);
        self.capture_ticks[row * MAX_HORIZON..(row + 1) * MAX_HORIZON].fill(0);
        self.capture_cursor[row] = 0;
        self.capture_count[row] = 0;
        self.seen_sequences[row] = 0;
        self.learned_total[row] = 0;
        let mut state = seed ^ (row as u64).wrapping_mul(0xd134_2543_de82_ef95);
        self.rng[row] = splitmix64(&mut state);
        Ok(())
    }

    pub(crate) fn reset_episode(&mut self, row: usize) -> Result<(), String> {
        if row >= self.batch {
            return Err("motor suffix resident differs".into());
        }
        self.capture_context[row * MAX_HORIZON * CONTEXT..(row + 1) * MAX_HORIZON * CONTEXT]
            .fill(0.0);
        self.capture_actions[row * MAX_HORIZON * ACTIONS..(row + 1) * MAX_HORIZON * ACTIONS]
            .fill(0.0);
        self.capture_outcomes[row * MAX_HORIZON * OUTCOMES..(row + 1) * MAX_HORIZON * OUTCOMES]
            .fill(0.0);
        self.capture_ticks[row * MAX_HORIZON..(row + 1) * MAX_HORIZON].fill(0);
        self.capture_cursor[row] = 0;
        self.capture_count[row] = 0;
        Ok(())
    }

    pub(crate) fn record_executed(
        &mut self,
        row: usize,
        tick: u64,
        context: &[f32],
        action: &[f32],
        outcome: &[f32],
    ) -> Result<(), String> {
        if row >= self.batch
            || context.len() != CONTEXT
            || action.len() != ACTIONS
            || outcome.len() != OUTCOMES
            || context
                .iter()
                .chain(action)
                .chain(outcome)
                .any(|x| !x.is_finite())
        {
            return Err("motor suffix execution differs".into());
        }
        if self.capture_count[row] > 0 {
            let prior = (self.capture_cursor[row] + MAX_HORIZON - 1) % MAX_HORIZON;
            if tick <= self.capture_ticks[row * MAX_HORIZON + prior] {
                return Err("motor suffix ticks are not increasing".into());
            }
        }
        let cursor = self.capture_cursor[row];
        self.capture_context
            [(row * MAX_HORIZON + cursor) * CONTEXT..(row * MAX_HORIZON + cursor + 1) * CONTEXT]
            .copy_from_slice(context);
        self.capture_actions
            [(row * MAX_HORIZON + cursor) * ACTIONS..(row * MAX_HORIZON + cursor + 1) * ACTIONS]
            .copy_from_slice(action);
        self.capture_outcomes
            [(row * MAX_HORIZON + cursor) * OUTCOMES..(row * MAX_HORIZON + cursor + 1) * OUTCOMES]
            .copy_from_slice(outcome);
        self.capture_ticks[row * MAX_HORIZON + cursor] = tick;
        self.capture_cursor[row] = (cursor + 1) % MAX_HORIZON;
        self.capture_count[row] = (self.capture_count[row] + 1).min(MAX_HORIZON);
        if self.capture_count[row] < 2 {
            return Ok(());
        }
        self.seen_sequences[row] += 1;
        let seen = self.seen_sequences[row];
        let slot = if let Some(slot) = (0..SLOTS).find(|s| !self.valid[row * SLOTS + *s]) {
            slot
        } else {
            let draw = splitmix64(&mut self.rng[row]) % seen;
            if draw >= SLOTS as u64 {
                return Ok(());
            }
            draw as usize
        };
        let index = row * SLOTS + slot;
        self.generation[index] = self.generation[index].wrapping_add(1).max(1);
        self.valid[index] = true;
        let length = self.capture_count[row];
        self.length[index] = length as u8;
        self.support[index] = 1;
        self.proposals[index] = 0;
        self.executions[index] = 0;
        let first = (self.capture_cursor[row] + MAX_HORIZON - length) % MAX_HORIZON;
        let first_context = (row * MAX_HORIZON + first) * CONTEXT;
        self.context[index * CONTEXT..(index + 1) * CONTEXT]
            .copy_from_slice(&self.capture_context[first_context..first_context + CONTEXT]);
        self.actions[index * MAX_HORIZON * ACTIONS..(index + 1) * MAX_HORIZON * ACTIONS].fill(0.0);
        self.outcomes[index * OUTCOMES..(index + 1) * OUTCOMES].fill(0.0);
        for step in 0..length {
            let source = (first + step) % MAX_HORIZON;
            let src = (row * MAX_HORIZON + source) * ACTIONS;
            let dst = (index * MAX_HORIZON + step) * ACTIONS;
            self.actions[dst..dst + ACTIONS]
                .copy_from_slice(&self.capture_actions[src..src + ACTIONS]);
            for j in 0..OUTCOMES {
                self.outcomes[index * OUTCOMES + j] +=
                    self.capture_outcomes[(row * MAX_HORIZON + source) * OUTCOMES + j];
            }
        }
        self.recorded_tick[index] = tick;
        self.learned_total[row] += 1;
        Ok(())
    }

    pub(crate) fn recall(
        &mut self,
        row: usize,
        context: &[f32],
        count: usize,
    ) -> Vec<RecalledSuffix> {
        if row >= self.batch || context.len() != CONTEXT || count == 0 {
            return vec![];
        }
        let mut ranked = Vec::with_capacity(SLOTS);
        for slot in 0..SLOTS {
            let index = row * SLOTS + slot;
            if !self.valid[index] || self.length[index] < 4 {
                continue;
            }
            let distance = (self.context[index * CONTEXT..(index + 1) * CONTEXT]
                .iter()
                .zip(context)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / CONTEXT as f32)
                .sqrt();
            let outcome = &self.outcomes[index * OUTCOMES..(index + 1) * OUTCOMES];
            let utility = outcome[0] - outcome[1] + outcome[2];
            let score = -distance + 0.25 * utility.tanh();
            ranked.push((score, self.recorded_tick[index], slot));
        }
        ranked.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        ranked.truncate(count);
        ranked
            .into_iter()
            .map(|(score, _, slot)| {
                let index = row * SLOTS + slot;
                self.proposals[index] = self.proposals[index].saturating_add(1);
                let mut actions = [0.0; MAX_HORIZON * ACTIONS];
                actions.copy_from_slice(
                    &self.actions
                        [index * MAX_HORIZON * ACTIONS..(index + 1) * MAX_HORIZON * ACTIONS],
                );
                let outcome = &self.outcomes[index * OUTCOMES..(index + 1) * OUTCOMES];
                RecalledSuffix {
                    slot,
                    generation: self.generation[index],
                    length: self.length[index] as usize,
                    recall_score: score,
                    empirical_utility: (outcome[0] - outcome[1] + outcome[2]).tanh(),
                    support: self.support[index],
                    actions,
                }
            })
            .collect()
    }

    pub(crate) fn note_executed(&mut self, row: usize, slot: usize, generation: u64) {
        if row < self.batch && slot < SLOTS {
            let index = row * SLOTS + slot;
            if self.valid[index] && self.generation[index] == generation {
                self.executions[index] = self.executions[index].saturating_add(1);
                self.support[index] = self.support[index].saturating_add(1);
            }
        }
    }

    pub(crate) fn counts(&self, row: usize) -> (usize, u64) {
        (
            (0..SLOTS)
                .filter(|slot| self.valid[row * SLOTS + *slot])
                .count(),
            self.learned_total[row],
        )
    }

    pub(crate) fn snapshot_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    pub(crate) fn restore_json(value: &str, batch: usize) -> Result<Self, String> {
        let result: Self = serde_json::from_str(value).map_err(|e| e.to_string())?;
        if result.format != FORMAT || result.batch != batch {
            return Err("motor suffix snapshot identity differs".into());
        }
        let expected = Self::new(batch, 0)?;
        if result.valid.len() != expected.valid.len()
            || result.generation.len() != expected.generation.len()
            || result.length.len() != expected.length.len()
            || result.context.len() != expected.context.len()
            || result.actions.len() != expected.actions.len()
            || result.outcomes.len() != expected.outcomes.len()
            || result.recorded_tick.len() != expected.recorded_tick.len()
            || result.support.len() != expected.support.len()
            || result.proposals.len() != expected.proposals.len()
            || result.executions.len() != expected.executions.len()
            || result.capture_context.len() != expected.capture_context.len()
            || result.capture_actions.len() != expected.capture_actions.len()
            || result.capture_outcomes.len() != expected.capture_outcomes.len()
            || result.capture_ticks.len() != expected.capture_ticks.len()
            || result.capture_cursor.len() != batch
            || result.capture_count.len() != batch
            || result.seen_sequences.len() != batch
            || result.rng.len() != batch
            || result.learned_total.len() != batch
            || result.capture_cursor.iter().any(|x| *x >= MAX_HORIZON)
            || result.capture_count.iter().any(|x| *x > MAX_HORIZON)
            || result.length.iter().any(|x| *x as usize > MAX_HORIZON)
            || result.valid.iter().enumerate().any(|(index, valid)| {
                *valid
                    && (result.length[index] < 2
                        || result.generation[index] == 0
                        || result.support[index] == 0)
            })
            || result
                .context
                .iter()
                .chain(&result.actions)
                .chain(&result.outcomes)
                .chain(&result.capture_context)
                .chain(&result.capture_actions)
                .chain(&result.capture_outcomes)
                .any(|x| !x.is_finite())
        {
            return Err("motor suffix snapshot shape differs".into());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_suffix_recall_and_restore_are_exact() {
        let mut memory = MotorSuffixMemory::new(1, 17).unwrap();
        for tick in 1..=8 {
            let mut context = [0.0; CONTEXT];
            context[0] = tick as f32 * 0.01;
            let mut action = [0.0; ACTIONS];
            action[0] = tick as f32 * 0.1;
            memory
                .record_executed(0, tick, &context, &action, &[0.2, 0.05, 0.1])
                .unwrap();
        }
        let query = [0.01; CONTEXT];
        let recalled = memory.recall(0, &query, 4);
        assert!(!recalled.is_empty());
        assert!((4..=8).contains(&recalled[0].length));
        assert!(recalled[0].empirical_utility > 0.0);
        let snapshot = memory.snapshot_json().unwrap();
        let mut restored = MotorSuffixMemory::restore_json(&snapshot, 1).unwrap();
        let replay = restored.recall(0, &query, 4);
        assert_eq!(recalled[0].slot, replay[0].slot);
        assert_eq!(recalled[0].generation, replay[0].generation);
        assert_eq!(recalled[0].actions, replay[0].actions);
    }
}
