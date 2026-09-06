//! Resident-private context-sensitive episodic retrieval.
//!
//! The achieved-goal reservoir owns episode contents and identities. This
//! learner owns only bounded retrieval statistics keyed by the exact
//! `(slot,generation)` identity, so replacement can never inherit credit.

use serde::{Deserialize, Serialize};

pub const CONTEXT: usize = 16;
const FORMAT: &str = "chreatures-contextual-episodic-retrieval-v1";
const MAX_BIAS: f64 = 0.30;
const LEARNING_RATE: f64 = 0.08;
const ELIGIBILITY_DECAY: f64 = 0.82;
const RETENTION: f64 = 0.9995;
const NOVELTY_GAIN: f64 = 0.06;
const SIMILARITY_GAIN: f64 = 0.10;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Pending {
    slot: usize,
    generation: u64,
    context: [f64; CONTEXT],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ContextualEpisodicLearner {
    schema: String,
    residents: usize,
    slots: usize,
    generation: Vec<u64>,
    context: Vec<f64>,
    value: Vec<f64>,
    eligibility: Vec<f64>,
    prediction_error: Vec<f64>,
    visits: Vec<u64>,
    pending: Vec<Option<Pending>>,
    updates: Vec<u64>,
}

impl ContextualEpisodicLearner {
    pub(crate) fn grow(&mut self, new_residents: usize) -> Result<(), String> {
        if new_residents <= self.residents {
            return Err("contextual episodic growth must append residents".into());
        }
        let cells = new_residents
            .checked_mul(self.slots)
            .ok_or("contextual episodic capacity overflow")?;
        self.generation.resize(cells, 0);
        self.context.resize(cells * CONTEXT, 0.0);
        self.value.resize(cells, 0.0);
        self.eligibility.resize(cells, 0.0);
        self.prediction_error.resize(cells, 0.0);
        self.visits.resize(cells, 0);
        self.pending.resize(new_residents, None);
        self.updates.resize(new_residents, 0);
        self.residents = new_residents;
        Ok(())
    }

    pub(crate) fn new(residents: usize, slots: usize) -> Result<Self, String> {
        if residents == 0 || slots == 0 {
            return Err("contextual episodic dimensions must be positive".into());
        }
        let cells = residents
            .checked_mul(slots)
            .ok_or("contextual episodic capacity overflow")?;
        Ok(Self {
            schema: FORMAT.into(),
            residents,
            slots,
            generation: vec![0; cells],
            context: vec![0.0; cells * CONTEXT],
            value: vec![0.0; cells],
            eligibility: vec![0.0; cells],
            prediction_error: vec![0.0; cells],
            visits: vec![0; cells],
            pending: vec![None; residents],
            updates: vec![0; residents],
        })
    }

    fn cell(&self, resident: usize, slot: usize) -> Result<usize, String> {
        if resident >= self.residents || slot >= self.slots {
            return Err("contextual episodic slot is outside the cohort".into());
        }
        Ok(resident * self.slots + slot)
    }

    fn validate_context(context: &[f64; CONTEXT]) -> Result<(), String> {
        if context.iter().any(|value| !value.is_finite()) {
            return Err("contextual episodic context must be finite".into());
        }
        Ok(())
    }

    pub(crate) fn replace(
        &mut self,
        resident: usize,
        slot: usize,
        generation: u64,
        context: [f64; CONTEXT],
    ) -> Result<(), String> {
        if generation == 0 {
            return Err("contextual episodic generation must be positive".into());
        }
        Self::validate_context(&context)?;
        let cell = self.cell(resident, slot)?;
        self.generation[cell] = generation;
        self.context[cell * CONTEXT..(cell + 1) * CONTEXT].copy_from_slice(&context);
        self.value[cell] = 0.0;
        self.eligibility[cell] = 0.0;
        self.prediction_error[cell] = 1.0;
        self.visits[cell] = 0;
        if self.pending[resident]
            .as_ref()
            .is_some_and(|pending| pending.slot == slot)
        {
            self.pending[resident] = None;
        }
        Ok(())
    }

    pub(crate) fn biases(
        &self,
        resident: usize,
        generations: &[u64],
        current: &[f64; CONTEXT],
    ) -> Result<Vec<f64>, String> {
        Self::validate_context(current)?;
        if generations.len() != self.slots {
            return Err("contextual episodic generation row has the wrong length".into());
        }
        let mut result = vec![0.0; self.slots];
        for slot in 0..self.slots {
            let cell = self.cell(resident, slot)?;
            if generations[slot] == 0 || generations[slot] != self.generation[cell] {
                continue;
            }
            let stored = &self.context[cell * CONTEXT..(cell + 1) * CONTEXT];
            let squared = stored
                .iter()
                .zip(current)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f64>()
                / CONTEXT as f64;
            let similarity = (-0.5 * squared).exp();
            let novelty = 1.0 / (1.0 + self.visits[cell] as f64).sqrt();
            let uncertainty = self.prediction_error[cell].sqrt().min(1.0);
            result[slot] = (self.value[cell]
                + SIMILARITY_GAIN * (similarity - 0.5)
                + NOVELTY_GAIN * novelty * uncertainty)
                .clamp(-MAX_BIAS, MAX_BIAS);
        }
        Ok(result)
    }

    pub(crate) fn begin(
        &mut self,
        resident: usize,
        slot: usize,
        generation: u64,
        context: [f64; CONTEXT],
    ) -> Result<(), String> {
        Self::validate_context(&context)?;
        let cell = self.cell(resident, slot)?;
        if generation == 0 || self.generation[cell] != generation {
            return Err("selected contextual episode identity differs".into());
        }
        self.visits[cell] = self.visits[cell].saturating_add(1);
        self.pending[resident] = Some(Pending {
            slot,
            generation,
            context,
        });
        Ok(())
    }

    pub(crate) fn observe(
        &mut self,
        resident: usize,
        normalized_return: f64,
        attributed: bool,
        learning_rate_gain: f64,
    ) -> Result<(), String> {
        if !normalized_return.is_finite() || normalized_return.abs() > 1.0 {
            return Err("contextual episodic return must be finite in [-1,1]".into());
        }
        if !learning_rate_gain.is_finite() || !(0.5..=1.5).contains(&learning_rate_gain) {
            return Err("contextual episodic learning gain is outside [0.5,1.5]".into());
        }
        let pending = self
            .pending
            .get_mut(resident)
            .ok_or("unknown contextual episodic resident")?
            .take();
        let Some(pending) = pending else {
            return Ok(());
        };
        let cell = self.cell(resident, pending.slot)?;
        if !attributed || self.generation[cell] != pending.generation {
            return Ok(());
        }
        let error = normalized_return - self.value[cell];
        self.eligibility[cell] = ELIGIBILITY_DECAY * self.eligibility[cell] + 1.0;
        self.value[cell] = (self.value[cell]
            + LEARNING_RATE * learning_rate_gain * self.eligibility[cell].min(4.0) * error)
            .clamp(-MAX_BIAS, MAX_BIAS);
        self.prediction_error[cell] =
            (0.95 * self.prediction_error[cell] + 0.05 * error * error).clamp(0.0, 1.0);
        for slot in 0..self.slots {
            let other = resident * self.slots + slot;
            if other != cell && self.generation[other] != 0 {
                self.value[other] *= RETENTION;
                self.eligibility[other] *= ELIGIBILITY_DECAY;
            }
        }
        self.updates[resident] = self.updates[resident].saturating_add(1);
        Ok(())
    }

    pub(crate) fn cancel(&mut self, resident: usize) -> Result<(), String> {
        self.pending
            .get_mut(resident)
            .ok_or("unknown contextual episodic resident")?
            .take();
        Ok(())
    }

    pub(crate) fn clear_resident(&mut self, resident: usize) -> Result<(), String> {
        if resident >= self.residents {
            return Err("unknown contextual episodic resident".into());
        }
        let range = resident * self.slots..(resident + 1) * self.slots;
        self.generation[range.clone()].fill(0);
        self.value[range.clone()].fill(0.0);
        self.eligibility[range.clone()].fill(0.0);
        self.prediction_error[range.clone()].fill(0.0);
        self.visits[range.clone()].fill(0);
        self.context[resident * self.slots * CONTEXT..(resident + 1) * self.slots * CONTEXT]
            .fill(0.0);
        self.pending[resident] = None;
        self.updates[resident] = 0;
        Ok(())
    }

    pub(crate) fn updates(&self) -> &[u64] {
        &self.updates
    }

    pub(crate) fn snapshot(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|error| error.to_string())
    }

    pub(crate) fn restore(value: &str, residents: usize, slots: usize) -> Result<Self, String> {
        let state: Self = serde_json::from_str(value).map_err(|error| error.to_string())?;
        if state.schema != FORMAT || state.residents != residents || state.slots != slots {
            return Err("contextual episodic snapshot belongs to another contract".into());
        }
        let cells = residents * slots;
        if state.generation.len() != cells
            || state.context.len() != cells * CONTEXT
            || state.value.len() != cells
            || state.eligibility.len() != cells
            || state.prediction_error.len() != cells
            || state.visits.len() != cells
            || state.pending.len() != residents
            || state.updates.len() != residents
            || state.context.iter().any(|value| !value.is_finite())
            || state
                .value
                .iter()
                .any(|value| !value.is_finite() || value.abs() > MAX_BIAS)
            || state
                .eligibility
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || state
                .prediction_error
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("contextual episodic snapshot values differ".into());
        }
        Ok(state)
    }
}
