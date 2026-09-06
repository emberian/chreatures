// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private lifetime corrections to inherited physical-consequence predictions.
//!
//! This is normalized LMS on experienced transitions, not policy training. The
//! inherited prediction and executed context are retained until that same
//! transition's outcome arrives. A hypothesis about an unexecuted action never
//! supplies an update. Reported error is a running prediction RMSE, not a
//! calibrated confidence interval or an estimate of an organism's welfare.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConsequenceTarget {
    pub name: String,
    pub unit: String,
    /// Physical units per normalized target unit.
    pub scale: f64,
    /// Maximum absolute private correction, in physical units.
    pub correction_limit: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConsequenceConfig {
    /// Binds the feature order and inherited law artifact, not just dimensions.
    pub inherited_identity: String,
    pub feature_names: Vec<String>,
    pub targets: Vec<ConsequenceTarget>,
    pub learning_rate: f64,
    pub error_decay: f64,
    /// Clip a single normalized innovation before updating weights.
    pub innovation_limit: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct PendingExperience {
    tick: u64,
    features: Vec<f64>,
    inherited: Vec<f64>,
    inside_domain: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Individual {
    weights: Vec<f64>,
    squared_error: Vec<f64>,
    updates: u64,
    out_of_domain: u64,
    last_completed_tick: Option<u64>,
    pending: Option<PendingExperience>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PersonalConsequences {
    schema: String,
    config: ConsequenceConfig,
    individuals: Vec<Individual>,
}

#[derive(Clone, Debug)]
pub struct PersonalEstimate {
    pub expected: Vec<f64>,
    pub correction: Vec<f64>,
    /// None until this individual has supplied an in-domain outcome.
    pub experienced_rmse: Option<Vec<f64>>,
    pub updates: u64,
    pub inside_domain: bool,
}

impl ConsequenceConfig {
    fn validate(&self) -> Result<(), String> {
        let unique = |names: Vec<&str>| {
            let mut ordered = names.clone();
            ordered.sort_unstable();
            ordered.dedup();
            ordered.len() == names.len() && names.iter().all(|s| !s.is_empty())
        };
        if self.inherited_identity.is_empty()
            || self.feature_names.is_empty()
            || self.feature_names.len() > 256
            || self.targets.is_empty()
            || self.targets.len() > 32
            || !unique(self.feature_names.iter().map(String::as_str).collect())
            || !unique(self.targets.iter().map(|t| t.name.as_str()).collect())
            || !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || !self.error_decay.is_finite()
            || !(0.0..1.0).contains(&self.error_decay)
            || !self.innovation_limit.is_finite()
            || self.innovation_limit <= 0.0
        {
            return Err("invalid private consequence-learning contract".into());
        }
        for target in &self.targets {
            if target.unit.is_empty()
                || !target.scale.is_finite()
                || target.scale <= 0.0
                || !target.correction_limit.is_finite()
                || target.correction_limit < 0.0
            {
                return Err("invalid physical consequence scale or bound".into());
            }
        }
        Ok(())
    }
}

impl PersonalConsequences {
    pub fn config(&self) -> &ConsequenceConfig {
        &self.config
    }
    pub fn new(batch: usize, config: ConsequenceConfig) -> Result<Self, String> {
        config.validate()?;
        if !(1..=4096).contains(&batch) {
            return Err("invalid private consequence cohort size".into());
        }
        let row = Individual {
            weights: vec![0.0; (config.feature_names.len() + 1) * config.targets.len()],
            squared_error: vec![0.0; config.targets.len()],
            updates: 0,
            out_of_domain: 0,
            last_completed_tick: None,
            pending: None,
        };
        Ok(Self {
            schema: "chreatures-private-consequences-v1".into(),
            config,
            individuals: vec![row; batch],
        })
    }

    fn check_context(&self, features: &[f64], inherited: &[f64]) -> Result<(), String> {
        if features.len() != self.config.feature_names.len()
            || inherited.len() != self.config.targets.len()
            || features.iter().any(|x| !x.is_finite() || x.abs() > 1.0)
            || inherited.iter().any(|x| !x.is_finite())
        {
            return Err(
                "consequence context must match ordered bounded features and targets".into(),
            );
        }
        Ok(())
    }

    pub fn estimate(
        &self,
        resident: usize,
        features: &[f64],
        inherited: &[f64],
        inside_domain: bool,
    ) -> Result<PersonalEstimate, String> {
        self.check_context(features, inherited)?;
        let individual = self.individuals.get(resident).ok_or("unknown resident")?;
        let width = features.len() + 1;
        let correction: Vec<_> = self
            .config
            .targets
            .iter()
            .enumerate()
            .map(|(k, target)| {
                if !inside_domain {
                    return 0.0;
                }
                let weights = &individual.weights[k * width..(k + 1) * width];
                let value = weights[width - 1]
                    + weights[..width - 1]
                        .iter()
                        .zip(features)
                        .map(|(w, x)| w * x)
                        .sum::<f64>();
                (value * target.scale).clamp(-target.correction_limit, target.correction_limit)
            })
            .collect();
        Ok(PersonalEstimate {
            expected: inherited
                .iter()
                .zip(&correction)
                .map(|(base, delta)| base + delta)
                .collect(),
            correction,
            experienced_rmse: (individual.updates > 0).then(|| {
                individual
                    .squared_error
                    .iter()
                    .zip(&self.config.targets)
                    .map(|(error, target)| error.sqrt() * target.scale)
                    .collect()
            }),
            updates: individual.updates,
            inside_domain,
        })
    }

    /// Call only once the candidate action has actually been committed.
    pub fn record_executed(
        &mut self,
        resident: usize,
        tick: u64,
        features: &[f64],
        inherited: &[f64],
        inside_domain: bool,
    ) -> Result<(), String> {
        self.check_context(features, inherited)?;
        let individual = self
            .individuals
            .get_mut(resident)
            .ok_or("unknown resident")?;
        if individual.pending.is_some()
            || individual
                .last_completed_tick
                .is_some_and(|previous| tick <= previous)
        {
            return Err(
                "a consequence is pending or this executed tick was already consumed".into(),
            );
        }
        individual.pending = Some(PendingExperience {
            tick,
            features: features.to_vec(),
            inherited: inherited.to_vec(),
            inside_domain,
        });
        Ok(())
    }

    /// Observed targets must describe the exact pending executed transition.
    pub fn observe(&mut self, resident: usize, tick: u64, actual: &[f64]) -> Result<(), String> {
        if actual.len() != self.config.targets.len() || actual.iter().any(|x| !x.is_finite()) {
            return Err("physical outcome does not match the consequence contract".into());
        }
        let individual = self
            .individuals
            .get_mut(resident)
            .ok_or("unknown resident")?;
        let pending = individual
            .pending
            .as_ref()
            .ok_or("no executed transition is pending")?;
        if pending.tick != tick {
            return Err("outcome tick differs from the executed transition".into());
        }
        if pending.inside_domain {
            let width = pending.features.len() + 1;
            let norm_squared = 1.0 + pending.features.iter().map(|x| x * x).sum::<f64>();
            let errors: Vec<_> = self
                .config
                .targets
                .iter()
                .enumerate()
                .map(|(k, target)| {
                    let weights = &individual.weights[k * width..(k + 1) * width];
                    let predicted = weights[width - 1]
                        + weights[..width - 1]
                            .iter()
                            .zip(&pending.features)
                            .map(|(w, x)| w * x)
                            .sum::<f64>();
                    (actual[k] - pending.inherited[k]) / target.scale - predicted
                })
                .collect();
            if errors.iter().any(|error| !(error * error).is_finite()) {
                return Err("physical outcome exceeds the configured numerical scale".into());
            }
            for (k, target) in self.config.targets.iter().enumerate() {
                let weights = &mut individual.weights[k * width..(k + 1) * width];
                let error = errors[k];
                let amount = self.config.learning_rate
                    * error.clamp(-self.config.innovation_limit, self.config.innovation_limit)
                    / norm_squared;
                for (weight, feature) in weights[..width - 1].iter_mut().zip(&pending.features) {
                    *weight += amount * feature;
                }
                weights[width - 1] += amount;
                // Projection bounds every possible prediction for features in [-1,1].
                let limit = target.correction_limit / target.scale / (width as f64).sqrt();
                let norm = weights.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > limit {
                    let ratio = limit / norm;
                    weights.iter_mut().for_each(|w| *w *= ratio);
                }
                individual.squared_error[k] = if individual.updates == 0 {
                    error * error
                } else {
                    self.config.error_decay * individual.squared_error[k]
                        + (1.0 - self.config.error_decay) * error * error
                };
            }
            individual.updates = individual.updates.saturating_add(1);
        } else {
            individual.out_of_domain = individual.out_of_domain.saturating_add(1);
        }
        individual.last_completed_tick = Some(tick);
        individual.pending = None;
        Ok(())
    }

    /// Discontinuity cancels pending credit; learned history remains private.
    pub fn cancel_pending(&mut self, resident: usize) -> Result<(), String> {
        self.individuals
            .get_mut(resident)
            .ok_or("unknown resident")?
            .pending = None;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|error| error.to_string())
    }

    pub fn restore(value: &str, config: &ConsequenceConfig, batch: usize) -> Result<Self, String> {
        config.validate()?;
        let state: Self = serde_json::from_str(value).map_err(|error| error.to_string())?;
        if state.schema != "chreatures-private-consequences-v1"
            || state.config != *config
            || state.individuals.len() != batch
        {
            return Err("private consequence snapshot belongs to another contract".into());
        }
        let width = config.feature_names.len() + 1;
        for individual in &state.individuals {
            if individual.weights.len() != width * config.targets.len()
                || individual.squared_error.len() != config.targets.len()
                || individual.weights.iter().any(|x| !x.is_finite())
                || individual
                    .squared_error
                    .iter()
                    .any(|x| !x.is_finite() || *x < 0.0)
            {
                return Err("malformed private consequence state".into());
            }
            for (weights, target) in individual.weights.chunks_exact(width).zip(&config.targets) {
                let bound = target.correction_limit / target.scale / (width as f64).sqrt();
                if weights.iter().map(|x| x * x).sum::<f64>().sqrt() > bound * (1.0 + 1e-12) {
                    return Err("private consequence weights exceed their inherited bound".into());
                }
            }
            if let Some(pending) = &individual.pending {
                state.check_context(&pending.features, &pending.inherited)?;
                if individual
                    .last_completed_tick
                    .is_some_and(|last| pending.tick <= last)
                {
                    return Err("pending consequence tick precedes completed experience".into());
                }
            }
        }
        Ok(state)
    }
}
