// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable GAM response mechanisms inherited only by a new genome birth.

use crate::gam_law::{LawBank, LawEstimate};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone, Deserialize)]
pub struct PopulationResponseBank {
    pub schema: String,
    pub feature_contract_sha256: String,
    pub fitted: LawBank,
    pub responses: Vec<ResponseRule>,
    #[serde(default)]
    pub budgets: Vec<BudgetGroup>,
    /// Populated only by `from_authenticated_path`; never trusted from JSON.
    #[serde(skip)]
    pub artifact_sha256: String,
    #[serde(default)]
    pub candidate_score: Option<CandidateScoreConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseRule {
    pub law: String,
    pub mechanism: String,
    pub unit: String,
    pub transform: ResponseTransform,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseTransform {
    PositiveSoftplus { ceiling: f64 },
    SignedTanh { magnitude: f64 },
    BudgetLogit,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BudgetGroup {
    pub name: String,
    pub members: Vec<String>,
    pub total: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CandidateScoreConfig {
    pub maximum_tilt: f64,
    pub terms: Vec<CandidateScoreTerm>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CandidateScoreTerm {
    pub mechanism: String,
    pub weight: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MechanismResponse {
    pub mechanism: String,
    pub unit: String,
    pub value: f64,
    pub uncertainty_bound: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PopulationEvaluation {
    pub out_of_domain: bool,
    pub responses: Vec<MechanismResponse>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CognitiveAdaptation {
    pub action_gain_delta: [f32; 12],
    pub retrieval_temperature_delta: f32,
}

pub const POPULATION_RESPONSE_FEATURES: usize = 28;
pub const POPULATION_HISTORY_WINDOW: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationHistorySnapshot {
    pub residents: usize,
    pub cursor: Vec<usize>,
    pub count: Vec<usize>,
    pub values: Vec<f32>,
}

/// Private causal physiology history. It deliberately stores only energy,
/// fatigue, structural integrity, and development fraction.
#[derive(Debug, Clone)]
pub struct PopulationHistory {
    state: PopulationHistorySnapshot,
}

impl PopulationHistory {
    pub fn new(residents: usize) -> Result<Self, String> {
        if residents == 0 {
            return Err("population history requires residents".into());
        }
        Ok(Self {
            state: PopulationHistorySnapshot {
                residents,
                cursor: vec![0; residents],
                count: vec![0; residents],
                values: vec![0.0; residents * POPULATION_HISTORY_WINDOW * 4],
            },
        })
    }

    pub fn summary(&self, resident: usize) -> Result<[f32; 4], String> {
        if resident >= self.state.residents {
            return Err("population history resident differs".into());
        }
        let count = self.state.count[resident];
        if count == 0 {
            return Ok([0.0; 4]);
        }
        let oldest = (self.state.cursor[resident] + POPULATION_HISTORY_WINDOW - count)
            % POPULATION_HISTORY_WINDOW;
        let oldest_base = (resident * POPULATION_HISTORY_WINDOW + oldest) * 4;
        let latest = (self.state.cursor[resident] + POPULATION_HISTORY_WINDOW - 1)
            % POPULATION_HISTORY_WINDOW;
        let latest_base = (resident * POPULATION_HISTORY_WINDOW + latest) * 4;
        let mut mean = [0.0f32; 2];
        for offset in 0..count {
            let slot = (oldest + offset) % POPULATION_HISTORY_WINDOW;
            let source = (resident * POPULATION_HISTORY_WINDOW + slot) * 4;
            mean[0] += self.state.values[source];
            mean[1] += self.state.values[source + 1];
        }
        Ok([
            mean[0] / count as f32,
            mean[1] / count as f32,
            self.state.values[latest_base + 2] - self.state.values[oldest_base + 2],
            self.state.values[latest_base + 3] - self.state.values[oldest_base + 3],
        ])
    }

    /// Advance history only after the matching physical consequence commits.
    pub fn record(&mut self, resident: usize, physiology12: &[f32; 12]) -> Result<(), String> {
        if resident >= self.state.residents || !physiology12.iter().all(|x| x.is_finite()) {
            return Err("population history resident or physiology differs".into());
        }
        let current = [
            physiology12[0],
            physiology12[2],
            physiology12[6],
            physiology12[7],
        ];
        let cursor = self.state.cursor[resident];
        let base = (resident * POPULATION_HISTORY_WINDOW + cursor) * 4;
        self.state.values[base..base + 4].copy_from_slice(&current);
        self.state.cursor[resident] = (cursor + 1) % POPULATION_HISTORY_WINDOW;
        self.state.count[resident] =
            (self.state.count[resident] + 1).min(POPULATION_HISTORY_WINDOW);
        Ok(())
    }

    pub fn snapshot(&self) -> PopulationHistorySnapshot {
        self.state.clone()
    }

    pub fn clear(&mut self, resident: usize) -> Result<(), String> {
        if resident >= self.state.residents {
            return Err("population history resident differs".into());
        }
        self.state.cursor[resident] = 0;
        self.state.count[resident] = 0;
        let start = resident * POPULATION_HISTORY_WINDOW * 4;
        self.state.values[start..start + POPULATION_HISTORY_WINDOW * 4].fill(0.0);
        Ok(())
    }

    pub fn grow(&mut self, residents: usize) -> Result<(), String> {
        if residents <= self.state.residents {
            return Err("population history growth must append residents".into());
        }
        self.state.cursor.resize(residents, 0);
        self.state.count.resize(residents, 0);
        self.state
            .values
            .resize(residents * POPULATION_HISTORY_WINDOW * 4, 0.0);
        self.state.residents = residents;
        Ok(())
    }

    pub fn restore(snapshot: PopulationHistorySnapshot) -> Result<Self, String> {
        if snapshot.residents == 0
            || snapshot.cursor.len() != snapshot.residents
            || snapshot.count.len() != snapshot.residents
            || snapshot.values.len() != snapshot.residents * POPULATION_HISTORY_WINDOW * 4
            || snapshot
                .cursor
                .iter()
                .any(|x| *x >= POPULATION_HISTORY_WINDOW)
            || snapshot
                .count
                .iter()
                .any(|x| *x > POPULATION_HISTORY_WINDOW)
            || !snapshot.values.iter().all(|x| x.is_finite())
        {
            return Err("population history snapshot differs".into());
        }
        Ok(Self { state: snapshot })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidentResponseReceipt {
    pub resident: usize,
    pub in_domain: bool,
    pub responses: Vec<MechanismResponse>,
}

impl ResidentResponseReceipt {
    pub fn mechanism(&self, name: &str) -> Option<&MechanismResponse> {
        self.responses
            .iter()
            .find(|response| response.mechanism == name)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PopulationResponseReceipt {
    pub artifact_sha256: String,
    pub feature_contract_sha256: String,
    pub residents: usize,
    pub in_domain: usize,
    pub out_of_domain: usize,
    pub evaluations: Vec<ResidentResponseReceipt>,
}

impl PopulationResponseBank {
    pub fn from_authenticated_json(
        json: &str,
        expected_sha256: &str,
        expected_feature_contract_sha256: &str,
    ) -> Result<Self, String> {
        let actual = format!("{:x}", Sha256::digest(json.as_bytes()));
        if actual != expected_sha256 {
            return Err(format!("population response artifact hash differs: expected {expected_sha256}, got {actual}"));
        }
        let mut bank: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        bank.artifact_sha256 = actual;
        bank.validate(expected_feature_contract_sha256)?;
        Ok(bank)
    }
    /// The expected hash is stored by an immutable candidate genome. Loading a
    /// different fitted mechanism into an existing life is therefore rejected.
    pub fn from_authenticated_path(
        path: impl AsRef<Path>,
        expected_sha256: &str,
        expected_feature_contract_sha256: &str,
    ) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if actual != expected_sha256 {
            return Err(format!("population response artifact hash differs: expected {expected_sha256}, got {actual}"));
        }
        let json = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
        Self::from_authenticated_json(json, &actual, expected_feature_contract_sha256)
    }

    pub fn validate(&self, expected_feature_contract_sha256: &str) -> Result<(), String> {
        if self.schema != "chreatures-population-response-bank-v1"
            || self.feature_contract_sha256 != expected_feature_contract_sha256
        {
            return Err("population response schema or feature contract differs".into());
        }
        self.fitted.validate()?;
        let law_index: HashMap<&str, usize> = self
            .fitted
            .laws
            .iter()
            .enumerate()
            .map(|(index, law)| (law.name.as_str(), index))
            .collect();
        let mut mechanisms = HashMap::new();
        for (index, response) in self.responses.iter().enumerate() {
            if !law_index.contains_key(response.law.as_str())
                || response.mechanism.is_empty()
                || response.unit.is_empty()
                || mechanisms
                    .insert(response.mechanism.as_str(), index)
                    .is_some()
            {
                return Err("population response law/mechanism mapping is invalid".into());
            }
            match response.transform {
                ResponseTransform::PositiveSoftplus { ceiling }
                | ResponseTransform::SignedTanh { magnitude: ceiling }
                    if !ceiling.is_finite() || ceiling <= 0.0 =>
                {
                    return Err(format!(
                        "invalid response ceiling for {}",
                        response.mechanism
                    ))
                }
                _ => {}
            }
        }
        let mut budget_members = HashMap::new();
        for group in &self.budgets {
            if group.name.is_empty()
                || !group.total.is_finite()
                || group.total <= 0.0
                || group.members.len() < 2
            {
                return Err("population response budget group is invalid".into());
            }
            for member in &group.members {
                let Some(index) = mechanisms.get(member.as_str()) else {
                    return Err(format!("budget member {member} has no response"));
                };
                if !matches!(
                    self.responses[*index].transform,
                    ResponseTransform::BudgetLogit
                ) || budget_members.insert(member, group.name.as_str()).is_some()
                {
                    return Err(format!(
                        "budget member {member} is duplicated or is not a logit"
                    ));
                }
            }
        }
        if self
            .responses
            .iter()
            .filter(|x| matches!(x.transform, ResponseTransform::BudgetLogit))
            .any(|x| !budget_members.contains_key(&x.mechanism))
        {
            return Err("every budget logit must belong to exactly one budget group".into());
        }
        if let Some(score) = &self.candidate_score {
            if !score.maximum_tilt.is_finite()
                || !(0.0..=0.5).contains(&score.maximum_tilt)
                || score.terms.is_empty()
                || score.terms.iter().any(|term| {
                    !mechanisms.contains_key(term.mechanism.as_str())
                        || !term.weight.is_finite()
                        || !term.scale.is_finite()
                        || term.scale <= 0.0
                })
            {
                return Err("population candidate score contract differs".into());
            }
        }
        Ok(())
    }

    pub fn candidate_score_tilts(
        &self,
        evaluations: &[PopulationEvaluation],
    ) -> Result<Vec<f32>, String> {
        let Some(config) = &self.candidate_score else {
            return Ok(vec![0.0; evaluations.len()]);
        };
        let mut raw = Vec::with_capacity(evaluations.len());
        for evaluation in evaluations {
            if evaluation.out_of_domain {
                raw.push(None);
                continue;
            }
            let values: HashMap<&str, f64> = evaluation
                .responses
                .iter()
                .map(|response| (response.mechanism.as_str(), response.value))
                .collect();
            let score = config
                .terms
                .iter()
                .map(|term| term.weight * values[term.mechanism.as_str()] / term.scale)
                .sum::<f64>();
            raw.push(Some(score));
        }
        let valid = raw.iter().flatten().copied().collect::<Vec<_>>();
        let center = if valid.is_empty() {
            0.0
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        };
        Ok(raw
            .into_iter()
            .map(|value| value.map_or(0.0, |x| (config.maximum_tilt * (x - center).tanh()) as f32))
            .collect())
    }

    pub fn evaluate(&self, raw_features: &[f64]) -> Result<PopulationEvaluation, String> {
        let estimates = self.fitted.evaluate(raw_features)?;
        let out_of_domain = estimates.iter().any(|x| x.out_of_domain);
        if out_of_domain {
            return Ok(PopulationEvaluation {
                out_of_domain: true,
                responses: Vec::new(),
            });
        }
        let by_law: HashMap<&str, &LawEstimate> = estimates
            .iter()
            .map(|estimate| (estimate.name.as_str(), estimate))
            .collect();
        let mut values = vec![0.0; self.responses.len()];
        let mut uncertainty = vec![0.0; self.responses.len()];
        for (index, response) in self.responses.iter().enumerate() {
            let estimate = by_law[response.law.as_str()];
            match response.transform {
                ResponseTransform::PositiveSoftplus { ceiling } => {
                    values[index] = softplus(estimate.expected).min(ceiling);
                    uncertainty[index] =
                        (logistic(estimate.expected) * estimate.uncertainty).min(ceiling);
                }
                ResponseTransform::SignedTanh { magnitude } => {
                    values[index] = magnitude * estimate.expected.tanh();
                    uncertainty[index] = magnitude * estimate.uncertainty;
                }
                ResponseTransform::BudgetLogit => values[index] = estimate.expected,
            }
        }
        let index: HashMap<&str, usize> = self
            .responses
            .iter()
            .enumerate()
            .map(|(i, x)| (x.mechanism.as_str(), i))
            .collect();
        for group in &self.budgets {
            let maximum = group
                .members
                .iter()
                .map(|name| values[index[name.as_str()]])
                .fold(f64::NEG_INFINITY, f64::max);
            let denominator: f64 = group
                .members
                .iter()
                .map(|name| (values[index[name.as_str()]] - maximum).exp())
                .sum();
            for name in &group.members {
                let i = index[name.as_str()];
                let fraction = (values[i] - maximum).exp() / denominator;
                values[i] = group.total * fraction;
                uncertainty[i] = group
                    .total
                    .min(by_law[self.responses[i].law.as_str()].uncertainty);
            }
        }
        Ok(PopulationEvaluation {
            out_of_domain: false,
            responses: self
                .responses
                .iter()
                .enumerate()
                .map(|(i, rule)| MechanismResponse {
                    mechanism: rule.mechanism.clone(),
                    unit: rule.unit.clone(),
                    value: values[i],
                    uncertainty_bound: uncertainty[i],
                })
                .collect(),
        })
    }

    /// Evaluate one tick for a cohort and return the exact adjustments plus
    /// authenticated artifact identity and domain coverage for private receipts.
    pub fn evaluate_cohort(
        &self,
        physiology12: &[[f32; 12]],
        history4: &[[f32; 4]],
        executed12: &[[f32; 12]],
    ) -> Result<PopulationResponseReceipt, String> {
        if self.artifact_sha256.is_empty() {
            return Err("population response receipts require an authenticated artifact".into());
        }
        if physiology12.len() != history4.len() || physiology12.len() != executed12.len() {
            return Err("population response cohort dimensions differ".into());
        }
        let mut in_domain = 0;
        let mut evaluations = Vec::with_capacity(physiology12.len());
        for resident in 0..physiology12.len() {
            let features = population_response_features(
                &physiology12[resident],
                &history4[resident],
                &executed12[resident],
            )?;
            let evaluation = self.evaluate(&features)?;
            if !evaluation.out_of_domain {
                in_domain += 1;
            }
            evaluations.push(ResidentResponseReceipt {
                resident,
                in_domain: !evaluation.out_of_domain,
                responses: evaluation.responses,
            });
        }
        Ok(PopulationResponseReceipt {
            artifact_sha256: self.artifact_sha256.clone(),
            feature_contract_sha256: self.feature_contract_sha256.clone(),
            residents: physiology12.len(),
            in_domain,
            out_of_domain: physiology12.len() - in_domain,
            evaluations,
        })
    }

    /// Frozen boundary proposed to the v4 cognitive controller. A bank without
    /// all thirteen explicitly named, signed-and-bounded mechanisms cannot act
    /// as a cognitive adapter.
    pub fn cognitive_adaptation(
        &self,
        evaluation: &PopulationEvaluation,
    ) -> Result<CognitiveAdaptation, String> {
        if evaluation.out_of_domain {
            return Err("out-of-domain population response cannot adapt cognition".into());
        }
        let values: HashMap<&str, f64> = evaluation
            .responses
            .iter()
            .map(|x| (x.mechanism.as_str(), x.value))
            .collect();
        let rules: HashMap<&str, &ResponseRule> = self
            .responses
            .iter()
            .map(|rule| (rule.mechanism.as_str(), rule))
            .collect();
        let mut action_gain_delta = [0.0f32; 12];
        for (index, output) in action_gain_delta.iter_mut().enumerate() {
            let name = format!("action_gain_delta_{index}");
            if !matches!(
                rules.get(name.as_str()).map(|rule| &rule.transform),
                Some(ResponseTransform::SignedTanh { .. })
            ) {
                return Err(format!("{name} is not a signed bounded response"));
            }
            *output = *values
                .get(name.as_str())
                .ok_or_else(|| format!("missing {name}"))? as f32;
        }
        if !matches!(
            rules
                .get("retrieval_temperature_delta")
                .map(|rule| &rule.transform),
            Some(ResponseTransform::SignedTanh { .. })
        ) {
            return Err("retrieval_temperature_delta is not a signed bounded response".into());
        }
        let retrieval_temperature_delta = *values
            .get("retrieval_temperature_delta")
            .ok_or("missing retrieval_temperature_delta")?
            as f32;
        Ok(CognitiveAdaptation {
            action_gain_delta,
            retrieval_temperature_delta,
        })
    }
}

/// Exact v1 feature order: physiology12, private history4, executed action12.
pub fn population_response_features(
    physiology12: &[f32; 12],
    history4: &[f32; 4],
    executed12: &[f32; 12],
) -> Result<[f64; POPULATION_RESPONSE_FEATURES], String> {
    let mut features = [0.0; POPULATION_RESPONSE_FEATURES];
    for (output, input) in features
        .iter_mut()
        .zip(physiology12.iter().chain(history4).chain(executed12))
    {
        if !input.is_finite() {
            return Err("population response feature is not finite".into());
        }
        *output = f64::from(*input);
    }
    Ok(features)
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
