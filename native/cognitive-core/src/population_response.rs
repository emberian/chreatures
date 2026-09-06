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
        let mut bank: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        bank.artifact_sha256 = actual;
        bank.validate(expected_feature_contract_sha256)?;
        Ok(bank)
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
        Ok(())
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
