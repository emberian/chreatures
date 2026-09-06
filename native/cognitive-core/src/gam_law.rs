// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable, bounded inference for experience-fitted GAM consequence laws.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Debug, Clone, Deserialize)]
pub struct LawBank {
    pub schema: String,
    pub source: Source,
    pub features: Vec<Feature>,
    pub laws: Vec<Law>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Source {
    pub model_library: String,
    pub model_version: String,
    pub model_source_commit: String,
    pub telemetry_sha256: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Feature {
    pub name: String,
    pub unit: String,
    pub mean: f64,
    pub scale: f64,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Law {
    pub name: String,
    pub unit: String,
    pub intercept: f64,
    pub residual_rmse: f64,
    pub target_scale: f64,
    pub conservative_residual_bound: f64,
    pub terms: Vec<Term>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Term {
    pub feature: usize,
    pub knots: Vec<f64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LawEstimate {
    pub name: String,
    pub expected: f64,
    pub uncertainty: f64,
    pub out_of_domain: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateScore {
    pub score: Option<f64>,
    pub out_of_domain: bool,
    pub consequences: Vec<LawEstimate>,
}

/// Current decision state permitted by the fitted-law contract.
pub struct DecisionContext<'a> {
    /// energy, gut, fatigue, tanh(speed/2), tanh(angular velocity/4), support
    pub physiology: &'a [f32; 6],
    /// Identity-bound MaleCNS readout rates. Identity is not itself a feature.
    pub neural: &'a [f32; 384],
}

/// Hypothetical physical command in the controller's canonical action order.
pub struct CandidateAction<'a> {
    /// thrust, yaw, gaze pitch, grip, signal low/mid/high, posture
    pub action: &'a [f32; 8],
    pub oral: f32,
}

impl LawBank {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let bank: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        bank.validate()?;
        Ok(bank)
    }

    pub fn from_authenticated_json_path(
        path: impl AsRef<Path>,
        expected_sha256: &str,
    ) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if actual != expected_sha256 {
            return Err(format!(
                "GAM law artifact hash differs: expected {expected_sha256}, got {actual}"
            ));
        }
        let bank: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        bank.validate()?;
        Ok(bank)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "chreatures-gam-consequence-law-bank-v1" || self.laws.is_empty() {
            return Err("unsupported or empty GAM consequence-law bank".into());
        }
        if self.source.model_library.is_empty()
            || self.source.model_version.is_empty()
            || self.source.model_source_commit.len() != 40
            || self.source.telemetry_sha256.is_empty()
            || self.source.telemetry_sha256.iter().any(|x| x.len() != 64)
        {
            return Err("law bank has incomplete model/data provenance".into());
        }
        for feature in &self.features {
            if feature.name.is_empty()
                || feature.unit.is_empty()
                || !feature.mean.is_finite()
                || !feature.scale.is_finite()
                || feature.scale <= 0.0
                || !feature.minimum.is_finite()
                || !feature.maximum.is_finite()
                || feature.minimum >= feature.maximum
            {
                return Err(format!("invalid domain for feature {}", feature.name));
            }
        }
        for law in &self.laws {
            if law.name.is_empty()
                || law.unit.is_empty()
                || !law.intercept.is_finite()
                || !law.residual_rmse.is_finite()
                || law.residual_rmse < 0.0
                || !law.target_scale.is_finite()
                || law.target_scale <= 0.0
                || !law.conservative_residual_bound.is_finite()
                || law.conservative_residual_bound < law.residual_rmse
            {
                return Err(format!("invalid law metadata for {}", law.name));
            }
            for term in &law.terms {
                if term.feature >= self.features.len()
                    || term.knots.len() < 2
                    || term.knots.len() != term.values.len()
                    || term.knots.windows(2).any(|x| x[0] >= x[1])
                    || term
                        .knots
                        .iter()
                        .chain(&term.values)
                        .any(|x| !x.is_finite())
                {
                    return Err(format!("invalid smooth term in {}", law.name));
                }
            }
        }
        Ok(())
    }

    pub fn evaluate(&self, raw_features: &[f64]) -> Result<Vec<LawEstimate>, String> {
        if raw_features.len() != self.features.len() {
            return Err(format!(
                "expected {} features, got {}",
                self.features.len(),
                raw_features.len()
            ));
        }
        let mut standardized = Vec::with_capacity(raw_features.len());
        let mut feature_ood = Vec::with_capacity(raw_features.len());
        for (raw, metadata) in raw_features.iter().zip(&self.features) {
            if !raw.is_finite() {
                return Err(format!("non-finite feature {}", metadata.name));
            }
            standardized.push((raw - metadata.mean) / metadata.scale);
            feature_ood.push(*raw < metadata.minimum || *raw > metadata.maximum);
        }
        Ok(self
            .laws
            .iter()
            .map(|law| {
                let mut expected = law.intercept;
                let mut out_of_domain = false;
                for term in &law.terms {
                    let x = standardized[term.feature];
                    out_of_domain |= feature_ood[term.feature]
                        || x < term.knots[0]
                        || x > *term.knots.last().unwrap();
                    expected += interpolate_clamped(x, &term.knots, &term.values);
                }
                LawEstimate {
                    name: law.name.clone(),
                    expected,
                    uncertainty: law.residual_rmse,
                    out_of_domain,
                }
            })
            .collect())
    }

    /// Exact feature order used by the v1 fitted artifact. Candidate evaluation
    /// substitutes only the hypothetical action while holding current state fixed.
    pub fn fitted_features(
        &self,
        context: &DecisionContext<'_>,
        candidate: &CandidateAction<'_>,
    ) -> Result<[f64; 12], String> {
        if self.features.len() != 12
            || self
                .features
                .iter()
                .map(|x| x.name.as_str())
                .collect::<Vec<_>>()
                != [
                    "energy",
                    "fatigue",
                    "body_speed",
                    "support",
                    "neural_activity",
                    "thrust",
                    "yaw",
                    "grip",
                    "oral",
                    "motor_magnitude",
                    "thrust_x_fatigue",
                    "yaw_x_speed",
                ]
        {
            return Err("law bank does not match the v1 body consequence feature contract".into());
        }
        let neural_activity = context.neural.iter().map(|x| f64::from(*x)).sum::<f64>() / 384.0;
        let motor_magnitude = candidate.action[..4]
            .iter()
            .map(|x| f64::from(*x).abs())
            .sum::<f64>()
            / 4.0;
        let fatigue = f64::from(context.physiology[2]);
        let speed = f64::from(context.physiology[3]);
        let thrust = f64::from(candidate.action[0]);
        let yaw = f64::from(candidate.action[1]);
        Ok([
            f64::from(context.physiology[0]),
            fatigue,
            speed,
            f64::from(context.physiology[5]),
            neural_activity,
            thrust,
            yaw,
            f64::from(candidate.action[3]),
            f64::from(candidate.oral),
            motor_magnitude,
            thrust * fatigue,
            yaw * speed,
        ])
    }

    /// Domain normalization for resident-private residual learners. Values are
    /// centered into [-1, 1]; an out-of-domain input is reported and clamped.
    pub fn normalize_private(&self, raw_features: &[f64]) -> Result<(Vec<f32>, bool), String> {
        if raw_features.len() != self.features.len() {
            return Err(format!(
                "expected {} features, got {}",
                self.features.len(),
                raw_features.len()
            ));
        }
        let mut out_of_domain = false;
        let mut normalized = Vec::with_capacity(raw_features.len());
        for (raw, feature) in raw_features.iter().zip(&self.features) {
            if !raw.is_finite() {
                return Err(format!("non-finite feature {}", feature.name));
            }
            out_of_domain |= *raw < feature.minimum || *raw > feature.maximum;
            let unit = 2.0 * (raw - feature.minimum) / (feature.maximum - feature.minimum) - 1.0;
            normalized.push(unit.clamp(-1.0, 1.0) as f32);
        }
        Ok((normalized, out_of_domain))
    }

    /// Weights are caller-owned organism preferences, in the same order as `laws`.
    /// Out-of-domain candidates are returned without a score and must not be promoted.
    pub fn score_candidate(
        &self,
        raw_features: &[f64],
        weights: &[f64],
    ) -> Result<CandidateScore, String> {
        if weights.len() != self.laws.len()
            || weights.iter().any(|x| !x.is_finite() || x.abs() > 10.0)
        {
            return Err("candidate weights must be finite, bounded, and match laws".into());
        }
        let consequences = self.evaluate(raw_features)?;
        let out_of_domain = consequences.iter().any(|x| x.out_of_domain);
        let score = (!out_of_domain).then(|| {
            consequences
                .iter()
                .zip(weights)
                .map(|(x, w)| x.expected * w)
                .sum()
        });
        Ok(CandidateScore {
            score,
            out_of_domain,
            consequences,
        })
    }

    pub fn score_body_candidate(
        &self,
        context: &DecisionContext<'_>,
        candidate: &CandidateAction<'_>,
        weights: &[f64],
    ) -> Result<CandidateScore, String> {
        let features = self.fitted_features(context, candidate)?;
        self.score_candidate(&features, weights)
    }
}

/// Authoritative one-tick targets fitted by the v1 law bank.
pub fn transition_targets(before: &[f32; 6], after: &[f32; 6]) -> [f64; 3] {
    [
        f64::from(after[3] - before[3]),
        f64::from(before[0] - after[0]),
        f64::from(before[2] - after[2]),
    ]
}

/// Per-tick consequence required to approach an achieved raw physiology goal
/// uniformly over its remaining horizon. The caller must provide at least one tick.
pub fn desired_goal_consequences(
    current: &[f32; 6],
    achieved_goal: &[f32; 6],
    remaining_ticks: u32,
) -> Result<[f64; 3], String> {
    if remaining_ticks == 0 {
        return Err("goal consequence horizon must contain at least one tick".into());
    }
    let horizon = f64::from(remaining_ticks);
    Ok([
        f64::from(achieved_goal[3] - current[3]) / horizon,
        f64::from(current[0] - achieved_goal[0]) / horizon,
        f64::from(current[2] - achieved_goal[2]) / horizon,
    ])
}

fn interpolate_clamped(x: f64, knots: &[f64], values: &[f64]) -> f64 {
    if x <= knots[0] {
        return values[0];
    }
    if x >= knots[knots.len() - 1] {
        return values[values.len() - 1];
    }
    let hi = knots.partition_point(|k| *k <= x);
    let lo = hi - 1;
    let fraction = (x - knots[lo]) / (knots[hi] - knots[lo]);
    values[lo] + fraction * (values[hi] - values[lo])
}
