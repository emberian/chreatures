// SPDX-License-Identifier: AGPL-3.0-or-later
#[path = "../../../native/cognitive-core/src/gam_law.rs"]
mod gam_law;

use gam_law::{CandidateAction, DecisionContext, LawBank};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

#[derive(Deserialize)]
struct VerificationCases {
    schema: String,
    tolerance: f64,
    cases: Vec<VerificationCase>,
}

#[derive(Deserialize)]
struct VerificationCase {
    name: String,
    features: Vec<f64>,
    expected: Vec<f64>,
    out_of_domain: bool,
}

#[derive(Serialize)]
struct VerificationReceipt {
    schema: &'static str,
    status: &'static str,
    artifact_sha256: String,
    cases_sha256: String,
    cases: usize,
    in_domain_cases: usize,
    out_of_domain_cases: usize,
    feature_contract_checked: bool,
    maximum_absolute_error: f64,
    tolerance: f64,
}

fn file_sha256(path: impl AsRef<Path>) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn verify(artifact_path: &str, cases_path: &str) -> Result<(), String> {
    let bank = LawBank::from_json_path(artifact_path)?;
    let physiology = [0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7];
    let neural = [0.25f32; 384];
    let action = [0.1f32, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let extracted = bank.fitted_features(
        &DecisionContext {
            physiology: &physiology,
            neural: &neural,
        },
        &CandidateAction {
            action: &action,
            oral: 0.9,
        },
    )?;
    let expected_features = [
        0.2, 0.4, 0.5, 0.7, 0.25, 0.1, -0.2, 0.4, 0.9, 0.25, 0.04, -0.1,
    ];
    if extracted
        .iter()
        .zip(expected_features)
        .any(|(actual, expected)| (actual - expected).abs() > 1e-6)
    {
        return Err("native fitted feature extraction contract differs".into());
    }
    let encoded = fs::read(cases_path).map_err(|error| error.to_string())?;
    let cases: VerificationCases =
        serde_json::from_slice(&encoded).map_err(|error| error.to_string())?;
    if cases.schema != "chreatures-gam-native-verification-cases-v1"
        || cases.cases.is_empty()
        || !cases.tolerance.is_finite()
        || cases.tolerance <= 0.0
    {
        return Err("native verification case contract differs".into());
    }
    let mut maximum_absolute_error = 0.0f64;
    let mut in_domain_cases = 0usize;
    let mut out_of_domain_cases = 0usize;
    for case in &cases.cases {
        if case.name.is_empty() || case.expected.len() != bank.laws.len() {
            return Err("native verification case shape differs".into());
        }
        let evaluated = bank.evaluate(&case.features)?;
        let actual_ood = evaluated.iter().any(|value| value.out_of_domain);
        if actual_ood != case.out_of_domain {
            return Err(format!("domain verdict differs for {}", case.name));
        }
        if actual_ood {
            out_of_domain_cases += 1;
        } else {
            in_domain_cases += 1;
        }
        for (actual, expected) in evaluated.iter().zip(&case.expected) {
            let error = (actual.expected - expected).abs();
            if !error.is_finite() || error > cases.tolerance {
                return Err(format!(
                    "native value differs for {} / {}: error={error:e}",
                    case.name, actual.name
                ));
            }
            maximum_absolute_error = maximum_absolute_error.max(error);
        }
    }
    let receipt = VerificationReceipt {
        schema: "chreatures-gam-native-evaluation-v1",
        status: "passed current native LawBank evaluation and domain checks",
        artifact_sha256: file_sha256(artifact_path)?,
        cases_sha256: format!("{:x}", Sha256::digest(encoded)),
        cases: cases.cases.len(),
        in_domain_cases,
        out_of_domain_cases,
        feature_contract_checked: true,
        maximum_absolute_error,
        tolerance: cases.tolerance,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let first = args.next().ok_or(
        "usage: chreatures-gam-mechanisms [--verify ARTIFACT CASES | [--sha256 DIGEST] ARTIFACT FEATURE...]",
    )?;
    if first == "--verify" {
        let artifact = args.next().ok_or("--verify requires an artifact")?;
        let cases = args.next().ok_or("--verify requires a case file")?;
        if args.next().is_some() {
            return Err("--verify accepts exactly an artifact and case file".into());
        }
        return verify(&artifact, &cases);
    }
    let bank = if first == "--sha256" {
        let expected = args.next().ok_or("--sha256 requires a digest")?;
        let artifact = args.next().ok_or("--sha256 requires an artifact")?;
        LawBank::from_authenticated_json_path(artifact, &expected)?
    } else {
        LawBank::from_json_path(first)?
    };
    let features = args
        .map(|value| value.parse::<f64>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let result = bank.score_candidate(&features, &vec![1.0; bank.laws.len()])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}
