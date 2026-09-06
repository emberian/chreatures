// SPDX-License-Identifier: AGPL-3.0-or-later
#[path = "../../../native/cognitive-core/src/gam_law.rs"]
mod gam_law;

use gam_law::LawBank;
use std::env;

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: chreatures-gam-mechanisms ARTIFACT FEATURE...")?;
    let bank = if first == "--sha256" {
        let expected = args.next().ok_or("--sha256 requires a digest")?;
        let artifact = args.next().ok_or("--sha256 requires an artifact")?;
        LawBank::from_authenticated_json_path(artifact, &expected)?
    } else {
        LawBank::from_json_path(first)?
    };
    let features = args
        .map(|x| x.parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let result = bank.score_candidate(&features, &vec![1.0; bank.laws.len()])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}
