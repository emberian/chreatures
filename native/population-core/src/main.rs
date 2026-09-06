use population_core::*;
use serde::{de::DeserializeOwned, Serialize};
use std::{env, fs};
fn read<T: DeserializeOwned>(path: &str) -> Result<T, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write<T: Serialize>(path: &str, value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    let temporary = format!("{path}.tmp-{}", std::process::id());
    fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    fs::rename(temporary, path).map_err(|e| e.to_string())
}
fn count(args: &[String]) -> Result<usize, String> {
    args.get(3)
        .ok_or_else(|| "count".to_string())?
        .parse()
        .map_err(|_| "count".into())
}
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("init") => {
            let config: SearchConfig = read(args.get(2).ok_or("config")?)?;
            let seed = args.get(4).ok_or("seed")?.parse().map_err(|_| "seed")?;
            write(args.get(3).ok_or("state")?, &SearchState::new(config, seed)?)
        }
        Some("register-environment") => {
            let path = args.get(2).ok_or("state")?;
            let mut state: SearchState = read(path)?;
            state.register_environment(read(args.get(3).ok_or("environment")?)?)?;
            write(path, &state)
        }
        Some("register-proposal-scores") => {
            let path = args.get(2).ok_or("state")?;
            let mut state: SearchState = read(path)?;
            state.register_proposal_scores(read(args.get(3).ok_or("scores")?)?)?;
            write(path, &state)
        }
        Some("ask") | Some("ask-transfers") | Some("ask-challenges") => {
            let path = args.get(2).ok_or("state")?;
            let mut state: SearchState = read(path)?;
            let output = match args[1].as_str() {
                "ask" => state.ask(count(&args)?)?,
                "ask-transfers" => state.ask_transfers(count(&args)?)?,
                _ => state.ask_challenges(count(&args)?)?,
            };
            write(path, &state)?;
            println!("{}", serde_json::to_string(&output).unwrap());
            Ok(())
        }
        Some("tell") => {
            let path = args.get(2).ok_or("state")?;
            let mut state: SearchState = read(path)?;
            let output = state.tell(read(args.get(3).ok_or("evaluation")?)?)?;
            write(path, &state)?;
            println!("{}", serde_json::to_string(&output).unwrap());
            Ok(())
        }
        Some("validate") => {
            let state: SearchState = read(args.get(2).ok_or("state")?)?;
            state.validate()
        }
        Some("environment-frontier") => {
            let state: SearchState = read(args.get(2).ok_or("state")?)?;
            state.validate()?;
            println!("{}", serde_json::to_string(&state.environment_frontier()).unwrap());
            Ok(())
        }
        Some("authorize-infrastructure-retry") => {
            let path = args.get(2).ok_or("state")?;
            let mut state: SearchState = read(path)?;
            state.authorize_infrastructure_retry(
                args.get(3).ok_or("candidate")?,
                args.get(4).ok_or("environment")?,
            )?;
            write(path, &state)
        }
        _ => Err("usage: init CONFIG STATE SEED | register-environment STATE ENV | register-proposal-scores STATE SCORES | ask STATE N | ask-transfers STATE N | ask-challenges STATE N | authorize-infrastructure-retry STATE CANDIDATE ENVIRONMENT | tell STATE EVAL | environment-frontier STATE | validate STATE".into()),
    }
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2)
    }
}
