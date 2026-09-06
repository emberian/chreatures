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
        Some("ask") | Some("ask-transfers") => {
            let path = args.get(2).ok_or("state")?;
            let mut state: SearchState = read(path)?;
            let output = if args[1] == "ask" { state.ask(count(&args)?)? } else { state.ask_transfers(count(&args)?)? };
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
        _ => Err("usage: init CONFIG STATE SEED | register-environment STATE ENV | ask STATE N | ask-transfers STATE N | tell STATE EVAL | validate STATE".into()),
    }
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2)
    }
}
