// SPDX-License-Identifier: AGPL-3.0-or-later
use chreatures_fly_world::{serve_stdio, NativeFlyWorld};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("native fly world: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut scene = None;
    let mut seed = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--scene") => scene = arguments.next().map(PathBuf::from),
            Some("--seed") => {
                let value = arguments.next().ok_or("--seed requires a value")?;
                seed = Some(
                    value
                        .to_string_lossy()
                        .parse::<u32>()
                        .map_err(|_| "invalid --seed")?,
                );
            }
            _ => return Err("arguments must be --scene PATH --seed U32".into()),
        }
    }
    let mut world = NativeFlyWorld::open(
        scene.ok_or("missing --scene")?,
        seed.ok_or("missing --seed")?,
    )?;
    serve_stdio(&mut world)
}
