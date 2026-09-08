// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native half of the joined resident persistence / Wasm parity probe.
//! Creates deterministic nonzero weights, not a trained-competence claim.
use _cognitive_core::developmental::DevelopmentalResidentCohort;
use serde_json::json;
use std::{fs, path::Path};
fn weights(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 32) as u32 as f64 / u32::MAX as f64 - 0.5) as f32 * 0.035
        })
        .collect()
}
fn write_floats(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap();
}
fn latent(tick: u64, batch: usize) -> Vec<f32> {
    (0..batch * 512)
        .map(|i| ((i * 31 + tick as usize * 17) % 251) as f32 / 125.0 - 1.0)
        .collect()
}
fn main() {
    let out = std::env::args().nth(1).expect("output fixture directory");
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let core = weights(
        3 * 256 * (524 + 256 + 2) + 128 * (768 + 1) + 256 * (396 + 1) + 48 * (256 + 1),
        173,
    );
    let predictor = weights(
        3 * (256 * (908 + 1) + 3 * 256 * (12 + 256 + 2) + 512 * (256 + 1)),
        979,
    );
    let sequence = weights(
        256 * (909 + 1)
            + 128 * (234 + 1)
            + 128 * (384 + 1)
            + 129
            + 128 * (512 + 1)
            + 129
            + 128 * (512 + 1)
            + 129,
        727,
    );
    let config = json!({"batch":2,"action_mode":"sample","action_seed":314,"suffix_seed":271,
        "tick_seconds":0.01,
        "context_policy_version":"signed-context12-v1",
        "private_learning_version":"context-consequence-v1",
        "core_sha256":"a".repeat(64),"predictor_sha256":"b".repeat(64),
        "sequence_control_version":1,"sequence_control_sha256":"c".repeat(64),"research_training":false});
    fs::write(out.join("config.json"), config.to_string()).unwrap();
    write_floats(&out.join("core.f32"), &core);
    write_floats(&out.join("predictor.f32"), &predictor);
    write_floats(&out.join("sequence.f32"), &sequence);
    let make = || {
        DevelopmentalResidentCohort::from_packed(
            2,
            "sample",
            314,
            271,
            0.01,
            "signed-context12-v1",
            "context-consequence-v1",
            &core,
            "a".repeat(64),
            &predictor,
            "b".repeat(64),
            &sequence,
            1,
            "c".repeat(64),
            false,
        )
        .unwrap()
    };
    let mut resident = make();
    let mut previous = vec![0.0; 24];
    let mut steps = Vec::new();
    for tick in 0..14 {
        let z = latent(tick, 2);
        let reset = [tick == 0, tick == 0];
        let context = resident
            .step_flat(&z, &previous, &[tick; 2], &reset)
            .unwrap();
        if tick == 4 {
            // Pending decisions survive save/load and cannot be stepped twice.
            let pending = resident.save_bytes().unwrap();
            let mut fork = make();
            fork.load_bytes(&pending).unwrap();
            assert!(fork.step_flat(&z, &previous, &[tick; 2], &reset).is_err());
            assert_eq!(pending, fork.save_bytes().unwrap());
            assert!(fork.acknowledge_flat(&[tick + 1; 2], &context).is_err());
            assert_eq!(pending, fork.save_bytes().unwrap());
            fork.acknowledge_flat(&[tick; 2], &context).unwrap();
        }
        resident.acknowledge_flat(&[tick; 2], &context).unwrap();
        steps.push(json!({"tick":tick,"z":z,"previous":previous,"reset":reset,"context":context,
            "diagnostics":serde_json::from_str::<serde_json::Value>(&resident.diagnostics_json().unwrap()).unwrap()}));
        previous = context;
        if tick == 6 {
            let saved = resident.save_bytes().unwrap();
            fs::write(out.join("native-checkpoint.json"), &saved).unwrap();
            let mut replay = make();
            replay.load_bytes(&saved).unwrap();
            assert_eq!(saved, replay.save_bytes().unwrap());
            let next = latent(tick + 1, 2);
            let a = resident
                .step_flat(&next, &previous, &[tick + 1; 2], &[false; 2])
                .unwrap();
            let b = replay
                .step_flat(&next, &previous, &[tick + 1; 2], &[false; 2])
                .unwrap();
            assert_eq!(a, b);
            assert_eq!(resident.save_bytes().unwrap(), replay.save_bytes().unwrap());
            resident.load_bytes(&saved).unwrap();
            let expanded = resident.expanded_portable(1, 19, 23).unwrap();
            assert_eq!(expanded.batch_size(), 3);
            let before = resident.snapshot_data().unwrap();
            let after = expanded.snapshot_data().unwrap();
            let before: serde_json::Value = serde_json::from_str(&before.private).unwrap();
            let after: serde_json::Value = serde_json::from_str(&after.private).unwrap();
            assert_eq!(
                before["state"].as_array().unwrap(),
                &after["state"].as_array().unwrap()[..512]
            );
        }
    }
    fs::write(out.join("steps.json"), serde_json::to_vec(&steps).unwrap()).unwrap();
    fs::write(
        out.join("native-final.json"),
        resident.save_bytes().unwrap(),
    )
    .unwrap();
    println!("native joined resident probe passed: 2 residents, 14 sampled ticks, pending+ack boundaries, exact restore/RNG continuation, cohort expansion");
}
