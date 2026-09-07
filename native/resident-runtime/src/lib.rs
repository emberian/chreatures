// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin browser host. All resident state and numeric mechanisms live in the
//! same cognitive-core crate loaded by Python; this module owns only ABI copies.
use _cognitive_core::developmental::DevelopmentalResidentCohort;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    batch: usize,
    action_mode: String,
    action_seed: u64,
    suffix_seed: u64,
    core_sha256: String,
    predictor_sha256: String,
    sequence_control_version: u64,
    sequence_control_sha256: String,
    #[serde(default)]
    research_training: bool,
}

fn error(message: String) -> JsValue {
    JsValue::from_str(&message)
}

/// A copy of one completed decision, stable across subsequent engine calls.
#[wasm_bindgen]
pub struct ResidentStep {
    proposed_command: Vec<f32>,
    diagnostics_json: String,
}
#[wasm_bindgen]
impl ResidentStep {
    #[wasm_bindgen(getter, js_name = proposedCommand)]
    pub fn proposed_command(&self) -> Vec<f32> {
        self.proposed_command.clone()
    }
    #[wasm_bindgen(getter, js_name = diagnosticsJson)]
    pub fn diagnostics_json(&self) -> String {
        self.diagnostics_json.clone()
    }
}

#[wasm_bindgen]
pub struct ResidentRuntime {
    core: DevelopmentalResidentCohort,
}

#[wasm_bindgen]
impl ResidentRuntime {
    /// Float32Array data is packed in the canonical training artifact order.
    /// Artifact checksums identify the immutable model; the asset loader verifies
    /// transport hashes before entering this host. No URLs or raw sensory ports.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        core: &[f32],
        predictor: &[f32],
        sequence: &[f32],
    ) -> Result<ResidentRuntime, JsValue> {
        let c: Config = serde_json::from_str(config_json).map_err(|e| error(e.to_string()))?;
        Ok(Self {
            core: DevelopmentalResidentCohort::from_packed(
                c.batch,
                &c.action_mode,
                c.action_seed,
                c.suffix_seed,
                core,
                c.core_sha256,
                predictor,
                c.predictor_sha256,
                sequence,
                c.sequence_control_version,
                c.sequence_control_sha256,
                c.research_training,
            )
            .map_err(error)?,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn batch(&self) -> usize {
        self.core.batch_size()
    }

    /// Inputs are resident-major B×512 and B×12; ticks are BigUint64Array,
    /// resets Uint8Array containing exactly 0 or 1. The command needs a receipt
    /// after physics execution, before another decision is allowed.
    #[wasm_bindgen(js_name = stepFlat)]
    pub fn step_flat(
        &mut self,
        z: &[f32],
        previous: &[f32],
        ticks: &[u64],
        reset: &[u8],
    ) -> Result<ResidentStep, JsValue> {
        if reset.iter().any(|&x| x > 1) {
            return Err(error("reset must contain only 0 or 1".into()));
        }
        let reset: Vec<bool> = reset.iter().map(|&x| x != 0).collect();
        let proposed_command = self
            .core
            .step_flat(z, previous, ticks, &reset)
            .map_err(error)?;
        let diagnostics_json = self.core.diagnostics_json().map_err(error)?;
        Ok(ResidentStep {
            proposed_command,
            diagnostics_json,
        })
    }

    pub fn acknowledge(
        &mut self,
        ticks: &[u64],
        delivered_command: &[f32],
    ) -> Result<Vec<u8>, JsValue> {
        self.core
            .acknowledge_flat(ticks, delivered_command)
            .map(|values| values.into_iter().map(u8::from).collect())
            .map_err(error)
    }

    #[wasm_bindgen(js_name = diagnosticsJson)]
    pub fn diagnostics_json(&self) -> Result<String, JsValue> {
        self.core.diagnostics_json().map_err(error)
    }

    #[wasm_bindgen(js_name = saveBytes)]
    pub fn save_bytes(&self) -> Result<Vec<u8>, JsValue> {
        self.core.save_bytes().map_err(error)
    }

    /// Restore validates all identities and state before committing any mutation.
    #[wasm_bindgen(js_name = loadBytes)]
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.core.load_bytes(bytes).map_err(error)
    }

    pub fn expanded(
        &self,
        additions: usize,
        action_seed: u64,
        suffix_seed: u64,
    ) -> Result<ResidentRuntime, JsValue> {
        Ok(Self {
            core: self
                .core
                .expanded_portable(additions, action_seed, suffix_seed)
                .map_err(error)?,
        })
    }
}
