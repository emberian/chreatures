// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native inference for the learned replacement, termination, and value heads.

use crate::{gemm_into, linear, tanh_all, Linear};
use serde::{Deserialize, Serialize};

pub(crate) const STATE: usize = 909;
pub(crate) const CHOICE: usize = 234;
pub(crate) const CANDIDATES: usize = 8;
const STATE_CODE: usize = 256;
const CHOICE_CODE: usize = 128;
const JOINT: usize = STATE_CODE + CHOICE_CODE * 2;
const FORMAT: &str = "chreatures-learned-sequence-control-private-v2";

#[derive(Clone)]
pub(crate) struct SequenceControlHeads {
    state_encoder: Linear,
    candidate_encoder: Linear,
    selector_hidden: Linear,
    selector_out: Linear,
    hazard_hidden: Linear,
    hazard_out: Linear,
    value_hidden: Linear,
    value_out: Linear,
}

#[derive(Clone, Debug)]
pub(crate) struct ControlDecision {
    pub hazard_logits: Vec<f32>,
    pub selector_logits: Vec<f32>,
    pub values: Vec<f32>,
    pub terminate: Vec<bool>,
    pub selected: Vec<i32>,
    pub hazard_mask: Vec<bool>,
    pub selector_mask: Vec<bool>,
    pub hazard_logp: Vec<f32>,
    pub selector_logp: Vec<f32>,
    pub logp: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivateState {
    format: String,
    batch: usize,
    policy_version: u64,
    policy_sha256: String,
    packed: Vec<f32>,
    rng: Vec<u64>,
}

#[derive(Clone)]
pub(crate) struct LearnedSequenceControl {
    batch: usize,
    pub policy_version: u64,
    pub policy_sha256: String,
    heads: SequenceControlHeads,
    packed: Vec<f32>,
    rng: Vec<u64>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn valid_identity(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}

fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

fn masked_logp(logits: &[f32], mask: &[bool], selected: usize) -> f32 {
    let maximum = logits
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(x, _)| *x)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(x, _)| (*x - maximum).exp())
        .sum::<f32>();
    logits[selected] - maximum - sum.ln()
}

impl LearnedSequenceControl {
    pub(crate) fn from_flat(
        batch: usize,
        packed: &[f32],
        policy_version: u64,
        policy_sha256: String,
        seed: u64,
    ) -> Result<Self, String> {
        if batch == 0
            || batch > 4096
            || !valid_identity(&policy_sha256)
            || packed.iter().any(|x| !x.is_finite())
        {
            return Err("learned sequence-control contract differs".into());
        }
        let mut cursor = 0;
        let heads = SequenceControlHeads {
            state_encoder: linear(packed, &mut cursor, STATE_CODE, STATE)
                .map_err(|e| e.to_string())?,
            candidate_encoder: linear(packed, &mut cursor, CHOICE_CODE, CHOICE)
                .map_err(|e| e.to_string())?,
            selector_hidden: linear(packed, &mut cursor, CHOICE_CODE, STATE_CODE + CHOICE_CODE)
                .map_err(|e| e.to_string())?,
            selector_out: linear(packed, &mut cursor, 1, CHOICE_CODE).map_err(|e| e.to_string())?,
            hazard_hidden: linear(packed, &mut cursor, CHOICE_CODE, JOINT)
                .map_err(|e| e.to_string())?,
            hazard_out: linear(packed, &mut cursor, 1, CHOICE_CODE).map_err(|e| e.to_string())?,
            value_hidden: linear(packed, &mut cursor, CHOICE_CODE, JOINT)
                .map_err(|e| e.to_string())?,
            value_out: linear(packed, &mut cursor, 1, CHOICE_CODE).map_err(|e| e.to_string())?,
        };
        if cursor != packed.len() {
            return Err("sequence-control weights have trailing values".into());
        }
        let mut rng = vec![0; batch];
        for (row, word) in rng.iter_mut().enumerate() {
            let mut s = seed ^ (row as u64).wrapping_mul(0xd134_2543_de82_ef95);
            *word = splitmix64(&mut s);
        }
        Ok(Self {
            batch,
            policy_version,
            policy_sha256,
            heads,
            packed: packed.to_vec(),
            rng,
        })
    }

    pub(crate) fn grow(&mut self, new_batch: usize, seed: u64) -> Result<(), String> {
        if new_batch <= self.batch || new_batch > 4096 {
            return Err("sequence-control growth differs".into());
        }
        self.rng.resize(new_batch, 0);
        for row in self.batch..new_batch {
            let mut s = seed ^ (row as u64).wrapping_mul(0xd134_2543_de82_ef95);
            self.rng[row] = splitmix64(&mut s);
        }
        self.batch = new_batch;
        Ok(())
    }

    pub(crate) fn clear_resident(&mut self, row: usize, seed: u64) -> Result<(), String> {
        if row >= self.batch {
            return Err("sequence-control resident differs".into());
        }
        let mut s = seed ^ (row as u64).wrapping_mul(0xd134_2543_de82_ef95);
        self.rng[row] = splitmix64(&mut s);
        Ok(())
    }

    fn logits(
        &self,
        state: &[f32],
        proposals: &[f32],
        active: &[f32],
        proposal_mask: &[bool],
        active_mask: &[bool],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if state.len() != self.batch * STATE
            || proposals.len() != self.batch * CANDIDATES * CHOICE
            || active.len() != self.batch * CHOICE
            || proposal_mask.len() != self.batch * CANDIDATES
            || active_mask.len() != self.batch
            || state
                .iter()
                .chain(proposals)
                .chain(active)
                .any(|x| !x.is_finite())
            || (0..self.batch).any(|b| !(0..CANDIDATES).any(|k| proposal_mask[b * CANDIDATES + k]))
        {
            return Err("sequence-control input tensors differ".into());
        }
        let mut s = vec![];
        gemm_into(state, self.batch, STATE, &self.heads.state_encoder, &mut s);
        tanh_all(&mut s);
        let mut c = vec![];
        gemm_into(
            proposals,
            self.batch * CANDIDATES,
            CHOICE,
            &self.heads.candidate_encoder,
            &mut c,
        );
        tanh_all(&mut c);
        let mut a = vec![];
        gemm_into(
            active,
            self.batch,
            CHOICE,
            &self.heads.candidate_encoder,
            &mut a,
        );
        tanh_all(&mut a);
        for b in 0..self.batch {
            if !active_mask[b] {
                a[b * CHOICE_CODE..(b + 1) * CHOICE_CODE].fill(0.0);
            }
        }
        let mut selector_in = vec![0.0; self.batch * CANDIDATES * (STATE_CODE + CHOICE_CODE)];
        let mut joint = vec![0.0; self.batch * JOINT];
        for b in 0..self.batch {
            let mut count = 0.0;
            for k in 0..CANDIDATES {
                let dst = (b * CANDIDATES + k) * (STATE_CODE + CHOICE_CODE);
                selector_in[dst..dst + STATE_CODE]
                    .copy_from_slice(&s[b * STATE_CODE..(b + 1) * STATE_CODE]);
                selector_in[dst + STATE_CODE..dst + STATE_CODE + CHOICE_CODE].copy_from_slice(
                    &c[(b * CANDIDATES + k) * CHOICE_CODE..(b * CANDIDATES + k + 1) * CHOICE_CODE],
                );
                if proposal_mask[b * CANDIDATES + k] {
                    count += 1.0;
                }
            }
            let dst = b * JOINT;
            joint[dst..dst + STATE_CODE].copy_from_slice(&s[b * STATE_CODE..(b + 1) * STATE_CODE]);
            joint[dst + STATE_CODE..dst + STATE_CODE + CHOICE_CODE]
                .copy_from_slice(&a[b * CHOICE_CODE..(b + 1) * CHOICE_CODE]);
            for k in 0..CANDIDATES {
                if proposal_mask[b * CANDIDATES + k] {
                    for j in 0..CHOICE_CODE {
                        joint[dst + STATE_CODE + CHOICE_CODE + j] +=
                            c[(b * CANDIDATES + k) * CHOICE_CODE + j] / count;
                    }
                }
            }
        }
        let mut sh = vec![];
        gemm_into(
            &selector_in,
            self.batch * CANDIDATES,
            STATE_CODE + CHOICE_CODE,
            &self.heads.selector_hidden,
            &mut sh,
        );
        tanh_all(&mut sh);
        let mut selector = vec![];
        gemm_into(
            &sh,
            self.batch * CANDIDATES,
            CHOICE_CODE,
            &self.heads.selector_out,
            &mut selector,
        );
        let mut hh = vec![];
        gemm_into(
            &joint,
            self.batch,
            JOINT,
            &self.heads.hazard_hidden,
            &mut hh,
        );
        tanh_all(&mut hh);
        let mut hazard = vec![];
        gemm_into(
            &hh,
            self.batch,
            CHOICE_CODE,
            &self.heads.hazard_out,
            &mut hazard,
        );
        let mut vh = vec![];
        gemm_into(&joint, self.batch, JOINT, &self.heads.value_hidden, &mut vh);
        tanh_all(&mut vh);
        let mut value = vec![];
        gemm_into(
            &vh,
            self.batch,
            CHOICE_CODE,
            &self.heads.value_out,
            &mut value,
        );
        if hazard
            .iter()
            .chain(&selector)
            .chain(&value)
            .any(|output| !output.is_finite())
        {
            return Err("sequence-control output tensors are nonfinite".into());
        }
        Ok((hazard, selector, value))
    }

    pub(crate) fn decide(
        &mut self,
        state: &[f32],
        proposals: &[f32],
        active: &[f32],
        proposal_mask: &[bool],
        active_mask: &[bool],
        sample: bool,
    ) -> Result<ControlDecision, String> {
        let (hazard_logits, selector_logits, values) =
            self.logits(state, proposals, active, proposal_mask, active_mask)?;
        let mut out = ControlDecision {
            hazard_logits,
            selector_logits,
            values,
            terminate: vec![false; self.batch],
            selected: vec![-1; self.batch],
            hazard_mask: active_mask.to_vec(),
            selector_mask: vec![false; self.batch],
            hazard_logp: vec![0.0; self.batch],
            selector_logp: vec![0.0; self.batch],
            logp: vec![0.0; self.batch],
        };
        for b in 0..self.batch {
            if active_mask[b] {
                let z = if sample {
                    unit(&mut self.rng[b]) < (1.0 / (1.0 + (-out.hazard_logits[b]).exp())) as f64
                } else {
                    out.hazard_logits[b] >= 0.0
                };
                out.terminate[b] = z;
                out.hazard_logp[b] = if z {
                    log_sigmoid(out.hazard_logits[b])
                } else {
                    log_sigmoid(-out.hazard_logits[b])
                };
                if !z {
                    out.logp[b] = out.hazard_logp[b];
                    continue;
                }
            }
            out.selector_mask[b] = true;
            let logits = &out.selector_logits[b * CANDIDATES..(b + 1) * CANDIDATES];
            let mask = &proposal_mask[b * CANDIDATES..(b + 1) * CANDIDATES];
            let selected = if sample {
                let max = logits
                    .iter()
                    .zip(mask)
                    .filter(|(_, m)| **m)
                    .map(|(x, _)| *x)
                    .fold(f32::NEG_INFINITY, f32::max);
                let total = logits
                    .iter()
                    .zip(mask)
                    .filter(|(_, m)| **m)
                    .map(|(x, _)| (*x - max).exp() as f64)
                    .sum::<f64>();
                let target = unit(&mut self.rng[b]) * total;
                let mut acc = 0.0;
                let mut pick = mask.iter().position(|x| *x).unwrap();
                for k in 0..CANDIDATES {
                    if mask[k] {
                        acc += (logits[k] - max).exp() as f64;
                        if target < acc {
                            pick = k;
                            break;
                        }
                    }
                }
                pick
            } else {
                let mut pick = mask.iter().position(|x| *x).unwrap();
                for k in pick + 1..CANDIDATES {
                    if mask[k] && logits[k] > logits[pick] {
                        pick = k
                    }
                }
                pick
            };
            out.selected[b] = selected as i32;
            out.selector_logp[b] = masked_logp(logits, mask, selected);
            out.logp[b] = out.hazard_logp[b] + out.selector_logp[b];
        }
        Ok(out)
    }

    /// Re-evaluate archived actions under these exact weights without advancing private RNG.
    pub(crate) fn likelihood(
        &self,
        state: &[f32],
        proposals: &[f32],
        active: &[f32],
        proposal_mask: &[bool],
        active_mask: &[bool],
        terminate: &[bool],
        selected: &[i32],
    ) -> Result<ControlDecision, String> {
        if terminate.len() != self.batch || selected.len() != self.batch {
            return Err("sequence-control archived decisions differ".into());
        }
        let (hazard_logits, selector_logits, values) =
            self.logits(state, proposals, active, proposal_mask, active_mask)?;
        let mut out = ControlDecision {
            hazard_logits,
            selector_logits,
            values,
            terminate: terminate.to_vec(),
            selected: selected.to_vec(),
            hazard_mask: active_mask.to_vec(),
            selector_mask: vec![false; self.batch],
            hazard_logp: vec![0.0; self.batch],
            selector_logp: vec![0.0; self.batch],
            logp: vec![0.0; self.batch],
        };
        for b in 0..self.batch {
            if !active_mask[b] && terminate[b] {
                return Err("termination is defined only for an active remainder".into());
            }
            if active_mask[b] {
                out.hazard_logp[b] = if terminate[b] {
                    log_sigmoid(out.hazard_logits[b])
                } else {
                    log_sigmoid(-out.hazard_logits[b])
                };
            }
            out.selector_mask[b] = !active_mask[b] || terminate[b];
            if out.selector_mask[b] {
                let chosen = usize::try_from(selected[b])
                    .ok()
                    .filter(|chosen| {
                        *chosen < CANDIDATES && proposal_mask[b * CANDIDATES + *chosen]
                    })
                    .ok_or("selected candidate is unavailable")?;
                out.selector_logp[b] = masked_logp(
                    &out.selector_logits[b * CANDIDATES..(b + 1) * CANDIDATES],
                    &proposal_mask[b * CANDIDATES..(b + 1) * CANDIDATES],
                    chosen,
                );
            } else if selected[b] != -1 {
                return Err("continued active remainder must use selected_candidate=-1".into());
            }
            out.logp[b] = out.hazard_logp[b] + out.selector_logp[b];
        }
        Ok(out)
    }

    pub(crate) fn decide_row(
        &mut self,
        row: usize,
        state: &[f32],
        proposals: &[f32],
        active: &[f32],
        proposal_mask: &[bool],
        active_mask: bool,
        sample: bool,
    ) -> Result<ControlDecision, String> {
        if row >= self.batch {
            return Err("sequence-control resident differs".into());
        }
        let mut one = Self {
            batch: 1,
            policy_version: self.policy_version.clone(),
            policy_sha256: self.policy_sha256.clone(),
            heads: self.heads.clone(),
            packed: self.packed.clone(),
            rng: vec![self.rng[row]],
        };
        let decision = one.decide(
            state,
            proposals,
            active,
            proposal_mask,
            &[active_mask],
            sample,
        )?;
        self.rng[row] = one.rng[0];
        Ok(decision)
    }

    #[cfg(test)]
    pub(crate) fn preview_value(
        &self,
        state: &[f32],
        proposals: &[f32],
        active: &[f32],
        proposal_mask: &[bool],
        active_mask: &[bool],
    ) -> Result<Vec<f32>, String> {
        Ok(self
            .logits(state, proposals, active, proposal_mask, active_mask)?
            .2)
    }

    pub(crate) fn replace(
        &mut self,
        packed: &[f32],
        new_version: u64,
        new_sha256: String,
        expected_version: u64,
        expected_sha256: &str,
    ) -> Result<(), String> {
        if self.policy_version != expected_version || self.policy_sha256 != expected_sha256 {
            return Err("sequence-control expected parent differs".into());
        }
        if new_version <= self.policy_version || new_sha256 == self.policy_sha256 {
            return Err("sequence-control successor identity differs".into());
        }
        let mut staged = Self::from_flat(self.batch, packed, new_version, new_sha256, 0)?;
        staged.rng = self.rng.clone();
        *self = staged;
        Ok(())
    }
    pub(crate) fn snapshot_json(&self) -> Result<String, String> {
        serde_json::to_string(&PrivateState {
            format: FORMAT.into(),
            batch: self.batch,
            policy_version: self.policy_version.clone(),
            policy_sha256: self.policy_sha256.clone(),
            packed: self.packed.clone(),
            rng: self.rng.clone(),
        })
        .map_err(|e| e.to_string())
    }
    pub(crate) fn restore_exact(value: &str) -> Result<Self, String> {
        let s: PrivateState = serde_json::from_str(value).map_err(|e| e.to_string())?;
        if s.format != FORMAT || s.rng.len() != s.batch {
            return Err("sequence-control snapshot identity differs".into());
        }
        let mut result = Self::from_flat(s.batch, &s.packed, s.policy_version, s.policy_sha256, 0)?;
        result.rng = s.rng;
        Ok(result)
    }
    pub(crate) fn restore_checked(&mut self, value: &str) -> Result<(), String> {
        let restored = Self::restore_exact(value)?;
        if restored.batch != self.batch
            || restored.policy_version != self.policy_version
            || restored.policy_sha256 != self.policy_sha256
            || restored.packed != self.packed
        {
            return Err("sequence-control snapshot parameters differ".into());
        }
        self.rng = restored.rng;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor_suffix::{MotorSuffixMemory, ACTIONS, CONTEXT, OUTCOMES};
    fn zeros() -> Vec<f32> {
        let n = STATE_CODE * STATE
            + STATE_CODE
            + CHOICE_CODE * CHOICE
            + CHOICE_CODE
            + CHOICE_CODE * (STATE_CODE + CHOICE_CODE)
            + CHOICE_CODE
            + CHOICE_CODE
            + 1
            + CHOICE_CODE * JOINT
            + CHOICE_CODE
            + CHOICE_CODE
            + 1
            + CHOICE_CODE * JOINT
            + CHOICE_CODE
            + CHOICE_CODE
            + 1;
        let mut p = vec![0.0; n];
        let hazard_bias = STATE_CODE * STATE
            + STATE_CODE
            + CHOICE_CODE * CHOICE
            + CHOICE_CODE
            + CHOICE_CODE * (STATE_CODE + CHOICE_CODE)
            + CHOICE_CODE
            + CHOICE_CODE
            + 1
            + CHOICE_CODE * JOINT
            + CHOICE_CODE
            + CHOICE_CODE;
        p[hazard_bias] = -(7.0f32).ln();
        p
    }
    #[test]
    fn likelihood_masks_and_restore_are_exact() {
        let mut c = LearnedSequenceControl::from_flat(2, &zeros(), 1, "a".repeat(64), 9).unwrap();
        let s = vec![0.0; 2 * STATE];
        let p = vec![0.0; 2 * CANDIDATES * CHOICE];
        let a = vec![0.0; 2 * CHOICE];
        let m = vec![true; 2 * CANDIDATES];
        let am = vec![true, false];
        let snap = c.snapshot_json().unwrap();
        let d = c.decide(&s, &p, &a, &m, &am, false).unwrap();
        assert!(!d.terminate[0]);
        assert_eq!(d.selected[0], -1);
        assert_eq!(d.selected[1], 0);
        assert!((d.hazard_logp[0] - (7.0f32 / 8.0).ln()).abs() < 1e-6);
        assert!((d.selector_logp[1] + (8.0f32).ln()).abs() < 1e-6);
        let restored = LearnedSequenceControl::restore_exact(&snap).unwrap();
        assert_eq!(
            restored.preview_value(&s, &p, &a, &m, &am).unwrap(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn joined_control_lifecycle_likelihood_and_restore() {
        let mut memory = MotorSuffixMemory::new(1, 31).unwrap();
        for tick in 1..=8 {
            memory
                .record_executed(
                    0,
                    tick,
                    &[0.0; CONTEXT],
                    &[tick as f32 / 10.0; ACTIONS],
                    &[0.2],
                )
                .unwrap();
        }
        let suffix = memory.recall(0, &[0.0; CONTEXT], 1).remove(0);
        memory.start(0, &suffix).unwrap();
        let mut control =
            LearnedSequenceControl::from_flat(1, &zeros(), 0, "b".repeat(64), 41).unwrap();
        let state = vec![0.0; STATE];
        let proposals = vec![0.0; CANDIDATES * CHOICE];
        let active = vec![0.0; CHOICE];
        let mask = vec![true; CANDIDATES];
        let decision = control
            .decide(&state, &proposals, &active, &mask, &[true], false)
            .unwrap();
        assert_eq!(decision.selected, [-1]);
        assert_eq!(decision.hazard_mask, [true]);
        assert_eq!(decision.selector_mask, [false]);
        let memory_snapshot = memory.snapshot_json().unwrap();
        let control_snapshot = control.snapshot_json().unwrap();
        memory.note_executed(0, 9, &[0.1; ACTIONS], &[0.3; OUTCOMES]);
        let restored_memory = MotorSuffixMemory::restore_json(&memory_snapshot, 1).unwrap();
        let mut restored_control =
            LearnedSequenceControl::from_flat(1, &zeros(), 0, "b".repeat(64), 99).unwrap();
        restored_control.restore_checked(&control_snapshot).unwrap();
        assert_eq!(restored_memory.active(0).unwrap().phase, 0);
        let replay = restored_control
            .likelihood(
                &state,
                &proposals,
                &active,
                &mask,
                &[true],
                &decision.terminate,
                &decision.selected,
            )
            .unwrap();
        assert_eq!(decision.hazard_logp, replay.hazard_logp);
        assert_eq!(decision.selector_logp, replay.selector_logp);
    }
}
