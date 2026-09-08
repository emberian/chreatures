//! Browser embodiment: physical truth stays inside this module and MuJoCo.
//! Only retina RGB and the fixed 110 body-local afferents cross to the CNS.
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
const SITES: usize = 1771;
const CHANNELS: usize = 110;
const MOTOR_CHANNELS: usize = 34;
const MUSCLES: usize = 24;
const JOINTS: usize = 12;
#[derive(Clone, Serialize, Deserialize)]
struct Controller {
    max_joint_torque: f64,
    hip_passive_damping: f64,
    knee_passive_damping: f64,
    posture_kp: f64,
    posture_kd: f64,
    max_posture_torque: f64,
}
#[derive(Clone, Serialize, Deserialize)]
struct Body {
    root: usize,
    head: usize,
    qpos: Vec<usize>,
    dofs: Vec<usize>,
    controller: Controller,
    physiology: [f64; 12],
    eyes: [[f64; 3]; 2],
}
#[derive(Clone, Serialize, Deserialize)]
struct Entity {
    body: usize,
    free: bool,
    food: f64,
    nutrition: f64,
    odor: i32,
    strength: f64,
    growth: f64,
    geoms: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Config {
    engine: String,
    source_mjcf_sha256: String,
    atlas_sha256: String,
    anatomical_sites: Vec<[i16; 3]>,
    supported_sites: Vec<bool>,
    bodies: Vec<Body>,
    entities: Vec<Entity>,
    screen_geom: usize,
}
#[derive(Clone, Serialize, Deserialize)]
struct Resident {
    physiology: [f64; 12],
    grip: Option<usize>,
    gaze: f64,
    signals: [f64; 3],
    work: f64,
    /// Engineered antagonist activation state. This synthetic body makes no NMJ claim.
    muscle_activation: [f64; MUSCLES],
    joint_fatigue: [f64; JOINTS],
}
#[derive(Clone, Serialize, Deserialize)]
struct Emission {
    origin: [f64; 3],
    born: f64,
    frequency_hz: f64,
    envelope: f64,
    duration: f64,
    kind: u8,
}
#[derive(Serialize, Deserialize)]
struct Saved {
    format: String,
    engine: String,
    model: String,
    atlas: String,
    time: f64,
    rng: u64,
    residents: Vec<Resident>,
    food: Vec<f64>,
    reservoir: Vec<f64>,
    emissions: Vec<Emission>,
    memory: String,
}
#[wasm_bindgen]
pub struct WorldCore {
    config: Config,
    state: Saved,
    directions: Vec<[f64; 3]>,
}
fn err(s: &str) -> JsValue {
    JsValue::from_str(s)
}
fn rotate(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[i * 3 + j] * v[j]).sum())
}
fn local(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[j * 3 + i] * v[j]).sum())
}
fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}
#[wasm_bindgen]
impl WorldCore {
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, seed: u32) -> Result<WorldCore, JsValue> {
        let c: Config =
            serde_json::from_str(config).map_err(|_| err("invalid world configuration"))?;
        if c.anatomical_sites.len() != SITES
            || c.supported_sites.len() != SITES
            || c.bodies.is_empty()
            || c.bodies.len() > 32
            || c.entities.len() > 96
            || c.anatomical_sites.iter().filter(|s| s[0] == 1).count() != 879
            || c.anatomical_sites.iter().filter(|s| s[0] == 2).count() != 892
            || c.anatomical_sites
                .iter()
                .any(|s| s[0] < 1 || s[0] > 2 || s[1].abs() > 100 || s[2].abs() > 100)
            || c.bodies.iter().any(|b| {
                b.root > 4096
                    || b.head > 4096
                    || b.qpos.len() != 12
                    || b.dofs.len() != 12
                    || b.qpos.iter().chain(&b.dofs).any(|i| *i > 8192)
                    || b.physiology
                        .iter()
                        .any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
                    || b.eyes
                        .iter()
                        .flatten()
                        .any(|v| !v.is_finite() || v.abs() > 1.0)
                    || b.controller.max_joint_torque <= 0.0
                    || b.controller.hip_passive_damping < 0.0
                    || b.controller.knee_passive_damping < 0.0
            })
            || c.entities.iter().any(|e| {
                e.body > 4096
                    || e.geoms.iter().any(|i| *i > 4096)
                    || e.food < 0.0
                    || e.nutrition < 0.0
                    || e.nutrition > 1.0
                    || e.growth < 0.0
            })
        {
            return Err(err("invalid anatomy/cohort"));
        }
        let directions = c
            .anatomical_sites
            .iter()
            .map(|s| {
                let sign = if s[0] == 1 { 1.0 } else { -1.0 };
                let q = s[1] as f64 - 18.85;
                let r = s[2] as f64 - 19.92;
                let az = (sign * (55.0 + 2.7 * q + 1.35 * r)).to_radians();
                let el = (-1.35 * q + 2.7 * r).to_radians();
                [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
            })
            .collect();
        let state = Saved {
            format: "chreatures-browser-world-state-v2".into(),
            engine: c.engine.clone(),
            model: c.source_mjcf_sha256.clone(),
            atlas: c.atlas_sha256.clone(),
            time: 0.0,
            rng: (seed as u64).max(1),
            residents: c
                .bodies
                .iter()
                .map(|b| Resident {
                    physiology: b.physiology,
                    grip: None,
                    gaze: 0.0,
                    signals: [0.0; 3],
                    work: 0.0,
                    muscle_activation: [0.0; MUSCLES],
                    joint_fatigue: [0.0; JOINTS],
                })
                .collect(),
            food: c.entities.iter().map(|e| e.food).collect(),
            reservoir: c.entities.iter().map(|e| e.food * 2.0).collect(),
            emissions: Vec::new(),
            memory: String::new(),
        };
        Ok(Self {
            config: c,
            state,
            directions,
        })
    }
    pub fn snapshot(&self) -> String {
        serde_json::to_string(&self.state).unwrap()
    }
    pub fn restore(&mut self, snapshot: &str) -> Result<(), JsValue> {
        let s: Saved = serde_json::from_str(snapshot).map_err(|_| err("invalid world snapshot"))?;
        if s.format != self.state.format
            || s.engine != self.state.engine
            || s.model != self.state.model
            || s.atlas != self.state.atlas
            || s.residents.len() != self.state.residents.len()
            || s.food.len() != self.state.food.len()
            || s.reservoir.len() != s.food.len()
            || s.reservoir.iter().any(|v| !v.is_finite() || *v < 0.0)
            || s.emissions.len() > 1024
            || s.emissions.iter().any(|e| {
                !finite(&e.origin)
                    || !e.born.is_finite()
                    || !e.frequency_hz.is_finite()
                    || !(40.0..=1600.0).contains(&e.frequency_hz)
                    || !e.envelope.is_finite()
                    || !(0.0..=1.0).contains(&e.envelope)
                    || !e.duration.is_finite()
                    || !(0.005..=20.0).contains(&e.duration)
                    || e.kind > 1
            })
            || !s.time.is_finite()
            || s.food.iter().any(|f| !f.is_finite() || *f < 0.0)
            || s.residents.iter().any(|r| {
                !finite(&r.physiology)
                    || !finite(&r.signals)
                    || !r.gaze.is_finite()
                    || !r.work.is_finite()
                    || !finite(&r.muscle_activation)
                    || !finite(&r.joint_fatigue)
                    || r.muscle_activation.iter().any(|v| !(0.0..=1.0).contains(v))
                    || r.joint_fatigue.iter().any(|v| !(0.0..=1.0).contains(v))
                    || r.physiology.iter().any(|v| *v < 0.0 || *v > 1.0)
                    || r.grip.is_some_and(|g| g >= s.food.len())
            })
        {
            return Err(err("snapshot identity or dimensions differ"));
        }
        self.state = s;
        Ok(())
    }
    /// Explicit append-only physical topology transaction; personal state is preserved.
    pub fn append_entity_config(&mut self, config: &str) -> Result<(), JsValue> {
        let next: Config = serde_json::from_str(config)
            .map_err(|_| err("invalid appended model configuration"))?;
        if next.engine != self.config.engine
            || next.atlas_sha256 != self.config.atlas_sha256
            || next.anatomical_sites != self.config.anatomical_sites
            || next.supported_sites != self.config.supported_sites
            || next.screen_geom != self.config.screen_geom
            || serde_json::to_string(&next.bodies).unwrap()
                != serde_json::to_string(&self.config.bodies).unwrap()
            || next.entities.len() != self.config.entities.len() + 1
            || next.entities.len() > 96
            || serde_json::to_string(&next.entities[..self.config.entities.len()]).unwrap()
                != serde_json::to_string(&self.config.entities).unwrap()
        {
            return Err(err("append topology contract differs"));
        }
        let added = next.entities.last().unwrap();
        if !added.food.is_finite()
            || added.food < 0.0
            || added.food > 10.0
            || !added.nutrition.is_finite()
            || added.nutrition < 0.0
            || added.nutrition > 1.0
            || !added.growth.is_finite()
            || added.growth < 0.0
            || added.growth > 0.1
            || !added.strength.is_finite()
            || added.strength < 0.0
            || added.strength > 4.0
        {
            return Err(err("invalid appended material"));
        }
        self.state.food.push(added.food);
        self.state.reservoir.push(added.food * 2.0);
        self.state.model = next.source_mjcf_sha256.clone();
        self.config = next;
        Ok(())
    }
    /// Opaque CNS-derived memory checkpoint supplied by the cognitive owner.
    pub fn set_memory(&mut self, memory: String) {
        self.state.memory = memory;
    }
    pub fn time(&self) -> f64 {
        self.state.time
    }
    pub fn food(&self) -> Vec<f64> {
        self.state.food.clone()
    }
    pub fn growth_scales(&self) -> Vec<f64> {
        self.config
            .entities
            .iter()
            .enumerate()
            .map(|(i, e)| {
                if e.food > 0.0 {
                    (self.state.food[i] / e.food).max(0.01).cbrt()
                } else {
                    1.0
                }
            })
            .collect()
    }
    /// Owner-created acoustic source at an actual 3D world location.
    pub fn visitor_sound(
        &mut self,
        position: &[f64],
        frequency_hz: f64,
        envelope: f64,
        duration: f64,
    ) -> Result<(), JsValue> {
        if position.len() != 3
            || !finite(position)
            || !frequency_hz.is_finite()
            || !(40.0..=1600.0).contains(&frequency_hz)
            || !envelope.is_finite()
            || !(0.0..=1.0).contains(&envelope)
            || !duration.is_finite()
            || !(0.005..=20.0).contains(&duration)
            || self.state.emissions.len() >= 1024
        {
            return Err(err("invalid visitor acoustic source"));
        }
        self.state.emissions.push(Emission {
            origin: position.try_into().unwrap(),
            born: self.state.time,
            frequency_hz,
            envelope,
            duration,
            kind: 0,
        });
        Ok(())
    }
    /// CNS motor head -> engineered antagonist muscles -> 12 physical hinge torques.
    pub fn actuation(
        &mut self,
        commands: &[f64],
        qpos: &[f64],
        qvel: &[f64],
        positions: &[f64],
        rotations: &[f64],
        velocities: &[f64],
        time: f64,
        dt: f64,
    ) -> Result<Vec<f64>, JsValue> {
        let n = self.config.bodies.len();
        if commands.len() != n * MOTOR_CHANNELS
            || velocities.len() != n * 6
            || !finite(commands)
            || !finite(qpos)
            || !finite(qvel)
            || !finite(positions)
            || !finite(rotations)
            || !finite(velocities)
            || !time.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || dt > 0.05
            || commands.chunks_exact(MOTOR_CHANNELS).any(|a| {
                a.iter().enumerate().any(|(k, v)| {
                    if k == 24 || k == 25 {
                        !(-1.0..=1.0).contains(v)
                    } else {
                        !(0.0..=1.0).contains(v)
                    }
                })
            })
        {
            return Err(err("invalid physical actuation packet"));
        }
        if self.config.bodies.iter().any(|b| {
            b.qpos.len() != 12
                || b.dofs.len() != 12
                || b.qpos.iter().any(|i| *i >= qpos.len())
                || b.dofs.iter().any(|i| *i >= qvel.len())
                || b.root * 9 + 9 > rotations.len()
                || b.root * 3 + 3 > positions.len()
        }) || self
            .config
            .entities
            .iter()
            .any(|e| e.body * 3 + 3 > positions.len())
        {
            return Err(err("physical model layout differs"));
        }
        let mut output = vec![0.0; n * 19];
        for row in 0..n {
            let b = &self.config.bodies[row];
            let c = &b.controller;
            let r = &mut self.state.residents[row];
            let a = &commands[row * MOTOR_CHANNELS..row * MOTOR_CHANNELS + MOTOR_CHANNELS];
            let out = &mut output[row * 19..row * 19 + 19];
            let resource = (0.12 + 0.88 * r.physiology[0]) * (0.55 + 0.45 * r.physiology[6]);
            let alpha = 1.0 - (-dt / 0.04).exp();
            for muscle in 0..MUSCLES {
                let excitation = a[muscle].clamp(0.0, 1.0);
                r.muscle_activation[muscle] += alpha * (excitation - r.muscle_activation[muscle]);
            }
            for j in 0..JOINTS {
                let angle = qpos[b.qpos[j]];
                let velocity = qvel[b.dofs[j]];
                let normalized_length = angle / if j % 2 == 0 { 0.733 } else { 1.012 };
                let force_length = (-1.35 * normalized_length * normalized_length).exp();
                let force_velocity = (1.0 - 0.18 * velocity.abs()).clamp(0.28, 1.0);
                let local_capacity = resource * (1.0 - 0.82 * r.joint_fatigue[j]);
                let positive = r.muscle_activation[j * 2];
                let negative = r.muscle_activation[j * 2 + 1];
                let active = (positive - negative)
                    * c.max_joint_torque
                    * local_capacity
                    * force_length
                    * force_velocity;
                let passive = -if j % 2 == 0 {
                    c.hip_passive_damping
                } else {
                    c.knee_passive_damping
                } * velocity
                    - c.max_joint_torque * 0.08 * normalized_length.powi(3);
                let torque = (active + passive).clamp(-c.max_joint_torque, c.max_joint_torque);
                out[j] = torque;
                let recruitment = 0.5 * (positive + negative);
                r.joint_fatigue[j] = (r.joint_fatigue[j]
                    + dt * (0.34 * recruitment * recruitment - 0.075 * (1.0 - recruitment)))
                    .clamp(0.0, 1.0);
                r.work += (active * velocity).abs() * dt + 0.02 * recruitment * dt;
            }
            let rot = &rotations[b.root * 9..b.root * 9 + 9];
            let vel = &velocities[row * 6..row * 6 + 6];
            let posture = a[25].clamp(-1.0, 1.0);
            let mut correction = [
                rot[5] * c.posture_kp - vel[0] * c.posture_kd,
                -rot[2] * c.posture_kp - vel[1] * c.posture_kd,
                0.0,
            ];
            correction[0] *= posture;
            correction[1] *= posture;
            let norm = correction[0].hypot(correction[1]);
            if norm > c.max_posture_torque {
                for x in &mut correction {
                    *x *= c.max_posture_torque / norm
                }
            }
            out[12..15].copy_from_slice(&correction);
            let p = &positions[b.root * 3..b.root * 3 + 3];
            if a[31] > 0.5 || a[26] < 0.1 {
                r.grip = None;
            } else if r.grip.is_none() {
                r.grip = self
                    .config
                    .entities
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.free)
                    .filter_map(|(i, e)| {
                        let ep = &positions[e.body * 3..e.body * 3 + 3];
                        let lp = local(rot, std::array::from_fn(|k| ep[k] - p[k]));
                        let d = dist(&lp, &[0.17, 0.0, 0.04]);
                        if d < 0.16 {
                            Some((i, d))
                        } else {
                            None
                        }
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|x| x.0);
            }
            out[15] = -1.0;
            if let Some(i) = r.grip {
                out[15] = i as f64;
                let e = &self.config.entities[i];
                let ep = &positions[e.body * 3..e.body * 3 + 3];
                let offset = rotate(rot, [0.17, 0.0, 0.04]);
                let force: [f64; 3] = std::array::from_fn(|k| 15.0 * (p[k] + offset[k] - ep[k]));
                let norm = dist(&force, &[0.0; 3]);
                for k in 0..3 {
                    out[16 + k] = force[k] * (8.0 / norm.max(8.0));
                }
            }
            r.gaze = (r.gaze + a[24].clamp(-1.0, 1.0) * dt).clamp(-0.7, 0.7);
        }
        Ok(output)
    }
    /// Whole-world ecological state advances once per control tick, not per visual frame.
    pub fn advance(&mut self, commands: &[f64], positions: &[f64], dt: f64) -> Result<(), JsValue> {
        let n = self.state.residents.len();
        if commands.len() != n * MOTOR_CHANNELS
            || !finite(commands)
            || !finite(positions)
            || !dt.is_finite()
            || dt <= 0.0
            || dt > 0.1
            || commands.chunks_exact(MOTOR_CHANNELS).any(|a| {
                a.iter().enumerate().any(|(k, v)| {
                    if k == 24 || k == 25 {
                        !(-1.0..=1.0).contains(v)
                    } else {
                        !(0.0..=1.0).contains(v)
                    }
                })
            })
            || self
                .config
                .bodies
                .iter()
                .any(|b| b.root * 3 + 3 > positions.len())
            || self
                .config
                .entities
                .iter()
                .any(|e| e.body * 3 + 3 > positions.len())
        {
            return Err(err("invalid ecological tick"));
        }
        self.state.rng ^= self.state.rng << 13;
        self.state.rng ^= self.state.rng >> 7;
        self.state.rng ^= self.state.rng << 17;
        for (i, e) in self.config.entities.iter().enumerate() {
            let growth = (e.growth * dt * (1.0 - self.state.food[i] / e.food.max(0.001)))
                .max(0.0)
                .min(self.state.reservoir[i]);
            self.state.reservoir[i] -= growth;
            self.state.food[i] = (self.state.food[i] + growth).clamp(0.0, e.food.max(0.0));
        }
        self.state
            .emissions
            .retain(|e| self.state.time - e.born < e.duration);
        for row in 0..n {
            let r = &mut self.state.residents[row];
            let a = &commands[row * MOTOR_CHANNELS..row * MOTOR_CHANNELS + MOTOR_CHANNELS];
            let p =
                &positions[self.config.bodies[row].root * 3..self.config.bodies[row].root * 3 + 3];
            if a[30] > 0.1 {
                for (i, e) in self.config.entities.iter().enumerate() {
                    if dist(p, &positions[e.body * 3..e.body * 3 + 3]) < 0.28 {
                        let eaten = (a[30].clamp(0.0, 1.0) * 0.12 * dt)
                            .min(self.state.food[i])
                            .min(1.0 - r.physiology[1]);
                        self.state.food[i] -= eaten;
                        r.physiology[1] += eaten * e.nutrition;
                    }
                }
            }
            let digested = r.physiology[1].min(0.018 * dt);
            r.physiology[1] -= digested;
            r.physiology[0] =
                (r.physiology[0] + digested * 0.8 - 0.0015 * dt - 0.025 * r.work).clamp(0.0, 1.0);
            r.physiology[2] = (r.physiology[2] + 0.05 * r.work - 0.006 * dt).clamp(0.0, 1.0);
            r.physiology[6] =
                (r.physiology[6] + (r.physiology[0] - 0.2) * 0.0005 * dt).clamp(0.0, 1.0);
            let allocate = a[33].clamp(0.0, 1.0) * 0.002 * dt * r.physiology[0];
            r.physiology[7] = (r.physiology[7] + allocate).min(1.0);
            r.physiology[0] = (r.physiology[0] - allocate).max(0.0);
            let secretion = (a[32].clamp(0.0, 1.0) * 0.01 * dt).min(r.physiology[8]);
            r.physiology[8] = (r.physiology[8] + 0.001 * dt - secretion).clamp(0.0, 1.0);
            if secretion > 0.0 && self.state.emissions.len() < 1024 {
                self.state.emissions.push(Emission {
                    origin: p.try_into().unwrap(),
                    born: self.state.time,
                    frequency_hz: 1600.0,
                    envelope: (secretion * 100.0).min(1.0),
                    duration: 8.0,
                    kind: 1,
                });
            }
            for k in 0..3 {
                r.signals[k] = a[27 + k].clamp(0.0, 1.0);
            }
            if r.signals.iter().any(|v| *v > 0.01) && self.state.emissions.len() < 1024 {
                for k in 0..3 {
                    if r.signals[k] > 0.01 && self.state.emissions.len() < 1024 {
                        self.state.emissions.push(Emission {
                            origin: p.try_into().unwrap(),
                            born: self.state.time,
                            frequency_hz: [80.0, 320.0, 1280.0][k],
                            envelope: r.signals[k],
                            duration: 0.12,
                            kind: 0,
                        });
                    }
                }
            }
            r.work = 0.0;
        }
        self.state.time += dt;
        Ok(())
    }
    /// Physics-only packet to MuJoCo: each eye is origin3 followed by 1771 direction3.
    pub fn rays(
        &self,
        row: usize,
        head_position: &[f64],
        head_rotation: &[f64],
    ) -> Result<Vec<f64>, JsValue> {
        if row >= self.config.bodies.len()
            || head_position.len() != 3
            || head_rotation.len() != 9
            || !finite(head_position)
            || !finite(head_rotation)
        {
            return Err(err("invalid head pose"));
        }
        let b = &self.config.bodies[row];
        let gaze = self.state.residents[row].gaze;
        let mut out = Vec::with_capacity(2 * (3 + SITES * 3));
        for eye in 0..2 {
            let origin = rotate(head_rotation, b.eyes[eye]);
            for k in 0..3 {
                out.push(head_position[k] + origin[k]);
            }
            for d in &self.directions {
                let tilted = [
                    gaze.cos() * d[0] - gaze.sin() * d[2],
                    d[1],
                    gaze.sin() * d[0] + gaze.cos() * d[2],
                ];
                out.extend(rotate(head_rotation, tilted));
            }
        }
        Ok(out)
    }
    /// Actual MuJoCo nearest hits, then physical screen UV emission (screen can be occluded).
    pub fn retina(
        &self,
        row: usize,
        rays: &[f64],
        hit_geoms: &[i32],
        distances: &[f64],
        geom_positions: &[f64],
        geom_rotations: &[f64],
        geom_sizes: &[f64],
        geom_colors: &[f64],
        frame: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let ng = geom_positions.len() / 3;
        if row >= self.config.bodies.len()
            || rays.len() != 2 * (3 + SITES * 3)
            || hit_geoms.len() != SITES * 2
            || distances.len() != SITES * 2
            || geom_rotations.len() != ng * 9
            || geom_sizes.len() != ng * 3
            || geom_colors.len() != ng * 4
            || width < 1
            || height < 1
            || width > 2048
            || height > 2048
            || frame.len() != width * height * 3
            || !finite(rays)
            || !finite(distances)
            || !finite(geom_positions)
            || !finite(geom_rotations)
            || !finite(geom_sizes)
            || !finite(geom_colors)
            || frame.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
            || hit_geoms.iter().any(|g| *g >= ng as i32)
        {
            return Err(err("invalid ray result"));
        }
        let mut rgb = vec![0.0; SITES * 3];
        for site in 0..SITES {
            if !self.config.supported_sites[site] {
                continue;
            }
            let eye = self.config.anatomical_sites[site][0] as usize - 1;
            let hit = hit_geoms[eye * SITES + site];
            let distance = distances[eye * SITES + site];
            let value = if hit < 0 || distance < 0.0 || distance > 3.2 {
                [0.015, 0.02, 0.025]
            } else {
                let g = hit as usize;
                if g == self.config.screen_geom {
                    let base = eye * (3 + SITES * 3);
                    let p: [f64; 3] = std::array::from_fn(|k| {
                        rays[base + k] + distance * rays[base + 3 + site * 3 + k]
                            - geom_positions[g * 3 + k]
                    });
                    let lp = local(&geom_rotations[g * 9..g * 9 + 9], p);
                    let ld = local(
                        &geom_rotations[g * 9..g * 9 + 9],
                        std::array::from_fn(|k| rays[base + 3 + site * 3 + k]),
                    );
                    if ld[0] <= 0.0 {
                        [0.02; 3]
                    } else {
                        let u = (0.5 - lp[1] / (2.0 * geom_sizes[g * 3 + 1])).clamp(0.0, 1.0);
                        let v = (0.5 - lp[2] / (2.0 * geom_sizes[g * 3 + 2])).clamp(0.0, 1.0);
                        let x = u * (width - 1) as f64;
                        let y = v * (height - 1) as f64;
                        let x0 = x.floor() as usize;
                        let y0 = y.floor() as usize;
                        let x1 = (x0 + 1).min(width - 1);
                        let y1 = (y0 + 1).min(height - 1);
                        let fx = (x - x0 as f64) as f32;
                        let fy = (y - y0 as f64) as f32;
                        std::array::from_fn(|k| {
                            let top = frame[(y0 * width + x0) * 3 + k] * (1.0 - fx)
                                + frame[(y0 * width + x1) * 3 + k] * fx;
                            let bot = frame[(y1 * width + x0) * 3 + k] * (1.0 - fx)
                                + frame[(y1 * width + x1) * 3 + k] * fx;
                            top * (1.0 - fy) + bot * fy
                        })
                    }
                } else {
                    std::array::from_fn(|k| geom_colors[g * 4 + k] as f32 * 0.72)
                }
            };
            rgb[site * 3..site * 3 + 3].copy_from_slice(&value);
        }
        Ok(rgb)
    }
    /// Body-local contract order exactly matches optic_retina.rs transduce_nonvisual.
    pub fn afferents(
        &self,
        positions: &[f64],
        rotations: &[f64],
        local_velocities: &[f64],
        contacts: &[f64],
        shade_distances: &[f64],
        qpos: &[f64],
        qvel: &[f64],
        joint_loads: &[f64],
        foot_contacts: &[f64],
    ) -> Result<Vec<f32>, JsValue> {
        let n = self.state.residents.len();
        if local_velocities.len() != n * 6
            || contacts.len() != n * 25
            || shade_distances.len() != n
            || joint_loads.len() != n * JOINTS
            || foot_contacts.len() != n * 6
            || !finite(shade_distances)
            || !finite(positions)
            || !finite(rotations)
            || !finite(local_velocities)
            || !finite(contacts)
            || !finite(qpos)
            || !finite(qvel)
            || !finite(joint_loads)
            || !finite(foot_contacts)
            || self.config.bodies.iter().any(|b| {
                b.root * 3 + 3 > positions.len()
                    || b.root * 9 + 9 > rotations.len()
                    || b.qpos.iter().any(|i| *i >= qpos.len())
                    || b.dofs.iter().any(|i| *i >= qvel.len())
            })
            || self
                .config
                .entities
                .iter()
                .any(|e| e.body * 3 + 3 > positions.len())
        {
            return Err(err("invalid afferent physics packet"));
        }
        let mut out = vec![0.0; n * CHANNELS];
        for row in 0..n {
            let b = &self.config.bodies[row];
            let r = &self.state.residents[row];
            let o = &mut out[row * CHANNELS..row * CHANNELS + CHANNELS];
            let p = &positions[b.root * 3..b.root * 3 + 3];
            let rot = &rotations[b.root * 9..b.root * 9 + 9];
            for eye in 0..2 {
                let offset = rotate(rot, [0.182, if eye == 0 { 0.038 } else { -0.038 }, 0.026]);
                let ap: [f64; 3] = std::array::from_fn(|k| p[k] + offset[k]);
                for (i, e) in self.config.entities.iter().enumerate() {
                    if e.odor >= 0 && e.odor < 3 {
                        let d = dist(&ap, &positions[e.body * 3..e.body * 3 + 3]);
                        let amount = e.strength * (self.state.food[i] / e.food.max(0.001));
                        o[eye * 3 + e.odor as usize] += (amount * (-d / 0.7).exp() / 4.0) as f32;
                    }
                }
            }
            for v in &mut o[..6] {
                *v = v.clamp(0.0, 1.0)
            }
            for (source, destination, scale) in [(3, 6, 4.0), (0, 12, 8.0)] {
                for k in 0..3 {
                    let v =
                        (local_velocities[row * 6 + source + k] / scale).clamp(-1.0, 1.0) as f32;
                    o[destination + k * 2] = v.max(0.0);
                    o[destination + k * 2 + 1] = (-v).max(0.0);
                }
            }
            let count = contacts[row * 25].clamp(0.0, 8.0) as usize;
            for i in 0..count {
                let normal = local(
                    rot,
                    std::array::from_fn(|k| contacts[row * 25 + 1 + i * 3 + k]),
                );
                for k in 0..3 {
                    o[18 + k * 2] = o[18 + k * 2].max(normal[k].max(0.0) as f32);
                    o[19 + k * 2] = o[19 + k * 2].max((-normal[k]).max(0.0) as f32);
                }
            }
            o[24] = count as f32 / 8.0;
            o[25] = if count > 0 { 1.0 } else { 0.0 };
            o[26] = o[25];
            for emission in &self.state.emissions {
                let distance = dist(p, &emission.origin);
                let age = self.state.time - emission.born;
                if emission.kind == 0 {
                    // Finite spherical propagation: 30m/s in this enlarged engineered world.
                    let arrived = age - distance / 30.0;
                    if (0.0..emission.duration).contains(&arrived) {
                        let attack = (arrived / 0.015).clamp(0.0, 1.0);
                        let release = ((emission.duration - arrived) / 0.03).clamp(0.0, 1.0);
                        let temporal = attack.min(release);
                        let log_frequency = emission.frequency_hz.ln();
                        let lo = 40.0_f64.ln();
                        let hi = 1600.0_f64.ln();
                        for k in 0..16 {
                            let centre = lo + (hi - lo) * k as f64 / 15.0;
                            let sigma = 0.35 * 2.0_f64.ln();
                            let tuning = (-0.5 * ((log_frequency - centre) / sigma).powi(2)).exp();
                            o[27 + k] += (emission.envelope * temporal * tuning
                                / (1.0 + distance * distance))
                                as f32;
                        }
                    }
                } else {
                    for eye in 0..2 {
                        let offset =
                            rotate(rot, [0.182, if eye == 0 { 0.038 } else { -0.038 }, 0.026]);
                        let ap: [f64; 3] = std::array::from_fn(|k| p[k] + offset[k]);
                        let d = dist(&ap, &emission.origin);
                        let spread = 0.08 + 0.04 * age;
                        for k in 0..3 {
                            o[eye * 3 + k] += (if k == 2 { emission.envelope } else { 0.0 }
                                * (-d * d / spread).exp()
                                * (-age / 8.0).exp()
                                / 4.0) as f32;
                        }
                    }
                }
            }
            for v in &mut o[..6] {
                *v = v.min(1.0);
            }
            for v in &mut o[27..43] {
                *v = v.min(1.0)
            }
            o[43] = if shade_distances[row] >= 0.0 && shade_distances[row] < 3.2 {
                1.0
            } else {
                0.0
            };
            for k in 0..12 {
                o[44 + k] = r.physiology[k] as f32;
                let angle_scale = if k % 2 == 0 { 0.733 } else { 1.012 };
                o[56 + k] = (qpos[b.qpos[k]] / angle_scale).clamp(-1.0, 1.0) as f32;
                o[68 + k] = (qvel[b.dofs[k]] / 20.0).clamp(-1.0, 1.0) as f32;
                o[80 + k] = (joint_loads[row * JOINTS + k] / b.controller.max_joint_torque)
                    .clamp(-1.0, 1.0) as f32;
                o[98 + k] = r.joint_fatigue[k] as f32;
            }
            for k in 0..6 {
                o[92 + k] = foot_contacts[row * 6 + k].clamp(0.0, 1.0) as f32;
            }
        }
        Ok(out)
    }
}
