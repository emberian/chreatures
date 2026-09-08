//! Full fly embodiment: MuJoCo owns mechanics, Rust owns transduction and ecology.
//! Only retina5313 and body807 leave this boundary for the actual MaleCNS.
mod fly_optics;
mod fly_acoustics;
mod fly_senses;
mod fly_types;
use chreatures_ecology_core as eco;
use fly_types::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Clone, Default, Serialize, Deserialize)]
struct HostEcology {
    growth_token: Option<String>,
    route_open_fraction: Vec<f64>,
    route_advection_m3_s: Vec<f64>,
    construction_sites: Vec<eco::ConstructionSite>,
    birth_sites: Vec<eco::BirthSite>,
    photon_exposures: Vec<eco::PhotonExposure>,
}
#[derive(Serialize, Deserialize)]
struct Envelope {
    format: String,
    state: Saved,
    ecology: String,
    host: HostEcology,
    acoustics: fly_acoustics::AcousticSnapshot,
}
struct Pending {
    state: Saved,
    token: String,
    previous_config: Option<Config>,
    acoustics: fly_acoustics::FlyAcoustics,
}

#[wasm_bindgen]
pub struct WorldCore {
    config: Config,
    state: Saved,
    ecology: eco::EcologyWorld,
    host: HostEcology,
    pending: Option<Pending>,
    acoustics: fly_acoustics::FlyAcoustics,
}
fn err(s: impl AsRef<str>) -> JsValue {
    JsValue::from_str(s.as_ref())
}
fn hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn commands_valid(a: &[f64], b: usize) -> bool {
    a.len() == b * MOTOR_CHANNELS
        && finite(a)
        && a.chunks_exact(MOTOR_CHANNELS).all(|r| {
            r.iter().enumerate().all(|(i, v)| {
                if i < 84 {
                    (-1.0..=1.0).contains(v)
                } else {
                    (0.0..=1.0).contains(v)
                }
            })
        })
}
fn mix_wing_afferents(out: &mut [f32], frame: &fly_acoustics::AcousticFrame) -> Result<(), String> {
    let n = out.len() / CHANNELS;
    if frame.antenna_airflow_local_mm_s.len() != n || frame.acoustic_bands.len() != n {
        return Err("endogenous acoustic cohort differs".into());
    }
    for (i, row) in out.chunks_exact_mut(CHANNELS).enumerate() {
        for antenna in 0..2 {
            for axis in 0..3 {
                row[40 + 3 * antenna + axis] += frame.antenna_airflow_local_mm_s[i][antenna][axis] as f32;
            }
        }
        for band in 0..16 { row[46 + band] += frame.acoustic_bands[i][band] as f32; }
    }
    if out.iter().any(|v| !v.is_finite()) { return Err("nonfinite endogenous afferent".into()); }
    Ok(())
}

#[wasm_bindgen]
impl WorldCore {
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, seed: u32) -> Result<WorldCore, JsValue> {
        let c: Config =
            serde_json::from_str(config).map_err(|e| err(format!("fly fixture: {e}")))?;
        if c.engine != ENGINE
            || [
                &c.source_mjcf_sha256,
                &c.atlas_sha256,
                &c.morphology_sha256,
                &c.sensory_schema_sha256,
                &c.actuator_schema_sha256,
            ]
            .iter()
            .any(|x| !hash(x))
            || c.bodies.is_empty()
            || c.bodies.len() > 32
            || c.ecology.pools.len() != 8
            || c.anatomical_sites.len() != SITES
            || c.supported_sites.len() != SITES
            || c.retinal_directions.len() != SITES
            || c.anatomical_sites.iter().any(|s| s[0] != 1 && s[0] != 2)
            || c.retinal_directions
                .iter()
                .any(|v| !finite(v) || (norm(*v) - 1.0).abs() > 1e-5)
            || c.geoms.iter().enumerate().any(|(i, g)| i != g.id)
            || c.screen_geom >= c.geoms.len()
            || !c.ray_distance_mm.is_finite()
            || c.ray_distance_mm <= 0.0
            || !finite(&c.airflow_mm_s)
            || !finite(&c.volatile_fraction)
            || !c.atp_per_model_work.is_finite()
            || c.atp_per_model_work < 0.0
            || c.volatile_fraction.iter().any(|x| !(0.0..=1.0).contains(x))
        {
            return Err(err("invalid current fly world contract"));
        }
        for b in &c.bodies {
            if b.qpos.len() != JOINTS
                || b.dofs.len() != JOINTS
                || b.segments.len() != SEGMENTS
                || b.actuators.len() != PHYSICAL_CONTROLS
                || b.neutral.len() != 84
                || b.control_ranges.len() != 84
                || !finite(&b.neutral)
                || b.wings.iter().any(|w| !b.segments.contains(w))
                || b.wing_centroid_local_mm.iter().any(|v| !finite(v))
                || !finite(&b.wing_source_gain)
                || b.wing_source_gain.iter().any(|v| *v < 0.0)
                || b.control_ranges.iter().zip(&b.neutral).any(|(range, q)| {
                    !finite(range) || range[0] >= range[1] || *q < range[0] || *q > range[1]
                })
                || b.feet
                    .iter()
                    .any(|f| f.is_empty() || f.iter().any(|id| !b.segments.contains(id)))
                || !b.mouth.contact_radius_mm.is_finite()
                || b.mouth.contact_radius_mm <= 0.0
                || !c.ecology.organisms.iter().any(|o| o.id == b.ecology_id)
            {
                return Err(err("incomplete anatomical fly body mapping"));
            }
        }
        let ecology = eco::EcologyWorld::new(c.ecology.clone()).map_err(|e| err(e.to_string()))?;
        let acoustics = fly_acoustics::FlyAcoustics::new(c.acoustics, c.bodies.len())
            .map_err(|e| err(e.to_string()))?;
        let state = Saved {
            format: "chreatures-fly-core-state-v4".into(),
            engine: c.engine.clone(),
            model: c.source_mjcf_sha256.clone(),
            atlas: c.atlas_sha256.clone(),
            time: 0.0,
            rng: (seed as u64).max(1),
            residents: c
                .bodies
                .iter()
                .map(|_| Resident {
                    fatigue: vec![0.0; MOTOR_CHANNELS],
                    pump_phase: 0.0,
                    pump_flow: 0.0,
                    salivary_flow: 0.0,
                    work: 0.0,
                    last_commands: vec![0.0; MOTOR_CHANNELS],
                })
                .collect(),
            emissions: Vec::new(),
            memory: String::new(),
            afferents: vec![0.0; c.bodies.len() * CHANNELS],
        };
        let host = HostEcology {
            route_open_fraction: c
                .ecology
                .routes
                .iter()
                .map(|r| r.base_open_fraction)
                .collect(),
            route_advection_m3_s: vec![0.0; c.ecology.routes.len()],
            ..HostEcology::default()
        };
        Ok(Self {
            config: c,
            state,
            ecology,
            host,
            pending: None,
            acoustics,
        })
    }
    pub fn time(&self) -> f64 {
        self.state.time
    }
    pub fn set_memory(&mut self, memory: String) -> Result<(), JsValue> {
        if self.pending.is_some() {
            return Err(err("ecological mutation pending"));
        }
        self.state.memory = memory;
        Ok(())
    }
    /// Append physical geometry only after the host has compiled and validated
    /// its candidate. Existing body/joint/actuator addresses cannot change.
    pub fn rebind_physics(&mut self, config: &str) -> Result<(), JsValue> {
        let next: Config = serde_json::from_str(config).map_err(|e| err(e.to_string()))?;
        let stable = |c: &Config| {
            let mut v = serde_json::to_value(c).unwrap();
            let m = v.as_object_mut().unwrap();
            for key in ["source_mjcf_sha256", "geoms", "material_bindings"] { m.remove(key); }
            v
        };
        if !hash(&next.source_mjcf_sha256) || stable(&next) != stable(&self.config)
            || next.geoms.len() < self.config.geoms.len()
            || next.geoms.iter().enumerate().any(|(i,g)|g.id!=i)
            || serde_json::to_value(&next.geoms[..self.config.geoms.len()]).unwrap()!=serde_json::to_value(&self.config.geoms).unwrap()
            || next.material_bindings.len()<self.config.material_bindings.len()
            || serde_json::to_value(&next.material_bindings[..self.config.material_bindings.len()]).unwrap()!=serde_json::to_value(&self.config.material_bindings).unwrap()
            || next.material_bindings.iter().any(|b|b.geoms.iter().any(|i|*i>=next.geoms.len())) {
            return Err(err("physical rebind changed existing world semantics"));
        }
        for binding in &next.material_bindings[self.config.material_bindings.len()..] {
            let present = match &binding.store {
                eco::StoreId::Region(id)=>self.config.ecology.regions.iter().any(|s|&s.id==id),
                eco::StoreId::Organism(id)=>self.ecology.state().organisms.iter().any(|s|&s.id==id),
                eco::StoreId::Packet(id)=>self.ecology.state().packets.iter().any(|s|&s.id==id),
                eco::StoreId::Structure(id)=>self.ecology.state().structures.iter().any(|s|&s.id==id)
                    ||self.ecology.pending_delta().is_some_and(|d|d.physical_creations.iter().any(|p|p.material_store==binding.store)),
            };
            if !present {return Err(err("physical rebind invented an unaccounted material store"));}
        }
        if let Some(p) = &mut self.pending {
            if p.previous_config.is_none(){p.previous_config=Some(self.config.clone());}
            p.state.model=next.source_mjcf_sha256.clone();
        } else {self.state.model=next.source_mjcf_sha256.clone();}
        self.config=next;Ok(())
    }
    pub fn afferents(&self) -> Vec<f32> {
        self.state.afferents.clone()
    }
    /// Observer-only finite material and physiology, never a resident input.
    pub fn ecology_observe(&self) -> String {
        serde_json::to_string(self.ecology.state()).unwrap()
    }
    pub fn actuator_state(&self) -> String {
        serde_json::to_string(&self.state.residents).unwrap()
    }
    pub fn food(&self) -> Vec<f64> {
        self.ecology
            .state()
            .packets
            .iter()
            .map(|p| p.material.quantity[1])
            .collect()
    }
    pub fn growth_scales(&self) -> Vec<f64> {
        self.ecology
            .state()
            .packets
            .iter()
            .map(|p| {
                let seed = self.config.ecology.packets.iter().find(|v| v.id == p.id);
                let initial = seed.map(|v| v.initial.iter().sum::<f64>()).unwrap_or(0.0);
                if initial > 0.0 {
                    (p.material.quantity.iter().sum::<f64>() / initial)
                        .max(0.0)
                        .cbrt()
                } else {
                    1.0
                }
            })
            .collect()
    }
    /// Pure learned-output-to-effective-servo conversion. No posture correction,
    /// gait, destination, abstract thrust or oral decision is supplied here.
    pub fn actuation(&self, commands: &[f64]) -> Result<Vec<f64>, JsValue> {
        if !commands_valid(commands, self.config.bodies.len()) {
            return Err(err("invalid CNS MOTOR92"));
        }
        let mut controls = vec![0.0; self.config.bodies.len() * PHYSICAL_CONTROLS];
        for (row, b) in self.config.bodies.iter().enumerate() {
            let a = &commands[row * MOTOR_CHANNELS..(row + 1) * MOTOR_CHANNELS];
            let out = &mut controls[row * PHYSICAL_CONTROLS..(row + 1) * PHYSICAL_CONTROLS];
            for j in 0..84 {
                let q0 = b.neutral[j];
                let [lo, hi] = b.control_ranges[j];
                out[j] = q0 + a[j] * if a[j] >= 0.0 { hi - q0 } else { q0 - lo };
            }
            out[84..90].copy_from_slice(&a[84..90]);
        }
        Ok(controls)
    }
    /// Effective force capacity belongs to body physiology. The host multiplies
    /// each original actuator force bound, preserving the desired servo target.
    pub fn actuator_capacity(&self) -> Result<Vec<f64>, JsValue> {
        let mut out = Vec::with_capacity(self.config.bodies.len() * PHYSICAL_CONTROLS);
        for (row, b) in self.config.bodies.iter().enumerate() {
            let internal = self
                .ecology
                .fly_interoception12(&b.ecology_id, &self.config.interoception)
                .map_err(|e| err(e.to_string()))?;
            let resource = (0.1 + 0.9 * internal[0]) * (0.25 + 0.75 * internal[5]);
            out.extend(
                self.state.residents[row].fatigue[..PHYSICAL_CONTROLS]
                    .iter()
                    .map(|f| resource * (1.0 - 0.8 * f)),
            );
        }
        Ok(out)
    }
    pub fn set_route_measurements(&mut self, open: &[f64], flows: &[f64]) -> Result<(), JsValue> {
        if self.pending.is_some()
            || open.len() != self.config.ecology.routes.len()
            || flows.len() != open.len()
            || !finite(open)
            || !finite(flows)
            || open.iter().any(|v| !(0.0..=1.0).contains(v))
        {
            return Err(err("invalid measured material routes"));
        }
        self.host.route_open_fraction = open.to_vec();
        self.host.route_advection_m3_s = flows.to_vec();
        Ok(())
    }
    /// Host-only geometry candidates and measured light. They are not goals or
    /// observations passed to any cognitive controller.
    pub fn set_ecology_sites(&mut self, sites: &str) -> Result<(), JsValue> {
        if self.pending.is_some() {
            return Err(err("ecological mutation pending"));
        }
        #[derive(Deserialize)]
        struct Sites {
            growth_token: Option<String>,
            construction_sites: Vec<eco::ConstructionSite>,
            birth_sites: Vec<eco::BirthSite>,
            photon_exposures: Vec<eco::PhotonExposure>,
        }
        let s: Sites = serde_json::from_str(sites).map_err(|e| err(e.to_string()))?;
        if s.construction_sites.len() > 4096
            || s.birth_sites.len() > 4096
            || s.photon_exposures.len() > 4096
        {
            return Err(err("too many physical ecology candidates"));
        }
        self.host.growth_token = s.growth_token;
        self.host.construction_sites = s.construction_sites;
        self.host.birth_sites = s.birth_sites;
        self.host.photon_exposures = s.photon_exposures;
        Ok(())
    }
    /// Local developmental dynamics propose physical growth. Only the host can
    /// measure geometry and accept a collision-free subset; no CNS input uses it.
    pub fn propose_growth(&mut self, input: &str) -> Result<String, JsValue> {
        if self.pending.is_some() { return Err(err("ecological mutation pending")); }
        let input: eco::GrowthInput = serde_json::from_str(input).map_err(|e| err(e.to_string()))?;
        if input.dt_s != DT { return Err(err("growth and physical tick differ")); }
        let proposal = self.ecology.propose_growth(&input).map_err(|e| err(e.to_string()))?;
        serde_json::to_string(&proposal).map_err(|e| err(e.to_string()))
    }
    pub fn discard_growth(&mut self, token: &str) -> Result<(), JsValue> {
        if self.pending.is_some() { return Err(err("ecological mutation pending")); }
        self.ecology.discard_growth(token).map_err(|e| err(e.to_string()))?;
        if self.host.growth_token.as_deref() == Some(token) {
            self.host.growth_token = None;
            self.host.construction_sites.clear();
            self.host.birth_sites.clear();
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sense_physics(
        &mut self,
        qpos: &[f64],
        qvel: &[f64],
        loads: &[f64],
        positions: &[f64],
        rotations: &[f64],
        velocities: &[f64],
        contacts: &[f64],
        offsets: &[u32],
        irradiance: &[f64],
    ) -> Result<(), JsValue> {
        if self.pending.is_some() {
            return Err(err("ecological mutation pending"));
        }
        let packet = PhysicalPacket {
            qpos,
            qvel,
            loads,
            positions,
            rotations,
            velocities,
            contacts,
            offsets,
            irradiance,
        };
        self.state.afferents =
            fly_senses::transduce(&self.config, &self.state, &self.ecology, &packet)
                .map_err(err)?
                .afferents;
        mix_wing_afferents(&mut self.state.afferents, &self.acoustics.frame()).map_err(err)?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_advance(
        &mut self,
        commands: &[f64],
        qpos: &[f64],
        qvel: &[f64],
        loads: &[f64],
        positions: &[f64],
        rotations: &[f64],
        velocities: &[f64],
        contacts: &[f64],
        offsets: &[u32],
        irradiance: &[f64],
        dt: f64,
    ) -> Result<String, JsValue> {
        if self.pending.is_some() || dt != DT || !commands_valid(commands, self.config.bodies.len())
        {
            return Err(err("invalid or overlapping fly tick"));
        }
        let packet = PhysicalPacket {
            qpos,
            qvel,
            loads,
            positions,
            rotations,
            velocities,
            contacts,
            offsets,
            irradiance,
        };
        let sensed = fly_senses::transduce(&self.config, &self.state, &self.ecology, &packet)
            .map_err(err)?;
        let period = self.config.acoustics.sample_period_s();
        if (period * packet.samples() as f64 - dt).abs() > 1e-10 {
            return Err(err("wing sampling cadence differs from physical packet"));
        }
        let mut acoustics = self.acoustics.clone();
        for sample in 0..packet.samples() {
            let motion = |body| {
                let velocity = packet.vel(sample, body);
                fly_acoustics::RigidMotion {
                    position_mm: packet.pos(sample, body),
                    world_from_local: packet.rot(sample, body).try_into().unwrap(),
                    angular_velocity_rad_s: velocity[..3].try_into().unwrap(),
                    linear_velocity_mm_s: velocity[3..].try_into().unwrap(),
                }
            };
            let bodies = self.config.bodies.iter().map(|b| fly_acoustics::ResidentKinematics {
                root: motion(b.root), wings: b.wings.map(motion),
                wing_centroid_local_mm: b.wing_centroid_local_mm,
                wing_source_gain: b.wing_source_gain,
                antenna_position_mm: std::array::from_fn(|i| {
                    let a = &b.olfactory_sites[i]; packet.site(sample, a.body, a.position)
                }),
                antenna_world_from_local: std::array::from_fn(|i| packet.rot(sample, b.olfactory_sites[i].body).try_into().unwrap()),
            }).collect::<Vec<_>>();
            acoustics.push_sample(self.state.time + (sample + 1) as f64 * period, &bodies)
                .map_err(|e| err(e.to_string()))?;
        }
        let mut next = self.state.clone();
        let mut exchanges = Vec::new();
        for (row, b) in self.config.bodies.iter().enumerate() {
            let a = &commands[row * MOTOR_CHANNELS..(row + 1) * MOTOR_CHANNELS];
            let r = &mut next.residents[row];
            r.last_commands.copy_from_slice(a);
            r.work = sensed.mechanical_work_rate[row] * dt;
            for (j, value) in a.iter().enumerate() {
                let recruitment = value.abs();
                r.fatigue[j] = (r.fatigue[j]
                    + dt * (0.25 * recruitment * recruitment - 0.08 * (1.0 - recruitment)))
                    .clamp(0.0, 1.0);
            }
            // Engineered finite pump: drive sets frequency, volume is transferred
            // only through measured mouth contact. The MN command supplies drive.
            let advance = std::f64::consts::TAU * 8.0 * a[90] * dt * (1.0 - 0.6 * r.fatigue[90]);
            r.pump_phase = (r.pump_phase + advance) % std::f64::consts::TAU;
            r.pump_flow = 0.0;
            r.salivary_flow = 0.0;
            let total = sensed.mouth_contacts[row]
                .iter()
                .map(|(_, f)| *f)
                .sum::<f64>()
                .max(1.0);
            for &(mi, fraction) in &sensed.mouth_contacts[row] {
                let material = &self.config.material_bindings[mi];
                let chemical = fly_senses::store_concentration(&self.ecology, &material.store);
                // 0.02 nL effective stroke volume; the phenomenological scale is
                // explicit and fitted later, not a measured fly ingestion law.
                let volume = 2e-14 * advance / std::f64::consts::TAU * fraction / total;
                let quantity: Vec<f64> = chemical.iter().map(|v| v * volume).collect();
                exchanges.push(eco::DirectedExchange {
                    event_id: format!("pump-{}-{mi}", b.id),
                    from: material.store.clone(),
                    to: eco::StoreId::Organism(b.ecology_id.clone()),
                    maximum_quantity: quantity,
                });
                let mut saliva = vec![0.0; 8];
                saliva[0] = a[91] * (1.0 - 0.6 * r.fatigue[91]) * dt * 0.002 * fraction / total;
                exchanges.push(eco::DirectedExchange {
                    event_id: format!("saliva-{}-{mi}", b.id),
                    from: eco::StoreId::Organism(b.ecology_id.clone()),
                    to: material.store.clone(),
                    maximum_quantity: saliva,
                });
            }
        }
        next.rng ^= next.rng << 13;
        next.rng ^= next.rng >> 7;
        next.rng ^= next.rng << 17;
        next.time += dt;
        next.emissions
            .retain(|e| next.time - e.born < e.duration + 1.0);
        let mut respiratory_contacts=Vec::new();
        for b in &self.config.bodies {
            let p=packet.pos(packet.samples()-1,b.root).map(|x|x*0.001);
            if let Some(region)=self.config.ecology.regions.iter().min_by(|a,b|{
                let da=(0..3).map(|k|(a.center_m[k]-p[k]).powi(2)).sum::<f64>();
                let db=(0..3).map(|k|(b.center_m[k]-p[k]).powi(2)).sum::<f64>();da.total_cmp(&db)
            }) {
                respiratory_contacts.push(eco::ContactEvent{a:eco::StoreId::Region(region.id.clone()),b:eco::StoreId::Organism(b.ecology_id.clone()),conductance_m3_s:vec![2e-14,0.0,0.0,0.0,1e-12,1e-12,0.0,1e-12]});
            }
        }
        // Anchored living surfaces exchange with their containing material cell.
        for organism in &self.ecology.state().organisms {
            if let Some(region)=&organism.anchored_region {
                respiratory_contacts.push(eco::ContactEvent{a:eco::StoreId::Region(region.clone()),b:eco::StoreId::Organism(organism.id.clone()),conductance_m3_s:vec![2e-12,2e-13,2e-13,2e-13,1e-12,1e-12,0.0,1e-12]});
            }
        }
        let input = eco::TickInput {
            growth_token: self.host.growth_token.clone(),
            dt_s: dt,
            route_open_fraction: self.host.route_open_fraction.clone(),
            route_advection_m3_s: self.host.route_advection_m3_s.clone(),
            contacts: respiratory_contacts,
            directed_exchanges: exchanges,
            metabolic_work_demands: self
                .config
                .bodies
                .iter()
                .enumerate()
                .map(|(row, b)| eco::MetabolicWorkDemand {
                    organism_id: b.ecology_id.clone(),
                    atp_quantity: next.residents[row].work * self.config.atp_per_model_work,
                })
                .collect(),
            photon_exposures: self.host.photon_exposures.clone(),
            construction_sites: self.host.construction_sites.clone(),
            birth_sites: self.host.birth_sites.clone(),
            packet_retirements: Vec::new(),
            afferent_sample_sites: Vec::new(),
        };
        let delta = self
            .ecology
            .prepare_step(&input)
            .map_err(|e| err(e.to_string()))?;
        for (row, b) in self.config.bodies.iter().enumerate() {
            for transfer in &delta.transfers {
                if transfer.detail_id.starts_with(&format!("pump-{}-", b.id)) {
                    next.residents[row].pump_flow += transfer.quantity.iter().sum::<f64>() / dt;
                }
                if transfer.detail_id.starts_with(&format!("saliva-{}-", b.id)) {
                    next.residents[row].salivary_flow += transfer.quantity.iter().sum::<f64>() / dt;
                }
            }
        }
        // Compute afferents from the candidate chemistry without committing the
        // authoritative world. The physical host must first accept its geometry.
        let mut projected = self.ecology.clone();
        projected
            .commit_step(&eco::CommitReceipt::accept_all(&delta))
            .map_err(|e| err(e.to_string()))?;
        match fly_senses::transduce(&self.config, &next, &projected, &packet) {
            Ok(s) => {
                next.afferents = s.afferents;
                if let Err(error) = mix_wing_afferents(&mut next.afferents, &acoustics.frame()) {
                    self.ecology.abort_step(&delta.token).map_err(|e| err(e.to_string()))?;
                    return Err(err(error));
                }
            },
            Err(e) => {
                self.ecology
                    .abort_step(&delta.token)
                    .map_err(|v| err(v.to_string()))?;
                return Err(err(e));
            }
        }
        self.pending = Some(Pending {
            state: next,
            token: delta.token.clone(),
            previous_config: None,
            acoustics,
        });
        serde_json::to_string(&delta).map_err(|e| err(e.to_string()))
    }
    pub fn commit_advance(&mut self, receipt: &str) -> Result<(), JsValue> {
        let r: eco::CommitReceipt =
            serde_json::from_str(receipt).map_err(|e| err(e.to_string()))?;
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| err("no prepared fly tick"))?;
        if r.token != pending.token {
            return Err(err("fly commit token differs"));
        }
        self.ecology
            .commit_step(&r)
            .map_err(|e| err(e.to_string()))?;
        let committed = self.pending.take().unwrap();
        self.state = committed.state;
        self.acoustics = committed.acoustics;
        Ok(())
    }
    pub fn abort_advance(&mut self, token: &str) -> Result<(), JsValue> {
        let p = self
            .pending
            .as_ref()
            .ok_or_else(|| err("no prepared fly tick"))?;
        if p.token != token {
            return Err(err("fly abort token differs"));
        }
        self.ecology
            .abort_step(token)
            .map_err(|e| err(e.to_string()))?;
        if let Some(previous)=self.pending.take().and_then(|p|p.previous_config){self.config=previous;}
        Ok(())
    }
    pub fn visitor_sound(
        &mut self,
        position: &[f64],
        frequency_hz: f64,
        envelope: f64,
        duration: f64,
    ) -> Result<(), JsValue> {
        if self.pending.is_some()
            || position.len() != 3
            || !finite(position)
            || !frequency_hz.is_finite()
            || !(40.0..=1600.0).contains(&frequency_hz)
            || !envelope.is_finite()
            || !(0.0..=1.0).contains(&envelope)
            || !duration.is_finite()
            || !(0.005..=20.0).contains(&duration)
            || self.state.emissions.len() >= 1024
        {
            return Err(err("invalid physical sound source"));
        }
        self.state.emissions.push(Emission {
            origin_mm: position.try_into().unwrap(),
            born: self.state.time,
            frequency_hz,
            envelope,
            duration,
        });
        Ok(())
    }
    pub fn rays(
        &self,
        row: usize,
        positions: &[f64],
        rotations: &[f64],
    ) -> Result<Vec<f64>, JsValue> {
        fly_optics::rays(&self.config, row, positions, rotations).map_err(err)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn retina(
        &self,
        row: usize,
        rays: &[f64],
        hits: &[i32],
        distances: &[f64],
        positions: &[f64],
        rotations: &[f64],
        sizes: &[f64],
        colors: &[f64],
        frame: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, JsValue> {
        fly_optics::retina(
            &self.config,
            row,
            rays,
            hits,
            distances,
            positions,
            rotations,
            sizes,
            colors,
            frame,
            width,
            height,
        )
        .map_err(err)
    }
    pub fn snapshot(&self) -> Result<String, JsValue> {
        if self.pending.is_some() {
            return Err(err(
                "cannot snapshot an uncommitted physical/ecological mutation",
            ));
        }
        let value = Envelope {
            format: "chreatures-fly-world-state-v4".into(),
            state: self.state.clone(),
            ecology: self
                .ecology
                .snapshot_json()
                .map_err(|e| err(e.to_string()))?,
            host: self.host.clone(),
            acoustics: self.acoustics.snapshot(),
        };
        serde_json::to_string(&value).map_err(|e| err(e.to_string()))
    }
    pub fn restore(&mut self, snapshot: &str) -> Result<(), JsValue> {
        let e: Envelope = serde_json::from_str(snapshot).map_err(|e| err(e.to_string()))?;
        if e.format != "chreatures-fly-world-state-v4"
            || e.state.format != "chreatures-fly-core-state-v4"
            || e.state.engine != self.config.engine
            || e.state.model != self.config.source_mjcf_sha256
            || e.state.atlas != self.config.atlas_sha256
            || e.state.residents.len() != self.config.bodies.len()
            || e.state.afferents.len() != self.config.bodies.len() * CHANNELS
            || !e.state.time.is_finite()
            || e.state.time < 0.0
            || e.state.afferents.iter().any(|x| !x.is_finite())
            || e.state.residents.iter().any(|r| {
                r.fatigue.len() != 92
                    || r.last_commands.len() != 92
                    || !finite(&r.fatigue)
                    || !finite(&r.last_commands)
                    || !finite(&[r.pump_phase, r.pump_flow, r.salivary_flow, r.work])
                    || r.fatigue.iter().any(|v| !(0.0..=1.0).contains(v))
            })
        {
            return Err(err("fly snapshot identity/state differs"));
        }
        let ecology =
            eco::EcologyWorld::from_snapshot_json(&e.ecology).map_err(|e| err(e.to_string()))?;
        if ecology.config_sha256() != self.ecology.config_sha256()
            || ecology.pending_delta().is_some()
            || (ecology.state().time_s - e.state.time).abs() > 1e-9
        {
            return Err(err("fly/ecology snapshot clock or configuration differs"));
        }
        if e.acoustics.resident_count != self.config.bodies.len()
            || match e.acoustics.last_sample_time_s {
                Some(time) => (time - e.state.time).abs() > 1e-9,
                None => e.state.time != 0.0,
            } {
            return Err(err("wing acoustics and physical snapshot clocks differ"));
        }
        let acoustics = fly_acoustics::FlyAcoustics::restore(self.config.acoustics, e.acoustics)
            .map_err(|e| err(e.to_string()))?;
        self.state = e.state;
        self.ecology = ecology;
        self.host = e.host;
        self.pending = None;
        self.acoustics = acoustics;
        Ok(())
    }
}
