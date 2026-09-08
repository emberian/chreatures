use crate::model::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

const EPS: f64 = 1.0e-12;
const LEDGER_TOLERANCE: f64 = 2.0e-9;
const MAX_POOLS: usize = 32;
const MAX_ELEMENTS: usize = 8;
const MAX_REACTIONS: usize = 128;
const MAX_REGIONS: usize = 512;
const MAX_ROUTES: usize = 2048;
const MAX_ORGANISMS: usize = 4096;
const MAX_PACKETS: usize = 8192;

#[derive(Debug, Clone, Default)]
struct Indices {
    regions: HashMap<String, usize>,
    routes: HashMap<String, usize>,
    organisms: HashMap<String, usize>,
    packets: HashMap<String, usize>,
    structures: HashMap<String, usize>,
}

impl Indices {
    fn build(config: &EcologyConfig, state: &WorldState) -> Self {
        Self {
            regions: config
                .regions
                .iter()
                .enumerate()
                .map(|(i, value)| (value.id.clone(), i))
                .collect(),
            routes: config
                .routes
                .iter()
                .enumerate()
                .map(|(i, value)| (value.id.clone(), i))
                .collect(),
            organisms: state
                .organisms
                .iter()
                .enumerate()
                .map(|(i, value)| (value.id.clone(), i))
                .collect(),
            packets: state
                .packets
                .iter()
                .enumerate()
                .map(|(i, value)| (value.id.clone(), i))
                .collect(),
            structures: state
                .structures
                .iter()
                .enumerate()
                .map(|(i, value)| (value.id.clone(), i))
                .collect(),
        }
    }
}

/// One closed ecological world. Calls are two-phase: `prepare_step` computes a
/// complete successor and physical delta, then `commit_step` installs it only
/// after an exact host receipt. `abort_step` discards it without changing time,
/// chemistry, private physiology, inheritance state, or RNG.
#[derive(Debug, Clone)]
pub struct EcologyWorld {
    envelope: WorldEnvelope,
    indices: Indices,
}

impl EcologyWorld {
    pub fn new(config: EcologyConfig) -> EcologyResult<Self> {
        validate_config(&config)?;
        let config_bytes = serde_json::to_vec(&config)?;
        let config_sha256 = sha256(&config_bytes);
        let organisms = config
            .organisms
            .iter()
            .enumerate()
            .map(|(index, seed)| OrganismState {
                id: seed.id.clone(),
                physics_binding: seed.physics_binding.clone(),
                anchored_region: seed.anchored_region.clone(),
                internal_volume_m3: seed.internal_volume_m3,
                capacity: seed.capacity.clone(),
                material: MaterialStore {
                    quantity: seed.initial.clone(),
                },
                atp: seed.atp,
                atp_capacity: seed.atp_capacity,
                genotype: seed.genotype.clone(),
                enzyme_activity: seed.genotype.enzyme_baseline.clone(),
                development_credit_s: 0.0,
                development_state: crate::growth::seed_state(splitmix_seed(
                    config.seed,
                    index as u64,
                )),
                reproduction_credit_s: 0.0,
                descendant_count: 0,
                maintenance_shortfall: 0.0,
                last_acclimation_atp_fraction: 0.0,
                generation: 0,
                rng: splitmix_seed(config.seed, index as u64),
            })
            .collect();
        let packets = config
            .packets
            .iter()
            .map(|seed| PacketState {
                id: seed.id.clone(),
                physics_binding: seed.physics_binding.clone(),
                volume_m3: seed.volume_m3,
                capacity: seed.capacity.clone(),
                material: MaterialStore {
                    quantity: seed.initial.clone(),
                },
                active: true,
            })
            .collect();
        let mut state = WorldState {
            time_s: 0.0,
            step_index: 0,
            regions: config
                .regions
                .iter()
                .map(|region| MaterialStore {
                    quantity: region.initial.clone(),
                })
                .collect(),
            organisms,
            packets,
            structures: Vec::new(),
            accounting: AccountingState {
                initial_elements: Vec::new(),
                last_elements: Vec::new(),
                last_residual: Vec::new(),
                maximum_absolute_residual: 0.0,
                captured_photon_energy: 0.0,
                dissipated_energy: 0.0,
                maintenance_atp_spent: 0.0,
                regulation_atp_spent: 0.0,
                physical_work_atp_spent: 0.0,
            },
        };
        let elements = element_totals(&config, &state);
        state.accounting.initial_elements = elements.clone();
        state.accounting.last_elements = elements;
        state.accounting.last_residual = vec![0.0; config.elements.len()];
        let envelope = WorldEnvelope {
            format: SNAPSHOT_FORMAT.into(),
            config_sha256,
            config,
            state,
            pending: None,
            growth: None,
        };
        let indices = Indices::build(&envelope.config, &envelope.state);
        Ok(Self { envelope, indices })
    }

    pub fn config(&self) -> &EcologyConfig {
        &self.envelope.config
    }

    pub fn config_sha256(&self) -> &str {
        &self.envelope.config_sha256
    }

    pub fn state(&self) -> &WorldState {
        &self.envelope.state
    }

    /// Project private committed physiology into the fixed BODY interoception
    /// order requested by the anatomical CNS contract. The returned values are
    /// afferent-source values for the body transducer, never a policy input.
    pub fn fly_interoception12(
        &self,
        organism_id: &str,
        spec: &FlyInteroceptionSpec,
    ) -> EcologyResult<[f64; 12]> {
        let organism = self
            .envelope
            .state
            .organisms
            .iter()
            .find(|value| value.id == organism_id)
            .ok_or_else(|| EcologyError("fly interoception organism is absent".into()))?;
        let ids = self
            .envelope
            .config
            .pools
            .iter()
            .enumerate()
            .map(|(index, value)| (value.id.as_str(), index))
            .collect::<HashMap<_, _>>();
        let resolve = |names: &[String]| -> EcologyResult<Vec<usize>> {
            names
                .iter()
                .map(|name| {
                    ids.get(name.as_str())
                        .copied()
                        .ok_or_else(|| EcologyError("fly interoception pool is absent".into()))
                })
                .collect()
        };
        let fraction = |indices: &[usize]| -> f64 {
            let amount = indices
                .iter()
                .map(|index| organism.material.quantity[*index])
                .sum::<f64>();
            let capacity = indices
                .iter()
                .map(|index| organism.capacity[*index])
                .sum::<f64>();
            if capacity > 0.0 {
                (amount / capacity).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        let gut = resolve(&spec.gut_pool_ids)?;
        let carbon = resolve(&spec.carbon_reserve_pool_ids)?;
        let nitrogen = resolve(&spec.nitrogen_reserve_pool_ids)?;
        let structural = resolve(&spec.structural_pool_ids)?;
        let osmotic = resolve(&spec.osmotic_pool_ids)?;
        let water = resolve(std::slice::from_ref(&spec.water_pool_id))?;
        let saliva = resolve(std::slice::from_ref(&spec.salivary_pool_id))?;
        let oxygen = resolve(std::slice::from_ref(&spec.oxygen_pool_id))?;
        if gut.is_empty()
            || carbon.is_empty()
            || nitrogen.is_empty()
            || structural.is_empty()
            || osmotic.is_empty()
            || !spec.osmotic_target_quantity_m3.is_finite()
            || !spec.osmotic_scale_quantity_m3.is_finite()
            || spec.osmotic_scale_quantity_m3 <= 0.0
        {
            return err("fly interoception mapping differs");
        }
        let reproductive = organism
            .genotype
            .reproduction
            .as_ref()
            .map(|program| {
                let time = (organism.reproduction_credit_s / program.interval_s).clamp(0.0, 1.0);
                let material = program
                    .material_endowment
                    .iter()
                    .zip(&organism.material.quantity)
                    .filter(|(required, _)| **required > 0.0)
                    .map(|(required, available)| (available / required).clamp(0.0, 1.0))
                    .fold(1.0, f64::min);
                let energy = if program.atp_cost + program.atp_endowment > 0.0 {
                    (organism.atp / (program.atp_cost + program.atp_endowment)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                time * material * energy
            })
            .unwrap_or(0.0);
        let osmotic_concentration = osmotic
            .iter()
            .map(|index| organism.material.quantity[*index] / organism.internal_volume_m3)
            .sum::<f64>();
        Ok([
            (organism.atp / organism.atp_capacity).clamp(0.0, 1.0),
            fraction(&gut),
            fraction(&carbon),
            fraction(&nitrogen),
            fraction(&water),
            fraction(&structural),
            fraction(&saliva),
            fraction(&oxygen),
            organism.maintenance_shortfall / (1.0 + organism.maintenance_shortfall),
            organism.last_acclimation_atp_fraction.clamp(0.0, 1.0),
            reproductive,
            ((osmotic_concentration - spec.osmotic_target_quantity_m3)
                / spec.osmotic_scale_quantity_m3)
                .tanh(),
        ])
    }

    /// Convert one host-only local sample into bounded odor and taste features
    /// for the body's chemical afferent adapter. Callers must inject the result
    /// through CNS afferents; this API grants no controller access.
    pub fn fly_chemical_projection(
        &self,
        sample: &AfferentTransductionSample,
        spec: &FlyChemicalSenseSpec,
    ) -> EcologyResult<FlyChemicalProjection> {
        let k = self.envelope.config.pools.len();
        if sample.concentration_quantity_m3.len() != k
            || spec.odor_permeability.len() != k
            || spec.half_saturation_quantity_m3.len() != k
            || !finite_nonnegative(&sample.concentration_quantity_m3)
            || !finite_nonnegative(&spec.odor_permeability)
            || spec.odor_permeability.iter().any(|value| *value > 1.0)
            || !finite_nonnegative(&spec.half_saturation_quantity_m3)
            || spec
                .half_saturation_quantity_m3
                .iter()
                .any(|value| *value <= 0.0)
        {
            return err("fly chemical afferent projection dimensions differ");
        }
        let taste = sample
            .concentration_quantity_m3
            .iter()
            .zip(&spec.half_saturation_quantity_m3)
            .map(|(concentration, half)| concentration / (half + concentration))
            .collect::<Vec<_>>();
        let odor = taste
            .iter()
            .zip(&spec.odor_permeability)
            .map(|(value, permeability)| value * permeability)
            .collect();
        Ok(FlyChemicalProjection { odor, taste })
    }

    /// Generate local candidate geometry without spending matter or advancing
    /// committed development/RNG. The host must measure final capsule clearance.
    pub fn propose_growth(&mut self, input: &GrowthInput) -> EcologyResult<GrowthProposals> {
        if self.envelope.pending.is_some() {
            return err("cannot propose growth during a prepared ecology step");
        }
        if let Some(plan) = &self.envelope.growth {
            if &plan.input == input {
                return Ok(plan.proposals.clone());
            }
            return err("a different growth proposal is already outstanding");
        }
        let plan = crate::growth::build(&self.envelope.config, &self.envelope.state, input)?;
        let output = plan.proposals.clone();
        self.envelope.growth = Some(plan);
        Ok(output)
    }

    pub fn discard_growth(&mut self, token: &str) -> EcologyResult<()> {
        if self.envelope.pending.is_some() {
            return err("abort the prepared step before discarding growth");
        }
        if self
            .envelope
            .growth
            .as_ref()
            .is_none_or(|p| p.proposals.token != token)
        {
            return err("growth discard token differs");
        }
        self.envelope.growth = None;
        Ok(())
    }

    pub fn pending_delta(&self) -> Option<&WorldDelta> {
        self.envelope.pending.as_ref().map(|value| &value.delta)
    }

    pub fn snapshot_json(&self) -> EcologyResult<String> {
        serde_json::to_string(&self.envelope).map_err(Into::into)
    }

    pub fn from_snapshot_json(snapshot: &str) -> EcologyResult<Self> {
        let envelope: WorldEnvelope = serde_json::from_str(snapshot)?;
        if envelope.format != SNAPSHOT_FORMAT {
            return err("ecology snapshot format differs");
        }
        validate_config(&envelope.config)?;
        let digest = sha256(&serde_json::to_vec(&envelope.config)?);
        if digest != envelope.config_sha256 {
            return err("ecology snapshot configuration hash differs");
        }
        validate_state(&envelope.config, &envelope.state)?;
        if let Some(pending) = &envelope.pending {
            if pending.token != pending.delta.token
                || pending.next_state.step_index != envelope.state.step_index + 1
                || pending.next_state.time_s <= envelope.state.time_s
            {
                return err("ecology pending snapshot differs");
            }
            validate_state(&envelope.config, &pending.next_state)?;
        }
        if let Some(plan) = &envelope.growth {
            if &crate::growth::build(&envelope.config, &envelope.state, &plan.input)? != plan {
                return err("ecology growth snapshot provenance or private transition differs");
            }
        }
        let indices = Indices::build(&envelope.config, &envelope.state);
        Ok(Self { envelope, indices })
    }

    pub fn validate_receipt(&self, receipt: &CommitReceipt) -> EcologyResult<()> {
        let pending = self
            .envelope
            .pending
            .as_ref()
            .ok_or_else(|| EcologyError("ecology world has no pending step".into()))?;
        if receipt.token != pending.token {
            return err("ecology commit token differs");
        }
        let expected_created = pending
            .delta
            .physical_creations
            .iter()
            .map(|value| value.proposal_id.as_str())
            .collect::<BTreeSet<_>>();
        let expected_removed = pending
            .delta
            .physical_removals
            .iter()
            .map(|value| value.proposal_id.as_str())
            .collect::<BTreeSet<_>>();
        let actual_created = receipt
            .created
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let actual_removed = receipt
            .removed
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if actual_created.len() != receipt.created.len()
            || actual_removed.len() != receipt.removed.len()
            || actual_created != expected_created
            || actual_removed != expected_removed
        {
            return err("ecology physical transaction receipt differs");
        }
        Ok(())
    }

    pub fn commit_step(&mut self, receipt: &CommitReceipt) -> EcologyResult<()> {
        self.validate_receipt(receipt)?;
        let pending = self.envelope.pending.take().unwrap();
        self.envelope.state = pending.next_state;
        self.envelope.growth = None;
        self.indices = Indices::build(&self.envelope.config, &self.envelope.state);
        Ok(())
    }

    pub fn abort_step(&mut self, token: &str) -> EcologyResult<()> {
        match &self.envelope.pending {
            Some(value) if value.token == token => {
                self.envelope.pending = None;
                Ok(())
            }
            _ => err("ecology abort token differs"),
        }
    }

    pub fn prepare_step(&mut self, input: &TickInput) -> EcologyResult<WorldDelta> {
        if self.envelope.pending.is_some() {
            return err("ecology world already has a pending step");
        }
        self.validate_input(input)?;
        crate::growth::validate_selection(self.envelope.growth.as_ref(), input)?;
        let mut digest = Sha256::new();
        digest.update(self.envelope.config_sha256.as_bytes());
        digest.update(self.envelope.state.step_index.to_le_bytes());
        digest.update(self.envelope.state.time_s.to_bits().to_le_bytes());
        digest.update(serde_json::to_vec(input)?);
        let token = format!("{:x}", digest.finalize());
        let before = element_totals(&self.envelope.config, &self.envelope.state);
        let mut next = self.envelope.state.clone();
        let mut transfers = Vec::new();
        let mut reactions = Vec::new();
        let mut physical_creations = Vec::new();
        let mut physical_removals = Vec::new();
        let mut blocked_events = Vec::new();

        self.advance_routes(input, &mut next, &mut transfers)?;
        self.advance_contacts(input, &mut next, &mut transfers)?;
        self.advance_metabolism(input, &mut next, &mut reactions)?;
        self.advance_physical_work(input, &mut next)?;
        self.advance_structure_decay(
            input.dt_s,
            &mut next,
            &mut reactions,
            &mut physical_removals,
        )?;
        self.advance_packet_retirements(
            input,
            &mut next,
            &mut transfers,
            &mut physical_removals,
            &mut blocked_events,
        )?;
        self.advance_development(
            input,
            &mut next,
            &mut transfers,
            &mut physical_creations,
            &mut blocked_events,
        )?;
        self.advance_reproduction(
            input,
            &mut next,
            &mut transfers,
            &mut physical_creations,
            &mut blocked_events,
        )?;

        if let Some(plan) = &self.envelope.growth {
            crate::growth::install(plan, &mut next, &physical_creations);
        }
        next.time_s += input.dt_s;
        next.step_index = next.step_index.saturating_add(1);
        let after = element_totals(&self.envelope.config, &next);
        let residual = after
            .iter()
            .zip(&before)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let maximum = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        let scale = before.iter().map(|value| value.abs()).fold(1.0, f64::max);
        if maximum > LEDGER_TOLERANCE * scale {
            return err("ecology step violated elemental conservation");
        }
        next.accounting.last_elements = after.clone();
        next.accounting.last_residual = residual.clone();
        next.accounting.maximum_absolute_residual =
            next.accounting.maximum_absolute_residual.max(maximum);
        let afferent_transduction_samples = self.sample_afferent_medium(input, &next)?;
        let delta = WorldDelta {
            format: DELTA_FORMAT.into(),
            token: token.clone(),
            step_index: next.step_index,
            time_start_s: self.envelope.state.time_s,
            time_end_s: next.time_s,
            transfers,
            reactions,
            physical_creations,
            physical_removals,
            afferent_transduction_samples,
            blocked_events,
            elemental_before: before,
            elemental_after: after,
            elemental_residual: residual,
        };
        self.envelope.pending = Some(PendingStep {
            token,
            next_state: next,
            delta: delta.clone(),
        });
        Ok(delta)
    }

    fn validate_input(&self, input: &TickInput) -> EcologyResult<()> {
        let k = self.envelope.config.pools.len();
        let route_count = self.envelope.config.routes.len();
        if !input.dt_s.is_finite()
            || input.dt_s <= 0.0
            || input.dt_s > 10.0
            || input.route_open_fraction.len() != route_count
            || input.route_advection_m3_s.len() != route_count
            || !finite(&input.route_open_fraction)
            || !finite(&input.route_advection_m3_s)
            || input
                .route_open_fraction
                .iter()
                .any(|value| !(0.0..=1.0).contains(value))
        {
            return err("ecology tick transport input differs");
        }
        for (flow, route) in input
            .route_advection_m3_s
            .iter()
            .zip(&self.envelope.config.routes)
        {
            if flow.abs() > route.hydraulic_capacity_m3_s + EPS {
                return err("ecology route flow exceeds physical capacity");
            }
        }
        let mut contact_pairs = HashSet::new();
        for contact in &input.contacts {
            self.store_view(&self.envelope.state, &contact.a)?;
            self.store_view(&self.envelope.state, &contact.b)?;
            let pair = if contact.a <= contact.b {
                (contact.a.clone(), contact.b.clone())
            } else {
                (contact.b.clone(), contact.a.clone())
            };
            if contact.a == contact.b
                || !contact_pairs.insert(pair)
                || contact.conductance_m3_s.len() != k
                || !finite_nonnegative(&contact.conductance_m3_s)
                || contact.conductance_m3_s.iter().any(|value| *value > 1.0)
            {
                return err("ecology measured-contact input differs");
            }
        }
        let mut directed_ids = HashSet::new();
        for exchange in &input.directed_exchanges {
            self.store_view(&self.envelope.state, &exchange.from)?;
            self.store_view(&self.envelope.state, &exchange.to)?;
            if !valid_id(&exchange.event_id)
                || !directed_ids.insert(exchange.event_id.as_str())
                || exchange.from == exchange.to
                || exchange.maximum_quantity.len() != k
                || !finite_nonnegative(&exchange.maximum_quantity)
            {
                return err("ecology directional measured exchange differs");
            }
        }
        let mut photon_ids = HashSet::new();
        for exposure in &input.photon_exposures {
            if !self.indices.organisms.contains_key(&exposure.organism_id)
                || !photon_ids.insert(exposure.organism_id.as_str())
                || !exposure.available_energy_per_s.is_finite()
                || exposure.available_energy_per_s < 0.0
            {
                return err("ecology photon exposure differs");
            }
        }
        let mut work_ids = HashSet::new();
        for demand in &input.metabolic_work_demands {
            if !self.indices.organisms.contains_key(&demand.organism_id)
                || !work_ids.insert(demand.organism_id.as_str())
                || !demand.atp_quantity.is_finite()
                || demand.atp_quantity < 0.0
            {
                return err("ecology measured physical-work demand differs");
            }
        }
        let mut event_ids = HashSet::new();
        let mut new_bindings = HashSet::new();
        for site in &input.construction_sites {
            if !valid_id(&site.site_id)
                || !event_ids.insert(site.site_id.as_str())
                || !self.indices.organisms.contains_key(&site.organism_id)
                || !self.indices.regions.contains_key(&site.region_id)
                || site
                    .route_hint
                    .as_ref()
                    .is_some_and(|value| !self.indices.routes.contains_key(value))
                || !valid_id(&site.host_template_id)
                || !valid_id(&site.physics_binding)
                || self.binding_exists(&site.physics_binding)
                || !new_bindings.insert(site.physics_binding.as_str())
                || !inside_world(&self.envelope.config, &site.position_m)
                || !valid_orientation(&site.orientation_xyzw)
            {
                return err("ecology construction site differs");
            }
        }
        let mut child_ids = HashSet::new();
        for site in &input.birth_sites {
            if !valid_id(&site.site_id)
                || !event_ids.insert(site.site_id.as_str())
                || !self.indices.organisms.contains_key(&site.parent_id)
                || !valid_id(&site.child_id)
                || self.indices.organisms.contains_key(&site.child_id)
                || !child_ids.insert(site.child_id.as_str())
                || !valid_id(&site.host_template_id)
                || !valid_id(&site.physics_binding)
                || self.binding_exists(&site.physics_binding)
                || !new_bindings.insert(site.physics_binding.as_str())
                || site
                    .anchored_region
                    .as_ref()
                    .is_some_and(|value| !self.indices.regions.contains_key(value))
                || !inside_world(&self.envelope.config, &site.position_m)
                || !valid_orientation(&site.orientation_xyzw)
                || !site.internal_volume_m3.is_finite()
                || site.internal_volume_m3 <= 0.0
                || site.capacity.len() != k
                || !finite_nonnegative(&site.capacity)
                || site.capacity.iter().any(|value| *value <= 0.0)
                || !site.atp_capacity.is_finite()
                || site.atp_capacity <= 0.0
                || !finite_nonnegative(&[site.nominal_radius_m, site.nominal_length_m])
                || site.nominal_radius_m <= 0.0
                || site.nominal_length_m <= 0.0
            {
                return err("ecology birth site differs");
            }
        }
        let mut retired = HashSet::new();
        for retirement in &input.packet_retirements {
            if !self.indices.packets.contains_key(&retirement.packet_id)
                || !self
                    .indices
                    .regions
                    .contains_key(&retirement.receiver_region_id)
                || !retired.insert(retirement.packet_id.as_str())
                || retirement.cause.is_empty()
                || retirement.cause.len() > 96
            {
                return err("ecology packet retirement differs");
            }
        }
        let mut sample_ids = HashSet::new();
        for sample in &input.afferent_sample_sites {
            if !valid_id(&sample.site_id)
                || !sample_ids.insert(sample.site_id.as_str())
                || !self.indices.regions.contains_key(&sample.region_id)
            {
                return err("ecology afferent sample site differs");
            }
        }
        Ok(())
    }

    fn binding_exists(&self, binding: &str) -> bool {
        self.envelope
            .state
            .organisms
            .iter()
            .any(|value| value.physics_binding == binding)
            || self
                .envelope
                .state
                .packets
                .iter()
                .any(|value| value.physics_binding == binding)
            || self
                .envelope
                .state
                .structures
                .iter()
                .any(|value| value.physics_binding == binding)
    }

    fn store_view(&self, state: &WorldState, id: &StoreId) -> EcologyResult<StoreView> {
        match id {
            StoreId::Region(name) => {
                let &index = self
                    .indices
                    .regions
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology region store is absent".into()))?;
                Ok(StoreView {
                    quantity: state.regions[index].quantity.clone(),
                    capacity: self.envelope.config.regions[index].capacity.clone(),
                    volume_m3: self.envelope.config.regions[index].volume_m3,
                    permeability: vec![1.0; self.envelope.config.pools.len()],
                })
            }
            StoreId::Organism(name) => {
                let index = *self
                    .indices
                    .organisms
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology organism store is absent".into()))?;
                let value = &state.organisms[index];
                Ok(StoreView {
                    quantity: value.material.quantity.clone(),
                    capacity: value.capacity.clone(),
                    volume_m3: value.internal_volume_m3,
                    permeability: value.genotype.membrane_permeability.clone(),
                })
            }
            StoreId::Packet(name) => {
                let index = *self
                    .indices
                    .packets
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology packet store is absent".into()))?;
                let value = &state.packets[index];
                if !value.active {
                    return err("ecology packet store is inactive");
                }
                Ok(StoreView {
                    quantity: value.material.quantity.clone(),
                    capacity: value.capacity.clone(),
                    volume_m3: value.volume_m3,
                    permeability: vec![1.0; self.envelope.config.pools.len()],
                })
            }
            StoreId::Structure(name) => {
                let index = *self
                    .indices
                    .structures
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology structure store is absent".into()))?;
                let value = &state.structures[index];
                if !value.active {
                    return err("ecology structure store is inactive");
                }
                Ok(StoreView {
                    quantity: value.material.quantity.clone(),
                    capacity: value.initial_material.clone(),
                    volume_m3: value.nominal_volume_m3,
                    permeability: vec![1.0; self.envelope.config.pools.len()],
                })
            }
        }
    }

    fn adjust_store(
        &self,
        state: &mut WorldState,
        id: &StoreId,
        change: &[f64],
    ) -> EcologyResult<()> {
        let quantities = match id {
            StoreId::Region(name) => {
                let index = *self
                    .indices
                    .regions
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology region store is absent".into()))?;
                &mut state.regions[index].quantity
            }
            StoreId::Organism(name) => {
                &mut state.organisms[*self
                    .indices
                    .organisms
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology organism store is absent".into()))?]
                .material
                .quantity
            }
            StoreId::Packet(name) => {
                &mut state.packets[*self
                    .indices
                    .packets
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology packet store is absent".into()))?]
                .material
                .quantity
            }
            StoreId::Structure(name) => {
                &mut state.structures[*self
                    .indices
                    .structures
                    .get(name)
                    .ok_or_else(|| EcologyError("ecology structure store is absent".into()))?]
                .material
                .quantity
            }
        };
        for (quantity, delta) in quantities.iter_mut().zip(change) {
            *quantity += delta;
            if quantity.abs() < EPS {
                *quantity = 0.0;
            }
            if *quantity < -EPS || !quantity.is_finite() {
                return err("ecology material adjustment became invalid");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct StoreView {
    quantity: Vec<f64>,
    capacity: Vec<f64>,
    volume_m3: f64,
    permeability: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ProposedTransfer {
    cause: TransferCause,
    detail_id: String,
    from: StoreId,
    to: StoreId,
    quantity: Vec<f64>,
}

impl EcologyWorld {
    fn advance_routes(
        &self,
        input: &TickInput,
        state: &mut WorldState,
        records: &mut Vec<TransferRecord>,
    ) -> EcologyResult<()> {
        let k = self.envelope.config.pools.len();
        let mut proposals = Vec::new();
        for (edge, route) in self.envelope.config.routes.iter().enumerate() {
            let a = self.indices.regions[&route.a];
            let b = self.indices.regions[&route.b];
            let openness = route.base_open_fraction * input.route_open_fraction[edge];
            let flow = input.route_advection_m3_s[edge] * openness;
            let mut a_to_b = vec![0.0; k];
            let mut b_to_a = vec![0.0; k];
            for pool in 0..k {
                let ca =
                    state.regions[a].quantity[pool] / self.envelope.config.regions[a].volume_m3;
                let cb =
                    state.regions[b].quantity[pool] / self.envelope.config.regions[b].volume_m3;
                let diffusive = self.envelope.config.pools[pool].diffusivity_m2_s
                    * route.cross_section_m2
                    / route.length_m
                    * openness
                    * (ca - cb);
                let advective = if flow >= 0.0 { flow * ca } else { flow * cb };
                let net = (diffusive + advective) * input.dt_s;
                if net >= 0.0 {
                    a_to_b[pool] = net;
                } else {
                    b_to_a[pool] = -net;
                }
            }
            if a_to_b.iter().any(|value| *value > 0.0) {
                proposals.push(ProposedTransfer {
                    cause: TransferCause::RouteDiffusionAdvection,
                    detail_id: route.id.clone(),
                    from: StoreId::Region(route.a.clone()),
                    to: StoreId::Region(route.b.clone()),
                    quantity: a_to_b,
                });
            }
            if b_to_a.iter().any(|value| *value > 0.0) {
                proposals.push(ProposedTransfer {
                    cause: TransferCause::RouteDiffusionAdvection,
                    detail_id: route.id.clone(),
                    from: StoreId::Region(route.b.clone()),
                    to: StoreId::Region(route.a.clone()),
                    quantity: b_to_a,
                });
            }
        }
        self.apply_simultaneous_transfers(state, proposals, records)
    }

    fn advance_contacts(
        &self,
        input: &TickInput,
        state: &mut WorldState,
        records: &mut Vec<TransferRecord>,
    ) -> EcologyResult<()> {
        let k = self.envelope.config.pools.len();
        let mut proposals = Vec::new();
        for (index, contact) in input.contacts.iter().enumerate() {
            let a = self.store_view(state, &contact.a)?;
            let b = self.store_view(state, &contact.b)?;
            let mut a_to_b = vec![0.0; k];
            let mut b_to_a = vec![0.0; k];
            for pool in 0..k {
                let ca = a.quantity[pool] / a.volume_m3;
                let cb = b.quantity[pool] / b.volume_m3;
                let conductance =
                    contact.conductance_m3_s[pool] * a.permeability[pool] * b.permeability[pool];
                let net = conductance * (ca - cb) * input.dt_s;
                if net >= 0.0 {
                    a_to_b[pool] = net;
                } else {
                    b_to_a[pool] = -net;
                }
            }
            let detail = format!("contact-{index:04}");
            if a_to_b.iter().any(|value| *value > 0.0) {
                proposals.push(ProposedTransfer {
                    cause: TransferCause::MeasuredContact,
                    detail_id: detail.clone(),
                    from: contact.a.clone(),
                    to: contact.b.clone(),
                    quantity: a_to_b,
                });
            }
            if b_to_a.iter().any(|value| *value > 0.0) {
                proposals.push(ProposedTransfer {
                    cause: TransferCause::MeasuredContact,
                    detail_id: detail,
                    from: contact.b.clone(),
                    to: contact.a.clone(),
                    quantity: b_to_a,
                });
            }
        }
        for exchange in &input.directed_exchanges {
            if exchange.maximum_quantity.iter().any(|value| *value > 0.0) {
                proposals.push(ProposedTransfer {
                    cause: TransferCause::MeasuredDirectedExchange,
                    detail_id: exchange.event_id.clone(),
                    from: exchange.from.clone(),
                    to: exchange.to.clone(),
                    quantity: exchange.maximum_quantity.clone(),
                });
            }
        }
        self.apply_simultaneous_transfers(state, proposals, records)
    }

    fn apply_simultaneous_transfers(
        &self,
        state: &mut WorldState,
        mut proposals: Vec<ProposedTransfer>,
        records: &mut Vec<TransferRecord>,
    ) -> EcologyResult<()> {
        let k = self.envelope.config.pools.len();
        let mut outgoing: BTreeMap<StoreId, Vec<f64>> = BTreeMap::new();
        let mut incoming: BTreeMap<StoreId, Vec<f64>> = BTreeMap::new();
        for proposal in &proposals {
            if !finite_nonnegative(&proposal.quantity) {
                return err("ecology transfer proposal overflowed");
            }
            let donor = outgoing
                .entry(proposal.from.clone())
                .or_insert_with(|| vec![0.0; k]);
            let receiver = incoming
                .entry(proposal.to.clone())
                .or_insert_with(|| vec![0.0; k]);
            for pool in 0..k {
                donor[pool] += proposal.quantity[pool];
                receiver[pool] += proposal.quantity[pool];
            }
        }
        let mut donor_factor: BTreeMap<StoreId, Vec<f64>> = BTreeMap::new();
        for (id, demand) in &outgoing {
            let view = self.store_view(state, id)?;
            donor_factor.insert(
                id.clone(),
                (0..k)
                    .map(|pool| {
                        if demand[pool] > 0.0 {
                            (view.quantity[pool] / demand[pool]).clamp(0.0, 1.0)
                        } else {
                            1.0
                        }
                    })
                    .collect(),
            );
        }
        let mut receiver_factor: BTreeMap<StoreId, Vec<f64>> = BTreeMap::new();
        for (id, demand) in &incoming {
            let view = self.store_view(state, id)?;
            receiver_factor.insert(
                id.clone(),
                (0..k)
                    .map(|pool| {
                        if demand[pool] > 0.0 {
                            ((view.capacity[pool] - view.quantity[pool]) / demand[pool])
                                .clamp(0.0, 1.0)
                        } else {
                            1.0
                        }
                    })
                    .collect(),
            );
        }
        for proposal in &mut proposals {
            for pool in 0..k {
                proposal.quantity[pool] *=
                    donor_factor[&proposal.from][pool] * receiver_factor[&proposal.to][pool];
            }
            if proposal.quantity.iter().all(|value| *value <= EPS) {
                continue;
            }
            let debit = proposal
                .quantity
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>();
            self.adjust_store(state, &proposal.from, &debit)?;
            self.adjust_store(state, &proposal.to, &proposal.quantity)?;
            records.push(TransferRecord {
                cause: proposal.cause.clone(),
                detail_id: proposal.detail_id.clone(),
                from: proposal.from.clone(),
                to: proposal.to.clone(),
                quantity: proposal.quantity.clone(),
            });
        }
        Ok(())
    }

    fn advance_metabolism(
        &self,
        input: &TickInput,
        state: &mut WorldState,
        records: &mut Vec<ReactionRecord>,
    ) -> EcologyResult<()> {
        let k = self.envelope.config.pools.len();
        let r = self.envelope.config.reactions.len();
        let photons = input
            .photon_exposures
            .iter()
            .map(|value| {
                (
                    value.organism_id.as_str(),
                    value.available_energy_per_s * input.dt_s,
                )
            })
            .collect::<HashMap<_, _>>();
        for organism in &mut state.organisms {
            let genotype = &organism.genotype;
            let atp_fraction = (organism.atp / organism.atp_capacity).clamp(0.0, 1.0);
            let mut target = vec![0.0; r];
            for (reaction_index, reaction) in self.envelope.config.reactions.iter().enumerate() {
                let mut availability = 1.0_f64;
                for pool in 0..k {
                    if reaction.stoichiometry[pool] < 0.0 {
                        let quantity = organism.material.quantity[pool];
                        let half = reaction.half_saturation_quantity[pool];
                        availability = availability.min(if half + quantity > 0.0 {
                            quantity / (half + quantity)
                        } else {
                            0.0
                        });
                    }
                }
                target[reaction_index] = (genotype.enzyme_baseline[reaction_index]
                    + genotype.enzyme_substrate_response[reaction_index] * (availability - 0.5)
                    + genotype.enzyme_atp_response[reaction_index] * (atp_fraction - 0.5))
                    .clamp(0.0, genotype.enzyme_maximum);
            }
            let target_sum = target.iter().sum::<f64>();
            if target_sum > genotype.enzyme_total_budget {
                let factor = genotype.enzyme_total_budget / target_sum;
                target.iter_mut().for_each(|value| *value *= factor);
            }
            let alpha = -(-input.dt_s / genotype.enzyme_time_constant_s).exp_m1();
            let requested_regulation = target
                .iter()
                .zip(&organism.enzyme_activity)
                .map(|(goal, current)| (alpha * (goal - current)).abs())
                .sum::<f64>()
                * genotype.enzyme_change_atp_cost;
            let regulation_factor = if requested_regulation > 0.0 {
                (organism.atp / requested_regulation).min(1.0)
            } else {
                1.0
            };
            for (reaction_index, activity) in organism.enzyme_activity.iter_mut().enumerate() {
                *activity += regulation_factor * alpha * (target[reaction_index] - *activity);
            }
            let regulation_paid = requested_regulation * regulation_factor;
            organism.atp -= regulation_paid;
            organism.last_acclimation_atp_fraction =
                (regulation_paid / organism.atp_capacity).clamp(0.0, 1.0);
            state.accounting.regulation_atp_spent += regulation_paid;
            state.accounting.dissipated_energy += regulation_paid;

            // Reactions compete against one shared pre-reaction state. Scarcity
            // in a growth product does not suppress an unrelated respiration
            // reaction, and no sequential reaction can spend a newly produced
            // pool during the same step.
            let mut raw = vec![0.0; r];
            let mut pool_demand = vec![0.0; k];
            let mut pool_production = vec![0.0; k];
            let mut atp_demand = 0.0;
            let mut photon_demand = 0.0;
            for (reaction_index, reaction) in self.envelope.config.reactions.iter().enumerate() {
                let mut saturation = 1.0_f64;
                for pool in 0..k {
                    if reaction.stoichiometry[pool] < 0.0 {
                        let quantity = organism.material.quantity[pool];
                        let half = reaction.half_saturation_quantity[pool];
                        saturation = saturation.min(if half + quantity > 0.0 {
                            quantity / (half + quantity)
                        } else {
                            0.0
                        });
                    }
                }
                raw[reaction_index] = reaction.base_rate_quantity_s
                    * organism.enzyme_activity[reaction_index]
                    * saturation
                    * input.dt_s;
                for pool in 0..k {
                    let amount = reaction.stoichiometry[pool] * raw[reaction_index];
                    if amount < 0.0 {
                        pool_demand[pool] -= amount;
                    } else {
                        pool_production[pool] += amount;
                    }
                }
                atp_demand += reaction.atp_cost_per_flux * raw[reaction_index];
                photon_demand += reaction.photon_cost_per_flux * raw[reaction_index];
            }
            let pool_factor = (0..k)
                .map(|pool| {
                    if pool_demand[pool] > 0.0 {
                        (organism.material.quantity[pool] / pool_demand[pool]).min(1.0)
                    } else {
                        1.0
                    }
                })
                .collect::<Vec<_>>();
            let product_factor = (0..k)
                .map(|pool| {
                    if pool_production[pool] > 0.0 {
                        ((organism.capacity[pool] - organism.material.quantity[pool])
                            / pool_production[pool])
                            .clamp(0.0, 1.0)
                    } else {
                        1.0
                    }
                })
                .collect::<Vec<_>>();
            let atp_factor = if atp_demand > 0.0 {
                (organism.atp / atp_demand).min(1.0)
            } else {
                1.0
            };
            let available_photons = photons.get(organism.id.as_str()).copied().unwrap_or(0.0);
            let photon_factor = if photon_demand > 0.0 {
                (available_photons / photon_demand).min(1.0)
            } else {
                1.0
            };
            for (reaction_index, reaction) in self.envelope.config.reactions.iter().enumerate() {
                let mut factor = atp_factor.min(photon_factor);
                for pool in 0..k {
                    if reaction.stoichiometry[pool] < 0.0 {
                        factor = factor.min(pool_factor[pool]);
                    } else if reaction.stoichiometry[pool] > 0.0 {
                        factor = factor.min(product_factor[pool]);
                    }
                }
                let flux = raw[reaction_index] * factor;
                if flux <= EPS {
                    continue;
                }
                for pool in 0..k {
                    organism.material.quantity[pool] += reaction.stoichiometry[pool] * flux;
                    if organism.material.quantity[pool].abs() < EPS {
                        organism.material.quantity[pool] = 0.0;
                    }
                }
                let atp_before = organism.atp;
                organism.atp += (reaction.atp_yield_per_flux - reaction.atp_cost_per_flux) * flux;
                let mut capacity_loss = 0.0;
                if organism.atp > organism.atp_capacity {
                    capacity_loss = organism.atp - organism.atp_capacity;
                    organism.atp = organism.atp_capacity;
                }
                if organism.atp < -EPS {
                    return err("ecology metabolism overspent ATP");
                }
                organism.atp = organism.atp.max(0.0);
                let photon_used = reaction.photon_cost_per_flux * flux;
                state.accounting.captured_photon_energy += photon_used;
                let chemical_delta = reaction
                    .stoichiometry
                    .iter()
                    .zip(&self.envelope.config.pools)
                    .map(|(coefficient, pool)| coefficient * pool.chemical_energy_per_quantity)
                    .sum::<f64>()
                    * flux;
                let atp_delta = organism.atp - atp_before;
                state.accounting.dissipated_energy +=
                    (photon_used - chemical_delta - atp_delta - capacity_loss).max(0.0)
                        + capacity_loss;
                records.push(ReactionRecord {
                    organism_id: organism.id.clone(),
                    reaction_id: reaction.id.clone(),
                    flux,
                });
            }
            let maintenance_needed = genotype.maintenance_atp_s * input.dt_s;
            let maintenance_paid = organism.atp.min(maintenance_needed);
            organism.atp -= maintenance_paid;
            state.accounting.maintenance_atp_spent += maintenance_paid;
            state.accounting.dissipated_energy += maintenance_paid;
            if maintenance_needed > 0.0 {
                let shortfall = (maintenance_needed - maintenance_paid) / maintenance_needed;
                organism.maintenance_shortfall =
                    (organism.maintenance_shortfall + shortfall - 0.1 * input.dt_s).max(0.0);
            }
        }
        Ok(())
    }

    fn advance_physical_work(
        &self,
        input: &TickInput,
        state: &mut WorldState,
    ) -> EcologyResult<()> {
        for demand in &input.metabolic_work_demands {
            let organism = state
                .organisms
                .iter_mut()
                .find(|value| value.id == demand.organism_id)
                .ok_or_else(|| EcologyError("physical-work organism is absent".into()))?;
            let paid = organism.atp.min(demand.atp_quantity);
            let shortfall = demand.atp_quantity - paid;
            organism.atp -= paid;
            organism.maintenance_shortfall += shortfall / organism.atp_capacity;
            state.accounting.physical_work_atp_spent += paid;
            state.accounting.dissipated_energy += paid;
        }
        Ok(())
    }

    fn advance_structure_decay(
        &self,
        tick_s: f64,
        state: &mut WorldState,
        records: &mut Vec<ReactionRecord>,
        removals: &mut Vec<PhysicalRemoval>,
    ) -> EcologyResult<()> {
        let structure_count = state.structures.len();
        for index in 0..structure_count {
            if !state.structures[index].active {
                continue;
            }
            let structure = state.structures[index].clone();
            let region_index = *self
                .indices
                .regions
                .get(&structure.region_id)
                .ok_or_else(|| EcologyError("ecology structure region is absent".into()))?;
            let mut fraction = structure.remaining_fraction
                * -(-tick_s / structure.decay_time_constant_s).exp_m1();
            if structure.remaining_fraction < 1.0e-7 {
                fraction = structure.remaining_fraction;
            }
            for pool in 0..self.envelope.config.pools.len() {
                let returned = structure.decay_return[pool];
                if returned > 0.0 {
                    let room = self.envelope.config.regions[region_index].capacity[pool]
                        - state.regions[region_index].quantity[pool];
                    fraction = fraction.min((room / returned).max(0.0));
                }
                let consumed = structure.initial_material[pool];
                if consumed > 0.0 {
                    fraction = fraction
                        .min((state.structures[index].material.quantity[pool] / consumed).max(0.0));
                }
            }
            if fraction <= EPS {
                continue;
            }
            for pool in 0..self.envelope.config.pools.len() {
                state.structures[index].material.quantity[pool] -=
                    structure.initial_material[pool] * fraction;
                state.regions[region_index].quantity[pool] +=
                    structure.decay_return[pool] * fraction;
                if state.structures[index].material.quantity[pool].abs() < EPS {
                    state.structures[index].material.quantity[pool] = 0.0;
                }
            }
            state.structures[index].remaining_fraction =
                (structure.remaining_fraction - fraction).max(0.0);
            records.push(ReactionRecord {
                organism_id: structure.owner_id.clone(),
                reaction_id: "construction-decay".into(),
                flux: fraction,
            });
            if state.structures[index].remaining_fraction == 0.0 {
                state.structures[index].active = false;
                removals.push(PhysicalRemoval {
                    proposal_id: format!(
                        "remove-{}-{}",
                        structure.id,
                        state.step_index.saturating_add(1)
                    ),
                    physics_binding: structure.physics_binding.clone(),
                    material_store: StoreId::Structure(structure.id),
                    cause: "resource-returned-decay".into(),
                });
            }
        }
        Ok(())
    }

    fn advance_packet_retirements(
        &self,
        input: &TickInput,
        state: &mut WorldState,
        transfers: &mut Vec<TransferRecord>,
        removals: &mut Vec<PhysicalRemoval>,
        blocked: &mut Vec<BlockedEvent>,
    ) -> EcologyResult<()> {
        for retirement in &input.packet_retirements {
            let packet_index = state
                .packets
                .iter()
                .position(|value| value.id == retirement.packet_id && value.active)
                .ok_or_else(|| EcologyError("retired ecology packet is inactive".into()))?;
            let region_index = self.indices.regions[&retirement.receiver_region_id];
            let amount = state.packets[packet_index].material.quantity.clone();
            let fits = amount.iter().enumerate().all(|(pool, value)| {
                state.regions[region_index].quantity[pool] + value
                    <= self.envelope.config.regions[region_index].capacity[pool] + EPS
            });
            if !fits {
                blocked.push(BlockedEvent {
                    event_id: retirement.packet_id.clone(),
                    reason: "receiver-region-capacity".into(),
                });
                continue;
            }
            for (pool, value) in amount.iter().enumerate() {
                state.regions[region_index].quantity[pool] += value;
                state.packets[packet_index].material.quantity[pool] = 0.0;
            }
            state.packets[packet_index].active = false;
            transfers.push(TransferRecord {
                cause: TransferCause::PacketExit,
                detail_id: retirement.cause.clone(),
                from: StoreId::Packet(retirement.packet_id.clone()),
                to: StoreId::Region(retirement.receiver_region_id.clone()),
                quantity: amount,
            });
            removals.push(PhysicalRemoval {
                proposal_id: format!(
                    "remove-packet-{}-{}",
                    retirement.packet_id,
                    state.step_index.saturating_add(1)
                ),
                physics_binding: state.packets[packet_index].physics_binding.clone(),
                material_store: StoreId::Packet(retirement.packet_id.clone()),
                cause: retirement.cause.clone(),
            });
        }
        Ok(())
    }

    fn advance_development(
        &self,
        input: &TickInput,
        state: &mut WorldState,
        transfers: &mut Vec<TransferRecord>,
        creations: &mut Vec<PhysicalCreation>,
        blocked: &mut Vec<BlockedEvent>,
    ) -> EcologyResult<()> {
        for organism in &mut state.organisms {
            if let Some(program) = &organism.genotype.development {
                organism.development_credit_s =
                    (organism.development_credit_s + input.dt_s).min(program.interval_s);
            }
        }
        let mut sites = input.construction_sites.iter().collect::<Vec<_>>();
        sites.sort_by(|a, b| a.site_id.cmp(&b.site_id));
        for site in sites {
            let organism_index = state
                .organisms
                .iter()
                .position(|value| value.id == site.organism_id)
                .unwrap();
            let Some(program) = state.organisms[organism_index].genotype.development.clone() else {
                continue;
            };
            let owned = state
                .structures
                .iter()
                .filter(|value| value.active && value.owner_id == site.organism_id)
                .count();
            if state.organisms[organism_index].development_credit_s + EPS < program.interval_s
                || owned >= program.maximum_structures
            {
                continue;
            }
            let funded = program
                .material_cost
                .iter()
                .zip(&state.organisms[organism_index].material.quantity)
                .all(|(cost, available)| cost <= &(available + EPS))
                && program.atp_cost <= state.organisms[organism_index].atp + EPS;
            if !funded {
                blocked.push(BlockedEvent {
                    event_id: site.site_id.clone(),
                    reason: "development-resource-or-atp-shortfall".into(),
                });
                continue;
            }
            let structure_id = format!(
                "structure-{}-{}-{}",
                site.organism_id,
                state.step_index.saturating_add(1),
                owned
            );
            for pool in 0..program.material_cost.len() {
                state.organisms[organism_index].material.quantity[pool] -=
                    program.material_cost[pool];
            }
            state.organisms[organism_index].atp -= program.atp_cost;
            state.organisms[organism_index].development_credit_s -= program.interval_s;
            state.structures.push(StructureState {
                id: structure_id.clone(),
                physics_binding: site.physics_binding.clone(),
                owner_id: site.organism_id.clone(),
                region_id: site.region_id.clone(),
                route_hint: site.route_hint.clone(),
                host_template_id: site.host_template_id.clone(),
                position_m: site.position_m,
                orientation_xyzw: site.orientation_xyzw,
                nominal_radius_m: program.nominal_radius_m,
                nominal_length_m: program.nominal_length_m,
                nominal_volume_m3: program.nominal_volume_m3,
                initial_material: program.material_cost.clone(),
                decay_return: program.decay_return.clone(),
                material: MaterialStore {
                    quantity: program.material_cost.clone(),
                },
                remaining_fraction: 1.0,
                decay_time_constant_s: program.decay_time_constant_s,
                active: true,
            });
            transfers.push(TransferRecord {
                cause: TransferCause::ConstructionAllocation,
                detail_id: site.site_id.clone(),
                from: StoreId::Organism(site.organism_id.clone()),
                to: StoreId::Structure(structure_id.clone()),
                quantity: program.material_cost,
            });
            creations.push(PhysicalCreation {
                proposal_id: format!("create-{structure_id}"),
                kind: PhysicalCreationKind::ConstructedGeometry,
                physics_binding: site.physics_binding.clone(),
                host_template_id: site.host_template_id.clone(),
                position_m: site.position_m,
                orientation_xyzw: site.orientation_xyzw,
                nominal_radius_m: program.nominal_radius_m,
                nominal_length_m: program.nominal_length_m,
                material_store: StoreId::Structure(structure_id),
                route_hint: site.route_hint.clone(),
            });
        }
        Ok(())
    }

    fn advance_reproduction(
        &self,
        input: &TickInput,
        state: &mut WorldState,
        transfers: &mut Vec<TransferRecord>,
        creations: &mut Vec<PhysicalCreation>,
        blocked: &mut Vec<BlockedEvent>,
    ) -> EcologyResult<()> {
        for organism in &mut state.organisms {
            if let Some(program) = &organism.genotype.reproduction {
                organism.reproduction_credit_s =
                    (organism.reproduction_credit_s + input.dt_s).min(program.interval_s);
            }
        }
        let initial_count = state.organisms.len();
        let mut sites = input.birth_sites.iter().collect::<Vec<_>>();
        sites.sort_by(|a, b| a.site_id.cmp(&b.site_id));
        for site in sites {
            let Some(parent_index) = state.organisms[..initial_count]
                .iter()
                .position(|value| value.id == site.parent_id)
            else {
                continue;
            };
            let Some(program) = state.organisms[parent_index].genotype.reproduction.clone() else {
                continue;
            };
            if state.organisms[parent_index].reproduction_credit_s + EPS < program.interval_s
                || state.organisms[parent_index].descendant_count >= program.maximum_descendants
            {
                continue;
            }
            let funded = program
                .material_endowment
                .iter()
                .zip(&state.organisms[parent_index].material.quantity)
                .all(|(cost, available)| cost <= &(available + EPS))
                && program.atp_cost + program.atp_endowment
                    <= state.organisms[parent_index].atp + EPS
                && program.atp_endowment <= site.atp_capacity + EPS
                && program
                    .material_endowment
                    .iter()
                    .zip(&site.capacity)
                    .all(|(amount, capacity)| amount <= &(capacity + EPS));
            if !funded {
                blocked.push(BlockedEvent {
                    event_id: site.site_id.clone(),
                    reason: "birth-resource-atp-or-capacity-shortfall".into(),
                });
                continue;
            }
            let (child_genotype, child_rng, generation) = {
                let parent = &mut state.organisms[parent_index];
                let genotype = mutate_genotype(
                    &parent.genotype,
                    &mut parent.rng,
                    program.mutation_fraction,
                    parent.generation.saturating_add(1),
                );
                let child_rng = splitmix64(&mut parent.rng);
                for pool in 0..program.material_endowment.len() {
                    parent.material.quantity[pool] -= program.material_endowment[pool];
                }
                parent.atp -= program.atp_cost + program.atp_endowment;
                parent.reproduction_credit_s -= program.interval_s;
                parent.descendant_count += 1;
                (genotype, child_rng, parent.generation.saturating_add(1))
            };
            state.organisms.push(OrganismState {
                id: site.child_id.clone(),
                physics_binding: site.physics_binding.clone(),
                anchored_region: site.anchored_region.clone(),
                internal_volume_m3: site.internal_volume_m3,
                capacity: site.capacity.clone(),
                material: MaterialStore {
                    quantity: program.material_endowment.clone(),
                },
                atp: program.atp_endowment,
                atp_capacity: site.atp_capacity,
                enzyme_activity: child_genotype.enzyme_baseline.clone(),
                genotype: child_genotype,
                development_credit_s: 0.0,
                development_state: crate::growth::seed_state(child_rng),
                reproduction_credit_s: 0.0,
                descendant_count: 0,
                maintenance_shortfall: 0.0,
                last_acclimation_atp_fraction: 0.0,
                generation,
                rng: child_rng,
            });
            transfers.push(TransferRecord {
                cause: TransferCause::BirthEndowment,
                detail_id: site.site_id.clone(),
                from: StoreId::Organism(site.parent_id.clone()),
                to: StoreId::Organism(site.child_id.clone()),
                quantity: program.material_endowment,
            });
            creations.push(PhysicalCreation {
                proposal_id: format!(
                    "create-offspring-{}-{}",
                    site.child_id,
                    state.step_index.saturating_add(1)
                ),
                kind: PhysicalCreationKind::OffspringBody,
                physics_binding: site.physics_binding.clone(),
                host_template_id: site.host_template_id.clone(),
                position_m: site.position_m,
                orientation_xyzw: site.orientation_xyzw,
                nominal_radius_m: site.nominal_radius_m,
                nominal_length_m: site.nominal_length_m,
                material_store: StoreId::Organism(site.child_id.clone()),
                route_hint: None,
            });
        }
        Ok(())
    }

    fn sample_afferent_medium(
        &self,
        input: &TickInput,
        state: &WorldState,
    ) -> EcologyResult<Vec<AfferentTransductionSample>> {
        input
            .afferent_sample_sites
            .iter()
            .map(|sample| {
                let region = self.indices.regions[&sample.region_id];
                let volume = self.envelope.config.regions[region].volume_m3;
                Ok(AfferentTransductionSample {
                    site_id: sample.site_id.clone(),
                    concentration_quantity_m3: state.regions[region]
                        .quantity
                        .iter()
                        .map(|value| value / volume)
                        .collect(),
                })
            })
            .collect()
    }
}

fn mutate_genotype(parent: &Genotype, rng: &mut u64, fraction: f64, generation: u32) -> Genotype {
    let mut child = parent.clone();
    let suffix = splitmix64(rng);
    let prefix = parent.lineage_id.chars().take(70).collect::<String>();
    child.lineage_id = format!("{prefix}-g{generation}-{:08x}", suffix as u32);
    let mut perturb_positive = |value: &mut f64, lower: f64, upper: f64| {
        let unit = ((splitmix64(rng) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        *value = (*value * (1.0 + fraction * (2.0 * unit - 1.0))).clamp(lower, upper);
    };
    for value in &mut child.enzyme_baseline {
        perturb_positive(value, 0.0, child.enzyme_maximum);
    }
    let total = child.enzyme_baseline.iter().sum::<f64>();
    if total > child.enzyme_total_budget {
        let scale = child.enzyme_total_budget / total;
        child
            .enzyme_baseline
            .iter_mut()
            .for_each(|value| *value *= scale);
    }
    for value in &mut child.membrane_permeability {
        perturb_positive(value, 0.0, 1.0);
    }
    perturb_positive(&mut child.enzyme_time_constant_s, 1.0e-3, 1.0e6);
    perturb_positive(&mut child.enzyme_change_atp_cost, 0.0, 1.0e6);
    perturb_positive(&mut child.maintenance_atp_s, 0.0, 1.0e6);
    if let Some(program) = &mut child.development {
        perturb_positive(
            &mut program.branch_angle_rad,
            0.0,
            std::f64::consts::FRAC_PI_2,
        );
        perturb_positive(&mut program.lateral_probability, 0.0, 1.0);
        perturb_positive(&mut program.phototropism, 0.0, 4.0);
        perturb_positive(&mut program.contact_avoidance, 0.0, 4.0);
        perturb_positive(&mut program.directional_persistence, 0.0, 1.0);
    }
    if let Some(program) = &mut child.reproduction {
        perturb_positive(&mut program.dispersal_distance_m, 1e-9, 1e6);
    }
    child
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchEnvelope {
    format: String,
    worlds: Vec<WorldEnvelope>,
}

/// Multi-world training host. Preparation is copy-then-swap and receipts are
/// all validated before any world commits, so a batch cannot be half advanced.
#[derive(Debug, Clone)]
pub struct EcologyBatch {
    worlds: Vec<EcologyWorld>,
}

impl EcologyBatch {
    pub fn new(configs: Vec<EcologyConfig>) -> EcologyResult<Self> {
        if configs.is_empty() || configs.len() > 4096 {
            return err("ecology batch size differs");
        }
        Ok(Self {
            worlds: configs
                .into_iter()
                .map(EcologyWorld::new)
                .collect::<EcologyResult<_>>()?,
        })
    }

    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    pub fn world(&self, index: usize) -> Option<&EcologyWorld> {
        self.worlds.get(index)
    }

    pub fn prepare_batch(&mut self, inputs: &[TickInput]) -> EcologyResult<Vec<WorldDelta>> {
        if inputs.len() != self.worlds.len() {
            return err("ecology batch input count differs");
        }
        let mut prepared = self.worlds.clone();
        let deltas = prepared
            .iter_mut()
            .zip(inputs)
            .map(|(world, input)| world.prepare_step(input))
            .collect::<EcologyResult<Vec<_>>>()?;
        self.worlds = prepared;
        Ok(deltas)
    }

    pub fn commit_batch(&mut self, receipts: &[CommitReceipt]) -> EcologyResult<()> {
        if receipts.len() != self.worlds.len() {
            return err("ecology batch receipt count differs");
        }
        for (world, receipt) in self.worlds.iter().zip(receipts) {
            world.validate_receipt(receipt)?;
        }
        for (world, receipt) in self.worlds.iter_mut().zip(receipts) {
            world.commit_step(receipt)?;
        }
        Ok(())
    }

    pub fn abort_batch(&mut self, tokens: &[String]) -> EcologyResult<()> {
        if tokens.len() != self.worlds.len() {
            return err("ecology batch abort count differs");
        }
        for (world, token) in self.worlds.iter().zip(tokens) {
            if world.pending_delta().map(|value| value.token.as_str()) != Some(token.as_str()) {
                return err("ecology batch abort token differs");
            }
        }
        for (world, token) in self.worlds.iter_mut().zip(tokens) {
            world.abort_step(token)?;
        }
        Ok(())
    }

    pub fn snapshot_json(&self) -> EcologyResult<String> {
        serde_json::to_string(&BatchEnvelope {
            format: BATCH_FORMAT.into(),
            worlds: self
                .worlds
                .iter()
                .map(|world| world.envelope.clone())
                .collect(),
        })
        .map_err(Into::into)
    }

    pub fn from_snapshot_json(snapshot: &str) -> EcologyResult<Self> {
        let envelope: BatchEnvelope = serde_json::from_str(snapshot)?;
        if envelope.format != BATCH_FORMAT
            || envelope.worlds.is_empty()
            || envelope.worlds.len() > 4096
        {
            return err("ecology batch snapshot format or size differs");
        }
        let worlds = envelope
            .worlds
            .into_iter()
            .map(|world| EcologyWorld::from_snapshot_json(&serde_json::to_string(&world)?))
            .collect::<EcologyResult<Vec<_>>>()?;
        Ok(Self { worlds })
    }
}

fn validate_config(config: &EcologyConfig) -> EcologyResult<()> {
    if config.format != CONFIG_FORMAT {
        return err("ecology configuration format differs");
    }
    let units = &config.coordinate_contract;
    if units.length_unit != "meter"
        || units.time_unit != "second"
        || units.material_unit != "abstract_conserved_quantity"
        || !finite(&units.world_min_m)
        || !finite(&units.world_max_m)
        || (0..3).any(|axis| units.world_min_m[axis] >= units.world_max_m[axis])
    {
        return err("ecology coordinate contract must declare finite SI space and time");
    }
    let k = config.pools.len();
    let e = config.elements.len();
    let r = config.reactions.len();
    if !(1..=MAX_POOLS).contains(&k)
        || !(1..=MAX_ELEMENTS).contains(&e)
        || !(1..=MAX_REACTIONS).contains(&r)
        || !(1..=MAX_REGIONS).contains(&config.regions.len())
        || config.routes.len() > MAX_ROUTES
        || config.organisms.len() > MAX_ORGANISMS
        || config.packets.len() > MAX_PACKETS
    {
        return err("ecology configuration dimensions differ");
    }
    unique_valid(config.elements.iter().map(String::as_str), "element")?;
    unique_valid(config.pools.iter().map(|value| value.id.as_str()), "pool")?;
    unique_valid(
        config.reactions.iter().map(|value| value.id.as_str()),
        "reaction",
    )?;
    unique_valid(
        config.regions.iter().map(|value| value.id.as_str()),
        "region",
    )?;
    unique_valid(config.routes.iter().map(|value| value.id.as_str()), "route")?;
    unique_valid(
        config.organisms.iter().map(|value| value.id.as_str()),
        "organism",
    )?;
    unique_valid(
        config.packets.iter().map(|value| value.id.as_str()),
        "packet",
    )?;
    let mut bindings = HashSet::new();
    for pool in &config.pools {
        if pool.elemental_composition.len() != e
            || !finite_nonnegative(&pool.elemental_composition)
            || !pool.chemical_energy_per_quantity.is_finite()
            || pool.chemical_energy_per_quantity < 0.0
            || !pool.diffusivity_m2_s.is_finite()
            || pool.diffusivity_m2_s < 0.0
            || pool.diffusivity_m2_s > 0.1
        {
            return err("ecology pool definition differs");
        }
    }
    for reaction in &config.reactions {
        if reaction.stoichiometry.len() != k
            || reaction.half_saturation_quantity.len() != k
            || !finite(&reaction.stoichiometry)
            || !finite_nonnegative(&reaction.half_saturation_quantity)
            || !finite_nonnegative(&[
                reaction.base_rate_quantity_s,
                reaction.atp_cost_per_flux,
                reaction.atp_yield_per_flux,
                reaction.photon_cost_per_flux,
            ])
            || !reaction.stoichiometry.iter().any(|value| *value < 0.0)
        {
            return err("ecology reaction definition differs");
        }
        for element in 0..e {
            let residual = reaction
                .stoichiometry
                .iter()
                .zip(&config.pools)
                .map(|(coefficient, pool)| coefficient * pool.elemental_composition[element])
                .sum::<f64>();
            if residual.abs() > 1.0e-10 {
                return err("ecology reaction violates elemental conservation");
            }
        }
        let chemical_delta = reaction
            .stoichiometry
            .iter()
            .zip(&config.pools)
            .map(|(coefficient, pool)| coefficient * pool.chemical_energy_per_quantity)
            .sum::<f64>();
        let dissipated = reaction.photon_cost_per_flux + reaction.atp_cost_per_flux
            - reaction.atp_yield_per_flux
            - chemical_delta;
        if dissipated < -1.0e-10 {
            return err("ecology reaction creates undeclared energy");
        }
    }
    let region_ids = config
        .regions
        .iter()
        .map(|value| value.id.as_str())
        .collect::<HashSet<_>>();
    for region in &config.regions {
        if !inside_world(config, &region.center_m)
            || !region.volume_m3.is_finite()
            || region.volume_m3 <= 0.0
            || !valid_store(&region.initial, &region.capacity, k)
        {
            return err("ecology region definition differs");
        }
    }
    let mut adjacency = vec![Vec::new(); config.regions.len()];
    let region_index = config
        .regions
        .iter()
        .enumerate()
        .map(|(i, value)| (value.id.as_str(), i))
        .collect::<HashMap<_, _>>();
    for route in &config.routes {
        let (Some(&a), Some(&b)) = (
            region_index.get(route.a.as_str()),
            region_index.get(route.b.as_str()),
        ) else {
            return err("ecology route endpoint is absent");
        };
        if a == b
            || !finite_nonnegative(&[
                route.length_m,
                route.cross_section_m2,
                route.hydraulic_capacity_m3_s,
                route.base_open_fraction,
            ])
            || route.length_m <= 0.0
            || route.cross_section_m2 <= 0.0
            || route.base_open_fraction > 1.0
        {
            return err("ecology route definition differs");
        }
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut seen = vec![false; config.regions.len()];
    let mut queue = VecDeque::from([0usize]);
    seen[0] = true;
    while let Some(a) = queue.pop_front() {
        for &b in &adjacency[a] {
            if !seen[b] {
                seen[b] = true;
                queue.push_back(b);
            }
        }
    }
    if seen.iter().any(|value| !value) {
        return err("ecology material-region graph is disconnected");
    }
    for seed in &config.organisms {
        if !valid_id(&seed.physics_binding)
            || !bindings.insert(seed.physics_binding.as_str())
            || seed
                .anchored_region
                .as_ref()
                .is_some_and(|value| !region_ids.contains(value.as_str()))
            || !seed.internal_volume_m3.is_finite()
            || seed.internal_volume_m3 <= 0.0
            || !valid_store(&seed.initial, &seed.capacity, k)
            || !seed.atp.is_finite()
            || !seed.atp_capacity.is_finite()
            || seed.atp < 0.0
            || seed.atp_capacity <= 0.0
            || seed.atp > seed.atp_capacity + EPS
        {
            return err("ecology organism seed differs");
        }
        validate_genotype(config, &seed.genotype)?;
    }
    for seed in &config.packets {
        if !valid_id(&seed.physics_binding)
            || !bindings.insert(seed.physics_binding.as_str())
            || !seed.volume_m3.is_finite()
            || seed.volume_m3 <= 0.0
            || !valid_store(&seed.initial, &seed.capacity, k)
        {
            return err("ecology packet seed differs");
        }
    }
    Ok(())
}

fn validate_genotype(config: &EcologyConfig, genotype: &Genotype) -> EcologyResult<()> {
    let k = config.pools.len();
    let r = config.reactions.len();
    if !valid_id(&genotype.lineage_id)
        || genotype.enzyme_baseline.len() != r
        || genotype.enzyme_substrate_response.len() != r
        || genotype.enzyme_atp_response.len() != r
        || genotype.membrane_permeability.len() != k
        || !finite_nonnegative(&genotype.enzyme_baseline)
        || !finite(&genotype.enzyme_substrate_response)
        || !finite(&genotype.enzyme_atp_response)
        || genotype
            .enzyme_substrate_response
            .iter()
            .chain(&genotype.enzyme_atp_response)
            .any(|value| value.abs() > 16.0)
        || !finite_nonnegative(&genotype.membrane_permeability)
        || genotype
            .membrane_permeability
            .iter()
            .any(|value| *value > 1.0)
        || !finite_nonnegative(&[
            genotype.enzyme_time_constant_s,
            genotype.enzyme_change_atp_cost,
            genotype.enzyme_maximum,
            genotype.enzyme_total_budget,
            genotype.maintenance_atp_s,
        ])
        || genotype.enzyme_time_constant_s <= 0.0
        || genotype.enzyme_maximum <= 0.0
        || genotype.enzyme_total_budget <= 0.0
        || genotype
            .enzyme_baseline
            .iter()
            .any(|value| *value > genotype.enzyme_maximum + EPS)
        || genotype.enzyme_baseline.iter().sum::<f64>() > genotype.enzyme_total_budget + EPS
    {
        return err("ecology inherited metabolic parameters differ");
    }
    if let Some(program) = &genotype.development {
        if program.material_cost.len() != k
            || program.decay_return.len() != k
            || !finite_nonnegative(&program.material_cost)
            || !finite_nonnegative(&program.decay_return)
            || program.material_cost.iter().sum::<f64>() <= 0.0
            || !finite_nonnegative(&[
                program.interval_s,
                program.atp_cost,
                program.branch_angle_rad,
                program.lateral_probability,
                program.phototropism,
                program.contact_avoidance,
                program.directional_persistence,
                program.nominal_radius_m,
                program.nominal_length_m,
                program.nominal_volume_m3,
                program.decay_time_constant_s,
            ])
            || program.interval_s <= 0.0
            || program.branch_angle_rad > std::f64::consts::FRAC_PI_2
            || program.lateral_probability > 1.0
            || program.phototropism > 4.0
            || program.contact_avoidance > 4.0
            || program.directional_persistence > 1.0
            || program.maximum_structures == 0
            || program.maximum_structures > 4096
            || program.nominal_radius_m <= 0.0
            || program.nominal_length_m <= 0.0
            || program.nominal_volume_m3 <= 0.0
            || program.decay_time_constant_s <= 0.0
        {
            return err("ecology developmental program differs");
        }
        if elemental_for_quantities(config, &program.material_cost)
            .iter()
            .zip(elemental_for_quantities(config, &program.decay_return))
            .any(|(a, b)| (a - b).abs() > 1.0e-10)
        {
            return err("ecology construction decay violates material conservation");
        }
    }
    if let Some(program) = &genotype.reproduction {
        if program.material_endowment.len() != k
            || !finite_nonnegative(&program.material_endowment)
            || program.material_endowment.iter().sum::<f64>() <= 0.0
            || !finite_nonnegative(&[
                program.interval_s,
                program.atp_cost,
                program.atp_endowment,
                program.mutation_fraction,
                program.dispersal_distance_m,
            ])
            || program.interval_s <= 0.0
            || program.dispersal_distance_m <= 0.0
            || program.maximum_descendants == 0
            || program.maximum_descendants > 4096
            || program.mutation_fraction > 0.5
        {
            return err("ecology reproduction program differs");
        }
    }
    Ok(())
}

fn validate_state(config: &EcologyConfig, state: &WorldState) -> EcologyResult<()> {
    let k = config.pools.len();
    if !state.time_s.is_finite()
        || state.time_s < 0.0
        || state.regions.len() != config.regions.len()
        || state.organisms.len() > MAX_ORGANISMS
        || state.packets.len() > MAX_PACKETS
        || state.structures.len() > MAX_PACKETS
    {
        return err("ecology snapshot state dimensions differ");
    }
    for (store, region) in state.regions.iter().zip(&config.regions) {
        if !valid_store(&store.quantity, &region.capacity, k) {
            return err("ecology regional snapshot material differs");
        }
    }
    unique_valid(
        state.organisms.iter().map(|value| value.id.as_str()),
        "organism",
    )?;
    unique_valid(
        state.packets.iter().map(|value| value.id.as_str()),
        "packet",
    )?;
    unique_valid(
        state.structures.iter().map(|value| value.id.as_str()),
        "structure",
    )?;
    let mut bindings = HashSet::new();
    for organism in &state.organisms {
        validate_genotype(config, &organism.genotype)?;
        if organism
            .development_state
            .apical_binding
            .as_ref()
            .is_some_and(|binding| {
                !state
                    .structures
                    .iter()
                    .any(|s| &s.physics_binding == binding && s.owner_id == organism.id)
            })
            || !bindings.insert(organism.physics_binding.as_str())
            || !valid_store(&organism.material.quantity, &organism.capacity, k)
            || organism.enzyme_activity.len() != config.reactions.len()
            || !finite_nonnegative(&organism.enzyme_activity)
            || !finite_nonnegative(&[
                organism.internal_volume_m3,
                organism.atp,
                organism.atp_capacity,
                organism.development_credit_s,
                organism.reproduction_credit_s,
                organism.maintenance_shortfall,
                organism.last_acclimation_atp_fraction,
            ])
            || organism.internal_volume_m3 <= 0.0
            || organism.atp_capacity <= 0.0
            || organism.atp > organism.atp_capacity + EPS
            || organism.last_acclimation_atp_fraction > 1.0
        {
            return err("ecology organism snapshot differs");
        }
    }
    for packet in &state.packets {
        if !bindings.insert(packet.physics_binding.as_str())
            || !valid_store(&packet.material.quantity, &packet.capacity, k)
            || !packet.volume_m3.is_finite()
            || packet.volume_m3 <= 0.0
        {
            return err("ecology packet snapshot differs");
        }
    }
    for structure in &state.structures {
        if !bindings.insert(structure.physics_binding.as_str())
            || structure.initial_material.len() != k
            || structure.decay_return.len() != k
            || structure.material.quantity.len() != k
            || !finite_nonnegative(&structure.initial_material)
            || !finite_nonnegative(&structure.decay_return)
            || !finite_nonnegative(&structure.material.quantity)
            || !finite_nonnegative(&[
                structure.remaining_fraction,
                structure.decay_time_constant_s,
                structure.nominal_radius_m,
                structure.nominal_length_m,
                structure.nominal_volume_m3,
            ])
            || structure.remaining_fraction > 1.0 + EPS
            || !inside_world(config, &structure.position_m)
            || !valid_orientation(&structure.orientation_xyzw)
        {
            return err("ecology structure snapshot differs");
        }
    }
    let totals = element_totals(config, state);
    if state.accounting.initial_elements.len() != config.elements.len()
        || state.accounting.last_elements.len() != config.elements.len()
        || state.accounting.last_residual.len() != config.elements.len()
        || !finite(&state.accounting.initial_elements)
        || !finite(&state.accounting.last_elements)
        || !finite(&state.accounting.last_residual)
        || !finite_nonnegative(&[
            state.accounting.maximum_absolute_residual,
            state.accounting.captured_photon_energy,
            state.accounting.dissipated_energy,
            state.accounting.maintenance_atp_spent,
            state.accounting.regulation_atp_spent,
            state.accounting.physical_work_atp_spent,
        ])
        || totals
            .iter()
            .zip(&state.accounting.last_elements)
            .any(|(a, b)| (a - b).abs() > LEDGER_TOLERANCE * a.abs().max(1.0))
        || totals
            .iter()
            .zip(&state.accounting.initial_elements)
            .any(|(a, b)| (a - b).abs() > LEDGER_TOLERANCE * a.abs().max(1.0))
    {
        return err("ecology accounting snapshot differs");
    }
    Ok(())
}

fn valid_store(initial: &[f64], capacity: &[f64], pools: usize) -> bool {
    initial.len() == pools
        && capacity.len() == pools
        && finite_nonnegative(initial)
        && finite_nonnegative(capacity)
        && capacity.iter().all(|value| *value > 0.0)
        && initial
            .iter()
            .zip(capacity)
            .all(|(amount, limit)| amount <= &(limit + EPS))
}

fn unique_valid<'a>(values: impl Iterator<Item = &'a str>, kind: &str) -> EcologyResult<()> {
    let mut seen = HashSet::new();
    for value in values {
        if !valid_id(value) || !seen.insert(value) {
            return err(&format!("ecology {kind} identities differ"));
        }
    }
    Ok(())
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphabetic()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
}

fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn finite_nonnegative(values: &[f64]) -> bool {
    values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
}

fn valid_orientation(value: &[f64; 4]) -> bool {
    finite(value) && value.iter().map(|item| item * item).sum::<f64>() > 0.25
}

fn inside_world(config: &EcologyConfig, point: &[f64; 3]) -> bool {
    finite(point)
        && (0..3).all(|axis| {
            point[axis] >= config.coordinate_contract.world_min_m[axis]
                && point[axis] <= config.coordinate_contract.world_max_m[axis]
        })
}

fn sha256(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn err<T>(message: &str) -> EcologyResult<T> {
    Err(EcologyError(message.into()))
}

fn splitmix_seed(seed: u64, stream: u64) -> u64 {
    let mut value = seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    splitmix64(&mut value)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn elemental_for_quantities(config: &EcologyConfig, quantities: &[f64]) -> Vec<f64> {
    (0..config.elements.len())
        .map(|element| {
            quantities
                .iter()
                .zip(&config.pools)
                .map(|(quantity, pool)| quantity * pool.elemental_composition[element])
                .sum()
        })
        .collect()
}

fn element_totals(config: &EcologyConfig, state: &WorldState) -> Vec<f64> {
    let mut pools = vec![0.0; config.pools.len()];
    for store in &state.regions {
        for (total, quantity) in pools.iter_mut().zip(&store.quantity) {
            *total += quantity;
        }
    }
    for organism in &state.organisms {
        for (total, quantity) in pools.iter_mut().zip(&organism.material.quantity) {
            *total += quantity;
        }
    }
    for packet in state.packets.iter().filter(|value| value.active) {
        for (total, quantity) in pools.iter_mut().zip(&packet.material.quantity) {
            *total += quantity;
        }
    }
    for structure in state.structures.iter().filter(|value| value.active) {
        for (total, quantity) in pools.iter_mut().zip(&structure.material.quantity) {
            *total += quantity;
        }
    }
    elemental_for_quantities(config, &pools)
}
