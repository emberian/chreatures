use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

pub const CONFIG_FORMAT: &str = "chreatures-ecology-config-v1";
pub const SNAPSHOT_FORMAT: &str = "chreatures-ecology-snapshot-v1";
pub const DELTA_FORMAT: &str = "chreatures-ecology-delta-v1";
pub const BATCH_FORMAT: &str = "chreatures-ecology-batch-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcologyError(pub String);

impl Display for EcologyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EcologyError {}

impl From<serde_json::Error> for EcologyError {
    fn from(value: serde_json::Error) -> Self {
        Self(value.to_string())
    }
}

pub type EcologyResult<T> = Result<T, EcologyError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoordinateContract {
    /// Must be `meter`. Millimetre-scale worlds use ordinary small SI values.
    pub length_unit: String,
    /// Must be `second`.
    pub time_unit: String,
    /// Synthetic conserved quantity used by the declared elemental ledger.
    pub material_unit: String,
    pub world_min_m: [f64; 3],
    pub world_max_m: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PoolSpec {
    pub id: String,
    pub elemental_composition: Vec<f64>,
    pub chemical_energy_per_quantity: f64,
    pub diffusivity_m2_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReactionSpec {
    pub id: String,
    /// Pool-major signed change for one unit of flux.
    pub stoichiometry: Vec<f64>,
    pub base_rate_quantity_s: f64,
    pub half_saturation_quantity: Vec<f64>,
    pub atp_cost_per_flux: f64,
    pub atp_yield_per_flux: f64,
    pub photon_cost_per_flux: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RegionSpec {
    pub id: String,
    pub center_m: [f64; 3],
    pub volume_m3: f64,
    pub capacity: Vec<f64>,
    pub initial: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RouteSpec {
    pub id: String,
    pub a: String,
    pub b: String,
    pub length_m: f64,
    pub cross_section_m2: f64,
    pub hydraulic_capacity_m3_s: f64,
    pub base_open_fraction: f64,
}

/// Inherited reaction and membrane response parameters. No field is a strategy
/// or ecological-role label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Genotype {
    pub lineage_id: String,
    pub enzyme_baseline: Vec<f64>,
    pub enzyme_substrate_response: Vec<f64>,
    pub enzyme_atp_response: Vec<f64>,
    pub enzyme_time_constant_s: f64,
    pub enzyme_change_atp_cost: f64,
    pub enzyme_maximum: f64,
    pub enzyme_total_budget: f64,
    pub membrane_permeability: Vec<f64>,
    pub maintenance_atp_s: f64,
    pub development: Option<DevelopmentProgram>,
    pub reproduction: Option<ReproductionProgram>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentProgram {
    pub interval_s: f64,
    pub material_cost: Vec<f64>,
    pub decay_return: Vec<f64>,
    pub atp_cost: f64,
    pub maximum_structures: usize,
    pub nominal_radius_m: f64,
    pub nominal_length_m: f64,
    pub nominal_volume_m3: f64,
    pub decay_time_constant_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReproductionProgram {
    pub interval_s: f64,
    pub material_endowment: Vec<f64>,
    pub atp_cost: f64,
    pub atp_endowment: f64,
    pub maximum_descendants: usize,
    /// Fractional bounded mutation of continuous metabolic parameters.
    pub mutation_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OrganismSeed {
    pub id: String,
    pub physics_binding: String,
    /// `Some` denotes a physically anchored body; it does not name a niche.
    pub anchored_region: Option<String>,
    pub internal_volume_m3: f64,
    pub capacity: Vec<f64>,
    pub initial: Vec<f64>,
    pub atp: f64,
    pub atp_capacity: f64,
    pub genotype: Genotype,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PacketSeed {
    pub id: String,
    pub physics_binding: String,
    pub volume_m3: f64,
    pub capacity: Vec<f64>,
    pub initial: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EcologyConfig {
    pub format: String,
    pub coordinate_contract: CoordinateContract,
    pub elements: Vec<String>,
    pub pools: Vec<PoolSpec>,
    pub reactions: Vec<ReactionSpec>,
    pub regions: Vec<RegionSpec>,
    pub routes: Vec<RouteSpec>,
    pub organisms: Vec<OrganismSeed>,
    pub packets: Vec<PacketSeed>,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MaterialStore {
    pub quantity: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OrganismState {
    pub id: String,
    pub physics_binding: String,
    pub anchored_region: Option<String>,
    pub internal_volume_m3: f64,
    pub capacity: Vec<f64>,
    pub material: MaterialStore,
    pub atp: f64,
    pub atp_capacity: f64,
    pub genotype: Genotype,
    /// Private acclimated state; descendants start from inherited baselines.
    pub enzyme_activity: Vec<f64>,
    pub development_credit_s: f64,
    pub reproduction_credit_s: f64,
    pub descendant_count: usize,
    pub maintenance_shortfall: f64,
    pub last_acclimation_atp_fraction: f64,
    pub generation: u32,
    pub rng: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PacketState {
    pub id: String,
    pub physics_binding: String,
    pub volume_m3: f64,
    pub capacity: Vec<f64>,
    pub material: MaterialStore,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StructureState {
    pub id: String,
    pub physics_binding: String,
    pub owner_id: String,
    pub region_id: String,
    pub route_hint: Option<String>,
    pub host_template_id: String,
    pub position_m: [f64; 3],
    pub orientation_xyzw: [f64; 4],
    pub nominal_radius_m: f64,
    pub nominal_length_m: f64,
    pub nominal_volume_m3: f64,
    pub initial_material: Vec<f64>,
    pub decay_return: Vec<f64>,
    pub material: MaterialStore,
    pub remaining_fraction: f64,
    pub decay_time_constant_s: f64,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AccountingState {
    pub initial_elements: Vec<f64>,
    pub last_elements: Vec<f64>,
    pub last_residual: Vec<f64>,
    pub maximum_absolute_residual: f64,
    pub captured_photon_energy: f64,
    pub dissipated_energy: f64,
    pub maintenance_atp_spent: f64,
    pub regulation_atp_spent: f64,
    pub physical_work_atp_spent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorldState {
    pub time_s: f64,
    pub step_index: u64,
    pub regions: Vec<MaterialStore>,
    pub organisms: Vec<OrganismState>,
    pub packets: Vec<PacketState>,
    pub structures: Vec<StructureState>,
    pub accounting: AccountingState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum StoreId {
    Region(String),
    Organism(String),
    Packet(String),
    Structure(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContactEvent {
    pub a: StoreId,
    pub b: StoreId,
    /// Pool-wise physical conductance. A host sets this to zero without a
    /// measured contact or open exchange surface.
    pub conductance_m3_s: Vec<f64>,
}

/// A physically gated active transfer. The physics host supplies a finite
/// upper bound only after a measured contact and actuator state (for example,
/// pharyngeal pumping or salivation). The ecology kernel still bounds it by
/// donor inventory and receiver capacity alongside competing exchanges.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DirectedExchange {
    pub event_id: String,
    pub from: StoreId,
    pub to: StoreId,
    pub maximum_quantity: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhotonExposure {
    pub organism_id: String,
    pub available_energy_per_s: f64,
}

/// ATP-equivalent demand derived by the physics host from measured actuator
/// work. The conversion coefficient lives in host configuration and makes no
/// joule-level biological calibration claim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetabolicWorkDemand {
    pub organism_id: String,
    pub atp_quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConstructionSite {
    pub site_id: String,
    pub organism_id: String,
    pub region_id: String,
    pub host_template_id: String,
    pub physics_binding: String,
    pub position_m: [f64; 3],
    pub orientation_xyzw: [f64; 4],
    /// Host-only hint used to associate the resulting geometry with a route
    /// clearance query. Route openness remains a measured input next tick.
    pub route_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BirthSite {
    pub site_id: String,
    pub parent_id: String,
    pub child_id: String,
    pub host_template_id: String,
    pub physics_binding: String,
    pub anchored_region: Option<String>,
    pub position_m: [f64; 3],
    pub orientation_xyzw: [f64; 4],
    pub internal_volume_m3: f64,
    pub capacity: Vec<f64>,
    pub atp_capacity: f64,
    pub nominal_radius_m: f64,
    pub nominal_length_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PacketRetirement {
    pub packet_id: String,
    pub receiver_region_id: String,
    pub cause: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AfferentSampleSite {
    pub site_id: String,
    pub region_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TickInput {
    pub dt_s: f64,
    /// Aligned to config.routes and measured by the physics host.
    pub route_open_fraction: Vec<f64>,
    /// Signed a->b volume flow, aligned to config.routes.
    pub route_advection_m3_s: Vec<f64>,
    pub contacts: Vec<ContactEvent>,
    pub directed_exchanges: Vec<DirectedExchange>,
    pub photon_exposures: Vec<PhotonExposure>,
    pub metabolic_work_demands: Vec<MetabolicWorkDemand>,
    pub construction_sites: Vec<ConstructionSite>,
    pub birth_sites: Vec<BirthSite>,
    pub packet_retirements: Vec<PacketRetirement>,
    pub afferent_sample_sites: Vec<AfferentSampleSite>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransferCause {
    RouteDiffusionAdvection,
    MeasuredContact,
    MeasuredDirectedExchange,
    ConstructionAllocation,
    BirthEndowment,
    PacketExit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TransferRecord {
    pub cause: TransferCause,
    pub detail_id: String,
    pub from: StoreId,
    pub to: StoreId,
    pub quantity: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReactionRecord {
    pub organism_id: String,
    pub reaction_id: String,
    pub flux: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalCreationKind {
    ConstructedGeometry,
    OffspringBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCreation {
    pub proposal_id: String,
    pub kind: PhysicalCreationKind,
    pub physics_binding: String,
    pub host_template_id: String,
    pub position_m: [f64; 3],
    pub orientation_xyzw: [f64; 4],
    pub nominal_radius_m: f64,
    pub nominal_length_m: f64,
    pub material_store: StoreId,
    /// Host-only association for the next physical clearance measurement.
    pub route_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRemoval {
    pub proposal_id: String,
    pub physics_binding: String,
    pub material_store: StoreId,
    pub cause: String,
}

/// Host-only chemical truth for conversion into masked CNS afferents. It is not
/// a controller input and intentionally carries neither object identities nor
/// world coordinates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AfferentTransductionSample {
    pub site_id: String,
    pub concentration_quantity_m3: Vec<f64>,
}

/// Mapping from a configurable chemistry to the fixed engineered fly
/// interoception order. Pool IDs are named only in the host adapter; resulting
/// values carry no object identity or world geometry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FlyInteroceptionSpec {
    pub gut_pool_ids: Vec<String>,
    pub carbon_reserve_pool_ids: Vec<String>,
    pub nitrogen_reserve_pool_ids: Vec<String>,
    pub water_pool_id: String,
    pub structural_pool_ids: Vec<String>,
    pub salivary_pool_id: String,
    pub oxygen_pool_id: String,
    pub osmotic_pool_ids: Vec<String>,
    pub osmotic_target_quantity_m3: f64,
    pub osmotic_scale_quantity_m3: f64,
}

/// Host-side concentration normalization before masked chemical afferents.
/// `odor_permeability` can encode the selected eight-pool garden mapping while
/// taste retains every locally contacted pool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FlyChemicalSenseSpec {
    pub odor_permeability: Vec<f64>,
    pub half_saturation_quantity_m3: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FlyChemicalProjection {
    pub odor: Vec<f64>,
    pub taste: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BlockedEvent {
    pub event_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorldDelta {
    pub format: String,
    pub token: String,
    pub step_index: u64,
    pub time_start_s: f64,
    pub time_end_s: f64,
    pub transfers: Vec<TransferRecord>,
    pub reactions: Vec<ReactionRecord>,
    pub physical_creations: Vec<PhysicalCreation>,
    pub physical_removals: Vec<PhysicalRemoval>,
    pub afferent_transduction_samples: Vec<AfferentTransductionSample>,
    pub blocked_events: Vec<BlockedEvent>,
    pub elemental_before: Vec<f64>,
    pub elemental_after: Vec<f64>,
    pub elemental_residual: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CommitReceipt {
    pub token: String,
    /// Exact proposal IDs successfully created by one atomic host transaction.
    pub created: Vec<String>,
    /// Exact proposal IDs successfully removed by that transaction.
    pub removed: Vec<String>,
}

impl CommitReceipt {
    pub fn accept_all(delta: &WorldDelta) -> Self {
        Self {
            token: delta.token.clone(),
            created: delta
                .physical_creations
                .iter()
                .map(|item| item.proposal_id.clone())
                .collect(),
            removed: delta
                .physical_removals
                .iter()
                .map(|item| item.proposal_id.clone())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PendingStep {
    pub token: String,
    pub next_state: WorldState,
    pub delta: WorldDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorldEnvelope {
    pub format: String,
    pub config_sha256: String,
    pub config: EcologyConfig,
    pub state: WorldState,
    pub pending: Option<PendingStep>,
}
