use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const STATE_FORMAT: &str = "chreatures-population-search-v2";
pub const GENOME_FORMAT: &str = "chreatures-population-genome-v2";
pub const DESCRIPTOR_VERSION: &str = "physical-descriptor-v2";
pub const QUALITY_VERSION: &str = "finite-life-quality-v2";
pub const VARIATION_VERSION: &str = "bounded-genome-variation-v2";

fn default_capacity() -> usize {
    4
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterSpec {
    pub name: String,
    pub group: String,
    pub low: f64,
    pub high: f64,
    pub mutation_sigma: f64,
    #[serde(default)]
    pub integer: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescriptorAxis {
    pub component: String,
    pub low: f64,
    pub high: f64,
    pub bins: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityTerm {
    pub component: String,
    pub scale: f64,
    pub weight: f64,
    pub direction: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    pub graph_sha256: String,
    pub port_spec_sha256: String,
    pub base_controller_sha256: String,
    pub developmental_base_sha256: String,
    pub population_adapter_bank_sha256: String,
    pub organism_interface_sha256: String,
    pub policy_adapter_count: u32,
    pub policy_adapter_rank: u32,
    pub parameter_specs: Vec<ParameterSpec>,
    pub founder_values: BTreeMap<String, f64>,
    pub descriptor_axes: Vec<DescriptorAxis>,
    pub quality_terms: Vec<QualityTerm>,
    #[serde(default = "default_capacity")]
    pub archive_members_per_cell: usize,
    pub variation_recipe_sha256: String,
    pub environment_probe_panel_sha256: String,
    pub environment_epoch: u64,
    pub environment_novelty_weight: f64,
    pub environment_cost_weight: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variation {
    pub operator: String,
    pub seed: u64,
    pub recipe_sha256: String,
    pub mutated: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Genome {
    pub format: String,
    pub sha256: String,
    pub parents: Vec<String>,
    pub graph_sha256: String,
    pub port_spec_sha256: String,
    pub base_controller_sha256: String,
    pub developmental_base_sha256: String,
    pub population_adapter_bank_sha256: String,
    pub organism_interface_sha256: String,
    pub policy_adapter_count: u32,
    pub policy_adapter_rank: u32,
    pub values: BTreeMap<String, f64>,
    pub variation: Variation,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentGenome {
    pub format: String,
    pub sha256: String,
    pub genome_sha256: String,
    pub genome_parents: Vec<String>,
    pub parents: Vec<String>,
    pub variation: EnvironmentVariation,
    pub topology_sha256: String,
    pub resource_sha256: String,
    pub profile_sha256: String,
    pub epoch: u64,
    pub descriptors: BTreeMap<String, f64>,
    pub generation_cost: EnvironmentGenerationCost,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentVariation {
    pub operator: String,
    pub seed: u64,
    pub recipe_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentGenerationCost {
    pub physical_geoms: u64,
    pub regions: u64,
    pub edges: u64,
    pub movables: u64,
    pub compartments: u64,
    pub normalized: f64,
}
impl EnvironmentGenome {
    fn compute_hash(&self) -> Result<String, String> {
        let mut value = self.clone();
        value.sha256.clear();
        digest(&value)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub candidate: Genome,
    pub environment_sha256: String,
    pub phase: String,
    pub selection: AssignmentEvidence,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssignmentEvidence {
    pub pair_valid_evaluations: u64,
    pub candidate_valid_environments: usize,
    pub environment_valid_evaluations: u64,
    pub transfer_gap: f64,
    pub environment_priority: f64,
    pub analyst_score: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationInput {
    pub evaluation_sha256: String,
    pub life_id: String,
    pub evaluation_seed: u64,
    pub committed_ticks: u64,
    pub trajectory_sha256: String,
    pub candidate_sha256: String,
    pub environment_sha256: String,
    pub status: String,
    pub metrics: BTreeMap<String, f64>,
    pub failure: String,
}
impl EvaluationInput {
    fn compute_hash(&self) -> Result<String, String> {
        let mut value = self.clone();
        value.evaluation_sha256.clear();
        digest(&value)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub evaluation_sha256: String,
    pub life_id: String,
    pub evaluation_seed: u64,
    pub committed_ticks: u64,
    pub trajectory_sha256: String,
    pub candidate_sha256: String,
    pub environment_sha256: String,
    pub status: String,
    pub failure: String,
    pub metrics: BTreeMap<String, f64>,
    pub descriptor: Option<Vec<f64>>,
    pub cell: Option<Vec<u32>>,
    pub quality: Option<f64>,
    pub archive_retained: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveMember {
    pub candidate_sha256: String,
    pub evaluation_sha256: String,
    pub quality: f64,
    pub descriptor: Vec<f64>,
    pub environment_sha256: String,
    pub valid_evaluation_count: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairHistory {
    pub candidate_sha256: String,
    pub environment_sha256: String,
    pub valid_evaluations: u64,
    pub infrastructure_failures: u64,
    pub organism_terminals: u64,
    pub quality_sum: f64,
    pub best_quality: Option<f64>,
    pub last_evaluation_sha256: String,
    pub last_sequence: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentEvidence {
    pub environment_sha256: String,
    pub valid_evaluations: u64,
    pub infrastructure_failures: u64,
    pub organism_terminals: u64,
    pub quality_sum: f64,
    pub quality_square_sum: f64,
    pub candidate_sha256: BTreeSet<String>,
    pub occupied_cells: BTreeSet<String>,
    pub last_sequence: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentFrontier {
    pub environment_sha256: String,
    pub parents: Vec<String>,
    pub valid_evaluations: u64,
    pub infrastructure_failures: u64,
    pub organism_terminals: u64,
    pub distinct_candidates: usize,
    pub occupied_cells: usize,
    pub mean_quality: Option<f64>,
    pub quality_stddev: Option<f64>,
    pub challenge_priority: f64,
    pub novelty: f64,
    pub descriptors: BTreeMap<String, f64>,
    pub generation_cost: EnvironmentGenerationCost,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalScores {
    pub format: String,
    pub sha256: String,
    pub artifact_sha256: String,
    pub scores: BTreeMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchState {
    pub format: String,
    pub config: SearchConfig,
    pub config_sha256: String,
    pub descriptor_version: String,
    pub quality_version: String,
    pub rng_state: u64,
    pub ask_count: u64,
    pub environment_cursor: u64,
    pub challenge_cursor: u64,
    pub evaluation_sequence: u64,
    pub genomes: BTreeMap<String, Genome>,
    pub environments: BTreeMap<String, EnvironmentGenome>,
    pub direct_completed: BTreeSet<String>,
    pub pending_assignments: BTreeSet<String>,
    pub pending_evidence: BTreeMap<String, AssignmentEvidence>,
    pub infrastructure_blocked: BTreeSet<String>,
    pub archive: BTreeMap<String, Vec<ArchiveMember>>,
    pub pair_histories: BTreeMap<String, PairHistory>,
    pub environment_evidence: BTreeMap<String, EnvironmentEvidence>,
    pub proposal_artifact_sha256: Option<String>,
    pub proposal_scores_sha256: Option<String>,
    pub proposal_scores: BTreeMap<String, f64>,
    pub evaluations: Vec<Evaluation>,
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn digest<T: Serialize>(v: &T) -> Result<String, String> {
    // serde_json maps are ordered without preserve_order, giving the same
    // recursive lexical-key canonical form used by the Python artifact host.
    let value = serde_json::to_value(v).map_err(|e| e.to_string())?;
    let b = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    Ok(hex(&Sha256::digest(b)))
}
fn valid_hash(v: &str) -> bool {
    v.len() == 64
        && v.bytes()
            .all(|b| b.is_ascii_hexdigit() && (!b.is_ascii_alphabetic() || b.is_ascii_lowercase()))
}
fn finite(v: f64) -> bool {
    v.is_finite()
}
fn reflected(value: f64, low: f64, high: f64) -> f64 {
    let width = high - low;
    let phase = (value - low).rem_euclid(2.0 * width);
    low + if phase <= width {
        phase
    } else {
        2.0 * width - phase
    }
}
fn private_name(v: &str) -> bool {
    let x = v.to_ascii_lowercase();
    if x.ends_with("_gain") {
        return false;
    }
    [
        "state",
        "memory",
        "optimizer",
        "rng",
        "rates",
        "support",
        "pools",
        "atp",
        "context",
        "history",
    ]
    .iter()
    .any(|k| x.split(['.', '/', '_']).any(|p| p == *k))
}
impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (n, h) in [
            ("graph", &self.graph_sha256),
            ("ports", &self.port_spec_sha256),
            ("controller", &self.base_controller_sha256),
            ("developmental base", &self.developmental_base_sha256),
            (
                "population adapter bank",
                &self.population_adapter_bank_sha256,
            ),
            ("organism interface", &self.organism_interface_sha256),
            ("variation recipe", &self.variation_recipe_sha256),
            (
                "environment probe panel",
                &self.environment_probe_panel_sha256,
            ),
        ] {
            if !valid_hash(h) {
                return Err(format!("invalid {n} SHA-256"));
            }
        }
        if self.policy_adapter_count == 0
            || self.policy_adapter_count > 4096
            || self.policy_adapter_rank == 0
            || self.policy_adapter_rank > 256
        {
            return Err("population adapter dimensions differ".into());
        }
        if !finite(self.environment_novelty_weight)
            || !(0.0..=1.0).contains(&self.environment_novelty_weight)
            || !finite(self.environment_cost_weight)
            || !(0.0..=1.0).contains(&self.environment_cost_weight)
        {
            return Err("environment search weights differ".into());
        }
        let adapter = self
            .parameter_specs
            .iter()
            .find(|spec| spec.name == "controller.policy_adapter_index")
            .ok_or("policy adapter locus is absent")?;
        if !adapter.integer
            || adapter.low != 0.0
            || adapter.high > f64::from(self.policy_adapter_count - 1)
        {
            return Err("policy adapter locus differs from bank capacity".into());
        }
        if self.parameter_specs.is_empty() {
            return Err("parameter_specs is empty".into());
        }
        if self.archive_members_per_cell == 0 || self.archive_members_per_cell > 32 {
            return Err("archive capacity outside 1..32".into());
        }
        let mut names = BTreeSet::new();
        for p in &self.parameter_specs {
            if !names.insert(&p.name)
                || private_name(&p.name)
                || !finite(p.low)
                || !finite(p.high)
                || !finite(p.mutation_sigma)
                || p.low > p.high
                || (!p.integer && p.low == p.high)
                || p.mutation_sigma <= 0.0
                || (!p.integer && p.mutation_sigma > 0.5 * (p.high - p.low))
                || (p.integer && p.mutation_sigma != 1.0)
            {
                return Err(format!("invalid parameter {}", p.name));
            }
            let v = *self
                .founder_values
                .get(&p.name)
                .ok_or_else(|| format!("missing founder {}", p.name))?;
            if !finite(v) || v < p.low || v > p.high {
                return Err(format!("founder {} outside bounds", p.name));
            }
            if p.integer && v.fract() != 0.0 {
                return Err(format!("integer founder {} differs", p.name));
            }
        }
        let simplex_groups: BTreeSet<_> = self
            .parameter_specs
            .iter()
            .filter(|p| p.group.starts_with("simplex:"))
            .map(|p| p.group.as_str())
            .collect();
        for group in simplex_groups {
            let specs: Vec<_> = self
                .parameter_specs
                .iter()
                .filter(|p| p.group == group)
                .collect();
            if specs.len() < 2
                || specs.iter().any(|p| p.low != 0.0 || p.high != 1.0)
                || (specs
                    .iter()
                    .map(|p| self.founder_values[&p.name])
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    > 1e-9
            {
                return Err(format!("invalid allocation simplex {group}"));
            }
        }
        if self.founder_values.len() != names.len() {
            return Err("founder has unknown parameters".into());
        }
        if self.descriptor_axes.is_empty() {
            return Err("descriptor axes empty".into());
        }
        for a in &self.descriptor_axes {
            if a.component.is_empty()
                || !finite(a.low)
                || !finite(a.high)
                || a.low >= a.high
                || a.bins < 2
                || a.bins > 256
            {
                return Err("invalid descriptor axis".into());
            }
        }
        if self.quality_terms.is_empty() {
            return Err("quality terms empty".into());
        }
        for q in &self.quality_terms {
            if q.component.is_empty()
                || !finite(q.scale)
                || q.scale <= 0.0
                || !finite(q.weight)
                || !finite(q.direction)
                || q.direction.abs() != 1.0
            {
                return Err("invalid quality term".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().max(1e-15).ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
    fn index(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
}
impl Genome {
    fn compute_hash(&self) -> Result<String, String> {
        let mut x = self.clone();
        x.sha256.clear();
        digest(&x)
    }
    pub fn validate(&self, c: &SearchConfig) -> Result<(), String> {
        if self.format != GENOME_FORMAT || self.compute_hash()? != self.sha256 {
            return Err("genome identity differs".into());
        }
        if self.parents.len() > 2 || self.parents.iter().any(|x| !valid_hash(x)) {
            return Err("invalid parents".into());
        }
        if self.graph_sha256 != c.graph_sha256
            || self.port_spec_sha256 != c.port_spec_sha256
            || self.base_controller_sha256 != c.base_controller_sha256
            || self.developmental_base_sha256 != c.developmental_base_sha256
            || self.population_adapter_bank_sha256 != c.population_adapter_bank_sha256
            || self.organism_interface_sha256 != c.organism_interface_sha256
            || self.policy_adapter_count != c.policy_adapter_count
            || self.policy_adapter_rank != c.policy_adapter_rank
        {
            return Err("genome base identity differs".into());
        }
        for p in &c.parameter_specs {
            let v = *self.values.get(&p.name).ok_or("genome parameter missing")?;
            if !finite(v) || v < p.low || v > p.high {
                return Err("genome parameter outside bounds".into());
            }
            if p.integer && v.fract() != 0.0 {
                return Err("integer genome parameter differs".into());
            }
        }
        if self.values.len() != c.parameter_specs.len() {
            return Err("unknown genome parameter".into());
        }
        let groups: BTreeSet<_> = c
            .parameter_specs
            .iter()
            .filter(|p| p.group.starts_with("simplex:"))
            .map(|p| p.group.as_str())
            .collect();
        for group in groups {
            let total: f64 = c
                .parameter_specs
                .iter()
                .filter(|p| p.group == group)
                .map(|p| self.values[&p.name])
                .sum();
            if (total - 1.0).abs() > 1e-9 {
                return Err(format!("genome allocation simplex differs: {group}"));
            }
        }
        Ok(())
    }
}
impl SearchState {
    pub fn new(config: SearchConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let config_sha256 = digest(&config)?;
        let mut founder = Genome {
            format: GENOME_FORMAT.into(),
            sha256: String::new(),
            parents: vec![],
            graph_sha256: config.graph_sha256.clone(),
            port_spec_sha256: config.port_spec_sha256.clone(),
            base_controller_sha256: config.base_controller_sha256.clone(),
            developmental_base_sha256: config.developmental_base_sha256.clone(),
            population_adapter_bank_sha256: config.population_adapter_bank_sha256.clone(),
            organism_interface_sha256: config.organism_interface_sha256.clone(),
            policy_adapter_count: config.policy_adapter_count,
            policy_adapter_rank: config.policy_adapter_rank,
            values: config.founder_values.clone(),
            variation: Variation {
                operator: "founder".into(),
                seed: 0,
                recipe_sha256: config.variation_recipe_sha256.clone(),
                mutated: vec![],
            },
        };
        founder.sha256 = founder.compute_hash()?;
        let genomes = BTreeMap::from([(founder.sha256.clone(), founder)]);
        Ok(Self {
            format: STATE_FORMAT.into(),
            config,
            config_sha256,
            descriptor_version: DESCRIPTOR_VERSION.into(),
            quality_version: QUALITY_VERSION.into(),
            rng_state: seed,
            ask_count: 0,
            environment_cursor: 0,
            challenge_cursor: 0,
            evaluation_sequence: 0,
            genomes,
            environments: BTreeMap::new(),
            direct_completed: BTreeSet::new(),
            pending_assignments: BTreeSet::new(),
            pending_evidence: BTreeMap::new(),
            infrastructure_blocked: BTreeSet::new(),
            archive: BTreeMap::new(),
            pair_histories: BTreeMap::new(),
            environment_evidence: BTreeMap::new(),
            proposal_artifact_sha256: None,
            proposal_scores_sha256: None,
            proposal_scores: BTreeMap::new(),
            evaluations: vec![],
        })
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.format != STATE_FORMAT
            || self.descriptor_version != DESCRIPTOR_VERSION
            || self.quality_version != QUALITY_VERSION
            || digest(&self.config)? != self.config_sha256
        {
            return Err("search identity differs".into());
        }
        self.config.validate()?;
        for g in self.genomes.values() {
            g.validate(&self.config)?;
            if g.parents
                .iter()
                .any(|parent| parent == &g.sha256 || !self.genomes.contains_key(parent))
            {
                return Err("saved genome ancestry differs".into());
            }
        }
        for (identity, environment) in &self.environments {
            if identity != &environment.sha256
                || environment.compute_hash()? != environment.sha256
                || environment.epoch != self.config.environment_epoch
            {
                return Err("saved environment identity differs".into());
            }
        }
        for (key, history) in &self.pair_histories {
            if key
                != &format!(
                    "{}:{}",
                    history.candidate_sha256, history.environment_sha256
                )
                || !self.genomes.contains_key(&history.candidate_sha256)
                || !self.environments.contains_key(&history.environment_sha256)
                || !finite(history.quality_sum)
            {
                return Err("saved candidate-environment history differs".into());
            }
        }
        for (identity, evidence) in &self.environment_evidence {
            if identity != &evidence.environment_sha256
                || !self.environments.contains_key(identity)
                || !finite(evidence.quality_sum)
                || !finite(evidence.quality_square_sum)
            {
                return Err("saved environment evidence differs".into());
            }
        }
        if self.environment_evidence.len() != self.environments.len() {
            return Err("environment evidence coverage differs".into());
        }
        if self.pending_assignments.len() != self.pending_evidence.len()
            || self
                .pending_assignments
                .iter()
                .any(|key| !self.pending_evidence.contains_key(key))
            || self.pending_evidence.iter().any(|(key, evidence)| {
                !key.contains(':')
                    || !finite(evidence.transfer_gap)
                    || !finite(evidence.environment_priority)
                    || evidence
                        .analyst_score
                        .is_some_and(|score| !finite(score) || score.abs() > 1.0)
            })
        {
            return Err("pending assignment evidence differs".into());
        }
        if self.infrastructure_blocked.iter().any(|key| {
            self.pair_histories
                .get(key)
                .is_none_or(|history| history.infrastructure_failures == 0)
        }) {
            return Err("infrastructure retry block differs".into());
        }
        for (niche, members) in &self.archive {
            if members.len() > self.config.archive_members_per_cell
                || members.iter().any(|member| {
                    !niche.starts_with(&format!("{}:", member.environment_sha256))
                        || !self.genomes.contains_key(&member.candidate_sha256)
                        || !self.environments.contains_key(&member.environment_sha256)
                        || !finite(member.quality)
                })
            {
                return Err("saved niche archive differs".into());
            }
        }
        if self
            .proposal_artifact_sha256
            .as_deref()
            .is_some_and(|x| !valid_hash(x))
            || self.proposal_artifact_sha256.is_some() != self.proposal_scores_sha256.is_some()
            || self.proposal_artifact_sha256.is_some() != !self.proposal_scores.is_empty()
            || self.proposal_scores.iter().any(|(key, score)| {
                let Some((candidate, environment)) = key.split_once(':') else {
                    return true;
                };
                !finite(*score)
                    || score.abs() > 1.0
                    || !self.genomes.contains_key(candidate)
                    || !self.environments.contains_key(environment)
            })
        {
            return Err("saved analyst proposal scores differ".into());
        }
        if self
            .proposal_scores_sha256
            .as_deref()
            .is_some_and(|value| !valid_hash(value))
        {
            return Err("saved analyst proposal receipt differs".into());
        }
        Ok(())
    }
    pub fn register_environment(&mut self, e: EnvironmentGenome) -> Result<(), String> {
        let descriptor_names = BTreeSet::from([
            "regional_scale".to_string(),
            "elevation_relief".to_string(),
            "resource_density".to_string(),
            "renewal_rate".to_string(),
            "connectivity".to_string(),
        ]);
        if e.format != "chreatures-environment-record-v2"
            || !valid_hash(&e.sha256)
            || !valid_hash(&e.genome_sha256)
            || e.parents.len() > 2
            || e.genome_parents.len() != e.parents.len()
            || e.parents.iter().any(|x| !valid_hash(x))
            || e.genome_parents.iter().any(|x| !valid_hash(x))
            || ![
                &e.topology_sha256,
                &e.resource_sha256,
                &e.profile_sha256,
                &e.variation.recipe_sha256,
            ]
            .iter()
            .all(|x| valid_hash(x))
            || e.descriptors.keys().cloned().collect::<BTreeSet<_>>() != descriptor_names
            || e.descriptors
                .values()
                .any(|value| !finite(*value) || !(0.0..=1.0).contains(value))
            || !finite(e.generation_cost.normalized)
            || !(0.0..=1.0).contains(&e.generation_cost.normalized)
        {
            return Err("invalid environment identity".into());
        }
        if e.epoch != self.config.environment_epoch {
            return Err("environment belongs to a different search epoch".into());
        }
        for (parent_record, parent_genome) in e.parents.iter().zip(&e.genome_parents) {
            let Some(registered_parent) = self.environments.get(parent_record) else {
                return Err("environment ancestry is unavailable or cyclic".into());
            };
            if parent_record == &e.sha256 || &registered_parent.genome_sha256 != parent_genome {
                return Err("environment record and genome ancestry differ".into());
            }
        }
        if e.compute_hash()? != e.sha256 {
            return Err("environment content hash differs".into());
        }
        let identity = e.sha256.clone();
        self.environment_evidence
            .entry(identity.clone())
            .or_insert_with(|| EnvironmentEvidence {
                environment_sha256: identity.clone(),
                valid_evaluations: 0,
                infrastructure_failures: 0,
                organism_terminals: 0,
                quality_sum: 0.0,
                quality_square_sum: 0.0,
                candidate_sha256: BTreeSet::new(),
                occupied_cells: BTreeSet::new(),
                last_sequence: 0,
            });
        self.environments.insert(e.sha256.clone(), e);
        Ok(())
    }
    pub fn register_proposal_scores(&mut self, scores: ProposalScores) -> Result<(), String> {
        if scores.format != "chreatures-population-challenge-scores-v1"
            || !valid_hash(&scores.sha256)
            || !valid_hash(&scores.artifact_sha256)
            || scores.scores.is_empty()
            || scores
                .scores
                .values()
                .any(|value| !finite(*value) || value.abs() > 1.0)
        {
            return Err("invalid analyst proposal scores".into());
        }
        let mut body = scores.clone();
        body.sha256.clear();
        if digest(&body)? != scores.sha256 {
            return Err("analyst proposal score content hash differs".into());
        }
        for key in scores.scores.keys() {
            let Some((candidate, environment)) = key.split_once(':') else {
                return Err("analyst proposal pair key differs".into());
            };
            if !self.genomes.contains_key(candidate) || !self.environments.contains_key(environment)
            {
                return Err("analyst proposal references unknown pair".into());
            }
        }
        self.proposal_artifact_sha256 = Some(scores.artifact_sha256);
        self.proposal_scores_sha256 = Some(scores.sha256);
        self.proposal_scores = scores.scores;
        Ok(())
    }
    pub fn authorize_infrastructure_retry(
        &mut self,
        candidate: &str,
        environment: &str,
    ) -> Result<(), String> {
        let key = format!("{candidate}:{environment}");
        if !self.genomes.contains_key(candidate)
            || !self.environments.contains_key(environment)
            || !self.infrastructure_blocked.remove(&key)
        {
            return Err("infrastructure retry pair is not blocked".into());
        }
        Ok(())
    }
    fn candidate_robustness(&self, candidate: &str) -> (usize, f64) {
        let means: Vec<_> = self
            .pair_histories
            .values()
            .filter(|history| {
                history.candidate_sha256 == candidate && history.valid_evaluations > 0
            })
            .map(|history| history.quality_sum / history.valid_evaluations as f64)
            .collect();
        let worst = means
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(f64::NEG_INFINITY);
        (means.len(), worst)
    }
    fn assignment_evidence(&self, candidate: &str, environment: &str) -> AssignmentEvidence {
        let key = format!("{candidate}:{environment}");
        let history = self.pair_histories.get(&key);
        let pair_mean = history
            .filter(|item| item.valid_evaluations > 0)
            .map_or(0.0, |item| item.quality_sum / item.valid_evaluations as f64);
        AssignmentEvidence {
            pair_valid_evaluations: history.map_or(0, |item| item.valid_evaluations),
            candidate_valid_environments: self.candidate_robustness(candidate).0,
            environment_valid_evaluations: self
                .environment_evidence
                .get(environment)
                .map_or(0, |item| item.valid_evaluations),
            transfer_gap: self.candidate_mean_quality(candidate) - pair_mean,
            environment_priority: self.environment_priority(environment),
            analyst_score: self.proposal_scores.get(&key).copied(),
        }
    }
    fn parents(&self) -> Vec<Genome> {
        // One representative per environment-conditioned niche comes first;
        // additional members follow by rank. This prevents a globally high
        // quality niche from exhausting the reproductive pool.
        let depth = self.archive.values().map(Vec::len).max().unwrap_or(0);
        let mut ordered = Vec::new();
        for rank in 0..depth {
            let mut layer = Vec::new();
            for members in self.archive.values() {
                if let Some(member) = members.get(rank) {
                    layer.push(member)
                }
            }
            layer.sort_by(|a, b| {
                let a_robust = self.candidate_robustness(&a.candidate_sha256);
                let b_robust = self.candidate_robustness(&b.candidate_sha256);
                b_robust
                    .0
                    .cmp(&a_robust.0)
                    .then_with(|| b_robust.1.total_cmp(&a_robust.1))
                    .then(a.candidate_sha256.cmp(&b.candidate_sha256))
            });
            ordered.extend(layer);
        }
        let hashes: Vec<_> = if ordered.is_empty() {
            self.genomes.keys().cloned().collect()
        } else {
            let mut seen = BTreeSet::new();
            ordered
                .iter()
                .filter_map(|member| {
                    seen.insert(member.candidate_sha256.clone())
                        .then(|| member.candidate_sha256.clone())
                })
                .collect()
        };
        hashes
            .into_iter()
            .filter_map(|h| self.genomes.get(&h).cloned())
            .collect()
    }
    pub fn ask(&mut self, n: usize) -> Result<Vec<Assignment>, String> {
        self.validate()?;
        if n == 0 || n > 4096 {
            return Err("ask count outside 1..4096".into());
        }
        if self.environments.is_empty() {
            return Err("no environments registered".into());
        }
        let pool = self.parents();
        let envs: Vec<_> = self.environments.keys().cloned().collect();
        let mut rng = Rng(self.rng_state);
        let mut out = vec![];
        for _ in 0..n {
            let p1 = &pool[rng.index(pool.len())];
            let use_two = pool.len() > 1 && rng.unit() < 0.35;
            let p2 = if use_two {
                Some(&pool[rng.index(pool.len())])
            } else {
                None
            };
            let seed = rng.next();
            let mut local = Rng(seed);
            let mut values = BTreeMap::new();
            let mut mutated = vec![];
            for spec in &self.config.parameter_specs {
                let mut v = *p1.values.get(&spec.name).unwrap();
                if let Some(other) = p2 {
                    if local.unit() < 0.5 {
                        v = *other.values.get(&spec.name).unwrap()
                    }
                    if !spec.integer && local.unit() < 0.2 {
                        v = 0.5 * (v + other.values[&spec.name])
                    }
                }
                if spec.low < spec.high && local.unit() < 0.35 {
                    v = if spec.integer {
                        (v + if local.unit() < 0.5 { -1.0 } else { 1.0 }).clamp(spec.low, spec.high)
                    } else {
                        reflected(
                            v + local.normal() * spec.mutation_sigma,
                            spec.low,
                            spec.high,
                        )
                    };
                    mutated.push(spec.name.clone())
                }
                values.insert(spec.name.clone(), v);
            }
            let simplex_groups: BTreeSet<_> = self
                .config
                .parameter_specs
                .iter()
                .filter(|p| p.group.starts_with("simplex:"))
                .map(|p| p.group.clone())
                .collect();
            for group in simplex_groups {
                let specs: Vec<_> = self
                    .config
                    .parameter_specs
                    .iter()
                    .filter(|p| p.group == group)
                    .collect();
                let total: f64 = specs.iter().map(|p| values[&p.name].max(1e-12)).sum();
                for spec in specs {
                    values.insert(spec.name.clone(), values[&spec.name].max(1e-12) / total);
                }
            }
            if mutated.is_empty() {
                let mutable: Vec<_> = self
                    .config
                    .parameter_specs
                    .iter()
                    .filter(|spec| spec.low < spec.high)
                    .collect();
                let spec = mutable[local.index(mutable.len())];
                let v = if spec.integer {
                    (values[&spec.name] + if local.unit() < 0.5 { -1.0 } else { 1.0 })
                        .clamp(spec.low, spec.high)
                } else {
                    reflected(
                        values[&spec.name] + local.normal() * spec.mutation_sigma,
                        spec.low,
                        spec.high,
                    )
                };
                values.insert(spec.name.clone(), v);
                mutated.push(spec.name.clone())
            }
            let parents = if let Some(x) = p2 {
                if x.sha256 == p1.sha256 {
                    vec![p1.sha256.clone()]
                } else {
                    vec![p1.sha256.clone(), x.sha256.clone()]
                }
            } else {
                vec![p1.sha256.clone()]
            };
            let mut g = Genome {
                format: GENOME_FORMAT.into(),
                sha256: String::new(),
                parents,
                graph_sha256: self.config.graph_sha256.clone(),
                port_spec_sha256: self.config.port_spec_sha256.clone(),
                base_controller_sha256: self.config.base_controller_sha256.clone(),
                developmental_base_sha256: self.config.developmental_base_sha256.clone(),
                population_adapter_bank_sha256: self.config.population_adapter_bank_sha256.clone(),
                organism_interface_sha256: self.config.organism_interface_sha256.clone(),
                policy_adapter_count: self.config.policy_adapter_count,
                policy_adapter_rank: self.config.policy_adapter_rank,
                values,
                variation: Variation {
                    operator: VARIATION_VERSION.into(),
                    seed,
                    recipe_sha256: self.config.variation_recipe_sha256.clone(),
                    mutated,
                },
            };
            g.sha256 = g.compute_hash()?;
            self.genomes.insert(g.sha256.clone(), g.clone());
            let env = envs[(self.environment_cursor as usize) % envs.len()].clone();
            self.environment_cursor += 1;
            let key = format!("{}:{env}", g.sha256);
            let phase = if self.direct_completed.contains(&key) {
                "eligible-fine-tune"
            } else {
                "direct-transfer"
            };
            let selection = self.assignment_evidence(&g.sha256, &env);
            self.pending_assignments.insert(key.clone());
            self.pending_evidence.insert(key, selection.clone());
            out.push(Assignment {
                candidate: g,
                environment_sha256: env,
                phase: phase.into(),
                selection: selection.clone(),
            });
            self.ask_count += 1;
        }
        self.rng_state = rng.0;
        Ok(out)
    }
    pub fn ask_transfers(&mut self, n: usize) -> Result<Vec<Assignment>, String> {
        self.validate()?;
        if n == 0 || n > 4096 {
            return Err("transfer ask count outside 1..4096".into());
        }
        let mut candidate_hashes: Vec<_> = self
            .archive
            .values()
            .flatten()
            .map(|member| member.candidate_sha256.clone())
            .collect();
        candidate_hashes.sort();
        candidate_hashes.dedup();
        let environments: Vec<_> = self.environments.keys().cloned().collect();
        if n % environments.len() != 0 {
            return Err("transfer ask must request complete environment waves".into());
        }
        let mut proposed = Vec::new();
        let mut counts = BTreeMap::<String, usize>::new();
        let mut balanced_len = 0;
        for candidate_sha256 in candidate_hashes {
            for environment_sha256 in &environments {
                let key = format!("{candidate_sha256}:{environment_sha256}");
                if self.direct_completed.contains(&key)
                    || self.pending_assignments.contains(&key)
                    || self.infrastructure_blocked.contains(&key)
                {
                    continue;
                }
                let selection = self.assignment_evidence(&candidate_sha256, environment_sha256);
                proposed.push(Assignment {
                    candidate: self.genomes[&candidate_sha256].clone(),
                    environment_sha256: (*environment_sha256).clone(),
                    phase: "direct-transfer".into(),
                    selection: selection.clone(),
                });
                *counts.entry((*environment_sha256).clone()).or_default() += 1;
            }
            let first = counts.get(&environments[0]).copied().unwrap_or(0);
            if proposed.len() <= n
                && first > 0
                && environments
                    .iter()
                    .all(|environment| counts.get(environment).copied().unwrap_or(0) == first)
            {
                balanced_len = proposed.len();
            }
            if proposed.len() >= n {
                break;
            }
        }
        proposed.truncate(balanced_len);
        for assignment in &proposed {
            let key = format!(
                "{}:{}",
                assignment.candidate.sha256, assignment.environment_sha256
            );
            self.pending_assignments.insert(key.clone());
            self.pending_evidence
                .insert(key, assignment.selection.clone());
        }
        Ok(proposed)
    }
    fn candidate_mean_quality(&self, candidate: &str) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for history in self.pair_histories.values() {
            if history.candidate_sha256 == candidate && history.valid_evaluations > 0 {
                sum += history.quality_sum;
                count += history.valid_evaluations;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
    fn environment_priority(&self, environment: &str) -> f64 {
        let Some(evidence) = self.environment_evidence.get(environment) else {
            return 0.0;
        };
        let novelty = self.environment_novelty(environment);
        let cost = self.environments[environment].generation_cost.normalized;
        if evidence.valid_evaluations == 0 {
            return 1.0 + self.config.environment_novelty_weight * novelty
                - self.config.environment_cost_weight * cost;
        }
        let count = evidence.valid_evaluations as f64;
        let mean = evidence.quality_sum / count;
        let variance = (evidence.quality_square_sum / count - mean * mean).max(0.0);
        // Difficulty and discrimination both matter. Infrastructure failures
        // are deliberately absent: they are execution evidence, not ecology.
        evidence.organism_terminals as f64 / count
            + variance.sqrt()
            + (-mean).max(0.0)
            + self.config.environment_novelty_weight * novelty
            - self.config.environment_cost_weight * cost
    }
    fn environment_novelty(&self, environment: &str) -> f64 {
        let current = &self.environments[environment].descriptors;
        self.environments
            .iter()
            .filter(|(identity, _)| identity.as_str() != environment)
            .map(|(_, other)| {
                let squared: f64 = current
                    .iter()
                    .map(|(name, value)| (value - other.descriptors[name]).powi(2))
                    .sum();
                (squared / current.len() as f64).sqrt()
            })
            .reduce(f64::min)
            .unwrap_or(1.0)
    }
    pub fn ask_challenges(&mut self, n: usize) -> Result<Vec<Assignment>, String> {
        self.validate()?;
        let environments: Vec<_> = self.environments.keys().cloned().collect();
        if n == 0 || n > 4096 || environments.is_empty() || n % environments.len() != 0 {
            return Err(
                "challenge ask must request complete environment waves within 1..4096".into(),
            );
        }
        let slots = n / environments.len();
        let mut candidates: Vec<_> = self
            .archive
            .values()
            .flatten()
            .map(|member| member.candidate_sha256.clone())
            .collect();
        candidates.sort();
        candidates.dedup();
        if candidates.len() < slots {
            return Err("challenge archive has too few distinct candidates".into());
        }
        let mut proposed = Vec::with_capacity(n);
        for environment in environments {
            let environment_priority = self.environment_priority(&environment);
            let cursor = self.challenge_cursor;
            let mut ranked: Vec<_> = candidates
                .iter()
                .filter(|candidate| {
                    let key = format!("{candidate}:{environment}");
                    !self.pending_assignments.contains(&key)
                        && !self.infrastructure_blocked.contains(&key)
                })
                .map(|candidate| {
                    let key = format!("{candidate}:{environment}");
                    let history = self.pair_histories.get(&key);
                    let repeats = history.map_or(0, |item| item.valid_evaluations);
                    let pair_mean = history
                        .filter(|item| item.valid_evaluations > 0)
                        .map_or(0.0, |item| item.quality_sum / item.valid_evaluations as f64);
                    let gap = self.candidate_mean_quality(candidate) - pair_mean;
                    let analyst = self.proposal_scores.get(&key).copied().unwrap_or(0.0);
                    let tie = hex(&Sha256::digest(format!("{cursor}:{key}").as_bytes()));
                    (
                        candidate.clone(),
                        repeats,
                        analyst,
                        environment_priority + gap,
                        tie,
                    )
                })
                .collect();
            ranked.sort_by(|a, b| {
                a.1.cmp(&b.1)
                    .then_with(|| b.2.total_cmp(&a.2))
                    .then_with(|| b.3.total_cmp(&a.3))
                    .then_with(|| a.4.cmp(&b.4))
            });
            if ranked.len() < slots {
                return Err("challenge assignments collide with pending work".into());
            }
            for (candidate, ..) in ranked.into_iter().take(slots) {
                let key = format!("{candidate}:{environment}");
                let selection = self.assignment_evidence(&candidate, &environment);
                proposed.push(Assignment {
                    candidate: self.genomes[&candidate].clone(),
                    environment_sha256: environment.clone(),
                    phase: if self.pair_histories.contains_key(&key) {
                        "history-challenge-repeat".into()
                    } else {
                        "history-challenge-transfer".into()
                    },
                    selection: selection.clone(),
                });
                self.pending_assignments.insert(key.clone());
                self.pending_evidence.insert(key, selection);
            }
            self.challenge_cursor += 1;
        }
        Ok(proposed)
    }
    pub fn environment_frontier(&self) -> Vec<EnvironmentFrontier> {
        let mut values: Vec<_> = self
            .environments
            .values()
            .map(|environment| {
                let evidence = &self.environment_evidence[&environment.sha256];
                let (mean, stddev) = if evidence.valid_evaluations == 0 {
                    (None, None)
                } else {
                    let count = evidence.valid_evaluations as f64;
                    let mean = evidence.quality_sum / count;
                    let variance = (evidence.quality_square_sum / count - mean * mean).max(0.0);
                    (Some(mean), Some(variance.sqrt()))
                };
                EnvironmentFrontier {
                    environment_sha256: environment.sha256.clone(),
                    parents: environment.parents.clone(),
                    valid_evaluations: evidence.valid_evaluations,
                    infrastructure_failures: evidence.infrastructure_failures,
                    organism_terminals: evidence.organism_terminals,
                    distinct_candidates: evidence.candidate_sha256.len(),
                    occupied_cells: evidence.occupied_cells.len(),
                    mean_quality: mean,
                    quality_stddev: stddev,
                    challenge_priority: self.environment_priority(&environment.sha256),
                    novelty: self.environment_novelty(&environment.sha256),
                    descriptors: environment.descriptors.clone(),
                    generation_cost: environment.generation_cost.clone(),
                }
            })
            .collect();
        values.sort_by(|a, b| {
            b.challenge_priority
                .total_cmp(&a.challenge_priority)
                .then(a.environment_sha256.cmp(&b.environment_sha256))
        });
        values
    }
    pub fn tell(&mut self, input: EvaluationInput) -> Result<Evaluation, String> {
        self.validate()?;
        if !valid_hash(&input.evaluation_sha256) {
            return Err("evaluation SHA-256 differs".into());
        }
        if !valid_hash(&input.life_id) {
            return Err("evaluation life identity differs".into());
        }
        if !valid_hash(&input.trajectory_sha256) {
            return Err("evaluation trajectory identity differs".into());
        }
        let valid_trajectory = input.status == "completed" || input.status == "organism-terminal";
        if input.committed_ticks == 0 && valid_trajectory {
            return Err("valid ecological evaluation has no committed ticks".into());
        }
        if !self.genomes.contains_key(&input.candidate_sha256) {
            return Err("evaluation candidate is unknown".into());
        }
        if !self.environments.contains_key(&input.environment_sha256) {
            return Err("evaluation environment is unknown".into());
        }
        if input.compute_hash()? != input.evaluation_sha256 {
            return Err("evaluation content hash differs".into());
        }
        if self
            .evaluations
            .iter()
            .any(|e| e.evaluation_sha256 == input.evaluation_sha256)
        {
            return Err("duplicate evaluation".into());
        }
        if input.metrics.values().any(|v| !finite(*v)) {
            return Err("nonfinite metric".into());
        }
        if !valid_trajectory && input.status != "infrastructure-failure" {
            return Err("evaluation status differs".into());
        }
        if !valid_trajectory && !input.metrics.is_empty() {
            return Err("infrastructure failure cannot carry ecological metrics".into());
        }
        if (input.status == "infrastructure-failure" && input.failure.is_empty())
            || (input.status == "completed" && !input.failure.is_empty())
        {
            return Err("evaluation failure detail differs".into());
        }
        let mut descriptor = vec![];
        let mut cell = vec![];
        let mut quality = 0.0;
        if valid_trajectory {
            for a in &self.config.descriptor_axes {
                let v = *input
                    .metrics
                    .get(&a.component)
                    .ok_or("descriptor component missing")?;
                descriptor.push(v);
                let t = ((v - a.low) / (a.high - a.low)).clamp(0.0, 1.0 - f64::EPSILON);
                cell.push((t * a.bins as f64).floor() as u32)
            }
            for q in &self.config.quality_terms {
                let v = *input
                    .metrics
                    .get(&q.component)
                    .ok_or("quality component missing")?;
                quality += q.weight * q.direction * (v / q.scale).clamp(-1.0, 1.0)
            }
        } else {
            quality = 0.0;
            descriptor = vec![0.0; self.config.descriptor_axes.len()];
            cell = vec![0; descriptor.len()]
        }
        let assignment_key = format!("{}:{}", input.candidate_sha256, input.environment_sha256);
        if !self.pending_assignments.contains(&assignment_key) {
            return Err("evaluation has no pending candidate-environment assignment".into());
        }
        self.pending_assignments.remove(&assignment_key);
        self.pending_evidence.remove(&assignment_key);
        if valid_trajectory {
            self.direct_completed.insert(assignment_key);
            self.infrastructure_blocked.remove(&format!(
                "{}:{}",
                input.candidate_sha256, input.environment_sha256
            ));
        }
        let key = cell
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(":");
        let mut retained = false;
        if valid_trajectory {
            let prior_valid = self
                .pair_histories
                .get(&format!(
                    "{}:{}",
                    input.candidate_sha256, input.environment_sha256
                ))
                .map_or(0, |history| history.valid_evaluations);
            for members in self.archive.values_mut() {
                for member in members
                    .iter_mut()
                    .filter(|member| member.candidate_sha256 == input.candidate_sha256)
                {
                    member.valid_evaluation_count = prior_valid + 1;
                }
            }
            let niche_key = format!("{}:{key}", input.environment_sha256);
            let members = self.archive.entry(niche_key).or_default();
            members.push(ArchiveMember {
                candidate_sha256: input.candidate_sha256.clone(),
                evaluation_sha256: input.evaluation_sha256.clone(),
                quality,
                descriptor: descriptor.clone(),
                environment_sha256: input.environment_sha256.clone(),
                valid_evaluation_count: prior_valid + 1,
            });
            members.sort_by(|a, b| {
                b.quality
                    .total_cmp(&a.quality)
                    .then(a.candidate_sha256.cmp(&b.candidate_sha256))
            });
            members.dedup_by(|a, b| a.candidate_sha256 == b.candidate_sha256);
            members.truncate(self.config.archive_members_per_cell);
            retained = members
                .iter()
                .any(|m| m.evaluation_sha256 == input.evaluation_sha256)
        }
        self.evaluation_sequence += 1;
        let pair_key = format!("{}:{}", input.candidate_sha256, input.environment_sha256);
        let history = self
            .pair_histories
            .entry(pair_key.clone())
            .or_insert_with(|| PairHistory {
                candidate_sha256: input.candidate_sha256.clone(),
                environment_sha256: input.environment_sha256.clone(),
                valid_evaluations: 0,
                infrastructure_failures: 0,
                organism_terminals: 0,
                quality_sum: 0.0,
                best_quality: None,
                last_evaluation_sha256: String::new(),
                last_sequence: 0,
            });
        if valid_trajectory {
            history.valid_evaluations += 1;
            history.quality_sum += quality;
            history.best_quality =
                Some(history.best_quality.map_or(quality, |old| old.max(quality)));
            if input.status == "organism-terminal" {
                history.organism_terminals += 1;
            }
        } else {
            history.infrastructure_failures += 1;
            self.infrastructure_blocked.insert(pair_key.clone());
        }
        history.last_evaluation_sha256 = input.evaluation_sha256.clone();
        history.last_sequence = self.evaluation_sequence;
        let evidence = self
            .environment_evidence
            .get_mut(&input.environment_sha256)
            .unwrap();
        if valid_trajectory {
            evidence.valid_evaluations += 1;
            evidence.quality_sum += quality;
            evidence.quality_square_sum += quality * quality;
            evidence
                .candidate_sha256
                .insert(input.candidate_sha256.clone());
            evidence.occupied_cells.insert(key);
            if input.status == "organism-terminal" {
                evidence.organism_terminals += 1;
            }
        } else {
            evidence.infrastructure_failures += 1;
        }
        evidence.last_sequence = self.evaluation_sequence;
        let e = Evaluation {
            evaluation_sha256: input.evaluation_sha256,
            life_id: input.life_id,
            evaluation_seed: input.evaluation_seed,
            committed_ticks: input.committed_ticks,
            trajectory_sha256: input.trajectory_sha256,
            candidate_sha256: input.candidate_sha256,
            environment_sha256: input.environment_sha256,
            status: input.status,
            failure: input.failure,
            metrics: input.metrics,
            descriptor: if valid_trajectory {
                Some(descriptor)
            } else {
                None
            },
            cell: if valid_trajectory { Some(cell) } else { None },
            quality: if valid_trajectory {
                Some(quality)
            } else {
                None
            },
            archive_retained: retained,
        };
        self.evaluations.push(e.clone());
        Ok(e)
    }
}
