use std::{
    collections::{HashMap, HashSet, hash_map::DefaultHasher},
    env,
    error::Error,
    fmt, fs,
    hash::BuildHasherDefault,
    path::PathBuf,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use serde_json::Value;
use sha2::{Digest, Sha256};
use universal_weave::{
    IndependentContents, MetadataWeave, Weave,
    independent::{IndependentNode, IndependentWeave},
    indexmap::IndexSet,
};

const SOURCE_COMMIT: &str = "7a5a0dabb94885e44ad8a6c4355c015d7f38020f";

/// Universal Weave requires Copy identifiers. This is the first 128 bits of
/// SHA-256(source_id), rendered as hex so JSON consumers never lose precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StableNodeId([u8; 16]);

impl StableNodeId {
    fn from_source_id(source_id: &str) -> Self {
        let digest = Sha256::digest(source_id.as_bytes());
        let mut bytes = [0_u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        Self(bytes)
    }

    fn parse(value: &str) -> Result<Self, String> {
        if value.len() != 32
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("stable node id must be 32 lowercase hexadecimal characters".to_owned());
        }
        let mut bytes = [0_u8; 16];
        for (index, target) in bytes.iter_mut().enumerate() {
            *target = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
                .map_err(|_| "invalid stable node id")?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for StableNodeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for StableNodeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for StableNodeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(&String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BlobRef {
    role: String,
    uri: String,
    sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verification: Option<String>,
}

impl BlobRef {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.role.trim().is_empty() {
            return Err("blob ref role must be nonempty".into());
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("blob ref {} has invalid lowercase SHA-256", self.role).into());
        }
        if self.uri != format!("urn:sha256:{}", self.sha256) {
            return Err(format!("blob ref {} URI does not match its SHA-256", self.role).into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EvidenceRecord {
    source_id: String,
    time: Value,
    record_type: String,
    text: String,
    artifact_uri: Option<String>,
    #[serde(default)]
    blob_refs: Vec<BlobRef>,
    source: Value,
}

impl IndependentContents for EvidenceRecord {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArchiveMetadata {
    schema_version: u32,
    library: String,
    source_commit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evidence_schema: Option<String>,
    #[serde(default)]
    archive_id: Option<String>,
    habitat_id: Option<String>,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ImportRequest {
    #[serde(default)]
    evidence_schema: Option<String>,
    #[serde(default)]
    archive_id: Option<String>,
    #[serde(default)]
    habitat_id: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    journal: Vec<Value>,
    #[serde(default)]
    evidence: Vec<ImportedEvidence>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImportedEvidence {
    id: String,
    #[serde(default)]
    time: Value,
    #[serde(default = "default_evidence_type")]
    record_type: String,
    text: String,
    #[serde(default)]
    artifact_uri: Option<String>,
    #[serde(default)]
    blob_refs: Vec<BlobRef>,
    #[serde(default)]
    parent_ids: Vec<String>,
    #[serde(default)]
    fields: Value,
}

fn default_evidence_type() -> String {
    "evidence".to_owned()
}

#[derive(Debug)]
struct PendingRecord {
    source_id: String,
    parent_ids: Vec<String>,
    time: Value,
    record_type: String,
    text: String,
    artifact_uri: Option<String>,
    blob_refs: Vec<BlobRef>,
    source: Value,
}

type EvidenceWeave = IndependentWeave<StableNodeId, EvidenceRecord, ArchiveMetadata, StableHasher>;
type EvidenceNode = IndependentNode<StableNodeId, EvidenceRecord, StableHasher>;
type StableHasher = BuildHasherDefault<DefaultHasher>;

fn ids(values: &[StableNodeId]) -> IndexSet<StableNodeId, StableHasher> {
    values.iter().copied().collect()
}

fn node(id: StableNodeId, parents: &[StableNodeId], contents: EvidenceRecord) -> EvidenceNode {
    EvidenceNode {
        id,
        from: ids(parents),
        to: IndexSet::default(),
        active: true,
        bookmarked: matches!(
            contents.record_type.as_str(),
            "evidence" | "gam_law_fit" | "model_promotion" | "snapshot"
        ),
        contents,
    }
}

fn insert(weave: &mut EvidenceWeave, node: EvidenceNode) -> Result<(), Box<dyn Error>> {
    let id = node.id;
    if weave.insert(node) {
        Ok(())
    } else {
        Err(format!("Universal Weave rejected node {id}").into())
    }
}

fn text_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("journal record requires string {field}").into())
}

fn string_list(value: &Value, field: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let Some(raw) = value.get(field) else {
        return Ok(Vec::new());
    };
    let values = raw
        .as_array()
        .ok_or_else(|| format!("{field} must be an array of strings"))?;
    values
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{field} must contain only strings").into())
        })
        .collect()
}

fn journal_pending(source: Value) -> Result<PendingRecord, Box<dyn Error>> {
    let source_id = text_field(&source, "id")?.to_owned();
    let time = source
        .get("time")
        .cloned()
        .ok_or("journal record requires time")?;
    let text = text_field(&source, "text")?.to_owned();
    let parent_ids = string_list(&source, "parent_ids")?;
    let blob_refs: Vec<BlobRef> = match source.get("blob_refs") {
        Some(value) => serde_json::from_value(value.clone())?,
        None => Vec::new(),
    };
    let record_type = source
        .get("record_type")
        .or_else(|| source.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("episode")
        .to_owned();
    let artifact_uri = source
        .get("artifact_uri")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(PendingRecord {
        source_id,
        parent_ids,
        time,
        record_type,
        text,
        artifact_uri,
        blob_refs,
        source,
    })
}

fn evidence_pending(evidence: ImportedEvidence) -> Result<PendingRecord, Box<dyn Error>> {
    let source = serde_json::to_value(&evidence)?;
    Ok(PendingRecord {
        source_id: evidence.id,
        parent_ids: evidence.parent_ids,
        time: evidence.time,
        record_type: evidence.record_type,
        text: evidence.text,
        artifact_uri: evidence.artifact_uri,
        blob_refs: evidence.blob_refs,
        source,
    })
}

const POPULATION_SCHEMA: &str = "chreatures-population-evidence-v1";
const POPULATION_TYPES: &[&str] = &[
    "population_run",
    "descriptor_epoch",
    "environment_probe_panel",
    "genome_candidate",
    "environment_candidate",
    "birth",
    "research_branch",
    "life_checkpoint",
    "evaluation_completed",
    "evaluation_failed",
    "archive_decision",
    "transfer_trial",
    "population_snapshot",
    "gam_fit_attempt",
    "interpretation_correction",
    "embodied_recording",
    "organism_transfer",
    "development_event",
    "environment_event",
    "interaction_event",
];

fn population_fields<'a>(
    record: &'a ImportedEvidence,
) -> Result<&'a serde_json::Map<String, Value>, Box<dyn Error>> {
    record
        .fields
        .as_object()
        .ok_or_else(|| format!("population record {} fields must be an object", record.id).into())
}

fn population_roles<'a>(
    record: &'a ImportedEvidence,
) -> Result<&'a serde_json::Map<String, Value>, Box<dyn Error>> {
    population_fields(record)?
        .get("parent_roles")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "population record {} requires fields.parent_roles",
                record.id
            )
            .into()
        })
}

fn required_string<'a>(
    record: &'a ImportedEvidence,
    name: &str,
) -> Result<&'a str, Box<dyn Error>> {
    population_fields(record)?
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("population record {} requires string {name}", record.id).into())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn private_genome_field(value: &Value, path: &str) -> Option<String> {
    match value {
        Value::Object(values) => values.iter().find_map(|(key, child)| {
            let child_path = format!("{path}.{key}");
            let normalized = key.to_ascii_lowercase().replace('-', "_");
            if normalized.split('_').any(|token| {
                [
                    "state",
                    "memory",
                    "optimizer",
                    "rng",
                    "history",
                    "checkpoint",
                    "rates",
                ]
                .contains(&token)
            }) {
                Some(child_path)
            } else {
                private_genome_field(child, &child_path)
            }
        }),
        Value::Array(values) => values
            .iter()
            .enumerate()
            .find_map(|(index, child)| private_genome_field(child, &format!("{path}[{index}]"))),
        _ => None,
    }
}

fn required_sha256<'a>(
    record: &'a ImportedEvidence,
    name: &str,
) -> Result<&'a str, Box<dyn Error>> {
    let value = required_string(record, name)?;
    if !valid_sha256(value) {
        return Err(format!("population record {} has invalid {name}", record.id).into());
    }
    Ok(value)
}

fn required_u64(record: &ImportedEvidence, name: &str) -> Result<u64, Box<dyn Error>> {
    population_fields(record)?
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("population record {} requires integer {name}", record.id).into())
}

fn require_blob_role(record: &ImportedEvidence, role: &str) -> Result<(), Box<dyn Error>> {
    let count = record
        .blob_refs
        .iter()
        .filter(|blob| blob.role == role)
        .count();
    if count != 1 {
        return Err(format!(
            "population record {} requires exactly one {role} blob",
            record.id
        )
        .into());
    }
    Ok(())
}

fn population_parents_for_role(
    record: &ImportedEvidence,
    role: &str,
) -> Result<Vec<String>, Box<dyn Error>> {
    let roles = population_roles(record)?;
    Ok(record
        .parent_ids
        .iter()
        .filter(|parent| roles.get(*parent).and_then(Value::as_str) == Some(role))
        .cloned()
        .collect())
}

fn population_parent_for_role(
    record: &ImportedEvidence,
    role: &str,
) -> Result<String, Box<dyn Error>> {
    let parents = population_parents_for_role(record, role)?;
    if parents.len() != 1 {
        return Err(format!(
            "population record {} requires exactly one parent for role {role}",
            record.id
        )
        .into());
    }
    Ok(parents[0].clone())
}

fn role_rule(
    record_type: &str,
    role: &str,
) -> Option<(&'static [&'static str], usize, Option<usize>)> {
    match (record_type, role) {
        ("descriptor_epoch", "campaign") => Some((&["population_run"], 1, Some(1))),
        ("descriptor_epoch", "previous_descriptor_epoch") => {
            Some((&["descriptor_epoch"], 0, Some(1)))
        }
        ("environment_probe_panel", "campaign") => Some((&["population_run"], 1, Some(1))),
        ("environment_probe_panel", "descriptor_epoch") => {
            Some((&["descriptor_epoch"], 1, Some(1)))
        }
        ("genome_candidate", "campaign") => Some((&["population_run"], 1, Some(1))),
        ("genome_candidate", "genome_parent") => Some((&["genome_candidate"], 0, Some(2))),
        ("genome_candidate", "inherited_law_fit") => Some((&["gam_fit_attempt"], 0, None)),
        ("environment_candidate", "campaign") => Some((&["population_run"], 1, Some(1))),
        ("environment_candidate", "probe_panel") => {
            Some((&["environment_probe_panel"], 1, Some(1)))
        }
        ("environment_candidate", "environment_parent") => {
            Some((&["environment_candidate"], 0, Some(2)))
        }
        ("birth", "candidate_genome") => Some((&["genome_candidate"], 1, Some(1))),
        ("birth", "environment") => Some((&["environment_candidate"], 1, Some(1))),
        ("birth", "physical_parent_birth") => Some((&["birth", "research_branch"], 0, Some(2))),
        ("research_branch", "source_checkpoint") => Some((&["life_checkpoint"], 1, Some(1))),
        ("research_branch", "candidate_genome") => Some((&["genome_candidate"], 1, Some(1))),
        ("research_branch", "environment") => Some((&["environment_candidate"], 1, Some(1))),
        ("life_checkpoint", "life_continuation") => {
            Some((&["birth", "research_branch", "life_checkpoint"], 1, Some(1)))
        }
        ("evaluation_completed", "life_continuation") => {
            Some((&["birth", "life_checkpoint"], 1, Some(1)))
        }
        ("evaluation_failed", "life_continuation") => {
            Some((&["birth", "life_checkpoint"], 0, Some(1)))
        }
        ("evaluation_failed", "planned_campaign") => Some((&["population_run"], 0, Some(1))),
        ("evaluation_completed" | "evaluation_failed", "candidate_genome") => {
            Some((&["genome_candidate"], 1, Some(1)))
        }
        ("evaluation_completed" | "evaluation_failed", "environment") => {
            Some((&["environment_candidate"], 1, Some(1)))
        }
        ("evaluation_completed" | "evaluation_failed", "descriptor_epoch") => {
            Some((&["descriptor_epoch"], 1, Some(1)))
        }
        ("evaluation_completed" | "evaluation_failed", "probe_panel") => {
            Some((&["environment_probe_panel"], 1, Some(1)))
        }
        ("archive_decision", "evaluated_candidate") => {
            Some((&["evaluation_completed", "evaluation_failed"], 1, Some(1)))
        }
        ("archive_decision", "descriptor_epoch") => Some((&["descriptor_epoch"], 1, Some(1))),
        ("transfer_trial", "source_evaluation" | "target_evaluation") => {
            Some((&["evaluation_completed", "evaluation_failed"], 1, Some(1)))
        }
        ("transfer_trial", "candidate_genome") => Some((&["genome_candidate"], 1, Some(1))),
        ("transfer_trial", "target_environment") => Some((&["environment_candidate"], 1, Some(1))),
        ("transfer_trial", "probe_panel") => Some((&["environment_probe_panel"], 1, Some(1))),
        ("population_snapshot", "campaign") => Some((&["population_run"], 1, Some(1))),
        ("population_snapshot", "archive_decision") => Some((&["archive_decision"], 0, None)),
        ("gam_fit_attempt", "source_evaluation") => {
            Some((&["evaluation_completed", "evaluation_failed"], 1, None))
        }
        ("interpretation_correction", "corrected_record") => {
            Some((&["gam_fit_attempt"], 1, Some(1)))
        }
        ("embodied_recording", "campaign") => Some((&["population_run"], 1, Some(1))),
        ("embodied_recording", "observed_environment") => {
            Some((&["environment_candidate"], 1, Some(1)))
        }
        ("embodied_recording", "observed_life") => Some((
            &["birth", "life_checkpoint", "evaluation_completed"],
            1,
            None,
        )),
        ("embodied_recording", "associated_law_fit") => Some((&["gam_fit_attempt"], 0, None)),
        (
            "organism_transfer" | "development_event" | "environment_event" | "interaction_event",
            "recording",
        ) => Some((&["embodied_recording"], 1, Some(1))),
        (
            "organism_transfer" | "development_event" | "environment_event" | "interaction_event",
            "actor_life",
        ) => Some((
            &["birth", "life_checkpoint", "evaluation_completed"],
            0,
            None,
        )),
        (
            "organism_transfer" | "development_event" | "environment_event" | "interaction_event",
            "observed_environment",
        ) => Some((&["environment_candidate"], 1, Some(1))),
        (
            "organism_transfer" | "development_event" | "environment_event" | "interaction_event",
            "associated_law_fit",
        ) => Some((&["gam_fit_attempt"], 0, None)),
        _ => None,
    }
}

fn validate_population_evidence(records: &[ImportedEvidence]) -> Result<(), Box<dyn Error>> {
    if records.is_empty() {
        return Err("population evidence has no records".into());
    }
    let by_id: HashMap<_, _> = records
        .iter()
        .map(|record| (record.id.as_str(), record))
        .collect();
    if by_id.len() != records.len() {
        return Err("population evidence has duplicate record ids".into());
    }
    if records
        .iter()
        .filter(|record| record.record_type == "population_run")
        .count()
        != 1
    {
        return Err("population evidence requires exactly one population_run".into());
    }
    let mut terminal_evaluations = HashSet::new();
    let mut archive_decisions = HashSet::new();
    let mut continued = HashSet::new();
    let mut life_ids = HashSet::new();
    let mut descriptor_epoch_ids = HashSet::new();
    let mut probe_panel_ids = HashSet::new();
    let mut epoch_indices: HashMap<u64, &ImportedEvidence> = HashMap::new();
    let mut embodied_event_counts: HashMap<String, u64> = HashMap::new();

    for record in records {
        if !POPULATION_TYPES.contains(&record.record_type.as_str()) {
            return Err(
                format!("unsupported population record type {}", record.record_type).into(),
            );
        }
        let roles = population_roles(record)?;
        let blob_roles: HashSet<_> = record
            .blob_refs
            .iter()
            .map(|blob| blob.role.as_str())
            .collect();
        if blob_roles.len() != record.blob_refs.len() {
            return Err(format!("population record {} repeats a blob role", record.id).into());
        }
        let parent_set: HashSet<_> = record.parent_ids.iter().map(String::as_str).collect();
        let role_set: HashSet<_> = roles.keys().map(String::as_str).collect();
        if parent_set.len() != record.parent_ids.len() || parent_set != role_set {
            return Err(format!(
                "population record {} parent_roles must exactly match parent_ids",
                record.id
            )
            .into());
        }
        if record.record_type == "population_run" && !record.parent_ids.is_empty() {
            return Err("population_run cannot have parents".into());
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for (parent_id, role_value) in roles {
            let role = role_value.as_str().ok_or_else(|| {
                format!(
                    "population record {} parent role must be a string",
                    record.id
                )
            })?;
            let parent = by_id.get(parent_id.as_str()).ok_or_else(|| {
                format!(
                    "population record {} parent {parent_id} is absent",
                    record.id
                )
            })?;
            let Some((allowed_types, _, _)) = role_rule(&record.record_type, role) else {
                return Err(
                    format!("population record {} has invalid role {role}", record.id).into(),
                );
            };
            if !allowed_types.contains(&parent.record_type.as_str()) {
                return Err(format!(
                    "population record {} role {role} cannot target {}",
                    record.id, parent.record_type
                )
                .into());
            }
            *counts.entry(role).or_default() += 1;
        }
        let possible_roles = [
            "campaign",
            "previous_descriptor_epoch",
            "descriptor_epoch",
            "genome_parent",
            "inherited_law_fit",
            "probe_panel",
            "environment_parent",
            "candidate_genome",
            "environment",
            "physical_parent_birth",
            "source_checkpoint",
            "life_continuation",
            "planned_campaign",
            "evaluated_candidate",
            "source_evaluation",
            "target_evaluation",
            "target_environment",
            "archive_decision",
            "observed_environment",
            "observed_life",
            "associated_law_fit",
            "recording",
            "actor_life",
        ];
        for role in possible_roles {
            if let Some((_, minimum, maximum)) = role_rule(&record.record_type, role) {
                let count = *counts.get(role).unwrap_or(&0);
                if count < minimum || maximum.is_some_and(|high| count > high) {
                    return Err(format!(
                        "population record {} has invalid count for role {role}",
                        record.id
                    )
                    .into());
                }
            }
        }
        if let Some(parent) = population_parents_for_role(record, "life_continuation")?.first() {
            if !continued.insert(parent.clone()) {
                return Err(format!("life continuation branches at {parent}").into());
            }
            let parent_life = required_string(by_id[parent.as_str()], "life_id")?;
            if required_string(record, "life_id")? != parent_life {
                return Err(format!("population record {} changes life_id", record.id).into());
            }
        }

        match record.record_type.as_str() {
            "population_run" => {
                required_sha256(record, "search_config_sha256")?;
                require_blob_role(record, "search_config")?;
            }
            "descriptor_epoch" => {
                let descriptor_epoch_id = required_string(record, "descriptor_epoch_id")?;
                if !descriptor_epoch_ids.insert(descriptor_epoch_id.to_owned()) {
                    return Err(format!(
                        "descriptor epoch identity {descriptor_epoch_id} is duplicated"
                    )
                    .into());
                }
                required_sha256(record, "descriptor_recipe_sha256")?;
                require_blob_role(record, "descriptor_recipe")?;
                let index = population_fields(record)?
                    .get("descriptor_epoch_index")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| format!("descriptor epoch {} has invalid index", record.id))?;
                if epoch_indices.insert(index, record).is_some() {
                    return Err(format!("descriptor epoch index {index} is duplicated").into());
                }
            }
            "environment_probe_panel" => {
                let probe_panel_id = required_string(record, "probe_panel_id")?;
                if !probe_panel_ids.insert(probe_panel_id.to_owned()) {
                    return Err(
                        format!("probe panel identity {probe_panel_id} is duplicated").into(),
                    );
                }
                required_sha256(record, "probe_panel_sha256")?;
                require_blob_role(record, "probe_policy_panel")?;
                let policies = population_fields(record)?
                    .get("policy_artifact_sha256s")
                    .and_then(Value::as_array)
                    .filter(|values| !values.is_empty())
                    .ok_or_else(|| format!("probe panel {} has no policies", record.id))?;
                if policies
                    .iter()
                    .any(|value| !value.as_str().is_some_and(valid_sha256))
                {
                    return Err(
                        format!("probe panel {} has an invalid policy hash", record.id).into(),
                    );
                }
                let unique_policies: HashSet<_> =
                    policies.iter().filter_map(Value::as_str).collect();
                if unique_policies.len() != policies.len() {
                    return Err(format!("probe panel {} repeats a policy", record.id).into());
                }
                let epoch_id = population_parent_for_role(record, "descriptor_epoch")?;
                if required_string(record, "descriptor_epoch_id")?
                    != required_string(by_id[epoch_id.as_str()], "descriptor_epoch_id")?
                {
                    return Err(
                        format!("probe panel {} crosses descriptor epochs", record.id).into(),
                    );
                }
            }
            "genome_candidate" => {
                for name in [
                    "genome_sha256",
                    "graph_sha256",
                    "port_spec_sha256",
                    "base_controller_sha256",
                    "developmental_base_sha256",
                    "population_adapter_bank_sha256",
                    "organism_interface_sha256",
                    "variation_recipe_sha256",
                ] {
                    required_sha256(record, name)?;
                }
                if required_u64(record, "policy_adapter_count")? == 0
                    || required_u64(record, "policy_adapter_rank")? == 0
                {
                    return Err(format!("genome {} has invalid adapter shape", record.id).into());
                }
                require_blob_role(record, "genome_artifact")?;
                if let Some(path) = private_genome_field(&record.fields, "fields") {
                    return Err(format!(
                        "genome record {} crosses private-state boundary in {path}",
                        record.id
                    )
                    .into());
                }
                let actual: Vec<_> = population_parents_for_role(record, "genome_parent")?
                    .iter()
                    .map(|id| required_sha256(by_id[id.as_str()], "genome_sha256"))
                    .collect::<Result<_, _>>()?;
                let declared = population_fields(record)?
                    .get("parent_genome_sha256s")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("genome {} lacks parent_genome_sha256s", record.id))?;
                if declared.iter().map(Value::as_str).collect::<Vec<_>>()
                    != actual.iter().map(|value| Some(*value)).collect::<Vec<_>>()
                {
                    return Err(
                        format!("genome {} parent hashes differ from edges", record.id).into(),
                    );
                }
                let inherited = population_parents_for_role(record, "inherited_law_fit")?;
                let declared_inherited = population_fields(record)?
                    .get("inherited_law_fit_ids")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("genome {} lacks inherited_law_fit_ids", record.id))?;
                if declared_inherited
                    .iter()
                    .map(Value::as_str)
                    .collect::<Vec<_>>()
                    != inherited
                        .iter()
                        .map(|value| Some(value.as_str()))
                        .collect::<Vec<_>>()
                {
                    return Err(format!(
                        "genome {} inherited law fits differ from edges",
                        record.id
                    )
                    .into());
                }
                for inherited_id in &inherited {
                    if required_string(by_id[inherited_id.as_str()], "status")? != "completed" {
                        return Err(format!(
                            "genome {} inherits an unsuccessful law fit",
                            record.id
                        )
                        .into());
                    }
                }
            }
            "environment_candidate" => {
                for name in [
                    "environment_sha256",
                    "topology_sha256",
                    "resource_sha256",
                    "profile_sha256",
                    "variation_recipe_sha256",
                    "probe_panel_sha256",
                ] {
                    required_sha256(record, name)?;
                }
                require_blob_role(record, "environment_artifact")?;
                required_u64(record, "environment_epoch")?;
                let actual: Vec<_> = population_parents_for_role(record, "environment_parent")?
                    .iter()
                    .map(|id| required_sha256(by_id[id.as_str()], "environment_sha256"))
                    .collect::<Result<_, _>>()?;
                let declared = population_fields(record)?
                    .get("parent_environment_sha256s")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        format!("environment {} lacks parent_environment_sha256s", record.id)
                    })?;
                if declared.iter().map(Value::as_str).collect::<Vec<_>>()
                    != actual.iter().map(|value| Some(*value)).collect::<Vec<_>>()
                {
                    return Err(format!(
                        "environment {} parent hashes differ from edges",
                        record.id
                    )
                    .into());
                }
                let panel_id = population_parent_for_role(record, "probe_panel")?;
                if required_sha256(record, "probe_panel_sha256")?
                    != required_sha256(by_id[panel_id.as_str()], "probe_panel_sha256")?
                {
                    return Err(format!("environment {} crosses probe panels", record.id).into());
                }
            }
            "birth" => {
                let life_id = required_string(record, "life_id")?;
                if !life_ids.insert(life_id.to_owned()) {
                    return Err(format!("life_id {life_id} has more than one birth").into());
                }
                let genome = population_parent_for_role(record, "candidate_genome")?;
                let environment = population_parent_for_role(record, "environment")?;
                if required_sha256(record, "genome_sha256")?
                    != required_sha256(by_id[genome.as_str()], "genome_sha256")?
                    || required_sha256(record, "environment_sha256")?
                        != required_sha256(by_id[environment.as_str()], "environment_sha256")?
                {
                    return Err(format!(
                        "birth {} artifact identities differ from edges",
                        record.id
                    )
                    .into());
                }
                let mode = required_string(record, "birth_mode")?;
                let physical = population_parents_for_role(record, "physical_parent_birth")?.len();
                if (mode == "experimental_initialization" && physical != 0)
                    || (mode == "embodied_reproduction" && physical == 0)
                    || !["experimental_initialization", "embodied_reproduction"].contains(&mode)
                {
                    return Err(format!(
                        "birth {} has inconsistent birth mode and physical parents",
                        record.id
                    )
                    .into());
                }
            }
            "research_branch" => {
                let life_id = required_string(record, "life_id")?;
                if !life_ids.insert(life_id.to_owned()) {
                    return Err(format!("life_id {life_id} has more than one life root").into());
                }
                let source_id = population_parent_for_role(record, "source_checkpoint")?;
                let source = by_id[source_id.as_str()];
                let source_root_id = population_parent_for_role(source, "life_continuation")?;
                let source_root = by_id[source_root_id.as_str()];
                let source_life_id = required_string(record, "source_life_id")?;
                if required_string(source, "life_id")? != source_life_id
                    || source_life_id == life_id
                {
                    return Err(format!(
                        "research branch {} does not establish a distinct life",
                        record.id
                    )
                    .into());
                }
                if required_u64(record, "source_tick")? != required_u64(source, "tick")? {
                    return Err(format!("research branch {} source tick differs", record.id).into());
                }
                let genome = population_parent_for_role(record, "candidate_genome")?;
                let environment = population_parent_for_role(record, "environment")?;
                if required_sha256(record, "genome_sha256")?
                    != required_sha256(by_id[genome.as_str()], "genome_sha256")?
                    || required_sha256(record, "environment_sha256")?
                        != required_sha256(by_id[environment.as_str()], "environment_sha256")?
                {
                    return Err(format!(
                        "research branch {} artifact identities differ from edges",
                        record.id
                    )
                    .into());
                }
                if required_string(record, "branch_mode")? != "authenticated_research_copy" {
                    return Err(format!("research branch {} has invalid mode", record.id).into());
                }
                if population_fields(record)?
                    .get("no_model_advance_during_migration")
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    return Err(
                        format!("research branch {} advanced model state", record.id).into(),
                    );
                }
                for field in [
                    "migration_receipt_sha256",
                    "migration_receipt_file_sha256",
                    "source_checkpoint_sha256",
                    "source_checkpoint_state_sha256",
                    "source_neural_snapshot_sha256",
                    "source_neural_payload_sha256",
                    "source_event_snapshot_sha256",
                    "source_event_head_sha256",
                    "target_initial_checkpoint_sha256",
                    "target_initial_checkpoint_state_sha256",
                    "target_neural_snapshot_sha256",
                    "target_neural_payload_sha256",
                    "from_engine_identity_sha256",
                    "to_engine_identity_sha256",
                    "world_instance_sha256",
                    "source_body_id_sha256",
                ] {
                    required_sha256(record, field)?;
                }
                if required_sha256(record, "source_checkpoint_sha256")?
                    != required_sha256(source, "checkpoint_sha256")?
                    || required_sha256(record, "source_checkpoint_state_sha256")?
                        != required_sha256(source, "checkpoint_state_sha256")?
                    || required_sha256(record, "source_neural_payload_sha256")?
                        != required_sha256(record, "target_neural_payload_sha256")?
                {
                    return Err(
                        format!("research branch {} migration hashes differ", record.id).into(),
                    );
                }
                if required_sha256(record, "genome_sha256")?
                    != required_sha256(source_root, "genome_sha256")?
                    || required_sha256(record, "environment_sha256")?
                        != required_sha256(source_root, "environment_sha256")?
                {
                    return Err(format!(
                        "research branch {} changes candidate or environment",
                        record.id
                    )
                    .into());
                }
                let public_body = required_u64(record, "public_body")?;
                let expected_id = format!(
                    "research-branch:{}:{public_body}",
                    required_sha256(record, "migration_receipt_sha256")?
                );
                if record.id != expected_id {
                    return Err(format!("research branch {} identity is invalid", record.id).into());
                }
                require_blob_role(record, "migration_receipt")?;
                let receipt_blob = record
                    .blob_refs
                    .iter()
                    .find(|blob| blob.role == "migration_receipt")
                    .expect("required migration receipt exists");
                if receipt_blob.sha256 != required_sha256(record, "migration_receipt_file_sha256")?
                {
                    return Err(
                        format!("research branch {} receipt blob differs", record.id).into(),
                    );
                }
            }
            "life_checkpoint" => {
                required_string(record, "life_id")?;
                required_sha256(record, "checkpoint_sha256")?;
                require_blob_role(record, "life_checkpoint")?;
            }
            "evaluation_completed" | "evaluation_failed" => {
                require_blob_role(record, "evaluation_result")?;
                require_blob_role(record, "evaluation_trace")?;
                let evaluation_id = required_string(record, "evaluation_id")?;
                if !terminal_evaluations.insert(evaluation_id.to_owned()) {
                    return Err(format!(
                        "evaluation {evaluation_id} has multiple terminal records"
                    )
                    .into());
                }
                required_string(record, "life_id")?;
                required_u64(record, "evaluation_seed")?;
                let committed_ticks = required_u64(record, "committed_ticks")?;
                let allocation_status = required_string(record, "allocation_status")?;
                let continuation_count =
                    population_parents_for_role(record, "life_continuation")?.len();
                let planned_count = population_parents_for_role(record, "planned_campaign")?.len();
                required_sha256(record, "genome_sha256")?;
                required_sha256(record, "environment_sha256")?;
                let trajectory_sha256 = required_sha256(record, "trajectory_sha256")?;
                let trace_blob = record
                    .blob_refs
                    .iter()
                    .find(|blob| blob.role == "evaluation_trace")
                    .expect("required trace blob exists");
                if trace_blob.sha256 != trajectory_sha256 {
                    return Err(
                        format!("evaluation {} trace blob identity differs", record.id).into(),
                    );
                }
                let genome = population_parent_for_role(record, "candidate_genome")?;
                let environment = population_parent_for_role(record, "environment")?;
                let epoch = population_parent_for_role(record, "descriptor_epoch")?;
                let panel = population_parent_for_role(record, "probe_panel")?;
                if required_sha256(record, "genome_sha256")?
                    != required_sha256(by_id[genome.as_str()], "genome_sha256")?
                    || required_sha256(record, "environment_sha256")?
                        != required_sha256(by_id[environment.as_str()], "environment_sha256")?
                    || required_string(record, "descriptor_epoch_id")?
                        != required_string(by_id[epoch.as_str()], "descriptor_epoch_id")?
                    || required_sha256(record, "probe_panel_sha256")?
                        != required_sha256(by_id[panel.as_str()], "probe_panel_sha256")?
                    || required_string(by_id[panel.as_str()], "descriptor_epoch_id")?
                        != required_string(by_id[epoch.as_str()], "descriptor_epoch_id")?
                {
                    return Err(
                        format!("evaluation {} provenance differs from edges", record.id).into(),
                    );
                }
                let status = required_string(record, "status")?;
                let status_matches = if record.record_type == "evaluation_completed" {
                    ["completed", "organism-terminal"].contains(&status)
                } else {
                    status == "infrastructure-failure"
                };
                if !status_matches {
                    return Err(format!(
                        "evaluation {} status differs from record type",
                        record.id
                    )
                    .into());
                }
                if status == "infrastructure-failure" {
                    required_string(record, "failure")?;
                    let valid_allocation = (allocation_status == "allocated"
                        && continuation_count == 1
                        && planned_count == 0)
                        || (allocation_status == "not_allocated"
                            && continuation_count == 0
                            && planned_count == 1
                            && committed_ticks == 0);
                    if !valid_allocation {
                        return Err(format!(
                            "failed evaluation {} has inconsistent allocation proof",
                            record.id
                        )
                        .into());
                    }
                } else {
                    if committed_ticks == 0
                        || allocation_status != "allocated"
                        || continuation_count != 1
                        || planned_count != 0
                    {
                        return Err(format!(
                            "completed evaluation {} has inconsistent allocation proof",
                            record.id
                        )
                        .into());
                    }
                    let descriptor = population_fields(record)?
                        .get("descriptor")
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("evaluation {} lacks descriptor", record.id))?;
                    let dimension = population_fields(by_id[epoch.as_str()])?
                        .get("descriptor_dimension")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| format!("descriptor epoch {epoch} lacks dimension"))?;
                    if descriptor.len() as u64 != dimension {
                        return Err(format!(
                            "evaluation {} descriptor dimension differs",
                            record.id
                        )
                        .into());
                    }
                }
            }
            "archive_decision" => {
                let evaluation_id = required_string(record, "evaluation_id")?;
                if !archive_decisions.insert(evaluation_id.to_owned()) {
                    return Err(format!(
                        "evaluation {evaluation_id} has multiple archive decisions"
                    )
                    .into());
                }
                let decision = required_string(record, "decision")?;
                if !["retained", "rejected", "replaced"].contains(&decision) {
                    return Err(format!("archive decision {} is invalid", record.id).into());
                }
                let evaluation = population_parent_for_role(record, "evaluated_candidate")?;
                let epoch = population_parent_for_role(record, "descriptor_epoch")?;
                if evaluation_id != required_string(by_id[evaluation.as_str()], "evaluation_id")?
                    || required_string(record, "descriptor_epoch_id")?
                        != required_string(by_id[epoch.as_str()], "descriptor_epoch_id")?
                    || required_string(by_id[evaluation.as_str()], "descriptor_epoch_id")?
                        != required_string(record, "descriptor_epoch_id")?
                {
                    return Err(format!("archive decision {} provenance differs", record.id).into());
                }
                if by_id[evaluation.as_str()].record_type == "evaluation_failed"
                    && decision != "rejected"
                {
                    return Err(format!(
                        "archive decision {} retains a failed evaluation",
                        record.id
                    )
                    .into());
                }
            }
            "transfer_trial" => {
                if population_fields(record)?.get("direct_before_fine_tuning")
                    != Some(&Value::Bool(true))
                {
                    return Err(format!(
                        "transfer {} was not recorded before fine tuning",
                        record.id
                    )
                    .into());
                }
                let source = population_parent_for_role(record, "source_evaluation")?;
                let target = population_parent_for_role(record, "target_evaluation")?;
                let genome = population_parent_for_role(record, "candidate_genome")?;
                let environment = population_parent_for_role(record, "target_environment")?;
                let panel = population_parent_for_role(record, "probe_panel")?;
                let genome_sha = required_sha256(by_id[genome.as_str()], "genome_sha256")?;
                if required_string(by_id[source.as_str()], "evaluation_id")?
                    == required_string(by_id[target.as_str()], "evaluation_id")?
                    || required_sha256(by_id[source.as_str()], "genome_sha256")? != genome_sha
                    || required_sha256(by_id[target.as_str()], "genome_sha256")? != genome_sha
                    || required_sha256(by_id[target.as_str()], "environment_sha256")?
                        != required_sha256(by_id[environment.as_str()], "environment_sha256")?
                    || required_sha256(by_id[target.as_str()], "probe_panel_sha256")?
                        != required_sha256(by_id[panel.as_str()], "probe_panel_sha256")?
                {
                    return Err(
                        format!("transfer {} provenance differs from edges", record.id).into(),
                    );
                }
            }
            "gam_fit_attempt" => {
                require_blob_role(record, "gam_fit_report")?;
                let status = required_string(record, "status")?;
                if !["completed", "failed"].contains(&status) {
                    return Err(format!("GAM fit {} has invalid status", record.id).into());
                }
                required_string(record, "unit_of_analysis")?;
                if status == "failed" {
                    required_string(record, "failure")?;
                    if record.blob_refs.iter().any(|blob| blob.role == "gam_law") {
                        return Err(
                            format!("failed GAM fit {} cannot mint a law", record.id).into()
                        );
                    }
                } else {
                    require_blob_role(record, "gam_law")?;
                }
            }
            "interpretation_correction" => {
                require_blob_role(record, "correction_receipt")?;
                let correction = required_sha256(record, "correction_sha256")?;
                let bank = required_sha256(record, "bank_sha256")?;
                let support = required_sha256(record, "original_support_report_sha256")?;
                let surface = required_sha256(record, "surface_sha256")?;
                if record.id != format!("interpretation-correction:{correction}") {
                    return Err(format!(
                        "interpretation correction {} is not keyed by its receipt",
                        record.id
                    )
                    .into());
                }
                let parent = population_parent_for_role(record, "corrected_record")?;
                if required_string(by_id[parent.as_str()], "status")? != "completed"
                    || bank != required_sha256(by_id[parent.as_str()], "bank_sha256")?
                    || support != required_sha256(by_id[parent.as_str()], "support_report_sha256")?
                    || surface != required_sha256(by_id[parent.as_str()], "surface_sha256")?
                    || required_string(record, "status")?
                        != "supplementary-interpretation-correction"
                {
                    return Err(format!(
                        "interpretation correction {} differs from its completed fit",
                        record.id
                    )
                    .into());
                }
                let blob = record
                    .blob_refs
                    .iter()
                    .find(|blob| blob.role == "correction_receipt")
                    .expect("required correction blob exists");
                if blob.sha256 != correction {
                    return Err(format!(
                        "interpretation correction {} receipt identity differs",
                        record.id
                    )
                    .into());
                }
            }
            "population_snapshot" => {
                require_blob_role(record, "population_search_state")?;
            }
            "embodied_recording" => {
                let recording = required_sha256(record, "recording_sha256")?;
                let content = required_sha256(record, "recording_content_sha256")?;
                if record.id != format!("embodied-recording:{content}") {
                    return Err(format!("recording {} is not keyed by content", record.id).into());
                }
                require_blob_role(record, "public_recording")?;
                let blob = record
                    .blob_refs
                    .iter()
                    .find(|blob| blob.role == "public_recording")
                    .expect("required recording blob exists");
                if blob.sha256 != recording {
                    return Err(format!("recording {} blob identity differs", record.id).into());
                }
                let transport = record
                    .blob_refs
                    .iter()
                    .find(|blob| blob.role == "public_recording_gzip")
                    .ok_or_else(|| format!("recording {} lacks gzip transport", record.id))?;
                if transport.sha256 != required_sha256(record, "recording_transport_sha256")?
                    || required_sha256(record, "recording_transport_decoded_sha256")? != recording
                    || required_string(record, "recording_transport_encoding")?
                        != "gzip-level9-mtime0-empty-filename"
                {
                    return Err(format!("recording {} transport differs", record.id).into());
                }
                if required_string(record, "recording_format")?
                    != "chreatures-living-reef-public-recording-v2"
                    || required_u64(record, "frame_count")? == 0
                    || required_u64(record, "resident_count")? == 0
                {
                    return Err(format!("recording {} has invalid extent", record.id).into());
                }
                let laws = population_parents_for_role(record, "associated_law_fit")?;
                for law in laws {
                    if required_string(by_id[law.as_str()], "status")? != "completed" {
                        return Err(
                            format!("recording {} cites a failed law fit", record.id).into()
                        );
                    }
                }
                if required_string(record, "law_relationship")? != "descriptive_association_only" {
                    return Err(format!("recording {} overstates fitted laws", record.id).into());
                }
            }
            "organism_transfer" | "development_event" | "environment_event"
            | "interaction_event" => {
                let public = required_sha256(record, "public_event_sha256")?;
                required_sha256(record, "source_event_sha256")?;
                required_sha256(record, "recording_content_sha256")?;
                if record.id != format!("embodied-event:{public}") {
                    return Err(
                        format!("event {} is not keyed by public receipt", record.id).into(),
                    );
                }
                required_string(record, "public_event_id")?;
                let kind = required_string(record, "kind")?;
                let expected = match kind {
                    "root-material-acquisition"
                    | "mobile-material-release"
                    | "colony-material-emission" => "organism_transfer",
                    "hatching" | "goal_episode_completed" | "research_continuation" => {
                        "development_event"
                    }
                    "developmental-growth-committed"
                    | "developmental-attachment-invalidated"
                    | "developmental-parts-removed"
                    | "visitor_material" => "environment_event",
                    "signal_emission" | "contact_begin" | "contact_end" | "visitor_stimulus" => {
                        "interaction_event"
                    }
                    _ => return Err(format!("event {} has unsupported kind", record.id).into()),
                };
                if record.record_type != expected {
                    return Err(format!("event {} kind differs from type", record.id).into());
                }
                let expected_scope = if kind == "research_continuation" {
                    "observer_provenance"
                } else {
                    "committed_world_event"
                };
                if required_string(record, "event_scope")? != expected_scope {
                    return Err(format!("event {} has invalid scope", record.id).into());
                }
                required_u64(record, "sequence")?;
                required_u64(record, "tick")?;
                let recording_id = population_parent_for_role(record, "recording")?;
                if required_sha256(record, "recording_content_sha256")?
                    != required_sha256(by_id[recording_id.as_str()], "recording_content_sha256")?
                    || population_parent_for_role(record, "observed_environment")?
                        != population_parent_for_role(
                            by_id[recording_id.as_str()],
                            "observed_environment",
                        )?
                {
                    return Err(format!("event {} crosses recording provenance", record.id).into());
                }
                *embodied_event_counts.entry(recording_id).or_default() += 1;
                for law in population_parents_for_role(record, "associated_law_fit")? {
                    if required_string(by_id[law.as_str()], "status")? != "completed" {
                        return Err(format!("event {} cites a failed law fit", record.id).into());
                    }
                }
                if required_string(record, "law_relationship")? != "descriptive_association_only" {
                    return Err(format!("event {} overstates fitted laws", record.id).into());
                }
            }
            _ => {}
        }
    }
    if archive_decisions != terminal_evaluations {
        return Err("terminal evaluations and archive decisions do not match".into());
    }
    for record in records
        .iter()
        .filter(|record| record.record_type == "embodied_recording")
    {
        if required_u64(record, "event_count")?
            != *embodied_event_counts.get(&record.id).unwrap_or(&0)
        {
            return Err(format!("recording {} event count differs", record.id).into());
        }
    }
    for (index, record) in &epoch_indices {
        let previous = population_parents_for_role(record, "previous_descriptor_epoch")?;
        if *index == 0 && !previous.is_empty() {
            return Err("descriptor epoch zero cannot have a predecessor".into());
        }
        if *index > 0 {
            if previous.len() != 1 {
                return Err(format!("descriptor epoch {index} requires one predecessor").into());
            }
            let parent_index = population_fields(by_id[previous[0].as_str()])?
                .get("descriptor_epoch_index")
                .and_then(Value::as_u64);
            if parent_index != Some(index - 1) {
                return Err(format!(
                    "descriptor epoch {index} predecessor is not epoch {}",
                    index - 1
                )
                .into());
            }
        }
    }
    Ok(())
}

fn import_weave(request: ImportRequest) -> Result<EvidenceWeave, Box<dyn Error>> {
    if let Some(schema) = request.evidence_schema.as_deref() {
        if schema != POPULATION_SCHEMA {
            return Err(format!("unsupported evidence schema {schema}").into());
        }
        if !request.journal.is_empty() {
            return Err("typed population evidence cannot include an untyped journal".into());
        }
        validate_population_evidence(&request.evidence)?;
    }
    let capacity = request.journal.len() + request.evidence.len();
    if capacity == 0 {
        return Err("import request has no records".into());
    }
    let mut pending = request
        .journal
        .into_iter()
        .map(journal_pending)
        .chain(request.evidence.into_iter().map(evidence_pending))
        .collect::<Result<Vec<_>, _>>()?;
    let mut source_to_node = HashMap::with_capacity(capacity);
    let mut node_to_source = HashMap::with_capacity(capacity);
    for record in &pending {
        if record.source_id.trim().is_empty() {
            return Err("source id must be nonempty".into());
        }
        let node_id = StableNodeId::from_source_id(&record.source_id);
        if source_to_node
            .insert(record.source_id.clone(), node_id)
            .is_some()
        {
            return Err(format!("duplicate source id {}", record.source_id).into());
        }
        if let Some(other) = node_to_source.insert(node_id, record.source_id.clone()) {
            return Err(format!(
                "stable node id collision between {other} and {}",
                record.source_id
            )
            .into());
        }
        for blob in &record.blob_refs {
            blob.validate()?;
        }
    }
    for record in &pending {
        for parent in &record.parent_ids {
            if !source_to_node.contains_key(parent) {
                return Err(format!("parent {parent} is absent").into());
            }
        }
    }

    let mut weave = EvidenceWeave::with_capacity(
        capacity,
        ArchiveMetadata {
            schema_version: 2,
            library: "universal-weave 0.5.0".to_owned(),
            source_commit: SOURCE_COMMIT.to_owned(),
            evidence_schema: request.evidence_schema,
            archive_id: request.archive_id,
            habitat_id: request.habitat_id,
            description: request.description.unwrap_or_else(|| {
                "external biography, model, snapshot, and law provenance".to_owned()
            }),
        },
    );
    let mut inserted = HashSet::with_capacity(capacity);
    while !pending.is_empty() {
        let before = pending.len();
        let mut blocked = Vec::new();
        for record in pending.drain(..) {
            if record
                .parent_ids
                .iter()
                .any(|parent| !inserted.contains(parent))
            {
                blocked.push(record);
                continue;
            }
            let id = source_to_node[&record.source_id];
            let parents: Vec<_> = record
                .parent_ids
                .iter()
                .map(|parent| source_to_node[parent])
                .collect();
            let source_id = record.source_id.clone();
            insert(
                &mut weave,
                node(
                    id,
                    &parents,
                    EvidenceRecord {
                        source_id: record.source_id,
                        time: record.time,
                        record_type: record.record_type,
                        text: record.text,
                        artifact_uri: record.artifact_uri,
                        blob_refs: record.blob_refs,
                        source: record.source,
                    },
                ),
            )?;
            inserted.insert(source_id);
        }
        if blocked.len() == before {
            let unresolved: Vec<_> = blocked.iter().map(|record| &record.source_id).collect();
            return Err(format!("parent graph contains a cycle: {unresolved:?}").into());
        }
        pending = blocked;
    }

    if !weave.validate() {
        return Err("imported Universal Weave DAG failed validation".into());
    }
    Ok(weave)
}

fn demo_weave() -> Result<EvidenceWeave, Box<dyn Error>> {
    import_weave(ImportRequest {
        archive_id: Some("synthetic-demo".to_owned()),
        habitat_id: Some("synthetic-demo".to_owned()),
        description: Some("explicitly synthetic native round-trip demonstration".to_owned()),
        evidence_schema: None,
        journal: vec![serde_json::json!({
            "id": "demo:episode:1",
            "time": 0.5,
            "kind": "observation",
            "text": "synthetic demonstration episode"
        })],
        evidence: vec![ImportedEvidence {
            id: "demo:evidence:1".to_owned(),
            time: serde_json::json!(0.5),
            record_type: "evidence".to_owned(),
            text: "synthetic demonstration evidence".to_owned(),
            artifact_uri: None,
            blob_refs: Vec::new(),
            parent_ids: vec!["demo:episode:1".to_owned()],
            fields: Value::Null,
        }],
    })
}

struct Args {
    input: Option<PathBuf>,
    output: PathBuf,
    compact_receipt: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut input = None;
    let mut output = PathBuf::from("../artifacts/weave/evidence.weave.json");
    let mut compact_receipt = false;
    let values: Vec<_> = env::args_os().skip(1).collect();
    let mut index = 0;
    while index < values.len() {
        match values[index].to_string_lossy().as_ref() {
            "--input" => {
                index += 1;
                input = Some(PathBuf::from(
                    values.get(index).ok_or("--input needs a path")?,
                ));
            }
            "--output" => {
                index += 1;
                output = PathBuf::from(values.get(index).ok_or("--output needs a path")?);
            }
            "--compact-receipt" => compact_receipt = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
        index += 1;
    }
    Ok(Args {
        input,
        output,
        compact_receipt,
    })
}

fn run(args: &Args) -> Result<Value, Box<dyn Error>> {
    let mut weave = if let Some(path) = &args.input {
        let request: ImportRequest = serde_json::from_slice(&fs::read(path)?)?;
        import_weave(request)?
    } else {
        demo_weave()?
    };
    let mut topological_order = Vec::new();
    weave.get_ordered_identifiers(&mut topological_order);

    let bytes = serde_json::to_vec(&weave)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, &bytes)?;

    let persisted = fs::read(&args.output)?;
    let reloaded: EvidenceWeave = serde_json::from_slice(&persisted)?;
    if !reloaded.validate() {
        return Err("Universal Weave failed validation after serialize/reload".into());
    }
    if reloaded != weave {
        return Err("Universal Weave changed across serialize/reload".into());
    }

    let records: Vec<Value> = topological_order
        .iter()
        .map(|id| {
            let record = reloaded.get_contents(id).expect("ordered node must exist");
            let parents: Vec<_> = reloaded
                .get_parents(id)
                .expect("ordered node must expose parents")
                .iter()
                .copied()
                .collect();
            let parent_source_ids: Vec<_> = parents
                .iter()
                .map(|parent| {
                    reloaded
                        .get_contents(parent)
                        .expect("parent node must exist")
                        .source_id
                        .clone()
                })
                .collect();
            serde_json::json!({
                "node_id": id,
                "source_id": record.source_id,
                "time": record.time,
                "record_type": record.record_type,
                "text": record.text,
                "artifact_uri": record.artifact_uri,
                "blob_refs": record.blob_refs,
                "parents": parents,
                "parent_source_ids": parent_source_ids,
                "source": record.source,
            })
        })
        .collect();
    let edges: Vec<Value> = records
        .iter()
        .flat_map(|record| {
            record["parent_source_ids"]
                .as_array()
                .expect("portable record parents must be an array")
                .iter()
                .map(|parent| {
                    let role = record["source"]["fields"]["parent_roles"]
                        .get(parent.as_str().unwrap_or_default())
                        .cloned()
                        .unwrap_or(Value::Null);
                    serde_json::json!({
                        "source": parent,
                        "target": record["source_id"],
                        "role": role,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let edge_count = edges.len();
    let multi_parent_nodes = records
        .iter()
        .filter(|record| {
            record["parents"]
                .as_array()
                .is_some_and(|parents| parents.len() > 1)
        })
        .count();

    let mut receipt = serde_json::json!({
        "format": "chreatures-portable-weave-biography-v1",
        "integration": "native-universal-weave-dag",
        "library": {
            "name": "universal-weave",
            "version": "0.5.0",
            "source_commit": SOURCE_COMMIT
        },
        "node_id_contract": "lowercase first-128-bit SHA-256 of source_id",
        "archive": reloaded.metadata(),
        "artifact": args.output,
        "bytes": persisted.len(),
        "node_count": reloaded.len(),
        "edge_count": edge_count,
        "multi_parent_nodes": multi_parent_nodes,
        "reload_equal": true,
        "validated_after_reload": true
    });
    if !args.compact_receipt {
        receipt["edges"] = Value::Array(edges);
        receipt["topological_order"] = serde_json::to_value(topological_order)?;
        receipt["records"] = Value::Array(records);
    }
    Ok(receipt)
}

fn main() -> Result<(), Box<dyn Error>> {
    let receipt = run(&parse_args()?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
