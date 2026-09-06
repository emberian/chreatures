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
    #[serde(default)]
    archive_id: Option<String>,
    habitat_id: Option<String>,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ImportRequest {
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

fn import_weave(request: ImportRequest) -> Result<EvidenceWeave, Box<dyn Error>> {
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
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut input = None;
    let mut output = PathBuf::from("../artifacts/weave/evidence.weave.json");
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
            other => return Err(format!("unknown argument {other}").into()),
        }
        index += 1;
    }
    Ok(Args { input, output })
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

    let bytes = serde_json::to_vec_pretty(&weave)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, &bytes)?;

    let persisted = fs::read(&args.output)?;
    let reloaded: EvidenceWeave = serde_json::from_slice(&persisted)?;
    if !reloaded.validate() || reloaded != weave {
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
                    serde_json::json!({
                        "source": parent,
                        "target": record["source_id"],
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

    Ok(serde_json::json!({
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
        "edges": edges,
        "multi_parent_nodes": multi_parent_nodes,
        "topological_order": topological_order,
        "records": records,
        "reload_equal": true,
        "validated_after_reload": true
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let receipt = run(&parse_args()?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
