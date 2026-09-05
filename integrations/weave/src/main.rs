use std::{collections::hash_map::RandomState, env, error::Error, fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use universal_weave::{
    IndependentContents, Weave,
    independent::{IndependentNode, IndependentWeave},
    indexmap::IndexSet,
};

const SOURCE_COMMIT: &str = "7a5a0dabb94885e44ad8a6c4355c015d7f38020f";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EvidenceRecord {
    source_id: String,
    time: Value,
    record_type: String,
    text: String,
    artifact_uri: Option<String>,
    source: Value,
}

impl IndependentContents for EvidenceRecord {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArchiveMetadata {
    schema_version: u32,
    library: String,
    source_commit: String,
    habitat_id: Option<String>,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ImportRequest {
    #[serde(default)]
    habitat_id: Option<String>,
    journal: Vec<Value>,
    #[serde(default)]
    evidence: Vec<ImportedEvidence>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImportedEvidence {
    id: String,
    #[serde(default)]
    time: Value,
    text: String,
    #[serde(default)]
    artifact_uri: Option<String>,
    #[serde(default)]
    parent_ids: Vec<String>,
}

type EvidenceWeave = IndependentWeave<u64, EvidenceRecord, ArchiveMetadata, RandomState>;
type EvidenceNode = IndependentNode<u64, EvidenceRecord, RandomState>;

fn ids(values: &[u64]) -> IndexSet<u64, RandomState> {
    values.iter().copied().collect()
}

fn node(id: u64, parents: &[u64], contents: EvidenceRecord) -> EvidenceNode {
    EvidenceNode {
        id,
        from: ids(parents),
        to: IndexSet::default(),
        active: true,
        bookmarked: contents.record_type == "evidence",
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

fn import_weave(request: ImportRequest) -> Result<EvidenceWeave, Box<dyn Error>> {
    let capacity = request.journal.len() + request.evidence.len();
    let mut weave = EvidenceWeave::with_capacity(
        capacity,
        ArchiveMetadata {
            schema_version: 1,
            library: "universal-weave 0.5.0".to_owned(),
            source_commit: SOURCE_COMMIT.to_owned(),
            habitat_id: request.habitat_id,
            description: "habitat journal and linked analysis evidence".to_owned(),
        },
    );
    let mut source_ids = std::collections::HashMap::with_capacity(capacity);

    for (offset, source) in request.journal.into_iter().enumerate() {
        let source_id = text_field(&source, "id")?.to_owned();
        if source_ids.contains_key(&source_id) {
            return Err(format!("duplicate source id {source_id}").into());
        }
        let time = source
            .get("time")
            .cloned()
            .ok_or("journal record requires time")?;
        let text = text_field(&source, "text")?.to_owned();
        let id = u64::try_from(offset + 1)?;
        insert(
            &mut weave,
            node(
                id,
                &[],
                EvidenceRecord {
                    source_id: source_id.clone(),
                    time,
                    record_type: "episode".to_owned(),
                    text,
                    artifact_uri: None,
                    source,
                },
            ),
        )?;
        source_ids.insert(source_id, id);
    }

    let first_evidence_id = source_ids.len() + 1;
    for (offset, evidence) in request.evidence.into_iter().enumerate() {
        if source_ids.contains_key(&evidence.id) {
            return Err(format!("duplicate source id {}", evidence.id).into());
        }
        let mut parents = Vec::with_capacity(evidence.parent_ids.len());
        for parent in &evidence.parent_ids {
            parents.push(
                *source_ids
                    .get(parent)
                    .ok_or_else(|| format!("evidence parent {parent} is absent"))?,
            );
        }
        let id = u64::try_from(first_evidence_id + offset)?;
        let source = serde_json::to_value(&evidence)?;
        insert(
            &mut weave,
            node(
                id,
                &parents,
                EvidenceRecord {
                    source_id: evidence.id.clone(),
                    time: evidence.time,
                    record_type: "evidence".to_owned(),
                    text: evidence.text,
                    artifact_uri: evidence.artifact_uri,
                    source,
                },
            ),
        )?;
        source_ids.insert(evidence.id, id);
    }

    if !weave.validate() {
        return Err("imported Universal Weave DAG failed validation".into());
    }
    Ok(weave)
}

fn demo_weave() -> Result<EvidenceWeave, Box<dyn Error>> {
    import_weave(ImportRequest {
        habitat_id: Some("synthetic-demo".to_owned()),
        journal: vec![serde_json::json!({
            "id": "demo:episode:1",
            "time": 0.5,
            "kind": "observation",
            "text": "synthetic demonstration episode"
        })],
        evidence: vec![ImportedEvidence {
            id: "demo:evidence:1".to_owned(),
            time: serde_json::json!(0.5),
            text: "synthetic demonstration evidence".to_owned(),
            artifact_uri: Some("artifact://synthetic/demo".to_owned()),
            parent_ids: vec!["demo:episode:1".to_owned()],
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
            serde_json::json!({
                "node_id": id,
                "source_id": record.source_id,
                "time": record.time,
                "record_type": record.record_type,
                "text": record.text,
                "parents": reloaded.get_parents(id),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "integration": "native-universal-weave-dag",
        "library": {
            "name": "universal-weave",
            "version": "0.5.0",
            "source_commit": SOURCE_COMMIT
        },
        "artifact": args.output,
        "bytes": persisted.len(),
        "node_count": reloaded.len(),
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
