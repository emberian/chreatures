# Project notebook deployment

The public notebook has one current binary-input mechanism. The tracked
`site/live/release-selection.json` identifies a single archive in the
`emberian/chreatures` GitHub releases area by URL, byte count and SHA-256. That
archive contains exactly three directories:

- `model/`: one authenticated browser model release and its `release.json`;
- `runtime/`: the resident JavaScript/Wasm pair exercised by the selected joined
  run and a reduced manifest that authenticates those two files;
- `physical/`: one output from `native/browser-world/stage_site.py`, including the
  actual fly fixture, 39 body meshes, MuJoCo runtime, shared WorldCore Wasm and its
  physical manifest.

The current selection is `initialized-untrained`. This describes initialization
state only and makes no claim of motor, feeding, recovery or ecological competence.
A fitted or otherwise changed model requires a new immutable archive, release tag
and reviewed selection manifest. Do not relabel an existing archive.

## Prepare a release input archive

First stage the root-selected physical release into an empty directory. Supply all
identities from the successful joined receipt; the command rejects a different
fixture, scene, source runtime or WorldCore Wasm:

```console
python3 native/browser-world/stage_site.py \
  --fixture-directory /path/to/selected-fixture \
  --runtime native/browser-world/runtime.mjs \
  --output /path/to/physical-stage \
  --source-revision FULL_SOURCE_REVISION \
  --expected-fixture-sha256 FIXTURE_SHA256 \
  --expected-scene-sha256 SCENE_SHA256 \
  --expected-runtime-sha256 RUNTIME_SHA256 \
  --expected-core-wasm-sha256 WORLDCORE_WASM_SHA256
```

The stage command replaces only its physical-runtime namespace and hardlinks the
39 authenticated mesh files. It rejects the former generic-garden engine, a mixed
body/actuator schema, and a MuJoCo or WorldCore mismatch.

Create the deterministic archive and its small tracked selection file from the
selected model, joined resident runtime and physical stage:

```console
python3 scripts/pages_release_bundle.py create \
  --model-directory /path/to/browser-model-release \
  --runtime-directory /path/to/successful-joined-live-directory \
  --physics-directory /path/to/physical-stage \
  --source-revision FULL_SOURCE_REVISION \
  --training-status initialized-untrained \
  --expected-model-release-sha256 MODEL_RELEASE_SHA256 \
  --expected-source-runtime-manifest-sha256 JOINED_RUNTIME_MANIFEST_SHA256 \
  --expected-physics-manifest-sha256 PHYSICAL_MANIFEST_SHA256 \
  --release-tag IMMUTABLE_RELEASE_TAG \
  --archive-name ARCHIVE_NAME.tar.gz \
  --bundle-root BUNDLE_ROOT \
  --archive-output /path/to/ARCHIVE_NAME.tar.gz \
  --selection-output site/live/release-selection.json
```

Archive entries are sorted and receive fixed ownership, modes and timestamps. A
second invocation over identical bytes must produce the same archive SHA-256. The
selection file contains no local paths, credentials, checkpoints or unrelated
research arrays.

## Review and publish

Before pushing the selection, create its named GitHub release and upload the exact
archive as the exact asset name in `archive.url`. Verify the remote asset's byte
count and SHA-256 against the selection. This ordering prevents a push to `main`
from selecting an asset that does not exist yet.

The `Publish project notebook` workflow then:

1. downloads only the selected archive from the authorized project release URL;
2. verifies its byte count and SHA-256 before extraction;
3. rejects links, special files, path traversal and an unexpected archive root;
4. verifies the model, resident-runtime and physical component manifests;
5. passes their paths, hashes and the explicit training status to
   `scripts/build_pages.py`;
6. uploads and deploys `dist/site/` only after the complete build succeeds.

For a local workflow-equivalent review, use the already-created archive without a
network fetch, then pass the printed paths and hashes to the builder:

```console
python3 scripts/pages_release_bundle.py fetch \
  --selection site/live/release-selection.json \
  --archive /path/to/ARCHIVE_NAME.tar.gz \
  --output /tmp/chreatures-pages-input

python3 scripts/build_pages.py \
  --revision FULL_SOURCE_REVISION \
  --model-directory /tmp/chreatures-pages-input/model \
  --runtime-directory /tmp/chreatures-pages-input/runtime \
  --expected-runtime-manifest-sha256 RUNTIME_MANIFEST_SHA256 \
  --physics-directory /tmp/chreatures-pages-input/physical \
  --expected-physics-manifest-sha256 PHYSICAL_MANIFEST_SHA256 \
  --model-training-status initialized-untrained \
  --expected-model-release-sha256 MODEL_RELEASE_SHA256
```

The build replaces `dist/site/`. It copies the intentionally public static site,
the allowlisted recorded PNGs and the pinned Three.js runtime, then overlays the
selected physical, resident and model releases. It removes generic-garden fixture
files and the superseded model lock from the output. It does not scan `runs/`,
model storage, checkpoints or other research datasets.

`build-info.json`, `live/runtime-manifest.json`, `live/model-selection.json` and
`live-publication.json` preserve the separate source, physical engine, scene,
WorldCore, resident runtime, model release and component revision boundaries. Site
links must remain relative so the same artifact works below
`https://emberian.github.io/chreatures/` and from a local static server:

```console
python3 -m http.server --directory dist/site 8080
```

The build rejects root-relative, escaping and missing local HTML/CSS references,
along with missing relative JavaScript imports and literal `fetch()` targets. The
workflow does not commit generated output. Before the first deployment, set the
repository's Pages source to **GitHub Actions**. GitHub-owned actions remain pinned
by commit, with their release versions recorded in workflow comments.
