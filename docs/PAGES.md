# Project notebook deployment

The public notebook is built from the intentionally public `site/` directory:

```console
python3 scripts/build_pages.py
```

The command replaces `dist/site/`. It also copies the four recorded garden PNGs
listed in `scripts/build_pages.py` to `assets/recorded/` and the five pinned Three.js
runtime/license files to `vendor/three/`. Nothing under `runs/`, model storage,
checkpoints, or other research datasets is scanned or copied. Files placed under
`site/assets/` are public by definition and should contain only selected, sanitized
recordings.

`build-info.json` records the exact source revision, UTC build time, and project
base path. Site links must be relative so the same artifact works below
`https://emberian.github.io/chreatures/` and in a local static server:

```console
python3 -m http.server --directory dist/site 8080
```

The build rejects root-relative, escaping, and missing local HTML/CSS references,
as well as relative JavaScript imports and literal `fetch()` targets that are not
present in the artifact.

The `Publish project notebook` workflow runs for relevant pushes to `main` and on
manual dispatch. Its build job can read repository contents and the existing Pages
configuration. Its deployment job can write Pages deployments and mint the
required OIDC identity token. Before the first run, set the repository's Pages
source to **GitHub Actions**. The workflow does not commit generated output.

The action releases are pinned by commit with their verified release versions in
workflow comments. They were checked against the official GitHub Pages custom
workflow documentation and the corresponding GitHub-owned action release APIs.
