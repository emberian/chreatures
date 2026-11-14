# Exact research data and history

On 2026-11-13, large saved research inputs were moved from ordinary branch/tag
history to `refs/evidence/research-data-20261113` in this same Git repository. All 43 distinct
versions of the 25 selected paths are retained byte-for-byte, including
unsuccessful episodes and old neural checkpoints. No simulation or training was
run. These inputs must be restored, not regenerated from today's dependencies.

## Restore saved inputs

```sh
python3 tools/research_data.py list
python3 tools/research_data.py restore
python3 tools/research_data.py verify
# Or materialize one path / directory:
python3 tools/research_data.py restore --only data/genomes
```

The initial fetch downloads the whole optional corpus; `--only` limits files
materialized. The pinned evidence commit is `a63d59cc0befc73d82f556b1821ae03452cb4892`.
Every restored file is checked against its original Git blob ID, SHA256, size
and executable mode. Existing differing files and symlink parents are refused.
The tool does not initialize missing weights or run a producer. Old research
recipes may require this restoration before their source is run.

For a different historical capture, pass `--original-commit FULL_OLD_COMMIT`.
`.evidence/history.json.xz` indexes all selected paths at all 267 original
commits; `.evidence/original-refs.json` records original tags, branch tips,
peeled commits and their archive refs. Use a separate checkout when restoring
a different capture; differing current files will never be overwritten.

## Original source identity

Ordinary branches and research/release tag histories were rewritten to omit
these payload paths. Named tags now point to compact representations; their
original objects and complete source histories remain fetchable under
`refs/archive/pre-shrink-20261113/heads/*` and `/tags/*`. The commit map is
`.evidence/commit-map.tsv.xz`. Source code outside the selected payloads was
retained in the historical rewrite.

```sh
git fetch --no-tags origin '+refs/archive/pre-shrink-20261113/tags/*:refs/archive/pre-shrink-20261113/tags/*'
git log refs/archive/pre-shrink-20261113/tags/v5-garden-2c040fd
```

Published release assets, their download URLs, scientific hashes and
`site/live/release-selection.json` were not edited. Their pinned original source
commits remain addressable through the archive refs. Frozen worlds, resident
weights/state and runtime selections have not been advanced or reinterpreted.
Original full histories were independently restored/fsck-checked and backed up
locally and on hbox in `~/repo-archives/2026-11-13/living-and-clutch/chreatures.bundle`.

## Public recordings

Public recordings stay in this repository. Four current JSON assets use gzip
wrappers, with exact decoded bytes and SHA256 receipts in
`site/assets/recording-storage.json`. `site/recorded-json.js` handles both packed
and HTTP-decoded responses. Existing large gzip recordings are unchanged.
Native atlas exports default to `.json.gz`; the living-reef recorder also
accepts `--output ...json.gz`. Compression does not change recorded floats,
identities, provenance or unsuccessful observations.

Existing shared local worktrees retain old history and were not modified.
Use fresh clones and reviewed patches; merging old history back restores the
bulk. Evidence fetches are explicit and add their retained data locally.
