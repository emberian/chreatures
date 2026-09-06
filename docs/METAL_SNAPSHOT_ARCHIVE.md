# Metal snapshot archive

Automatic Metal brain snapshots are about 8 MB each. The archive tool preserves
their history in content-addressed storage on hbox while keeping deletion an
explicit, separately requested operation.

The default command is read-only:

```bash
.venv/bin/python scripts/archive_metal_snapshots.py
```

It scans `runs/metal-terrarium/brain` for names matching the automatic
`world-UUID-SEQUENCE.npz` form. Manual names, migration snapshots, and other
NPZ artifacts never become candidates. It scans JSON, backup, and manifest
files under `runs` for snapshot names and SHA-256 values, protects every match,
and always protects the newest automatic snapshot for each world UUID. The
default minimum age is one hour. Add repeated `--snapshot-dir` and
`--reference-root` arguments when another local world is in scope.

To copy eligible history without deleting local files:

```bash
.venv/bin/python scripts/archive_metal_snapshots.py --apply
```

Each object is hashed locally, copied to
`hbox:/tank/chreatures/archives/metal-snapshots/objects/HH/SHA256.npz`, and
checked remotely for both SHA-256 and byte size before its temporary remote
name is atomically renamed. Only then is the local
`runs/metal-archive/catalog.json` updated atomically. The catalog is excluded
from reference discovery because it records archive availability rather than a
world checkpoint dependency.

Local removal requires both flags:

```bash
.venv/bin/python scripts/archive_metal_snapshots.py --apply --delete-local
```

Immediately before each removal, the tool rescans live references and newest
world snapshots, then verifies the source file's size, modification time, and
digest again. If a world pointer changed to reference that object, or the file
changed, removal stops. Already archived manual/pinned artifacts remain local.
Use `--limit N` for bounded operator batches.

Restore an archived object by its cataloged digest. Without `--restore-to`, the
tool restores to the first recorded original path:

```bash
.venv/bin/python scripts/archive_metal_snapshots.py \
  --restore a762ca17902a2d5db6d2f28ad3c5aad036b120bb0d5755e43542c6e1765fd6ad
```

Restore refuses to overwrite a different local file and verifies fetched byte
size and digest before atomically placing it. Supplying `--restore-to` is useful
for an audit copy.

On 2026-09-05, an operational proof archived the two oldest safe automatic
snapshots and verified both remote receipts while leaving their originals
local. A fetch of one object to a temporary audit path reproduced its
7,953,754 bytes and SHA-256 exactly. No live service operation was issued.

On 2026-09-06, a bounded relocation archived the oldest 540 eligible automatic
snapshots from the terrarium and nursery: exactly 4,295,070,392 bytes (4.0001
GiB). Every remote file was independently rechecked for SHA-256 and size before
reporting completion. Referenced, newest and young snapshots stayed local;
manual names and current recycling snapshots were outside this operation.
The local catalog retains each original path and remote object address.

Batch scans now cache hashes and parsed reference tokens only while a file's
resolved path, device, inode, size, mtime and ctime remain unchanged. Every
deletion still refreshes the reference-file listing and metadata, and a changed
or new checkpoint is parsed immediately. Unreadable existing references abort
the operation. Each selected snapshot is independently rehashed before upload
and before removal. A temporary private SSH connection avoids repeated
authentication handshakes and is closed at the end of the operation.
