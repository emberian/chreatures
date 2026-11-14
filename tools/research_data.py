#!/usr/bin/env python3
"""Restore exact saved research inputs from this repository's optional evidence ref."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(*args, **kwargs):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], **kwargs)


def blob_id(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def destination(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe research path: ' + name)
    target = ROOT / name
    for parent in target.parents:
        if parent == ROOT:
            break
        if parent.is_symlink():
            raise ValueError('Symlink parent: ' + str(parent))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['restore', 'verify', 'list'])
    parser.add_argument('--only', default='', help='One original path or a directory prefix')
    parser.add_argument('--original-commit', help='Restore the retained version at an original full commit ID')
    args = parser.parse_args()
    index = json.loads((ROOT / '.evidence/index.json').read_text())
    if args.original_commit:
        history = json.loads(lzma.decompress((ROOT / '.evidence/history.json.xz').read_bytes()))
        if args.original_commit not in history:
            raise ValueError('Original commit is not indexed; inspect .evidence/original-refs.json')
        entries = history[args.original_commit]
    else:
        entries = index['entries']
    entries = [e for e in entries if not args.only or e['path'] == args.only or e['path'].startswith(args.only.rstrip('/') + '/')]
    if not entries:
        raise ValueError('No matching retained research inputs')
    if args.action == 'list':
        for entry in entries:
            print(entry['path'] + '\t' + str(entry['size']) + '\t' + entry['oid'])
        return
    # Refuse every conflict before materializing any selected input.
    missing = []
    for entry in entries:
        target = destination(entry['path'])
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or blob_id(target.read_bytes()) != entry['oid']:
                raise ValueError('Refusing to overwrite differing research data: ' + entry['path'])
            if bool(target.stat().st_mode & 0o111) != (entry['mode'] == '100755'):
                raise ValueError('Research data mode differs: ' + entry['path'])
        else:
            missing.append(entry)
    if args.action == 'verify':
        print(json.dumps({'matching': len(entries) - len(missing), 'missing': len(missing)}))
        if missing:
            raise SystemExit(1)
        return
    if not missing:
        print(json.dumps({'restored': 0, 'already_matching': len(entries)}))
        return
    if subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', index['commit'] + '^{commit}'], capture_output=True).returncode:
        print('Fetching the exact saved research corpus from origin; no experiments are run.', file=sys.stderr)
        subprocess.run(['git', '-C', str(ROOT), 'fetch', '--no-tags', 'origin', index['ref']], check=True)
        if git('rev-parse', 'FETCH_HEAD').decode().strip() != index['commit']:
            raise ValueError('Evidence ref differs from its pinned commit')
        subprocess.run(['git', '-C', str(ROOT), 'update-ref', index['ref'], index['commit']], check=True)
    for entry in missing:
        raw = git('show', index['commit'] + ':objects/' + entry['oid'])
        if len(raw) != entry['size'] or blob_id(raw) != entry['oid'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Retained research bytes differ: ' + entry['path'])
        target = destination(entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        target.chmod(0o755 if entry['mode'] == '100755' else 0o644)
    print(json.dumps({'restored': len(missing), 'already_matching': len(entries) - len(missing)}))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
