#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the shared Rust resident as a browser ES module or a Node test module."""
import argparse
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, default=root.parents[1] / "site/live/pkg")
parser.add_argument("--target-dir", type=Path, default=Path(os.environ.get("CARGO_TARGET_DIR", root / "target")))
parser.add_argument("--target", choices=("web", "nodejs"), default="web")
args = parser.parse_args()
args.target_dir = args.target_dir.resolve()
env = os.environ.copy()
env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C target-feature=+simd128"
subprocess.run(["cargo", "build", "--manifest-path", str(root / "Cargo.toml"),
                "--target", "wasm32-unknown-unknown", "--target-dir", str(args.target_dir), "--release", "--locked"], env=env, check=True)
args.out_dir.mkdir(parents=True, exist_ok=True)
subprocess.run([os.environ.get("WASM_BINDGEN", "wasm-bindgen"), "--target", args.target,
                "--out-dir", str(args.out_dir), "--out-name", "resident_runtime",
                str(args.target_dir / "wasm32-unknown-unknown/release/resident_runtime.wasm")], check=True)
