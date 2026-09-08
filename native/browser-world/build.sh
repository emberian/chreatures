#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Pinned bindgen API and canonical MuJoCo package; no global runtime changes.
test "$(wasm-bindgen --version)" = "wasm-bindgen 0.2.127"
npm ci --ignore-scripts
cargo build --locked --release --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/chreatures_browser_world.wasm --out-dir pkg --target web
# Integration and publication are explicit joined steps, not hidden build side effects.
if [ "${1:-}" = "--check" ]; then node test.mjs; fi
if [ "${1:-}" = "--stage" ]; then python3 stage_site.py; fi
