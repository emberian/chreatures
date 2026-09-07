#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Pinned bindgen API and canonical MuJoCo package; no global runtime changes.
test "$(wasm-bindgen --version)" = "wasm-bindgen 0.2.127"
npm ci --ignore-scripts
cargo build --locked --release --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/chreatures_browser_world.wasm --out-dir pkg --target web
node test.mjs
python3 stage_site.py
