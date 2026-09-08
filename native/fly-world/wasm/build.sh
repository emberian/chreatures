#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
set -eu

: "${MUJOCO_SOURCE_DIR:?must name the pinned MuJoCo 3.12.0 checkout}"
: "${MUJOCO_EMSCRIPTEN_BUILD:?must name its Emscripten build directory}"
: "${CARGO_TARGET_DIR:?must be a scoped fly-world Emscripten target directory}"
: "${OUTPUT_DIR:?must be a scoped output directory}"

test "$(git -C "$MUJOCO_SOURCE_DIR" rev-parse HEAD)" = 13827e9ee56f097f57acf69ae52b078f9839682d
test -f "$MUJOCO_EMSCRIPTEN_BUILD/lib/libmujoco.a"
em++ --version | head -n 1 | grep '6.0.9' >/dev/null
mkdir -p "$OUTPUT_DIR"

MUJOCO_INCLUDE_DIR="$MUJOCO_SOURCE_DIR/include" \
MUJOCO_EMSCRIPTEN_LIB="$MUJOCO_EMSCRIPTEN_BUILD/lib/libmujoco.a" \
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
cargo build --release --lib --target wasm32-unknown-emscripten \
  --manifest-path "$(dirname "$0")/../Cargo.toml"

RUST_ARCHIVE="$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/libchreatures_fly_world.a"
test -f "$RUST_ARCHIVE"

em++ "$RUST_ARCHIVE" \
  -Wl,--start-group \
  -Wl,--whole-archive \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/libmujoco.a" \
  -Wl,--no-whole-archive \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/libccd.a" \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/libqhullstatic_r.a" \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/libtinyxml2.a" \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/libtinyobjloader.a" \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/liblodepng.a" \
  "$MUJOCO_EMSCRIPTEN_BUILD/lib/libminiz.a" \
  -Wl,--end-group \
  -O3 -flto -msimd128 -ffp-contract=off -fexceptions -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=5242880 -sFILESYSTEM=1 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,web,worker \
  -sEXPORTED_RUNTIME_METHODS='["FS","HEAPU8","HEAPF32","HEAPF64"]' \
  -sEXPORTED_FUNCTIONS='["_malloc","_free","_chreatures_fly_world_open","_chreatures_fly_world_close","_chreatures_fly_world_metadata_length","_chreatures_fly_world_metadata","_chreatures_fly_world_topology_revision","_chreatures_fly_world_set_screen","_chreatures_fly_world_visitor_sound","_chreatures_fly_world_visitor_force","_chreatures_fly_world_schedule_interaction","_chreatures_fly_world_prepare_interaction_tick","_chreatures_fly_world_interaction_status","_chreatures_fly_world_interaction_json","_chreatures_fly_world_screen_revision","_chreatures_fly_world_screen_width","_chreatures_fly_world_screen_height","_chreatures_fly_world_screen_length","_chreatures_fly_world_screen_frame","_chreatures_fly_world_ecology_status","_chreatures_fly_world_ecology_json","_chreatures_fly_world_set_routes","_chreatures_fly_world_insert_object","_chreatures_fly_world_mutation_json","_chreatures_fly_world_sample","_chreatures_fly_world_advance","_chreatures_fly_world_snapshot_length","_chreatures_fly_world_snapshot","_chreatures_fly_world_restore","_chreatures_fly_world_observe","_chreatures_fly_world_observation_json","_chreatures_fly_world_observation_f64","_chreatures_fly_world_observation_f32","_chreatures_fly_world_geometry","_chreatures_fly_world_geometry_json","_chreatures_fly_world_geometry_f64","_chreatures_fly_world_geometry_i32","_chreatures_fly_world_last_error_length","_chreatures_fly_world_last_error"]' \
  -o "$OUTPUT_DIR/chreatures-fly-world.mjs"

MJS_SHA=$(shasum -a 256 "$OUTPUT_DIR/chreatures-fly-world.mjs" | awk '{print $1}')
WASM_SHA=$(shasum -a 256 "$OUTPUT_DIR/chreatures-fly-world.wasm" | awk '{print $1}')
cat >"$OUTPUT_DIR/build-receipt.json" <<EOF
{
  "format": "chreatures-unified-fly-world-wasm-build-v1",
  "mujoco_source_revision": "13827e9ee56f097f57acf69ae52b078f9839682d",
  "emscripten": "6.0.9",
  "c_cxx_flags": ["-O3", "-flto", "-msimd128", "-ffp-contract=off", "-fexceptions"],
  "rust_flags": ["-C", "target-feature=+simd128"],
  "mujoco_wasm_threads": false,
  "mjs_sha256": "$MJS_SHA",
  "wasm_sha256": "$WASM_SHA"
}
EOF
printf '%s  %s\n%s  %s\n' "$MJS_SHA" "$OUTPUT_DIR/chreatures-fly-world.mjs" \
  "$WASM_SHA" "$OUTPUT_DIR/chreatures-fly-world.wasm"
