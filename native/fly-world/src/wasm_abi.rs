// SPDX-License-Identifier: AGPL-3.0-or-later
//! Flat Emscripten ABI for one persistent native fly world.

use crate::{GeometrySample, NativeFlyWorld, ResearchSample};
use serde_json::json;
use std::cell::RefCell;
use std::mem::{align_of, size_of};
use std::path::Path;
use std::ptr::NonNull;

extern "C" {
    fn emscripten_get_heap_size() -> usize;
}

struct Entry {
    world: NativeFlyWorld,
    snapshot: Vec<u8>,
    metadata: Vec<u8>,
    observation_json: Vec<u8>,
    observation: Option<ResearchSample>,
    geometry_json: Vec<u8>,
    geometry: Option<GeometrySample>,
    mutation_json: Vec<u8>,
}

thread_local! {
    static WORLDS: RefCell<Vec<Option<Entry>>> = const { RefCell::new(Vec::new()) };
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn fail(error: impl ToString) -> i32 {
    LAST_ERROR.with(|slot| {
        let mut bytes = slot.borrow_mut();
        bytes.clear();
        bytes.extend_from_slice(error.to_string().as_bytes());
    });
    -1
}

fn with_entry<R>(
    handle: u32,
    operation: impl FnOnce(&mut Entry) -> Result<R, String>,
) -> Result<R, String> {
    if handle == 0 {
        return Err("world handle is zero".into());
    }
    WORLDS.with(|worlds| {
        let mut worlds = worlds.borrow_mut();
        let entry = worlds
            .get_mut(handle as usize - 1)
            .and_then(Option::as_mut)
            .ok_or_else(|| "world handle is closed or unknown".to_string())?;
        operation(entry)
    })
}

fn checked_region<T>(pointer: *const T, length: usize, label: &str) -> Result<(), String> {
    if length == 0 {
        return Ok(());
    }
    if pointer.is_null() {
        return Err(format!("{label} pointer is null"));
    }
    if (pointer as usize) % align_of::<T>() != 0 {
        return Err(format!("{label} pointer is misaligned"));
    }
    let bytes = length
        .checked_mul(size_of::<T>())
        .ok_or_else(|| format!("{label} byte length overflows"))?;
    let end = (pointer as usize)
        .checked_add(bytes)
        .ok_or_else(|| format!("{label} address overflows"))?;
    if end > unsafe { emscripten_get_heap_size() } {
        return Err(format!("{label} exceeds Wasm linear memory"));
    }
    Ok(())
}

unsafe fn input<'a, T>(pointer: *const T, length: usize, label: &str) -> Result<&'a [T], String> {
    checked_region(pointer, length, label)?;
    let pointer = if length == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        pointer as *mut T
    };
    Ok(std::slice::from_raw_parts(pointer, length))
}

unsafe fn output<'a, T>(
    pointer: *mut T,
    length: usize,
    required: usize,
    label: &str,
) -> Result<&'a mut [T], String> {
    if length != required {
        return Err(format!(
            "{label} length differs: got {length}, expected {required}"
        ));
    }
    checked_region(pointer, length, label)?;
    let pointer = if length == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        pointer
    };
    Ok(std::slice::from_raw_parts_mut(pointer, length))
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_open(
    path: *const u8,
    length: usize,
    seed: u32,
) -> u32 {
    let result: Result<Entry, String> = (|| {
        let bytes = input(path, length, "scene path")?;
        let text = std::str::from_utf8(bytes).map_err(|_| "scene path is not UTF-8")?;
        let world = NativeFlyWorld::open(Path::new(text), seed)?;
        let metadata = serde_json::to_vec(&json!({
            "format": "chreatures-fly-world-wasm-v1",
            "engine": world.engine(),
            "residents": world.residents(),
            "scene_sha256": world.scene_sha256(),
            "fixture_sha256": world.fixture_sha256(),
            "fixture": world.ready_fixture(),
            "control_dt_s": 0.01,
            "physics_substeps": 100,
            "motor_channels_per_resident": 92,
            "optic_values_per_resident": 5313,
            "body_values_per_resident": 807,
        }))
        .map_err(|error| error.to_string())?;
        Ok(Entry {
            world,
            snapshot: Vec::new(),
            metadata,
            observation_json: Vec::new(),
            observation: None,
            geometry_json: Vec::new(),
            geometry: None,
            mutation_json: Vec::new(),
        })
    })();
    match result {
        Ok(entry) => WORLDS.with(|worlds| {
            let mut worlds = worlds.borrow_mut();
            // Never reuse a slot: a stale handle must never address a new life.
            if worlds.len() >= u32::MAX as usize {
                fail("world handle space is exhausted");
                return 0;
            }
            worlds.push(Some(entry));
            worlds.len() as u32
        }),
        Err(error) => {
            fail(error);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_close(handle: u32) -> i32 {
    if handle == 0 {
        return fail("world handle is zero");
    }
    WORLDS.with(|worlds| {
        let mut worlds = worlds.borrow_mut();
        match worlds.get_mut(handle.saturating_sub(1) as usize) {
            Some(slot @ Some(_)) => {
                *slot = None;
                0
            }
            _ => fail("world handle is closed or unknown"),
        }
    })
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_metadata_length(handle: u32) -> usize {
    with_entry(handle, |entry| Ok(entry.metadata.len())).unwrap_or_else(|error| {
        fail(error);
        0
    })
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_topology_revision(handle: u32) -> u32 {
    with_entry(handle, |entry| Ok(entry.world.topology_revision() as u32)).unwrap_or_else(|error| {
        fail(error);
        u32::MAX
    })
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_metadata(
    handle: u32,
    pointer: *mut u8,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        output(pointer, length, entry.metadata.len(), "metadata output")?
            .copy_from_slice(&entry.metadata);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_set_screen(
    handle: u32,
    pointer: *const f32,
    length: usize,
    width: usize,
    height: usize,
) -> i32 {
    let frame = match input(pointer, length, "screen input") {
        Ok(values) => values.to_vec(),
        Err(error) => return fail(error),
    };
    match with_entry(handle, |entry| {
        entry.world.set_screen(&frame, width, height)
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_visitor_sound(
    handle: u32,
    position_pointer: *const f64,
    position_length: usize,
    frequency: f64,
    envelope: f64,
    duration: f64,
) -> i32 {
    let position = match input(position_pointer, position_length, "sound position") {
        Ok(values) if values.len() == 3 => [values[0], values[1], values[2]],
        Ok(_) => return fail("sound position length differs"),
        Err(error) => return fail(error),
    };
    match with_entry(handle, |entry| {
        entry
            .world
            .visitor_sound(position, frequency, envelope, duration)
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_visitor_force(
    handle: u32,
    entity_pointer: *const u8,
    entity_length: usize,
    force_pointer: *const f64,
    force_length: usize,
) -> i32 {
    let entity = match input(entity_pointer, entity_length, "visitor entity").and_then(|bytes| {
        std::str::from_utf8(bytes).map_err(|_| "visitor entity is not UTF-8".to_string())
    }) {
        Ok(value) => value.to_owned(),
        Err(error) => return fail(error),
    };
    let force = match input(force_pointer, force_length, "visitor force") {
        Ok(values) if values.len() == 3 => [values[0], values[1], values[2]],
        Ok(_) => return fail("visitor force length differs"),
        Err(error) => return fail(error),
    };
    match with_entry(handle, |entry| {
        entry.world.queue_visitor_force(&entity, force)
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_set_routes(
    handle: u32,
    open_pointer: *const f64,
    open_length: usize,
    flow_pointer: *const f64,
    flow_length: usize,
) -> i32 {
    let open = match input(open_pointer, open_length, "route openness") {
        Ok(values) => values.to_vec(),
        Err(error) => return fail(error),
    };
    let flow = match input(flow_pointer, flow_length, "route flow") {
        Ok(values) => values.to_vec(),
        Err(error) => return fail(error),
    };
    match with_entry(handle, |entry| entry.world.set_routes(&open, &flow)) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_insert_object(
    handle: u32,
    position_pointer: *const f64,
    position_length: usize,
    size_pointer: *const f64,
    size_length: usize,
    shape: u32,
    rgba_pointer: *const f64,
    rgba_length: usize,
    food: f64,
    odor: i32,
) -> usize {
    let position = match input(position_pointer, position_length, "object position") {
        Ok(values) if values.len() == 3 => [values[0], values[1], values[2]],
        Ok(_) => {
            fail("object position length differs");
            return 0;
        }
        Err(error) => {
            fail(error);
            return 0;
        }
    };
    let size = match input(size_pointer, size_length, "object size") {
        Ok(values) if values.len() == 3 => [values[0], values[1], values[2]],
        Ok(_) => {
            fail("object size length differs");
            return 0;
        }
        Err(error) => {
            fail(error);
            return 0;
        }
    };
    let rgba = match input(rgba_pointer, rgba_length, "object color") {
        Ok(values) if values.len() == 4 => [values[0], values[1], values[2], values[3]],
        Ok(_) => {
            fail("object color length differs");
            return 0;
        }
        Err(error) => {
            fail(error);
            return 0;
        }
    };
    let shape = match shape {
        0 => "box",
        1 => "sphere",
        2 => "capsule",
        3 => "cylinder",
        4 => "ellipsoid",
        _ => {
            fail("unknown object shape");
            return 0;
        }
    };
    match with_entry(handle, |entry| {
        let receipt = entry
            .world
            .insert_visitor_object(position, size, shape, rgba, food, odor)?;
        entry.mutation_json = serde_json::to_vec(&receipt).map_err(|error| error.to_string())?;
        entry.geometry = None;
        Ok(entry.mutation_json.len())
    }) {
        Ok(length) => length,
        Err(error) => {
            fail(error);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_mutation_json(
    handle: u32,
    pointer: *mut u8,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        output(
            pointer,
            length,
            entry.mutation_json.len(),
            "mutation output",
        )?
        .copy_from_slice(&entry.mutation_json);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_sample(
    handle: u32,
    optic_pointer: *mut f32,
    optic_length: usize,
    body_pointer: *mut f32,
    body_length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        let expected_optic = entry.world.residents() * 5313;
        let expected_body = entry.world.residents() * 807;
        output(optic_pointer, optic_length, expected_optic, "optic output")?;
        output(body_pointer, body_length, expected_body, "body output")?;
        let sample = entry.world.sample()?;
        output(
            optic_pointer,
            optic_length,
            sample.optic.len(),
            "optic output",
        )?
        .copy_from_slice(&sample.optic);
        output(body_pointer, body_length, sample.body.len(), "body output")?
            .copy_from_slice(&sample.body);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_advance(
    handle: u32,
    motor_pointer: *const f32,
    motor_length: usize,
    dt: f64,
) -> i32 {
    let motor = match input(motor_pointer, motor_length, "motor input") {
        Ok(values) => values.to_vec(),
        Err(error) => return fail(error),
    };
    match with_entry(handle, |entry| {
        let expected = entry.world.residents() * 92;
        if motor.len() != expected {
            return Err(format!(
                "motor length differs: got {}, expected {expected}",
                motor.len()
            ));
        }
        let commands: Vec<f64> = motor.iter().map(|&value| f64::from(value)).collect();
        entry.world.advance(&commands, dt)
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_snapshot_length(handle: u32) -> usize {
    with_entry(handle, |entry| {
        entry.snapshot = entry.world.snapshot()?;
        Ok(entry.snapshot.len())
    })
    .unwrap_or_else(|error| {
        fail(error);
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_snapshot(
    handle: u32,
    pointer: *mut u8,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        output(pointer, length, entry.snapshot.len(), "snapshot output")?
            .copy_from_slice(&entry.snapshot);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_restore(
    handle: u32,
    pointer: *const u8,
    length: usize,
) -> i32 {
    let snapshot = match input(pointer, length, "snapshot input") {
        Ok(values) => values.to_vec(),
        Err(error) => return fail(error),
    };
    match with_entry(handle, |entry| entry.world.restore(&snapshot)) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_observe(handle: u32) -> usize {
    with_entry(handle, |entry| {
        let sample = entry.world.research_sample()?;
        entry.observation_json = serde_json::to_vec(&json!({
            "format": "chreatures-fly-world-observation-v1",
            "time": sample.time,
            "numeric_lengths": {
                "qpos_f64": sample.qpos.len(),
                "qvel_f64": sample.qvel.len(),
                "body_positions_f64": sample.body_positions.len(),
                "body_quaternions_f64": sample.body_quaternions.len(),
                "body_rotations_f64": sample.body_rotations.len(),
                "geom_positions_f64": sample.geom_positions.len(),
                "geom_rotations_f64": sample.geom_rotations.len(),
                "geom_sizes_f64": sample.geom_sizes.len(),
                "geom_colors_f64": sample.geom_colors.len(),
                "sensor_data_f64": sample.sensor_data.len(),
                "controls_f64": sample.controls.len(),
                "entity_positions_f32": sample.entity_positions.len(),
                "optic_f32": sample.optic.len(),
                "body_f32": sample.body.len(),
            },
            "entity_ids": &sample.entity_ids,
            "body_map": &sample.body_map,
            "ecology": &sample.ecology,
            "actuator_state": &sample.actuator_state,
        }))
        .map_err(|error| error.to_string())?;
        entry.observation = Some(sample);
        Ok(entry.observation_json.len())
    })
    .unwrap_or_else(|error| {
        fail(error);
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_observation_f64(
    handle: u32,
    field: u32,
    pointer: *mut f64,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        let sample = entry
            .observation
            .as_ref()
            .ok_or("observe must be called first")?;
        let (values, label): (&[f64], &str) = match field {
            0 => (&sample.qpos, "qpos"),
            1 => (&sample.qvel, "qvel"),
            2 => (&sample.body_positions, "body positions"),
            3 => (&sample.body_quaternions, "body quaternions"),
            4 => (&sample.body_rotations, "body rotations"),
            5 => (&sample.sensor_data, "sensor data"),
            6 => (&sample.controls, "controls"),
            7 => (&sample.geom_positions, "geom positions"),
            8 => (&sample.geom_rotations, "geom rotations"),
            9 => (&sample.geom_sizes, "geom sizes"),
            10 => (&sample.geom_colors, "geom colors"),
            _ => return Err("unknown f64 observation field".into()),
        };
        output(pointer, length, values.len(), label)?.copy_from_slice(values);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_observation_f32(
    handle: u32,
    field: u32,
    pointer: *mut f32,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        let sample = entry
            .observation
            .as_ref()
            .ok_or("observe must be called first")?;
        let (values, label): (&[f32], &str) = match field {
            0 => (&sample.entity_positions, "entity positions"),
            1 => (&sample.optic, "optic"),
            2 => (&sample.body, "body"),
            _ => return Err("unknown f32 observation field".into()),
        };
        output(pointer, length, values.len(), label)?.copy_from_slice(values);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_observation_json(
    handle: u32,
    pointer: *mut u8,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        output(
            pointer,
            length,
            entry.observation_json.len(),
            "observation output",
        )?
        .copy_from_slice(&entry.observation_json);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_geometry(handle: u32) -> usize {
    with_entry(handle, |entry| {
        let sample = entry.world.geometry_sample()?;
        entry.geometry_json = serde_json::to_vec(&json!({
            "format": "chreatures-fly-world-geometry-v1",
            "topology_revision": sample.topology_revision,
            "model_sha256": sample.model_sha256,
            "metadata": sample.metadata,
            "f64_lengths": [
                sample.geom_size.len(), sample.geom_position.len(),
                sample.geom_quaternion.len(), sample.geom_rgba.len(),
                sample.material_rgba.len(), sample.mesh_vertices.len(),
                sample.mesh_normals.len(),
            ],
            "i32_lengths": [
                sample.geom_body_id.len(), sample.geom_type.len(),
                sample.geom_material_id.len(), sample.geom_data_id.len(),
                sample.mesh_vertex_address.len(),
                sample.mesh_vertex_count.len(), sample.mesh_normal_address.len(),
                sample.mesh_normal_count.len(), sample.mesh_face_address.len(),
                sample.mesh_face_count.len(), sample.mesh_faces.len(),
            ],
        }))
        .map_err(|error| error.to_string())?;
        entry.geometry = Some(sample);
        Ok(entry.geometry_json.len())
    })
    .unwrap_or_else(|error| {
        fail(error);
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_geometry_json(
    handle: u32,
    pointer: *mut u8,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        output(
            pointer,
            length,
            entry.geometry_json.len(),
            "geometry JSON output",
        )?
        .copy_from_slice(&entry.geometry_json);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_geometry_f64(
    handle: u32,
    field: u32,
    pointer: *mut f64,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        let sample = entry
            .geometry
            .as_ref()
            .ok_or("geometry must be called first")?;
        let (values, label): (&[f64], &str) = match field {
            0 => (&sample.geom_size, "geom size"),
            1 => (&sample.geom_position, "geom position"),
            2 => (&sample.geom_quaternion, "geom quaternion"),
            3 => (&sample.geom_rgba, "geom RGBA"),
            4 => (&sample.material_rgba, "material RGBA"),
            5 => (&sample.mesh_vertices, "mesh vertices"),
            6 => (&sample.mesh_normals, "mesh normals"),
            _ => return Err("unknown f64 geometry field".into()),
        };
        output(pointer, length, values.len(), label)?.copy_from_slice(values);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_geometry_i32(
    handle: u32,
    field: u32,
    pointer: *mut i32,
    length: usize,
) -> i32 {
    match with_entry(handle, |entry| {
        let sample = entry
            .geometry
            .as_ref()
            .ok_or("geometry must be called first")?;
        let (values, label): (&[i32], &str) = match field {
            0 => (&sample.geom_body_id, "geom body"),
            1 => (&sample.geom_type, "geom type"),
            2 => (&sample.geom_material_id, "geom material"),
            3 => (&sample.geom_data_id, "geom data"),
            4 => (&sample.mesh_vertex_address, "mesh vertex address"),
            5 => (&sample.mesh_vertex_count, "mesh vertex count"),
            6 => (&sample.mesh_normal_address, "mesh normal address"),
            7 => (&sample.mesh_normal_count, "mesh normal count"),
            8 => (&sample.mesh_face_address, "mesh face address"),
            9 => (&sample.mesh_face_count, "mesh face count"),
            10 => (&sample.mesh_faces, "mesh faces"),
            _ => return Err("unknown i32 geometry field".into()),
        };
        output(pointer, length, values.len(), label)?.copy_from_slice(values);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => fail(error),
    }
}

#[no_mangle]
pub extern "C" fn chreatures_fly_world_last_error_length() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().len())
}

#[no_mangle]
pub unsafe extern "C" fn chreatures_fly_world_last_error(pointer: *mut u8, length: usize) -> i32 {
    LAST_ERROR.with(|slot| {
        let bytes = slot.borrow();
        match output(pointer, length, bytes.len(), "error output") {
            Ok(target) => {
                target.copy_from_slice(&bytes);
                0
            }
            Err(_) => -1,
        }
    })
}
