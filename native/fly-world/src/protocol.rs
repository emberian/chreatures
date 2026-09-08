// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::host::{NativeFlyWorld, ResearchSample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{self, BufRead, Write};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
fn base64_encode(input: &[u8]) -> String {
    let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let value = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        output.push(B64[((value >> 18) & 63) as usize] as char);
        output.push(B64[((value >> 12) & 63) as usize] as char);
        output.push(if chunk.len() > 1 {
            B64[((value >> 6) & 63) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            B64[(value & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    if input.len() % 4 != 0 || !input.is_ascii() {
        return Err("invalid base64 framing".into());
    }
    let mut reverse = [u8::MAX; 256];
    for (i, value) in B64.iter().enumerate() {
        reverse[*value as usize] = i as u8;
    }
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(input.len() / 4 * 3);
    for (block_index, block) in bytes.chunks_exact(4).enumerate() {
        let last = block_index + 1 == bytes.len() / 4;
        let padding = usize::from(block[3] == b'=') + usize::from(block[2] == b'=');
        if padding > 0 && !last || padding == 2 && block[3] != b'=' {
            return Err("invalid base64 padding".into());
        }
        let mut values = [0u32; 4];
        for i in 0..4 {
            if block[i] == b'=' {
                continue;
            }
            let decoded = reverse[block[i] as usize];
            if decoded == u8::MAX {
                return Err("invalid base64 alphabet".into());
            }
            values[i] = u32::from(decoded);
        }
        let value = values[0] << 18 | values[1] << 12 | values[2] << 6 | values[3];
        output.push((value >> 16) as u8);
        if padding < 2 {
            output.push((value >> 8) as u8);
        }
        if padding < 1 {
            output.push(value as u8);
        }
    }
    Ok(output)
}
fn encode_f32(values: &[f32]) -> Value {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    json!({"dtype":"<f4","length":values.len(),"base64":base64_encode(&bytes)})
}
fn encode_f64(values: &[f64]) -> Value {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    json!({"dtype":"<f8","length":values.len(),"base64":base64_encode(&bytes)})
}
fn decode_f32(encoded: &str, count: usize) -> Result<Vec<f32>, String> {
    let bytes = base64_decode(encoded)?;
    if bytes.len() != count * 4 {
        return Err("float32 tensor byte length differs".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn packet(sample: ResearchSample) -> Value {
    json!({
        "optic": encode_f32(&sample.optic), "body": encode_f32(&sample.body),
        "qpos": encode_f64(&sample.qpos), "qvel": encode_f64(&sample.qvel),
        "bodyPositions": encode_f64(&sample.body_positions),
        "bodyQuaternions": encode_f64(&sample.body_quaternions),
        "bodyRotations": encode_f64(&sample.body_rotations),
        "sensordata": encode_f64(&sample.sensor_data), "ctrl": encode_f64(&sample.controls),
        "entityIds": sample.entity_ids, "entityPosition": encode_f32(&sample.entity_positions),
        "bodyMap": sample.body_map, "ecology": sample.ecology,
        "actuatorState": sample.actuator_state, "time": sample.time,
    })
}
fn write_response(output: &mut impl Write, value: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *output, value).map_err(|e| e.to_string())?;
    output.write_all(b"\n").map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())
}

pub fn serve_stdio(world: &mut NativeFlyWorld) -> Result<(), String> {
    let snapshot = world.snapshot()?;
    let ready = json!({
        "ok":true, "event":"ready", "residents":world.residents(), "engine":world.engine(),
        "fixture_sha256":world.fixture_sha256(), "scene_xml_sha256":world.scene_sha256(),
        "native_host":"chreatures-native-fly-world-v1",
        "initial_snapshot_sha256":sha256(&snapshot), "fixture":world.ready_fixture(),
    });
    let stdin = io::stdin();
    let mut output = io::stdout().lock();
    write_response(&mut output, &ready)?;
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let request: Value =
            serde_json::from_str(&line).map_err(|e| format!("request JSON: {e}"))?;
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let result = match request.get("command").and_then(Value::as_str) {
            Some("sample") => world
                .research_sample()
                .map(|sample| json!({"id":id,"ok":true,"sample":packet(sample)})),
            Some("advance") => (|| {
                let encoded = request
                    .get("motor92_base64")
                    .and_then(Value::as_str)
                    .ok_or("advance motor missing")?;
                let motor = decode_f32(encoded, world.residents() * 92)?;
                world.advance(
                    &motor.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                    0.01,
                )?;
                Ok(json!({"id":id,"ok":true,"time":world.time()}))
            })(),
            Some("stimulus") => (|| {
                let screen = request.get("screen").ok_or("stimulus screen missing")?;
                let width = screen
                    .get("width")
                    .and_then(Value::as_u64)
                    .ok_or("screen width missing")? as usize;
                let height = screen
                    .get("height")
                    .and_then(Value::as_u64)
                    .ok_or("screen height missing")? as usize;
                let bytes = base64_decode(
                    screen
                        .get("rgb_f32_base64")
                        .and_then(Value::as_str)
                        .ok_or("screen bytes missing")?,
                )?;
                if screen.get("sha256").and_then(Value::as_str) != Some(&sha256(&bytes)) {
                    return Err("stimulus screen checksum differs".into());
                }
                let frame = bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                world.set_screen(&frame, width, height)?;
                let sound = request.get("sound").ok_or("stimulus sound missing")?;
                let position: [f64; 3] = serde_json::from_value(
                    sound
                        .get("position_mm")
                        .cloned()
                        .ok_or("sound position missing")?,
                )
                .map_err(|_| "sound position differs")?;
                let frequency = sound
                    .get("frequency_hz")
                    .and_then(Value::as_f64)
                    .ok_or("sound frequency missing")?;
                let envelope = sound
                    .get("envelope")
                    .and_then(Value::as_f64)
                    .ok_or("sound envelope missing")?;
                let duration = sound
                    .get("duration_s")
                    .and_then(Value::as_f64)
                    .ok_or("sound duration missing")?;
                world.visitor_sound(position, frequency, envelope, duration)?;
                Ok(
                    json!({"id":id,"ok":true,"world_time_s":world.time(),"screen_sha256":sha256(&bytes)}),
                )
            })(),
            Some("routes") => (|| {
                let open: Vec<f64> = serde_json::from_value(
                    request
                        .get("open_fraction")
                        .cloned()
                        .ok_or("route openness missing")?,
                )
                .map_err(|_| "route openness differs")?;
                let flow: Vec<f64> = serde_json::from_value(
                    request
                        .get("advection_m3_s")
                        .cloned()
                        .ok_or("route flow missing")?,
                )
                .map_err(|_| "route flow differs")?;
                world.set_routes(&open, &flow)?;
                Ok(json!({"id":id,"ok":true}))
            })(),
            Some("visitor_force") => (|| {
                let entity = request
                    .get("entity_id")
                    .and_then(Value::as_str)
                    .ok_or("visitor entity missing")?;
                let force: [f64; 3] = serde_json::from_value(
                    request
                        .get("force")
                        .cloned()
                        .ok_or("visitor force missing")?,
                )
                .map_err(|_| "visitor force differs")?;
                world.queue_visitor_force(entity, force)?;
                Ok(json!({"id":id,"ok":true}))
            })(),
            Some("snapshot") => world
                .snapshot()
                .map(|value| json!({"id":id,"ok":true,"snapshot_base64":base64_encode(&value)})),
            Some("restore") => request
                .get("snapshot_base64")
                .and_then(Value::as_str)
                .ok_or_else(|| "snapshot bytes missing".into())
                .and_then(base64_decode)
                .and_then(|value| world.restore(&value))
                .map(|_| json!({"id":id,"ok":true,"time":world.time()})),
            Some("close") => {
                write_response(&mut output, &json!({"id":id,"ok":true}))?;
                return Ok(());
            }
            _ => Err("unknown bridge command".into()),
        };
        write_response(
            &mut output,
            &result.unwrap_or_else(|error: String| json!({"id":id,"ok":false,"error":error})),
        )?;
    }
    Ok(())
}
