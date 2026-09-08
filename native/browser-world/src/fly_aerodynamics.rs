//! Allocation-stable browser/native boundary for actual wing aerodynamic loads.
//!
//! Ordering is resident-major, then left/right wing. The wrapper consumes
//! current MuJoCo body arrays and emits model-unit force plus torque about each
//! wing's MuJoCo `xipos` center of mass for `xfrc_applied`. Its internal blade
//! evaluation still uses 24 spanwise elements. It owns no recurrent state.

use chreatures_fly_aerodynamics::{
    neuromechfly_geometry, AeroConfig, ElementLoad, Mat3, ModelUnitScale, PreparedWing, Vec3,
    WingKinematics, WingLoad,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub const WINGS_PER_RESIDENT: usize = 2;
pub const ELEMENTS_PER_WING: usize = neuromechfly_geometry::ELEMENT_COUNT;
/// Force xyz and torque xyz about MuJoCo `xipos`, all in model units.
pub const WRENCH_STRIDE: usize = 6;
/// Total force, lift force, drag force, root torque, power W, max Re, active count.
pub const DIAGNOSTIC_STRIDE: usize = 15;
pub const AERODYNAMIC_SCHEMA_SHA256: &str =
    "88893421111b6339091cb2c97368704f1ee51fcbf4cbdcbf3a793f16cf78d9fd";

const CANONICAL_AERODYNAMICS: &str =
    include_str!("../../fly-aerodynamics/assets/aerodynamics-model-v1.json");

#[derive(Deserialize)]
struct FixtureBody {
    id: String,
    segments: Vec<usize>,
    wings: [usize; WINGS_PER_RESIDENT],
}

#[derive(Deserialize)]
struct Fixture {
    wing_aerodynamics: Value,
    airflow_mm_s: [f64; 3],
    physics_dt: f64,
    bodies: Vec<FixtureBody>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AeroWorld {
    prepared: [PreparedWing; WINGS_PER_RESIDENT],
    meters_per_model_length: f64,
    meters_per_second_per_model_velocity: f64,
    radians_per_second_per_model_angular_velocity: f64,
    model_force_units_per_newton: f64,
    model_torque_units_per_newton_meter: f64,
    airflow_world_m_s: Vec3,
    wing_bodies: Vec<i32>,
    maximum_body_id: usize,
    element_scratch: [ElementLoad; ELEMENTS_PER_WING],
    wrenches: Vec<f64>,
    diagnostics: Vec<f64>,
}

fn vec3(values: &[f64]) -> Vec3 {
    Vec3::new(values[0], values[1], values[2])
}

fn copy_vec3(output: &mut [f64], value: Vec3) {
    output.copy_from_slice(&[value.x, value.y, value.z]);
}

fn error(message: impl Into<String>) -> String {
    message.into()
}

// JSON.stringify writes integral doubles as integers. Compare the exact
// represented numerical values, not serde_json's integer/float storage tag.
// Keys, array order, strings and nonnumeric types still match exactly.
fn same_json_values(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => a.len() == b.len() &&
            a.iter().zip(b).all(|(a, b)| same_json_values(a, b)),
        (Value::Object(a), Value::Object(b)) => a.len() == b.len() &&
            a.iter().all(|(key, value)| b.get(key).is_some_and(|other| same_json_values(value, other))),
        _ => a == b,
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl AeroWorld {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(fixture_json: &str) -> Result<Self, String> {
        let fixture: Fixture =
            serde_json::from_str(fixture_json).map_err(|e| error(format!("aero fixture: {e}")))?;
        let canonical: Value = serde_json::from_str(CANONICAL_AERODYNAMICS)
            .map_err(|e| error(format!("embedded aero identity: {e}")))?;
        if !same_json_values(&fixture.wing_aerodynamics, &canonical)
            || fixture
                .wing_aerodynamics
                .get("aerodynamic_schema_sha256")
                .and_then(Value::as_str)
                != Some(AERODYNAMIC_SCHEMA_SHA256)
            || fixture.physics_dt != 0.0001
            || fixture.bodies.is_empty()
            || fixture.bodies.len() > 32
            || fixture.airflow_mm_s.iter().any(|value| !value.is_finite())
        {
            return Err(error("invalid frozen wing aerodynamic contract"));
        }

        let mut distinct = HashSet::with_capacity(fixture.bodies.len() * WINGS_PER_RESIDENT);
        let mut wing_bodies = Vec::with_capacity(fixture.bodies.len() * WINGS_PER_RESIDENT);
        let mut maximum_body_id = 0;
        for body in &fixture.bodies {
            if body.id.is_empty()
                || body.wings[0] == body.wings[1]
                || body.wings.iter().any(|wing| !body.segments.contains(wing))
            {
                return Err(error("invalid compiled wing body mapping"));
            }
            for &wing in &body.wings {
                if !distinct.insert(wing) || wing > i32::MAX as usize {
                    return Err(error("duplicate or out-of-range compiled wing body"));
                }
                maximum_body_id = maximum_body_id.max(wing);
                wing_bodies.push(wing as i32);
            }
        }

        let wing_count = wing_bodies.len();
        let config = AeroConfig::default();
        let units = ModelUnitScale::NEUROMECHFLY_MM_GRAM_INFERRED;
        units.validate().map_err(|e| error(e.to_string()))?;
        let prepared = [
            PreparedWing::new(config, &neuromechfly_geometry::LEFT_WING)
                .map_err(|e| error(e.to_string()))?,
            PreparedWing::new(config, &neuromechfly_geometry::RIGHT_WING)
                .map_err(|e| error(e.to_string()))?,
        ];
        Ok(Self {
            prepared,
            meters_per_model_length: units.meters_per_length_unit,
            meters_per_second_per_model_velocity: units.meters_per_length_unit
                / units.seconds_per_time_unit,
            radians_per_second_per_model_angular_velocity: 1.0 / units.seconds_per_time_unit,
            model_force_units_per_newton: units
                .model_force_units_per_newton()
                .map_err(|e| error(e.to_string()))?,
            model_torque_units_per_newton_meter: units
                .model_torque_units_per_newton_meter()
                .map_err(|e| error(e.to_string()))?,
            airflow_world_m_s: Vec3::new(
                fixture.airflow_mm_s[0] * units.meters_per_length_unit,
                fixture.airflow_mm_s[1] * units.meters_per_length_unit,
                fixture.airflow_mm_s[2] * units.meters_per_length_unit,
            ),
            wing_bodies,
            maximum_body_id,
            element_scratch: [ElementLoad::default(); ELEMENTS_PER_WING],
            wrenches: vec![0.0; wing_count * WRENCH_STRIDE],
            diagnostics: vec![0.0; wing_count * DIAGNOSTIC_STRIDE],
        })
    }

    pub fn resident_count(&self) -> usize {
        self.wing_bodies.len() / WINGS_PER_RESIDENT
    }

    pub fn wing_count(&self) -> usize {
        self.wing_bodies.len()
    }

    pub fn wrench_len(&self) -> usize {
        self.wrenches.len()
    }

    pub fn diagnostic_len(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn wing_body_len(&self) -> usize {
        self.wing_bodies.len()
    }

    /// Evaluate all resident wings from full MuJoCo body arrays.
    ///
    /// `body_velocities` uses MuJoCo `mj_objectVelocity` ordering: angular xyz,
    /// then linear xyz, all in the world frame.
    pub fn evaluate_bulk(
        &mut self,
        body_positions: &[f64],
        body_com_positions: &[f64],
        body_rotations: &[f64],
        body_velocities: &[f64],
    ) -> Result<(), String> {
        let required_bodies = self.maximum_body_id + 1;
        if body_positions.len() < required_bodies * 3
            || body_com_positions.len() < required_bodies * 3
            || body_rotations.len() < required_bodies * 9
            || body_velocities.len() < required_bodies * 6
        {
            return Err(error("MuJoCo body array is shorter than compiled wing IDs"));
        }

        for wing_index in 0..self.wing_bodies.len() {
            let body = self.wing_bodies[wing_index] as usize;
            let position = &body_positions[body * 3..body * 3 + 3];
            let com_position = &body_com_positions[body * 3..body * 3 + 3];
            let rotation = &body_rotations[body * 9..body * 9 + 9];
            let velocity = &body_velocities[body * 6..body * 6 + 6];
            let kinematics = WingKinematics {
                origin_world_m: vec3(position) * self.meters_per_model_length,
                rotation_world_from_local: Mat3::new(
                    rotation.try_into().expect("nine-value rotation slice"),
                ),
                linear_velocity_world_m_s: vec3(&velocity[3..6])
                    * self.meters_per_second_per_model_velocity,
                angular_velocity_world_rad_s: vec3(&velocity[0..3])
                    * self.radians_per_second_per_model_angular_velocity,
                airflow_world_m_s: self.airflow_world_m_s,
            };
            let load = self.prepared[wing_index % WINGS_PER_RESIDENT]
                .evaluate_into(&kinematics, &mut self.element_scratch)
                .map_err(|e| error(e.to_string()))?;
            self.pack_wrench(
                wing_index,
                load,
                kinematics.origin_world_m,
                vec3(com_position),
            )?;
            self.pack_diagnostics(wing_index, load)?;
        }
        Ok(())
    }

    /// Copy the persistent per-wing force/COM-torque buffer for `xfrc_applied`.
    pub fn copy_wrenches(&self, output: &mut [f64]) -> Result<(), String> {
        if output.len() != self.wrenches.len() {
            return Err(error("aerodynamic wrench output shape differs"));
        }
        output.copy_from_slice(&self.wrenches);
        Ok(())
    }

    /// Copy the persistent resident/wing observer-only diagnostic buffer.
    ///
    /// Each row contains total force xyz, lift xyz, drag xyz, torque xyz about
    /// the wing body origin (all model units), actuator power against air in W,
    /// maximum Reynolds number, and active element count.
    pub fn copy_diagnostics(&self, output: &mut [f64]) -> Result<(), String> {
        if output.len() != self.diagnostics.len() {
            return Err(error("aerodynamic diagnostic output shape differs"));
        }
        output.copy_from_slice(&self.diagnostics);
        Ok(())
    }

    /// Copy compiled wing body IDs, matching wrench rows.
    pub fn copy_wing_bodies(&self, output: &mut [i32]) -> Result<(), String> {
        if output.len() != self.wing_body_len() {
            return Err(error("aerodynamic wing-body output shape differs"));
        }
        output.copy_from_slice(&self.wing_bodies);
        Ok(())
    }
}

impl AeroWorld {
    pub fn wrenches(&self) -> &[f64] {
        &self.wrenches
    }

    pub fn diagnostics(&self) -> &[f64] {
        &self.diagnostics
    }

    pub fn wing_bodies(&self) -> &[i32] {
        &self.wing_bodies
    }

    fn pack_wrench(
        &mut self,
        wing_index: usize,
        load: WingLoad,
        origin_world_m: Vec3,
        com_position_model: Vec3,
    ) -> Result<(), String> {
        if !com_position_model.is_finite() {
            return Err(error("nonfinite MuJoCo wing center of mass"));
        }
        let com_world_m = com_position_model * self.meters_per_model_length;
        let offset = wing_index * WRENCH_STRIDE;
        let total = load.total_force_world_n * self.model_force_units_per_newton;
        let torque = load.torque_about_world_point(origin_world_m, com_world_m)
            * self.model_torque_units_per_newton_meter;
        copy_vec3(&mut self.wrenches[offset..offset + 3], total);
        copy_vec3(&mut self.wrenches[offset + 3..offset + 6], torque);
        Ok(())
    }

    fn pack_diagnostics(&mut self, wing_index: usize, load: WingLoad) -> Result<(), String> {
        let offset = wing_index * DIAGNOSTIC_STRIDE;
        let row = &mut self.diagnostics[offset..offset + DIAGNOSTIC_STRIDE];
        let total = load.total_force_world_n * self.model_force_units_per_newton;
        let lift = load.lift_force_world_n * self.model_force_units_per_newton;
        let drag = load.drag_force_world_n * self.model_force_units_per_newton;
        let torque = load.torque_about_origin_world_n_m * self.model_torque_units_per_newton_meter;
        copy_vec3(&mut row[0..3], total);
        copy_vec3(&mut row[3..6], lift);
        copy_vec3(&mut row[6..9], drag);
        copy_vec3(&mut row[9..12], torque);
        row[12] = load.actuator_power_against_air_w;
        row[13] = load.maximum_reynolds_number;
        row[14] = load.active_element_count as f64;
        Ok(())
    }
}
