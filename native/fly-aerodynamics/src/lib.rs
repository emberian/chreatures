//! Native quasi-steady loads for the imported NeuroMechFly articulated wings.
//!
//! This crate only maps measured wing kinematics and airflow to forces. It has
//! no oscillator, controller, target trajectory, flight reward, or hidden
//! state. Inputs and outputs use SI so the host must declare its model-unit
//! conversion explicitly.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

pub mod neuromechfly_geometry;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let norm = self.norm();
        (norm > 0.0 && norm.is_finite()).then(|| self / norm)
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self * (1.0 / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// Row-major rotation from the wing body-local frame to the world frame.
    pub row_major: [f64; 9],
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        row_major: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    pub const fn new(row_major: [f64; 9]) -> Self {
        Self { row_major }
    }

    pub fn rotate(self, vector: Vec3) -> Vec3 {
        let m = self.row_major;
        Vec3::new(
            m[0] * vector.x + m[1] * vector.y + m[2] * vector.z,
            m[3] * vector.x + m[4] * vector.y + m[5] * vector.z,
            m[6] * vector.x + m[7] * vector.y + m[8] * vector.z,
        )
    }

    fn is_rotation(self) -> bool {
        if !self.row_major.iter().all(|value| value.is_finite()) {
            return false;
        }
        let x = Vec3::new(self.row_major[0], self.row_major[3], self.row_major[6]);
        let y = Vec3::new(self.row_major[1], self.row_major[4], self.row_major[7]);
        let z = Vec3::new(self.row_major[2], self.row_major[5], self.row_major[8]);
        let tolerance = 1.0e-6;
        (x.norm_squared() - 1.0).abs() <= tolerance
            && (y.norm_squared() - 1.0).abs() <= tolerance
            && (z.norm_squared() - 1.0).abs() <= tolerance
            && x.dot(y).abs() <= tolerance
            && x.dot(z).abs() <= tolerance
            && y.dot(z).abs() <= tolerance
            && (x.cross(y).dot(z) - 1.0).abs() <= tolerance
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelUnitScale {
    pub meters_per_length_unit: f64,
    pub kilograms_per_mass_unit: f64,
    pub seconds_per_time_unit: f64,
}

impl ModelUnitScale {
    /// Engineering interpretation for the imported NeuroMechFly MJCF.
    ///
    /// FlyGym names millimetres and seconds but does not name its mass unit.
    /// The gram interpretation makes its 0.00102531 total mass 1.02531 mg.
    pub const NEUROMECHFLY_MM_GRAM_INFERRED: Self = Self {
        meters_per_length_unit: 1.0e-3,
        kilograms_per_mass_unit: 1.0e-3,
        seconds_per_time_unit: 1.0,
    };

    pub fn validate(self) -> Result<(), AeroError> {
        if self.meters_per_length_unit.is_finite()
            && self.meters_per_length_unit > 0.0
            && self.kilograms_per_mass_unit.is_finite()
            && self.kilograms_per_mass_unit > 0.0
            && self.seconds_per_time_unit.is_finite()
            && self.seconds_per_time_unit > 0.0
        {
            Ok(())
        } else {
            Err(AeroError::InvalidUnitScale)
        }
    }

    pub fn length_to_si(self, value: Vec3) -> Result<Vec3, AeroError> {
        self.validate()?;
        Ok(value * self.meters_per_length_unit)
    }

    pub fn length_to_model(self, value_m: Vec3) -> Result<Vec3, AeroError> {
        self.validate()?;
        Ok(value_m / self.meters_per_length_unit)
    }

    pub fn velocity_to_si(self, value: Vec3) -> Result<Vec3, AeroError> {
        self.validate()?;
        Ok(value * (self.meters_per_length_unit / self.seconds_per_time_unit))
    }

    pub fn angular_velocity_to_si(self, value: Vec3) -> Result<Vec3, AeroError> {
        self.validate()?;
        Ok(value / self.seconds_per_time_unit)
    }

    pub fn force_to_model(self, force_n: Vec3) -> Result<Vec3, AeroError> {
        self.validate()?;
        Ok(force_n
            / (self.kilograms_per_mass_unit * self.meters_per_length_unit
                / (self.seconds_per_time_unit * self.seconds_per_time_unit)))
    }

    pub fn torque_to_model(self, torque_n_m: Vec3) -> Result<Vec3, AeroError> {
        self.validate()?;
        Ok(torque_n_m
            / (self.kilograms_per_mass_unit
                * self.meters_per_length_unit
                * self.meters_per_length_unit
                / (self.seconds_per_time_unit * self.seconds_per_time_unit)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroConfig {
    pub air_density_kg_m3: f64,
    pub kinematic_viscosity_m2_s: f64,
    pub lift_scale: f64,
    pub drag_scale: f64,
    pub minimum_speed_m_s: f64,
}

impl Default for AeroConfig {
    fn default() -> Self {
        Self {
            air_density_kg_m3: 1.225,
            kinematic_viscosity_m2_s: 1.48e-5,
            lift_scale: 1.0,
            drag_scale: 1.0,
            minimum_speed_m_s: 1.0e-6,
        }
    }
}

impl AeroConfig {
    fn validate(self) -> Result<(), AeroError> {
        if self.air_density_kg_m3.is_finite()
            && self.air_density_kg_m3 > 0.0
            && self.kinematic_viscosity_m2_s.is_finite()
            && self.kinematic_viscosity_m2_s > 0.0
            && self.lift_scale.is_finite()
            && self.lift_scale >= 0.0
            && self.drag_scale.is_finite()
            && self.drag_scale >= 0.0
            && self.minimum_speed_m_s.is_finite()
            && self.minimum_speed_m_s >= 0.0
        {
            Ok(())
        } else {
            Err(AeroError::InvalidConfig)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WingSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BladeElement {
    pub center_local_m: Vec3,
    pub area_m2: f64,
    pub chord_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WingGeometry {
    pub semantic_id: &'static str,
    pub side: WingSide,
    pub source_asset_sha256: &'static str,
    pub chord_local: Vec3,
    pub span_local: Vec3,
    pub normal_local: Vec3,
    pub planform_area_m2: f64,
    pub elements: &'static [BladeElement],
}

impl WingGeometry {
    fn validate(self) -> Result<(), AeroError> {
        let tolerance = 1.0e-6;
        let frame_valid = self.chord_local.is_finite()
            && self.span_local.is_finite()
            && self.normal_local.is_finite()
            && (self.chord_local.norm_squared() - 1.0).abs() <= tolerance
            && (self.span_local.norm_squared() - 1.0).abs() <= tolerance
            && (self.normal_local.norm_squared() - 1.0).abs() <= tolerance
            && self.chord_local.dot(self.span_local).abs() <= tolerance
            && self.chord_local.dot(self.normal_local).abs() <= tolerance
            && self.span_local.dot(self.normal_local).abs() <= tolerance
            && (self
                .chord_local
                .cross(self.span_local)
                .dot(self.normal_local)
                - 1.0)
                .abs()
                <= tolerance;
        let elements_valid = !self.elements.is_empty()
            && self.elements.iter().all(|element| {
                element.center_local_m.is_finite()
                    && element.area_m2.is_finite()
                    && element.area_m2 > 0.0
                    && element.chord_m.is_finite()
                    && element.chord_m > 0.0
            });
        if frame_valid
            && elements_valid
            && self.planform_area_m2.is_finite()
            && self.planform_area_m2 > 0.0
        {
            Ok(())
        } else {
            Err(AeroError::InvalidGeometry)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WingKinematics {
    pub origin_world_m: Vec3,
    pub rotation_world_from_local: Mat3,
    pub linear_velocity_world_m_s: Vec3,
    pub angular_velocity_world_rad_s: Vec3,
    pub airflow_world_m_s: Vec3,
}

impl WingKinematics {
    /// Convert declared model length/time quantities to SI. Angles are radians;
    /// the rotation must already be body-local to world.
    pub fn from_model_units(
        scale: ModelUnitScale,
        origin_world_model: Vec3,
        rotation_world_from_local: Mat3,
        linear_velocity_world_model_s: Vec3,
        angular_velocity_world_rad_s: Vec3,
        airflow_world_model_s: Vec3,
    ) -> Result<Self, AeroError> {
        Ok(Self {
            origin_world_m: scale.length_to_si(origin_world_model)?,
            rotation_world_from_local,
            linear_velocity_world_m_s: scale.velocity_to_si(linear_velocity_world_model_s)?,
            angular_velocity_world_rad_s: scale
                .angular_velocity_to_si(angular_velocity_world_rad_s)?,
            airflow_world_m_s: scale.velocity_to_si(airflow_world_model_s)?,
        })
    }

    fn validate(self) -> Result<(), AeroError> {
        if self.origin_world_m.is_finite()
            && self.rotation_world_from_local.is_rotation()
            && self.linear_velocity_world_m_s.is_finite()
            && self.angular_velocity_world_rad_s.is_finite()
            && self.airflow_world_m_s.is_finite()
        {
            Ok(())
        } else {
            Err(AeroError::InvalidKinematics)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ElementLoad {
    pub application_point_world_m: Vec3,
    pub total_force_world_n: Vec3,
    pub lift_force_world_n: Vec3,
    pub drag_force_world_n: Vec3,
    pub angle_of_attack_rad: f64,
    pub reynolds_number: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WingLoad {
    pub total_force_world_n: Vec3,
    pub lift_force_world_n: Vec3,
    pub drag_force_world_n: Vec3,
    pub torque_about_origin_world_n_m: Vec3,
    /// Positive means mechanical work must be supplied against aerodynamic load.
    pub actuator_power_against_air_w: f64,
    pub maximum_reynolds_number: f64,
    pub active_element_count: usize,
}

impl WingLoad {
    /// Shift the equivalent torque from the wing-body origin to another world
    /// point, such as MuJoCo's body center of mass.
    pub fn torque_about_world_point(self, wing_origin_world_m: Vec3, target_world_m: Vec3) -> Vec3 {
        self.torque_about_origin_world_n_m
            + (wing_origin_world_m - target_world_m).cross(self.total_force_world_n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AeroError {
    InvalidConfig,
    InvalidGeometry,
    InvalidKinematics,
    InvalidUnitScale,
    OutputTooShort { required: usize, supplied: usize },
    NonFiniteResult,
}

impl fmt::Display for AeroError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("invalid aerodynamic configuration"),
            Self::InvalidGeometry => formatter.write_str("invalid wing geometry"),
            Self::InvalidKinematics => formatter.write_str("invalid wing kinematics"),
            Self::InvalidUnitScale => formatter.write_str("invalid model-unit scale"),
            Self::OutputTooShort { required, supplied } => write!(
                formatter,
                "element output is too short: requires {required}, supplied {supplied}"
            ),
            Self::NonFiniteResult => formatter.write_str("aerodynamic calculation was non-finite"),
        }
    }
}

/// Dickinson/Sane translational coefficient fit, with angle in radians.
pub fn translational_coefficients(angle_of_attack_rad: f64) -> (f64, f64) {
    let lift = 0.225 + 1.58 * (2.13 * angle_of_attack_rad - 0.14).sin();
    let drag = 1.92 - 1.55 * (2.04 * angle_of_attack_rad - 0.17).cos();
    (lift, drag.max(0.0))
}

/// Evaluate translational quasi-steady loads without allocating.
///
/// The host must call this from current physics-step kinematics. A 100 Hz
/// controller observation is too slow to represent wingbeat aerodynamics.
pub fn evaluate_into(
    config: &AeroConfig,
    geometry: &WingGeometry,
    kinematics: &WingKinematics,
    element_loads: &mut [ElementLoad],
) -> Result<WingLoad, AeroError> {
    config.validate()?;
    geometry.validate()?;
    kinematics.validate()?;
    if element_loads.len() < geometry.elements.len() {
        return Err(AeroError::OutputTooShort {
            required: geometry.elements.len(),
            supplied: element_loads.len(),
        });
    }

    let chord_world = kinematics
        .rotation_world_from_local
        .rotate(geometry.chord_local);
    let span_world = kinematics
        .rotation_world_from_local
        .rotate(geometry.span_local);
    let normal_world = kinematics
        .rotation_world_from_local
        .rotate(geometry.normal_local);
    let mut result = WingLoad::default();

    for (element, output) in geometry.elements.iter().zip(element_loads.iter_mut()) {
        *output = ElementLoad::default();
        let radius_world = kinematics
            .rotation_world_from_local
            .rotate(element.center_local_m);
        output.application_point_world_m = kinematics.origin_world_m + radius_world;
        let point_velocity_world = kinematics.linear_velocity_world_m_s
            + kinematics.angular_velocity_world_rad_s.cross(radius_world);
        let relative_flow_world = point_velocity_world - kinematics.airflow_world_m_s;
        let planar_flow_world =
            relative_flow_world - span_world * relative_flow_world.dot(span_world);
        let speed_m_s = planar_flow_world.norm();
        if speed_m_s <= config.minimum_speed_m_s {
            continue;
        }
        let travel_direction = planar_flow_world / speed_m_s;
        let mut angle = travel_direction
            .dot(normal_world)
            .atan2(travel_direction.dot(chord_world));
        if angle < 0.0 {
            angle += core::f64::consts::PI;
        }
        let (lift_coefficient, drag_coefficient) = translational_coefficients(angle);
        let pressure_area =
            0.5 * config.air_density_kg_m3 * speed_m_s * speed_m_s * element.area_m2;
        let drag_force = -travel_direction * (pressure_area * drag_coefficient * config.drag_scale);
        let lift_direction = travel_direction
            .cross(span_world)
            .normalized()
            .ok_or(AeroError::NonFiniteResult)?;
        let lift_force = lift_direction * (pressure_area * lift_coefficient * config.lift_scale);
        let total_force = lift_force + drag_force;
        let reynolds_number = speed_m_s * element.chord_m / config.kinematic_viscosity_m2_s;

        output.total_force_world_n = total_force;
        output.lift_force_world_n = lift_force;
        output.drag_force_world_n = drag_force;
        output.angle_of_attack_rad = angle;
        output.reynolds_number = reynolds_number;

        result.total_force_world_n += total_force;
        result.lift_force_world_n += lift_force;
        result.drag_force_world_n += drag_force;
        result.torque_about_origin_world_n_m += radius_world.cross(total_force);
        result.actuator_power_against_air_w -= total_force.dot(point_velocity_world);
        result.maximum_reynolds_number = result.maximum_reynolds_number.max(reynolds_number);
        result.active_element_count += 1;
    }

    let finite = result.total_force_world_n.is_finite()
        && result.lift_force_world_n.is_finite()
        && result.drag_force_world_n.is_finite()
        && result.torque_about_origin_world_n_m.is_finite()
        && result.actuator_power_against_air_w.is_finite()
        && result.maximum_reynolds_number.is_finite();
    if finite {
        Ok(result)
    } else {
        Err(AeroError::NonFiniteResult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromechfly_geometry::{ELEMENT_COUNT, LEFT_WING, RIGHT_WING};

    fn kinematics(linear_velocity: Vec3, airflow: Vec3) -> WingKinematics {
        WingKinematics {
            origin_world_m: Vec3::ZERO,
            rotation_world_from_local: Mat3::IDENTITY,
            linear_velocity_world_m_s: linear_velocity,
            angular_velocity_world_rad_s: Vec3::ZERO,
            airflow_world_m_s: airflow,
        }
    }

    fn near(left: f64, right: f64, tolerance: f64) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    fn near_vec(left: Vec3, right: Vec3, tolerance: f64) {
        near(left.x, right.x, tolerance);
        near(left.y, right.y, tolerance);
        near(left.z, right.z, tolerance);
    }

    #[test]
    fn author_geometry_and_load_invariants() {
        let config = AeroConfig::default();
        let mut element_loads = [ElementLoad::default(); ELEMENT_COUNT];

        let still = evaluate_into(
            &config,
            &LEFT_WING,
            &kinematics(Vec3::ZERO, Vec3::ZERO),
            &mut element_loads,
        )
        .unwrap();
        assert_eq!(still, WingLoad::default());

        let moving = evaluate_into(
            &config,
            &LEFT_WING,
            &kinematics(Vec3::new(0.75, 0.0, 0.15), Vec3::ZERO),
            &mut element_loads,
        )
        .unwrap();
        assert_eq!(moving.active_element_count, ELEMENT_COUNT);
        assert!(moving.drag_force_world_n.dot(Vec3::new(0.75, 0.0, 0.15)) < 0.0);
        assert!(moving.maximum_reynolds_number > 1.0);

        let shifted = evaluate_into(
            &config,
            &LEFT_WING,
            &kinematics(Vec3::new(4.75, -2.0, 1.15), Vec3::new(4.0, -2.0, 1.0)),
            &mut element_loads,
        )
        .unwrap();
        near_vec(
            moving.total_force_world_n,
            shifted.total_force_world_n,
            1.0e-18,
        );
        near_vec(
            moving.torque_about_origin_world_n_m,
            shifted.torque_about_origin_world_n_m,
            1.0e-21,
        );

        near(LEFT_WING.planform_area_m2, RIGHT_WING.planform_area_m2, 0.0);
        for (left, right) in LEFT_WING.elements.iter().zip(RIGHT_WING.elements) {
            near(left.center_local_m.x, right.center_local_m.x, 0.0);
            near(left.center_local_m.y, -right.center_local_m.y, 0.0);
            near(left.center_local_m.z, right.center_local_m.z, 0.0);
            near(left.area_m2, right.area_m2, 0.0);
        }

        let units = ModelUnitScale::NEUROMECHFLY_MM_GRAM_INFERRED;
        near_vec(
            units.force_to_model(Vec3::new(1.0e-6, 0.0, 0.0)).unwrap(),
            Vec3::new(1.0, 0.0, 0.0),
            2.0e-16,
        );
        near_vec(
            units.torque_to_model(Vec3::new(1.0e-9, 0.0, 0.0)).unwrap(),
            Vec3::new(1.0, 0.0, 0.0),
            2.0e-16,
        );

        // Diagnostic scale only: finite loads are not evidence of flight.
        eprintln!(
            "left-wing diagnostic: area_mm2={:.9}, force_uN={:?}, torque_nNmm={:?}, max_Re={:.3}",
            LEFT_WING.planform_area_m2 * 1.0e6,
            moving.total_force_world_n * 1.0e6,
            moving.torque_about_origin_world_n_m * 1.0e9,
            moving.maximum_reynolds_number,
        );
    }
}
