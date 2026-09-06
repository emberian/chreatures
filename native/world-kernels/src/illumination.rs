//! Native recurring solar illumination law.
//!
//! The kernel owns the cycle clock and computes one smooth physical sun state.
//! Python only applies that state to MuJoCo and forwards the same irradiance and
//! direction to batched occlusion sampling.  No chemical pool is changed here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::TAU;

#[pyclass]
pub struct SolarCycle {
    period_seconds: f64,
    phase_offset_cycles: f64,
    path_azimuth_radians: f64,
    peak_irradiance: f64,
    diffuse_fraction: f64,
    twilight_sine: f64,
    orbit_radius_m: f64,
    center_m: [f64; 3],
    clock_seconds: f64,
}

impl SolarCycle {
    fn current_state(&self) -> (f64, Vec<f64>, Vec<f64>, f64, f64, f64) {
        let cycles =
            (self.clock_seconds / self.period_seconds + self.phase_offset_cycles).rem_euclid(1.0);
        let angle = TAU * cycles;
        let horizontal = angle.cos();
        let vertical = angle.sin();
        let (azimuth_sine, azimuth_cosine) = self.path_azimuth_radians.sin_cos();
        let toward_sun = [
            horizontal * azimuth_cosine,
            horizontal * azimuth_sine,
            vertical,
        ];

        // Cubic smoothstep begins at the configured sub-horizon twilight
        // angle and reaches full irradiance at solar zenith.  Both endpoints
        // have zero slope, avoiding an authored square-wave discontinuity.
        let ramp = ((vertical + self.twilight_sine) / (1.0 + self.twilight_sine)).clamp(0.0, 1.0);
        let daylight = ramp * ramp * (3.0 - 2.0 * ramp);
        let irradiance = self.peak_irradiance * daylight;
        let diffuse = irradiance * self.diffuse_fraction;
        let direct = irradiance - diffuse;
        let position = vec![
            self.center_m[0] + self.orbit_radius_m * toward_sun[0],
            self.center_m[1] + self.orbit_radius_m * toward_sun[1],
            self.center_m[2] + self.orbit_radius_m * toward_sun[2],
        ];
        (
            self.clock_seconds,
            position,
            toward_sun.to_vec(),
            irradiance,
            direct,
            diffuse,
        )
    }
}

#[pymethods]
impl SolarCycle {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        period_seconds: f64,
        phase_offset_cycles: f64,
        path_azimuth_degrees: f64,
        peak_irradiance: f64,
        diffuse_fraction: f64,
        twilight_degrees: f64,
        orbit_radius_m: f64,
        center_m: Vec<f64>,
    ) -> PyResult<Self> {
        let scalars = [
            period_seconds,
            phase_offset_cycles,
            path_azimuth_degrees,
            peak_irradiance,
            diffuse_fraction,
            twilight_degrees,
            orbit_radius_m,
        ];
        if scalars.iter().any(|value| !value.is_finite())
            || !(10.0..=604_800.0).contains(&period_seconds)
            || !(0.0..1.0).contains(&phase_offset_cycles)
            || !(-360.0..=360.0).contains(&path_azimuth_degrees)
            || !(0.0..=1.0).contains(&peak_irradiance)
            || !(0.0..=1.0).contains(&diffuse_fraction)
            || !(0.1..=30.0).contains(&twilight_degrees)
            || !(0.1..=1000.0).contains(&orbit_radius_m)
            || center_m.len() != 3
            || center_m.iter().any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err("invalid physical solar cycle"));
        }
        Ok(Self {
            period_seconds,
            phase_offset_cycles,
            path_azimuth_radians: path_azimuth_degrees.to_radians(),
            peak_irradiance,
            diffuse_fraction,
            twilight_sine: twilight_degrees.to_radians().sin(),
            orbit_radius_m,
            center_m: [center_m[0], center_m[1], center_m[2]],
            clock_seconds: 0.0,
        })
    }

    fn state(&self) -> (f64, Vec<f64>, Vec<f64>, f64, f64, f64) {
        self.current_state()
    }

    fn advance(&mut self, dt_seconds: f64) -> PyResult<(f64, Vec<f64>, Vec<f64>, f64, f64, f64)> {
        if !dt_seconds.is_finite() || !(0.0..=60.0).contains(&dt_seconds) {
            return Err(PyValueError::new_err(
                "solar cycle step must be finite and in [0, 60] seconds",
            ));
        }
        let next = self.clock_seconds + dt_seconds;
        if !next.is_finite() || next > 1.0e15 {
            return Err(PyValueError::new_err("solar cycle clock overflow"));
        }
        self.clock_seconds = next;
        Ok(self.current_state())
    }

    fn clock_seconds(&self) -> f64 {
        self.clock_seconds
    }

    fn restore_clock(
        &mut self,
        clock_seconds: f64,
    ) -> PyResult<(f64, Vec<f64>, Vec<f64>, f64, f64, f64)> {
        if !clock_seconds.is_finite() || !(0.0..=1.0e15).contains(&clock_seconds) {
            return Err(PyValueError::new_err("invalid solar cycle clock"));
        }
        self.clock_seconds = clock_seconds;
        Ok(self.current_state())
    }
}
