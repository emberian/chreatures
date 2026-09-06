//! Conservative finite-volume transport with reusable, world-owned scratch.
//!
//! The arithmetic order follows FieldEnvironment's reference face passes:
//! x, y, z; subtract every outgoing face, then add every incoming face.
//! No fast-math reassociation or fused multiply-add is requested.

use numpy::{
    PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyReadwriteArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;

type BarrierArrays<'py> = (
    PyReadonlyArray3<'py, f64>,
    PyReadonlyArray3<'py, f64>,
    PyReadonlyArray3<'py, f64>,
);

#[pyclass]
pub struct TransportSolver {
    shape: [usize; 3],
    channels: usize,
    spacing: [f64; 3],
    diffusion: Vec<f64>,
    change: Vec<f64>,
    flux: Vec<f64>,
}

#[pymethods]
impl TransportSolver {
    #[new]
    fn new(
        shape_xyz: [usize; 3],
        channels: usize,
        spacing: [f64; 3],
        diffusion: Vec<f64>,
    ) -> PyResult<Self> {
        if shape_xyz.iter().any(|&v| !(4..=256).contains(&v))
            || shape_xyz.iter().product::<usize>() > 2_500_000
            || !(1..=32).contains(&channels)
            || spacing.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || diffusion.len() != channels
            || diffusion
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=2.0).contains(v))
        {
            return Err(PyValueError::new_err(
                "invalid transport grid, spacing, or channel diffusion",
            ));
        }
        let cells = shape_xyz.iter().product::<usize>() * channels;
        Ok(Self {
            shape: shape_xyz,
            channels,
            spacing,
            diffusion,
            change: vec![0.0; cells],
            flux: vec![0.0; cells],
        })
    }

    #[pyo3(signature = (dt, current, flow, permeability, solid, barriers=None))]
    fn step(
        &mut self,
        dt: f64,
        mut current: PyReadwriteArray4<'_, f64>,
        flow: PyReadonlyArray4<'_, f64>,
        permeability: PyReadonlyArray3<'_, f64>,
        solid: PyReadonlyArray3<'_, bool>,
        barriers: Option<BarrierArrays<'_>>,
    ) -> PyResult<()> {
        let [nx, ny, nz] = self.shape;
        if !dt.is_finite()
            || dt <= 0.0
            || dt > 10.0
            || current.shape() != [self.channels, nz, ny, nx]
            || flow.shape() != [3, nz, ny, nx]
            || permeability.shape() != [nz, ny, nx]
            || solid.shape() != [nz, ny, nx]
        {
            return Err(PyValueError::new_err(
                "transport timestep or array shape mismatch",
            ));
        }
        let c = current.as_slice_mut()?;
        let f = flow.as_slice()?;
        let p = permeability.as_slice()?;
        let s = solid.as_slice()?;
        let barrier_slices = match barriers.as_ref() {
            Some((x, y, z)) => {
                if x.shape() != [nz, ny, nx - 1]
                    || y.shape() != [nz, ny - 1, nx]
                    || z.shape() != [nz - 1, ny, nx]
                {
                    return Err(PyValueError::new_err(
                        "transport barrier face shape mismatch",
                    ));
                }
                Some([x.as_slice()?, y.as_slice()?, z.as_slice()?])
            }
            None => None,
        };
        if c.iter().any(|v| !v.is_finite() || *v < 0.0)
            || f.iter().any(|v| !v.is_finite() || v.abs() > 50.0)
            || p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || barrier_slices.as_ref().is_some_and(|faces| {
                faces.iter().any(|face| {
                    face.iter()
                        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                })
            })
        {
            return Err(PyValueError::new_err(
                "transport arrays contain invalid values",
            ));
        }
        self.transport(dt, c, f, p, s, barrier_slices);
        let minimum = c.iter().fold(0.0_f64, |a, &b| a.min(b));
        if minimum < -1e-10 || c.iter().any(|v| !v.is_finite()) {
            return Err(PyFloatingPointError::new_err(format!(
                "invalid transported concentration, minimum {minimum}; check CFL"
            )));
        }
        for value in c {
            *value = value.max(0.0);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn deposit_sources<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        fraction: f64,
        mut current: PyReadwriteArray4<'_, f64>,
        solid: PyReadonlyArray3<'_, bool>,
        positions: PyReadonlyArray2<'_, f64>,
        previous: PyReadonlyArray2<'_, f64>,
        rates: PyReadonlyArray1<'_, f64>,
        spreads: PyReadonlyArray1<'_, f64>,
        channels: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let count = rates.len();
        let [nx, ny, nz] = self.shape;
        if !dt.is_finite()
            || dt <= 0.0
            || !fraction.is_finite()
            || !(0.0..=1.0).contains(&fraction)
            || current.shape() != [self.channels, nz, ny, nx]
            || solid.shape() != [nz, ny, nx]
            || positions.shape() != [count, 3]
            || previous.shape() != [count, 3]
            || spreads.len() != count
            || channels.len() != count
        {
            return Err(PyValueError::new_err("invalid source batch dimensions"));
        }
        let mut injected = vec![0.0; self.channels];
        let mut outside = vec![0.0; self.channels];
        self.deposit_source_batch(
            dt,
            fraction,
            current.as_slice_mut()?,
            solid.as_slice()?,
            positions.as_slice()?,
            previous.as_slice()?,
            rates.as_slice()?,
            spreads.as_slice()?,
            channels.as_slice()?,
            &mut injected,
            &mut outside,
        )?;
        Ok((
            PyArray1::from_vec(py, injected),
            PyArray1::from_vec(py, outside),
        ))
    }
}

impl TransportSolver {
    #[allow(clippy::too_many_arguments)]
    fn deposit_source_batch(
        &self,
        dt: f64,
        fraction: f64,
        current: &mut [f64],
        solid: &[bool],
        positions: &[f64],
        previous: &[f64],
        rates: &[f64],
        spreads: &[f64],
        channels: &[i32],
        injected: &mut [f64],
        outside: &mut [f64],
    ) -> PyResult<()> {
        let [nx, ny, nz] = self.shape;
        let cells = nx * ny * nz;
        let size = [
            nx as f64 * self.spacing[0],
            ny as f64 * self.spacing[1],
            nz as f64 * self.spacing[2],
        ];
        let mut weighted = Vec::<(usize, f64)>::new();
        for source in 0..rates.len() {
            let rate = rates[source];
            let spread = spreads[source];
            let channel = channels[source];
            if !rate.is_finite()
                || rate < 0.0
                || !spread.is_finite()
                || spread < 0.0
                || channel < 0
                || channel as usize >= self.channels
            {
                return Err(PyValueError::new_err("invalid source batch value"));
            }
            let mut position = [0.0; 3];
            for axis in 0..3 {
                let now = positions[source * 3 + axis];
                let before = previous[source * 3 + axis];
                if !now.is_finite() || !before.is_finite() {
                    return Err(PyValueError::new_err("source positions must be finite"));
                }
                position[axis] = before + fraction * (now - before);
            }
            let mass = rate * dt;
            if mass == 0.0 {
                continue;
            }
            let channel = channel as usize;
            if (0..3).any(|axis| position[axis] < 0.0 || position[axis] > size[axis]) {
                outside[channel] += mass;
                continue;
            }
            weighted.clear();
            if spread <= 0.0 {
                let scaled = [
                    position[0] / self.spacing[0] - 0.5,
                    position[1] / self.spacing[1] - 0.5,
                    position[2] / self.spacing[2] - 0.5,
                ];
                let lower = [
                    scaled[0].floor() as isize,
                    scaled[1].floor() as isize,
                    scaled[2].floor() as isize,
                ];
                for dz in 0..=1 {
                    for dy in 0..=1 {
                        for dx in 0..=1 {
                            let index = [lower[0] + dx, lower[1] + dy, lower[2] + dz];
                            if index[0] < 0
                                || index[1] < 0
                                || index[2] < 0
                                || index[0] >= nx as isize
                                || index[1] >= ny as isize
                                || index[2] >= nz as isize
                            {
                                continue;
                            }
                            let cell = (index[2] as usize * ny + index[1] as usize) * nx
                                + index[0] as usize;
                            if solid[cell] {
                                continue;
                            }
                            let fx = scaled[0] - lower[0] as f64;
                            let fy = scaled[1] - lower[1] as f64;
                            let fz = scaled[2] - lower[2] as f64;
                            let weight = (if dx == 1 { fx } else { 1.0 - fx })
                                * (if dy == 1 { fy } else { 1.0 - fy })
                                * (if dz == 1 { fz } else { 1.0 - fz });
                            if weight > 0.0 {
                                weighted.push((cell, weight));
                            }
                        }
                    }
                }
            } else {
                let radius = 3.5 * spread;
                for z in 0..nz {
                    let cz = (z as f64 + 0.5) * self.spacing[2];
                    if (cz - position[2]).abs() > radius {
                        continue;
                    }
                    for y in 0..ny {
                        let cy = (y as f64 + 0.5) * self.spacing[1];
                        if (cy - position[1]).abs() > radius {
                            continue;
                        }
                        for x in 0..nx {
                            let cx = (x as f64 + 0.5) * self.spacing[0];
                            if (cx - position[0]).abs() > radius {
                                continue;
                            }
                            let cell = (z * ny + y) * nx + x;
                            if solid[cell] {
                                continue;
                            }
                            let distance2 = (cx - position[0]).powi(2)
                                + (cy - position[1]).powi(2)
                                + (cz - position[2]).powi(2);
                            weighted.push((cell, (-distance2 / (2.0 * spread * spread)).exp()));
                        }
                    }
                }
            }
            if weighted.is_empty() {
                let mut nearest = None;
                let mut best = f64::INFINITY;
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            let cell = (z * ny + y) * nx + x;
                            if solid[cell] {
                                continue;
                            }
                            let distance = ((x as f64 + 0.5) * self.spacing[0] - position[0])
                                .powi(2)
                                + ((y as f64 + 0.5) * self.spacing[1] - position[1]).powi(2)
                                + ((z as f64 + 0.5) * self.spacing[2] - position[2]).powi(2);
                            if distance < best {
                                best = distance;
                                nearest = Some(cell);
                            }
                        }
                    }
                }
                let cell =
                    nearest.ok_or_else(|| PyValueError::new_err("field has no fluid cells"))?;
                weighted.push((cell, 1.0));
            }
            let total: f64 = weighted.iter().map(|item| item.1).sum();
            for &(cell, weight) in &weighted {
                current[channel * cells + cell] +=
                    mass * (weight / total) / (self.spacing[0] * self.spacing[1] * self.spacing[2]);
            }
            injected[channel] += mass;
        }
        Ok(())
    }

    fn transport(
        &mut self,
        dt: f64,
        current: &mut [f64],
        flow: &[f64],
        permeability: &[f64],
        solid: &[bool],
        barriers: Option<[&[f64]; 3]>,
    ) {
        let [nx, ny, nz] = self.shape;
        let cells = nx * ny * nz;
        self.change.fill(0.0);
        for axis in 0..3 {
            let stride = [1, nx, nx * ny][axis];
            let lengths = [
                nx - usize::from(axis == 0),
                ny - usize::from(axis == 1),
                nz - usize::from(axis == 2),
            ];
            let spacing = self.spacing[axis];
            // Flux uses the immutable beginning-of-substep concentration.
            for channel in 0..self.channels {
                let offset = channel * cells;
                let diffusion = self.diffusion[channel];
                let mut face = 0;
                for z in 0..lengths[2] {
                    for y in 0..lengths[1] {
                        for x in 0..lengths[0] {
                            let i = (z * ny + y) * nx + x;
                            let j = i + stride;
                            // SAFETY: step validates all contiguous shapes.
                            // The face extents exclude the last cell on this
                            // axis, hence i and j=i+stride are < cells. Channel
                            // and flow offsets are bounded by their validated
                            // leading dimensions. `face` walks the validated
                            // axis-face array exactly once. Scratch is allocated
                            // for channels*cells at construction. Removing these
                            // redundant inner bounds checks allows vectorization.
                            unsafe {
                                let mut p = permeability
                                    .get_unchecked(i)
                                    .min(*permeability.get_unchecked(j))
                                    * if !solid.get_unchecked(i) && !solid.get_unchecked(j) {
                                        1.0
                                    } else {
                                        0.0
                                    };
                                if let Some(faces) = barriers {
                                    p *= faces[axis].get_unchecked(face);
                                }
                                let left = *current.get_unchecked(offset + i);
                                let right = *current.get_unchecked(offset + j);
                                let diffusive =
                                    diffusion * p * (left - right) / (spacing * spacing);
                                let velocity = 0.5
                                    * (flow.get_unchecked(axis * cells + i)
                                        + flow.get_unchecked(axis * cells + j));
                                let upwind = if velocity >= 0.0 { left } else { right };
                                let advective = p * velocity * upwind / spacing;
                                *self.flux.get_unchecked_mut(offset + i) = diffusive + advective;
                            }
                            face += 1;
                        }
                    }
                }
            }
            // These are deliberately two passes: incoming-before-outgoing can
            // change float64 rounding at interior cells.
            for channel in 0..self.channels {
                let offset = channel * cells;
                for z in 0..lengths[2] {
                    for y in 0..lengths[1] {
                        let start = offset + (z * ny + y) * nx;
                        for x in 0..lengths[0] {
                            self.change[start + x] -= self.flux[start + x];
                        }
                    }
                }
            }
            for channel in 0..self.channels {
                let offset = channel * cells;
                for z in 0..lengths[2] {
                    for y in 0..lengths[1] {
                        let start = offset + (z * ny + y) * nx;
                        for x in 0..lengths[0] {
                            self.change[start + x + stride] += self.flux[start + x];
                        }
                    }
                }
            }
        }
        for (value, change) in current.iter_mut().zip(&self.change) {
            *value += dt * change;
        }
    }
}
