use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

unsafe extern "C" {
    fn chreatures_mujoco_header_version() -> i32;
    fn chreatures_mujoco_runtime_version() -> i32;
    fn chreatures_contact_batch(
        model: *const std::ffi::c_void,
        data: *mut std::ffi::c_void,
        capacity: i32,
        timestep: f64,
        impulse_limit: f64,
        work_limit: f64,
        geom1: *mut i32,
        geom2: *mut i32,
        positions: *mut f64,
        normals: *mut f64,
        relative_speed: *mut f64,
        impulse: *mut f64,
        impact_work: *mut f64,
        contact_force_norm: *mut f64,
        geom_count: i32,
        geom_resident: *const i32,
        geom_entity: *const i32,
        resident_body: *const i32,
        resident_z: *const f64,
        resident_count: i32,
        participant_resident: *mut i32,
        participant_entity: *mut i32,
        participant_side: *mut i8,
    ) -> i32;
}

#[pyclass]
pub struct ContactBatch {
    geom1: Vec<i32>,
    geom2: Vec<i32>,
    positions: Vec<f64>,
    normals: Vec<f64>,
    relative_speed: Vec<f64>,
    impulse: Vec<f64>,
    impact_work: Vec<f64>,
    contact_force_norm: Vec<f64>,
    geom_resident: Vec<i32>,
    geom_entity: Vec<i32>,
    geom_shape: Vec<i32>,
    resident_body: Vec<i32>,
    participant_resident: Vec<i32>,
    participant_entity: Vec<i32>,
    participant_side: Vec<i8>,
    tick_limit: usize,
    tick_keys: HashSet<(i32, i32, i32, i64, i64, i64)>,
    tick_resident: Vec<i32>,
    tick_counterpart_resident: Vec<i32>,
    tick_counterpart_entity: Vec<i32>,
    tick_shape: Vec<i32>,
    tick_point: Vec<f64>,
    tick_impulse: Vec<f64>,
    tick_work: Vec<f64>,
    tick_speed: Vec<f64>,
}

impl ContactBatch {
    fn reserve(&mut self, count: usize) {
        self.geom1.resize(count, 0);
        self.geom2.resize(count, 0);
        self.positions.resize(count * 3, 0.0);
        self.normals.resize(count * 3, 0.0);
        self.relative_speed.resize(count, 0.0);
        self.impulse.resize(count, 0.0);
        self.impact_work.resize(count, 0.0);
        self.contact_force_norm.resize(count, 0.0);
        self.participant_resident.resize(count * 2, -1);
        self.participant_entity.resize(count * 2, -1);
        self.participant_side.resize(count * 2, -1);
    }
}

#[pymethods]
impl ContactBatch {
    #[new]
    #[pyo3(signature = (capacity=256))]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 || capacity > 1_000_000 {
            return Err(PyValueError::new_err(
                "contact capacity must be in 1..=1000000",
            ));
        }
        let (header, runtime) = unsafe {
            (
                chreatures_mujoco_header_version(),
                chreatures_mujoco_runtime_version(),
            )
        };
        if header != runtime {
            return Err(PyValueError::new_err(format!(
                "MuJoCo native ABI differs: compiled {header}, loaded {runtime}"
            )));
        }
        let mut value = Self {
            geom1: Vec::new(),
            geom2: Vec::new(),
            positions: Vec::new(),
            normals: Vec::new(),
            relative_speed: Vec::new(),
            impulse: Vec::new(),
            impact_work: Vec::new(),
            contact_force_norm: Vec::new(),
            geom_resident: Vec::new(),
            geom_entity: Vec::new(),
            geom_shape: Vec::new(),
            resident_body: Vec::new(),
            participant_resident: Vec::new(),
            participant_entity: Vec::new(),
            participant_side: Vec::new(),
            tick_limit: 0,
            tick_keys: HashSet::new(),
            tick_resident: Vec::new(),
            tick_counterpart_resident: Vec::new(),
            tick_counterpart_entity: Vec::new(),
            tick_shape: Vec::new(),
            tick_point: Vec::new(),
            tick_impulse: Vec::new(),
            tick_work: Vec::new(),
            tick_speed: Vec::new(),
        };
        value.reserve(capacity);
        Ok(value)
    }

    fn bind_metadata(
        &mut self,
        geom_resident: PyReadonlyArray1<'_, i32>,
        geom_entity: PyReadonlyArray1<'_, i32>,
        geom_shape: PyReadonlyArray1<'_, i32>,
        resident_body: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        let geom_resident = geom_resident.as_slice()?;
        let geom_entity = geom_entity.as_slice()?;
        let geom_shape = geom_shape.as_slice()?;
        let resident_body = resident_body.as_slice()?;
        if geom_resident.is_empty()
            || geom_resident.len() != geom_entity.len()
            || geom_resident.len() != geom_shape.len()
            || resident_body.is_empty()
            || resident_body.len() > 32
            || geom_resident
                .iter()
                .any(|value| *value < -1 || (*value >= 0 && *value as usize >= resident_body.len()))
            || geom_entity.iter().any(|value| *value < -1)
            || geom_shape.iter().any(|value| *value < -1)
            || resident_body.iter().any(|value| *value < 0)
        {
            return Err(PyValueError::new_err("invalid contact metadata binding"));
        }
        self.geom_resident = geom_resident.to_vec();
        self.geom_entity = geom_entity.to_vec();
        self.geom_shape = geom_shape.to_vec();
        self.resident_body = resident_body.to_vec();
        Ok(())
    }

    fn begin_tick(&mut self, limit: usize) -> PyResult<()> {
        if limit == 0 || limit > 65_536 {
            return Err(PyValueError::new_err("invalid packed contact limit"));
        }
        self.tick_limit = limit;
        self.tick_keys.clear();
        self.tick_resident.clear();
        self.tick_counterpart_resident.clear();
        self.tick_counterpart_entity.clear();
        self.tick_shape.clear();
        self.tick_point.clear();
        self.tick_impulse.clear();
        self.tick_work.clear();
        self.tick_speed.clear();
        Ok(())
    }

    #[pyo3(signature = (model_address, data_address, contact_count, timestep, impulse_limit, work_limit, resident_z))]
    fn evaluate<'py>(
        &mut self,
        py: Python<'py>,
        model_address: usize,
        data_address: usize,
        contact_count: usize,
        timestep: f64,
        impulse_limit: f64,
        work_limit: f64,
        resident_z: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<i32>>,
        Bound<'py, PyArray2<i32>>,
        Bound<'py, PyArray2<i8>>,
    )> {
        let resident_z = resident_z.as_slice()?;
        if model_address == 0 || data_address == 0 {
            return Err(PyValueError::new_err(
                "MuJoCo model and data addresses must be nonzero",
            ));
        }
        if self.geom_resident.is_empty()
            || resident_z.len() != self.resident_body.len()
            || resident_z.iter().any(|value| !value.is_finite())
            || contact_count > 1_000_000
            || !timestep.is_finite()
            || timestep <= 0.0
            || !impulse_limit.is_finite()
            || impulse_limit < 0.0
            || !work_limit.is_finite()
            || work_limit < 0.0
        {
            return Err(PyValueError::new_err(
                "invalid contact batch dimensions or limits",
            ));
        }
        if contact_count > self.geom1.len() {
            self.reserve(contact_count.next_power_of_two());
        }
        let count = unsafe {
            chreatures_contact_batch(
                model_address as *const std::ffi::c_void,
                data_address as *mut std::ffi::c_void,
                contact_count as i32,
                timestep,
                impulse_limit,
                work_limit,
                self.geom1.as_mut_ptr(),
                self.geom2.as_mut_ptr(),
                self.positions.as_mut_ptr(),
                self.normals.as_mut_ptr(),
                self.relative_speed.as_mut_ptr(),
                self.impulse.as_mut_ptr(),
                self.impact_work.as_mut_ptr(),
                self.contact_force_norm.as_mut_ptr(),
                self.geom_resident.len() as i32,
                self.geom_resident.as_ptr(),
                self.geom_entity.as_ptr(),
                self.resident_body.as_ptr(),
                resident_z.as_ptr(),
                resident_z.len() as i32,
                self.participant_resident.as_mut_ptr(),
                self.participant_entity.as_mut_ptr(),
                self.participant_side.as_mut_ptr(),
            )
        };
        if count < 0 || count as usize != contact_count {
            return Err(PyValueError::new_err(format!(
                "native contact aggregation failed ({count}; expected {contact_count})",
            )));
        }
        let n = count as usize;
        if self.tick_limit > 0 {
            for index in 0..n {
                for side in 0..2 {
                    if self.tick_resident.len() >= self.tick_limit {
                        break;
                    }
                    let resident = self.participant_resident[index * 2 + side];
                    if resident < 0 {
                        continue;
                    }
                    let key = (
                        resident,
                        self.geom1[index],
                        self.geom2[index],
                        (self.positions[index * 3] * 10_000.0).round_ties_even() as i64,
                        (self.positions[index * 3 + 1] * 10_000.0).round_ties_even() as i64,
                        (self.positions[index * 3 + 2] * 10_000.0).round_ties_even() as i64,
                    );
                    if !self.tick_keys.insert(key) {
                        continue;
                    }
                    let other_geom = if side == 0 {
                        self.geom2[index]
                    } else {
                        self.geom1[index]
                    };
                    self.tick_resident.push(resident);
                    self.tick_counterpart_resident
                        .push(self.geom_resident[other_geom as usize]);
                    self.tick_counterpart_entity
                        .push(self.geom_entity[other_geom as usize]);
                    self.tick_shape.push(self.geom_shape[other_geom as usize]);
                    self.tick_point
                        .extend_from_slice(&self.positions[index * 3..index * 3 + 3]);
                    self.tick_impulse.push(self.impulse[index]);
                    self.tick_work.push(self.impact_work[index]);
                    self.tick_speed.push(self.relative_speed[index]);
                }
            }
        }
        let positions = Array2::from_shape_vec((n, 3), self.positions[..n * 3].to_vec())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let normals = Array2::from_shape_vec((n, 3), self.normals[..n * 3].to_vec())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let participant_resident =
            Array2::from_shape_vec((n, 2), self.participant_resident[..n * 2].to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let participant_entity =
            Array2::from_shape_vec((n, 2), self.participant_entity[..n * 2].to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let participant_side =
            Array2::from_shape_vec((n, 2), self.participant_side[..n * 2].to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((
            PyArray1::from_slice(py, &self.geom1[..n]),
            PyArray1::from_slice(py, &self.geom2[..n]),
            positions.into_pyarray(py),
            normals.into_pyarray(py),
            PyArray1::from_slice(py, &self.relative_speed[..n]),
            PyArray1::from_slice(py, &self.impulse[..n]),
            PyArray1::from_slice(py, &self.impact_work[..n]),
            PyArray1::from_slice(py, &self.contact_force_norm[..n]),
            participant_resident.into_pyarray(py),
            participant_entity.into_pyarray(py),
            participant_side.into_pyarray(py),
        ))
    }

    #[allow(clippy::type_complexity)]
    fn finish_tick<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let count = self.tick_resident.len();
        let points = Array2::from_shape_vec((count, 3), self.tick_point.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((
            PyArray1::from_slice(py, &self.tick_resident),
            PyArray1::from_slice(py, &self.tick_counterpart_resident),
            PyArray1::from_slice(py, &self.tick_counterpart_entity),
            PyArray1::from_slice(py, &self.tick_shape),
            points.into_pyarray(py),
            PyArray1::from_slice(py, &self.tick_impulse),
            PyArray1::from_slice(py, &self.tick_work),
            PyArray1::from_slice(py, &self.tick_speed),
        ))
    }
}
