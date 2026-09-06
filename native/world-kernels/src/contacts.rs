use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
        };
        value.reserve(capacity);
        Ok(value)
    }

    #[pyo3(signature = (model_address, data_address, contact_count, timestep, impulse_limit, work_limit))]
    fn evaluate<'py>(
        &mut self,
        py: Python<'py>,
        model_address: usize,
        data_address: usize,
        contact_count: usize,
        timestep: f64,
        impulse_limit: f64,
        work_limit: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        if model_address == 0 || data_address == 0 {
            return Err(PyValueError::new_err(
                "MuJoCo model and data addresses must be nonzero",
            ));
        }
        if contact_count > 1_000_000
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
            )
        };
        if count < 0 || count as usize != contact_count {
            return Err(PyValueError::new_err(
                "MuJoCo contact count changed during native evaluation",
            ));
        }
        let n = count as usize;
        let positions = Array2::from_shape_vec((n, 3), self.positions[..n * 3].to_vec())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let normals = Array2::from_shape_vec((n, 3), self.normals[..n * 3].to_vec())
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
        ))
    }
}
