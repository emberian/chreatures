use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const CONTROLLER_COUNT: usize = 15;
const DYNAMIC_COUNT: usize = 5;

unsafe extern "C" {
    fn chreatures_mujoco_header_version() -> i32;
    fn chreatures_mujoco_runtime_version() -> i32;
    fn chreatures_actuation_batch(
        model: *const std::ffi::c_void,
        data: *mut std::ffi::c_void,
        residents: i32,
        legs: i32,
        roots: *const i32,
        qpos: *const i32,
        dofs: *const i32,
        sides: *const f64,
        phases: *const f64,
        controller: *const f64,
        dynamic: *const f64,
        grips: *const i32,
        world_time: f64,
        timestep: f64,
        phase_kind: i32,
        work: *mut f64,
    ) -> i32;
}

#[pyclass]
pub struct ActuationCohort {
    residents: usize,
    legs: usize,
    roots: Vec<i32>,
    qpos: Vec<i32>,
    dofs: Vec<i32>,
    sides: Vec<f64>,
    phases: Vec<f64>,
    controller: Vec<f64>,
    dynamic: Vec<f64>,
    grips: Vec<i32>,
    work: Vec<f64>,
}

#[pymethods]
impl ActuationCohort {
    #[staticmethod]
    fn identity() -> &'static str {
        "native-articulated-cohort-v1"
    }

    #[new]
    fn new(
        roots: Vec<i32>,
        qpos: Vec<i32>,
        dofs: Vec<i32>,
        sides: Vec<f64>,
        phases: Vec<f64>,
        controller: Vec<f64>,
    ) -> PyResult<Self> {
        let residents = roots.len();
        if residents == 0
            || residents > 32
            || qpos.len() != dofs.len()
            || qpos.len() % (residents * 2) != 0
        {
            return Err(PyValueError::new_err("invalid actuation cohort layout"));
        }
        let legs = qpos.len() / (residents * 2);
        if sides.len() != residents * legs
            || phases.len() != residents * legs
            || controller.len() != residents * CONTROLLER_COUNT
        {
            return Err(PyValueError::new_err(
                "invalid actuation cohort coefficient dimensions",
            ));
        }
        let (header, runtime) = unsafe {
            (
                chreatures_mujoco_header_version(),
                chreatures_mujoco_runtime_version(),
            )
        };
        if header != runtime {
            return Err(PyValueError::new_err("MuJoCo native ABI differs"));
        }
        Ok(Self {
            residents,
            legs,
            roots,
            qpos,
            dofs,
            sides,
            phases,
            controller,
            dynamic: vec![0.0; residents * DYNAMIC_COUNT],
            grips: vec![-1; residents],
            work: vec![0.0; residents],
        })
    }

    fn begin_tick(
        &mut self,
        dynamic: PyReadonlyArray1<'_, f64>,
        grips: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        let dynamic = dynamic.as_slice()?;
        let grips = grips.as_slice()?;
        if dynamic.len() != self.dynamic.len()
            || grips.len() != self.residents
            || dynamic.iter().any(|value| !value.is_finite())
            || grips.iter().any(|value| *value < -1)
        {
            return Err(PyValueError::new_err(
                "invalid dynamic actuation dimensions",
            ));
        }
        self.dynamic.copy_from_slice(dynamic);
        self.grips.copy_from_slice(grips);
        self.work.fill(0.0);
        Ok(())
    }

    fn apply_phase(
        &mut self,
        py: Python<'_>,
        model_address: usize,
        data_address: usize,
        world_time: f64,
        timestep: f64,
        phase_kind: i32,
    ) -> PyResult<()> {
        if model_address == 0
            || data_address == 0
            || !world_time.is_finite()
            || !timestep.is_finite()
            || timestep <= 0.0
        {
            return Err(PyValueError::new_err("invalid native actuation call"));
        }
        let result = py.detach(|| unsafe {
            chreatures_actuation_batch(
                model_address as *const _,
                data_address as *mut _,
                self.residents as i32,
                self.legs as i32,
                self.roots.as_ptr(),
                self.qpos.as_ptr(),
                self.dofs.as_ptr(),
                self.sides.as_ptr(),
                self.phases.as_ptr(),
                self.controller.as_ptr(),
                self.dynamic.as_ptr(),
                self.grips.as_ptr(),
                world_time,
                timestep,
                phase_kind,
                self.work.as_mut_ptr(),
            )
        });
        if result != self.residents as i32 {
            return Err(PyValueError::new_err("native actuation failed"));
        }
        Ok(())
    }

    fn apply_gait(
        &mut self,
        py: Python<'_>,
        model_address: usize,
        data_address: usize,
        world_time: f64,
        timestep: f64,
    ) -> PyResult<()> {
        self.apply_phase(py, model_address, data_address, world_time, timestep, 0)
    }

    fn apply_grip(
        &mut self,
        py: Python<'_>,
        model_address: usize,
        data_address: usize,
        world_time: f64,
        timestep: f64,
    ) -> PyResult<()> {
        self.apply_phase(py, model_address, data_address, world_time, timestep, 1)
    }

    fn finish_tick<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.work)
    }
}
