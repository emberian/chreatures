//! Batched bilateral MuJoCo rays for the anatomical optic sampler.
//!
//! This class owns only physical scene tracing.  The paired `OpticRetina`
//! owns the versioned eye calibration and screen-texture transduction.  Python
//! crosses each native boundary once per cohort and never iterates over rays.

use numpy::{
    ndarray::{Array2, Array3},
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const SITES: usize = 1771;
const MAX_RESIDENTS: usize = 4096;

unsafe extern "C" {
    fn chreatures_mujoco_header_version() -> i32;
    fn chreatures_mujoco_runtime_version() -> i32;
    fn chreatures_optic_scene_bind(
        model: *const std::ffi::c_void,
        residents: i32,
        head_geoms: *const i32,
        excluded_body: i32,
    ) -> i32;
    fn chreatures_optic_scene_sample(
        model: *const std::ffi::c_void,
        data: *mut std::ffi::c_void,
        residents: i32,
        head_geoms: *const i32,
        site_sides: *const i16,
        left_sites: i32,
        eye_origins: *const f64,
        ray_directions: *const f64,
        illumination: *const f64,
        maximum_range: f64,
        excluded_body: i32,
        background_rgb: *const f32,
        direction_scratch: *mut f64,
        distance_output: *mut f64,
        geom_output: *mut i32,
        rgb_output: *mut f32,
    ) -> i32;
}

/// Persistent scene-ray binding for one fixed MuJoCo resident cohort.
#[pyclass]
pub struct SceneRayBatch {
    model_address: usize,
    residents: usize,
    left_sites: usize,
    maximum_range: f64,
    excluded_body: i32,
    head_geoms: Vec<i32>,
    site_sides: Vec<i16>,
    eye_origins: Vec<f64>,
    ray_directions: Vec<f64>,
    background_rgb: Vec<f32>,
    direction_scratch: Vec<f64>,
    distance_output: Vec<f64>,
    geom_output: Vec<i32>,
    rgb_output: Vec<f32>,
}

#[pymethods]
impl SceneRayBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_address: usize,
        head_geoms: PyReadonlyArray1<'_, i32>,
        site_sides: PyReadonlyArray1<'_, i16>,
        eye_origins: PyReadonlyArray2<'_, f64>,
        ray_directions: PyReadonlyArray2<'_, f64>,
        maximum_range: f64,
        excluded_body: i32,
        background_rgb: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        let heads = head_geoms.as_slice()?;
        let sides = site_sides.as_slice()?;
        let origins = eye_origins.as_slice()?;
        let directions = ray_directions.as_slice()?;
        let background = background_rgb.as_slice()?;
        let residents = heads.len();
        let left_sites = sides.iter().take_while(|side| **side == 1).count();
        if model_address == 0
            || residents == 0
            || residents > MAX_RESIDENTS
            || site_sides.shape() != [SITES]
            || left_sites == 0
            || left_sites == SITES
            || sides[..left_sites].iter().any(|side| *side != 1)
            || sides[left_sites..].iter().any(|side| *side != 2)
            || eye_origins.shape() != [2, 3]
            || ray_directions.shape() != [SITES, 3]
            || origins
                .iter()
                .chain(directions)
                .any(|value| !value.is_finite())
            || directions.chunks_exact(3).any(|direction| {
                let norm = direction.iter().map(|value| value * value).sum::<f64>();
                (norm - 1.0).abs() > 1e-8
            })
            || !maximum_range.is_finite()
            || maximum_range <= 0.0
            || excluded_body < -1
            || background.len() != 3
            || background
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(PyValueError::new_err("invalid bilateral scene-ray binding"));
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
        let bound = unsafe {
            chreatures_optic_scene_bind(
                model_address as *const _,
                residents as i32,
                heads.as_ptr(),
                excluded_body,
            )
        };
        if bound != residents as i32 {
            return Err(PyValueError::new_err(
                "scene-ray cohort does not match the bound MuJoCo model",
            ));
        }
        Ok(Self {
            model_address,
            residents,
            left_sites,
            maximum_range,
            excluded_body,
            head_geoms: heads.to_vec(),
            site_sides: sides.to_vec(),
            eye_origins: origins.to_vec(),
            ray_directions: directions.to_vec(),
            background_rgb: background.to_vec(),
            direction_scratch: vec![0.0; SITES * 3],
            distance_output: vec![f64::INFINITY; residents * SITES],
            geom_output: vec![-1; residents * SITES],
            rgb_output: vec![0.0; residents * SITES * 3],
        })
    }

    #[staticmethod]
    fn identity() -> &'static str {
        "native-bilateral-mujoco-scene-rays-v1"
    }

    #[getter]
    fn residents(&self) -> usize {
        self.residents
    }

    #[getter]
    fn rays_per_resident(&self) -> usize {
        SITES
    }

    /// Return nearest distance, RGB, and private geometry IDs in anatomical order.
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        model_address: usize,
        data_address: usize,
        illumination: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<i32>>,
    )> {
        let light = illumination.as_slice()?;
        if model_address == 0
            || model_address != self.model_address
            || data_address == 0
            || light.len() != self.residents
            || light
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(PyValueError::new_err("invalid bilateral scene-ray sample"));
        }
        let sampled = py.detach(|| unsafe {
            chreatures_optic_scene_sample(
                model_address as *const _,
                data_address as *mut _,
                self.residents as i32,
                self.head_geoms.as_ptr(),
                self.site_sides.as_ptr(),
                self.left_sites as i32,
                self.eye_origins.as_ptr(),
                self.ray_directions.as_ptr(),
                light.as_ptr(),
                self.maximum_range,
                self.excluded_body,
                self.background_rgb.as_ptr(),
                self.direction_scratch.as_mut_ptr(),
                self.distance_output.as_mut_ptr(),
                self.geom_output.as_mut_ptr(),
                self.rgb_output.as_mut_ptr(),
            )
        });
        if sampled != self.residents as i32 {
            return Err(PyValueError::new_err(format!(
                "native bilateral scene-ray sample failed with status {sampled}"
            )));
        }
        let distance =
            Array2::from_shape_vec((self.residents, SITES), self.distance_output.clone())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let rgb = Array3::from_shape_vec((self.residents, SITES, 3), self.rgb_output.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let geom = Array2::from_shape_vec((self.residents, SITES), self.geom_output.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((
            distance.into_pyarray(py),
            rgb.into_pyarray(py),
            geom.into_pyarray(py),
        ))
    }
}
