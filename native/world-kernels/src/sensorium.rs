//! Body-bound cohort retina for the live MuJoCo model.
//!
//! Immutable resident/head bindings and angular ray templates are copied once
//! when the model is bound. Each sample then crosses Python once, releases the
//! GIL, transforms every ray from the current physical head pose and gaze, and
//! invokes `mj_multiRay` exactly once per resident. MuJoCo remains authoritative
//! for current poses, collision geometry, materials, and self-occlusion.

use numpy::{
    ndarray::{Array2, Array4},
    IntoPyArray, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const COMPONENTS: usize = 4;
const COARSE_ELEVATIONS: usize = 5;
const COARSE_AZIMUTHS: usize = 16;
const MAX_RESIDENTS: usize = 4096;
const MAX_RAYS: usize = 1_000_000;

unsafe extern "C" {
    fn chreatures_mujoco_header_version() -> i32;
    fn chreatures_mujoco_runtime_version() -> i32;
    fn chreatures_retina_bind(
        model: *const std::ffi::c_void,
        residents: i32,
        root_geoms: *const i32,
        head_geoms: *const i32,
    ) -> i32;
    fn chreatures_retina_cohort(
        model: *const std::ffi::c_void,
        data: *mut std::ffi::c_void,
        residents: i32,
        root_geoms: *const i32,
        head_geoms: *const i32,
        gaze_pitch: *const f64,
        illumination: *const f64,
        ray_count: i32,
        peripheral_rays: i32,
        ray_templates: *const f64,
        coarse_elevation_offsets: *const i32,
        maximum_range: f64,
        direction_scratch: *mut f64,
        distance_scratch: *mut f64,
        geom_scratch: *mut i32,
        coarse_output: *mut f32,
        rich_output: *mut f32,
    ) -> i32;
}

/// Persistent bindings and scratch for one fixed resident cohort.
#[pyclass]
pub struct RetinaCohort {
    model_address: usize,
    residents: usize,
    ray_count: usize,
    peripheral_rays: usize,
    maximum_range: f64,
    profile_sha256: String,
    root_geoms: Vec<i32>,
    head_geoms: Vec<i32>,
    ray_templates: Vec<f64>,
    coarse_elevation_offsets: Vec<i32>,
    direction_scratch: Vec<f64>,
    distance_scratch: Vec<f64>,
    geom_scratch: Vec<i32>,
    coarse_output: Vec<f32>,
    rich_output: Vec<f32>,
}

#[pymethods]
impl RetinaCohort {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_address: usize,
        root_geoms: PyReadonlyArray1<'_, i32>,
        head_geoms: PyReadonlyArray1<'_, i32>,
        ray_templates: PyReadonlyArray2<'_, f64>,
        peripheral_rays: usize,
        coarse_elevation_offsets: PyReadonlyArray1<'_, i32>,
        maximum_range: f64,
        profile_sha256: String,
    ) -> PyResult<Self> {
        let roots = root_geoms.as_slice()?;
        let heads = head_geoms.as_slice()?;
        let templates = ray_templates.as_slice()?;
        let offsets = coarse_elevation_offsets.as_slice()?;
        let residents = roots.len();
        let dimensions = ray_templates.shape();
        let ray_count = dimensions.first().copied().unwrap_or(0);
        if model_address == 0
            || residents == 0
            || residents > MAX_RESIDENTS
            || heads.len() != residents
            || dimensions.len() != 2
            || dimensions[1] != 3
            || ray_count == 0
            || ray_count > MAX_RAYS
            || peripheral_rays != 8 * 32
            || peripheral_rays > ray_count
            || offsets != [0, 2, 3, 5, 6, 8]
            || !maximum_range.is_finite()
            || maximum_range <= 0.0
            || profile_sha256.len() != 64
            || !profile_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || templates.iter().any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err("invalid retina cohort binding"));
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
            chreatures_retina_bind(
                model_address as *const _,
                residents as i32,
                roots.as_ptr(),
                heads.as_ptr(),
            )
        };
        if bound != residents as i32 {
            return Err(PyValueError::new_err(
                "retina cohort does not match the bound MuJoCo model",
            ));
        }
        Ok(Self {
            model_address,
            residents,
            ray_count,
            peripheral_rays,
            maximum_range,
            profile_sha256: profile_sha256.to_ascii_lowercase(),
            root_geoms: roots.to_vec(),
            head_geoms: heads.to_vec(),
            ray_templates: templates.to_vec(),
            coarse_elevation_offsets: offsets.to_vec(),
            direction_scratch: vec![0.0; ray_count * 3],
            distance_scratch: vec![-1.0; ray_count],
            geom_scratch: vec![-1; ray_count],
            coarse_output: vec![0.0; residents * COARSE_ELEVATIONS * COARSE_AZIMUTHS * COMPONENTS],
            rich_output: vec![0.0; residents * ray_count * COMPONENTS],
        })
    }

    #[staticmethod]
    fn identity() -> &'static str {
        "native-body-retina-cohort-v1"
    }

    #[getter]
    fn profile_sha256(&self) -> &str {
        &self.profile_sha256
    }

    #[getter]
    fn residents(&self) -> usize {
        self.residents
    }

    #[getter]
    fn rays_per_resident(&self) -> usize {
        self.ray_count
    }

    /// Return `(coarse_5x16, rich_packed)` for the entire resident cohort.
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        model_address: usize,
        data_address: usize,
        gaze_pitch: PyReadonlyArray1<'_, f64>,
        illumination: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<f32>>)> {
        let gaze = gaze_pitch.as_slice()?;
        let light = illumination.as_slice()?;
        if model_address == 0
            || model_address != self.model_address
            || data_address == 0
            || gaze.len() != self.residents
            || light.len() != self.residents
            || gaze.iter().any(|value| !value.is_finite())
            || light
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(PyValueError::new_err("invalid retina cohort sample"));
        }
        let sampled = py.detach(|| unsafe {
            chreatures_retina_cohort(
                model_address as *const _,
                data_address as *mut _,
                self.residents as i32,
                self.root_geoms.as_ptr(),
                self.head_geoms.as_ptr(),
                gaze.as_ptr(),
                light.as_ptr(),
                self.ray_count as i32,
                self.peripheral_rays as i32,
                self.ray_templates.as_ptr(),
                self.coarse_elevation_offsets.as_ptr(),
                self.maximum_range,
                self.direction_scratch.as_mut_ptr(),
                self.distance_scratch.as_mut_ptr(),
                self.geom_scratch.as_mut_ptr(),
                self.coarse_output.as_mut_ptr(),
                self.rich_output.as_mut_ptr(),
            )
        });
        if sampled != self.residents as i32 {
            return Err(PyValueError::new_err(format!(
                "native retina sample failed with status {sampled}"
            )));
        }
        let coarse = Array4::from_shape_vec(
            (
                self.residents,
                COARSE_ELEVATIONS,
                COARSE_AZIMUTHS,
                COMPONENTS,
            ),
            self.coarse_output.clone(),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let rich = Array2::from_shape_vec(
            (self.residents, self.ray_count * COMPONENTS),
            self.rich_output.clone(),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((coarse.into_pyarray(py), rich.into_pyarray(py)))
    }
}
