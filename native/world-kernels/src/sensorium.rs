use numpy::{ndarray::Array3, IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ELEVATIONS: usize = 5;
const AZIMUTHS: usize = 16;
const COMPONENTS: usize = 4;
const RAY_COUNT: usize = ELEVATIONS * AZIMUTHS;

/// Convert one exact MuJoCo multi-ray result into the physical retinal raster.
///
/// Collision remains authoritative in MuJoCo.  The model arrays are borrowed
/// on every call because ecology can change material colours without changing
/// model topology.  Arithmetic is deliberately f64 and ordered like the
/// historical Python receptor loop.
#[pyfunction]
#[pyo3(signature = (distances, geom_ids, geom_matid, mat_rgba, geom_rgba, illumination, max_range))]
pub fn transduce_retina<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray1<'py, f64>,
    geom_ids: PyReadonlyArray1<'py, i32>,
    geom_matid: PyReadonlyArray1<'py, i32>,
    mat_rgba: PyReadonlyArray2<'py, f32>,
    geom_rgba: PyReadonlyArray2<'py, f32>,
    illumination: f64,
    max_range: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let distances = distances
        .as_slice()
        .map_err(|_| PyValueError::new_err("retinal distances must be contiguous"))?;
    let geom_ids = geom_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("retinal geom ids must be contiguous"))?;
    let geom_matid = geom_matid
        .as_slice()
        .map_err(|_| PyValueError::new_err("geom material ids must be contiguous"))?;
    let mat_rgba = mat_rgba.as_array();
    let geom_rgba = geom_rgba.as_array();
    if distances.len() != RAY_COUNT || geom_ids.len() != RAY_COUNT {
        return Err(PyValueError::new_err(
            "retinal ray result must contain 80 entries",
        ));
    }
    if geom_rgba.nrows() != geom_matid.len()
        || geom_rgba.ncols() < 3
        || mat_rgba.ncols() < 3
        || !illumination.is_finite()
        || !(0.0..=1.0).contains(&illumination)
        || !max_range.is_finite()
        || max_range <= 0.0
    {
        return Err(PyValueError::new_err(
            "invalid retinal model arrays or limits",
        ));
    }

    let light = 0.45_f64 + 0.55_f64 * illumination;
    let mut output = vec![0.0_f64; RAY_COUNT * COMPONENTS];
    for index in 0..RAY_COUNT {
        let distance = distances[index];
        let geom_id = geom_ids[index];
        if !distance.is_finite() {
            return Err(PyValueError::new_err("retinal distance must be finite"));
        }
        // mj_multiRay's cutoff is only a broad-phase optimization and can
        // report a farther exact hit.  Keep the public threshold here.
        if distance < 0.0 || distance > max_range || geom_id < 0 {
            continue;
        }
        let geom = geom_id as usize;
        if geom >= geom_matid.len() {
            return Err(PyValueError::new_err(
                "retinal geom id is outside the model",
            ));
        }
        let material = geom_matid[geom];
        let base = index * COMPONENTS;
        for channel in 0..3 {
            let source = if material >= 0 {
                let material = material as usize;
                if material >= mat_rgba.nrows() {
                    return Err(PyValueError::new_err(
                        "retinal material id is outside the model",
                    ));
                }
                mat_rgba[[material, channel]] as f64
            } else {
                geom_rgba[[geom, channel]] as f64
            };
            if !source.is_finite() {
                return Err(PyValueError::new_err("retinal colour must be finite"));
            }
            output[base + channel] = (source * light).min(1.0);
        }
        output[base + 3] = (1.0_f64 - distance / max_range).max(0.0);
    }
    let result = Array3::from_shape_vec((ELEVATIONS, AZIMUTHS, COMPONENTS), output)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(result.into_pyarray(py))
}
