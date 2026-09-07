//! Thin Python host for the portable anatomical adapter.
use super::*;
use numpy::{
    ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

fn host_fail(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}
#[pymethods]
impl CnsAdapter {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parameters: PyReadonlyArray1<'_, f32>,
        neutral_drive: PyReadonlyArray1<'_, f32>,
        receptor_rows: PyReadonlyArray1<'_, u32>,
        receptor_types: PyReadonlyArray1<'_, u32>,
        site_ptr: PyReadonlyArray1<'_, u32>,
        site_indices: PyReadonlyArray1<'_, u32>,
        site_weights: PyReadonlyArray1<'_, f32>,
        body_rows: PyReadonlyArray1<'_, u32>,
        neuron_types: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Self> {
        Self::from_packed(
            parameters.as_slice()?,
            neutral_drive.as_slice()?,
            receptor_rows.as_slice()?,
            receptor_types.as_slice()?,
            site_ptr.as_slice()?,
            site_indices.as_slice()?,
            site_weights.as_slice()?,
            body_rows.as_slice()?,
            neuron_types.as_slice()?,
        )
        .map_err(PyValueError::new_err)
    }

    fn encode_afferents<'py>(
        &self,
        py: Python<'py>,
        sensory: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = sensory.shape();
        if shape.len() != 2 || shape[1] != INPUT || !(1..=32).contains(&shape[0]) {
            return Err(host_fail("CNS sensory input must be [B,5356], B in1..32"));
        }
        let batch = shape[0];
        let values = sensory.as_slice()?;
        let result = py
            .detach(|| self.encode_afferents_flat(values, batch))
            .map_err(PyValueError::new_err)?;
        Ok(Array2::from_shape_vec((batch, N), result)
            .unwrap()
            .into_pyarray(py))
    }

    fn project_rates<'py>(
        &self,
        py: Python<'py>,
        rates: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = rates.shape();
        if shape.len() != 2 || shape[1] != N || !(1..=32).contains(&shape[0]) {
            return Err(host_fail("CNS rates must be [B,165122], B in1..32"));
        }
        let batch = shape[0];
        let values = rates.as_slice()?;
        let result = py
            .detach(|| self.project_rates_flat(values, batch))
            .map_err(PyValueError::new_err)?;
        Ok(Array2::from_shape_vec((batch, LATENT), result)
            .unwrap()
            .into_pyarray(py))
    }

    fn effective_dynamics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (name, values) in [
            "baseline",
            "recurrent_gain",
            "tau",
            "adaptation_gain",
            "adaptation_tau",
        ]
        .iter()
        .zip(self.effective_dynamics_flat())
        {
            result.set_item(
                *name,
                numpy::ndarray::Array1::from_vec(values).into_pyarray(py),
            )?;
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_rates<'py>(
        &self,
        py: Python<'py>,
        graph_crow: PyReadonlyArray1<'py, u32>,
        graph_col: PyReadonlyArray1<'py, u32>,
        graph_weight: PyReadonlyArray1<'py, f32>,
        drive: PyReadonlyArray2<'py, f32>,
        rate: PyReadonlyArray2<'py, f32>,
        adaptation: PyReadonlyArray2<'py, f32>,
        support: PyReadonlyArray2<'py, f32>,
        dt: f32,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let shape = rate.shape();
        if shape.len() != 2 || shape[1] != N || !(1..=32).contains(&shape[0]) {
            return Err(host_fail("V2 CNS state must be [B,165122], B in1..32"));
        }
        let batch = shape[0];
        if drive.shape() != shape || adaptation.shape() != shape || support.shape() != shape {
            return Err(host_fail("V2 drive/rate/adaptation/support shapes differ"));
        }
        let graph_crow = graph_crow.as_slice()?;
        let graph_col = graph_col.as_slice()?;
        let graph_weight = graph_weight.as_slice()?;
        let drive = drive.as_slice()?;
        let rate = rate.as_slice()?;
        let adaptation = adaptation.as_slice()?;
        let support = support.as_slice()?;
        let (next_rate, next_adaptation, next_support) = py.detach(|| {
            self.step_rates_flat(
                graph_crow,
                graph_col,
                graph_weight,
                drive,
                rate,
                adaptation,
                support,
                dt,
                batch,
            )
            .map_err(PyValueError::new_err)
        })?;
        let array = |values| {
            Array2::from_shape_vec((batch, N), values)
                .unwrap()
                .into_pyarray(py)
        };
        Ok((
            array(next_rate),
            array(next_adaptation),
            array(next_support),
        ))
    }
}
