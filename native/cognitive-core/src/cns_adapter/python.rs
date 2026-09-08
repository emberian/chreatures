use super::*;
use numpy::{
    ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pymethods]
impl CnsAdapter {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        p: PyReadonlyArray1<'_, f32>,
        n: PyReadonlyArray1<'_, f32>,
        rr: PyReadonlyArray1<'_, u32>,
        rt: PyReadonlyArray1<'_, u32>,
        ptr: PyReadonlyArray1<'_, u32>,
        si: PyReadonlyArray1<'_, u32>,
        sw: PyReadonlyArray1<'_, f32>,
        br: PyReadonlyArray1<'_, u32>,
        bm: PyReadonlyArray2<'_, f32>,
        cr: PyReadonlyArray1<'_, u32>,
        mr: PyReadonlyArray1<'_, u32>,
        mm: PyReadonlyArray2<'_, f32>,
        nt: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Self> {
        Self::from_packed(
            p.as_slice()?,
            n.as_slice()?,
            rr.as_slice()?,
            rt.as_slice()?,
            ptr.as_slice()?,
            si.as_slice()?,
            sw.as_slice()?,
            br.as_slice()?,
            bm.as_slice()?,
            cr.as_slice()?,
            mr.as_slice()?,
            mm.as_slice()?,
            nt.as_slice()?,
        )
        .map_err(PyValueError::new_err)
    }
    fn encode_drive<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray2<'py, f32>,
        c: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let sh = s.shape();
        if sh.len() != 2
            || sh[1] != INPUT
            || c.shape() != [sh[0], CTX]
            || !(1..=32).contains(&sh[0])
        {
            return Err(PyValueError::new_err(
                "V3 sensory [B,5423], context [B,12] required",
            ));
        }
        let b = sh[0];
        let v = self
            .encode_drive_flat(s.as_slice()?, c.as_slice()?, b)
            .map_err(PyValueError::new_err)?;
        Ok(Array2::from_shape_vec((b, N), v).unwrap().into_pyarray(py))
    }
    fn outputs<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let sh = r.shape();
        if sh.len() != 2 || sh[1] != N {
            return Err(PyValueError::new_err("V3 rates must be [B,165122]"));
        }
        let b = sh[0];
        let (m, l) = self
            .outputs_flat(r.as_slice()?, b)
            .map_err(PyValueError::new_err)?;
        Ok((
            Array2::from_shape_vec((b, MOTOR), m)
                .unwrap()
                .into_pyarray(py),
            Array2::from_shape_vec((b, LATENT), l)
                .unwrap()
                .into_pyarray(py),
        ))
    }
}
