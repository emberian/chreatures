//! Funded embryo maturation with deferred, generation-qualified hatch offers.

use numpy::{IntoPyArray, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

const MAGIC: &[u8; 8] = b"CHLIF1\0\0";
const EPS: f64 = 1e-12;

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[pyclass]
pub struct LifecycleCohort {
    residents: usize,
    config_sha256: String,
    material_target: Vec<f64>,
    energy_target: Vec<f64>,
    maturation_rate: Vec<f64>,
    maturity: Vec<f64>,
    pending_serial: Vec<i64>,
    birth_count: Vec<u64>,
    next_serial: u64,
}

#[pymethods]
impl LifecycleCohort {
    #[new]
    fn new(
        config_sha256: String,
        material_target: PyReadonlyArray1<'_, f64>,
        energy_target: PyReadonlyArray1<'_, f64>,
        maturation_rate: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let residents = material_target.len();
        if !(1..=32).contains(&residents)
            || energy_target.shape() != [residents]
            || maturation_rate.shape() != [residents]
            || !valid_hash(&config_sha256)
        {
            return Err(PyValueError::new_err(
                "lifecycle configuration identity or dimensions differ",
            ));
        }
        let material_target = material_target.as_slice()?.to_vec();
        let energy_target = energy_target.as_slice()?.to_vec();
        let maturation_rate = maturation_rate.as_slice()?.to_vec();
        if [&material_target, &energy_target, &maturation_rate]
            .iter()
            .any(|values| {
                values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            })
        {
            return Err(PyValueError::new_err(
                "lifecycle targets and rates must be finite and positive",
            ));
        }
        Ok(Self {
            residents,
            config_sha256,
            material_target,
            energy_target,
            maturation_rate,
            maturity: vec![0.0; residents],
            pending_serial: vec![-1; residents],
            birth_count: vec![0; residents],
            next_serial: 1,
        })
    }

    #[getter]
    fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    fn expanded(
        &self,
        config_sha256: String,
        material_target: f64,
        energy_target: f64,
        maturation_rate: f64,
    ) -> PyResult<Self> {
        if self.residents >= 32
            || !valid_hash(&config_sha256)
            || [material_target, energy_target, maturation_rate]
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(
                "lifecycle expansion identity or traits differ",
            ));
        }
        let mut material = self.material_target.clone();
        material.push(material_target);
        let mut energy = self.energy_target.clone();
        energy.push(energy_target);
        let mut rates = self.maturation_rate.clone();
        rates.push(maturation_rate);
        let mut maturity = self.maturity.clone();
        maturity.push(0.0);
        let mut pending = self.pending_serial.clone();
        pending.push(-1);
        let mut births = self.birth_count.clone();
        births.push(0);
        Ok(Self {
            residents: self.residents + 1,
            config_sha256,
            material_target: material,
            energy_target: energy,
            maturation_rate: rates,
            maturity,
            pending_serial: pending,
            birth_count: births,
            next_serial: self.next_serial,
        })
    }

    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        brood_material: PyReadonlyArray1<'_, f64>,
        brood_energy: PyReadonlyArray1<'_, f64>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if brood_material.shape() != [self.residents]
            || brood_energy.shape() != [self.residents]
            || !dt.is_finite()
            || dt <= 0.0
            || dt > 1.0
        {
            return Err(PyValueError::new_err(
                "lifecycle state arrays or timestep differ",
            ));
        }
        let material = brood_material.as_slice()?;
        let energy = brood_energy.as_slice()?;
        let mut funding = vec![0.0; self.residents];
        for resident in 0..self.residents {
            if !material[resident].is_finite()
                || material[resident] < 0.0
                || !energy[resident].is_finite()
                || energy[resident] < 0.0
            {
                return Err(PyValueError::new_err(
                    "lifecycle funding must be finite and nonnegative",
                ));
            }
            funding[resident] = (material[resident] / self.material_target[resident])
                .min(energy[resident] / self.energy_target[resident])
                .clamp(0.0, 1.0);
            if self.pending_serial[resident] < 0 {
                self.maturity[resident] = (self.maturity[resident]
                    + dt * self.maturation_rate[resident] * funding[resident])
                    .clamp(0.0, 1.0);
                if self.maturity[resident] >= 1.0 - EPS {
                    if self.next_serial > i64::MAX as u64 {
                        return Err(PyRuntimeError::new_err(
                            "lifecycle offer serial capacity exhausted",
                        ));
                    }
                    self.pending_serial[resident] = self.next_serial as i64;
                    self.next_serial += 1;
                    self.maturity[resident] = 1.0;
                }
            }
        }
        let result = PyDict::new(py);
        result.set_item("funding", funding.into_pyarray(py))?;
        result.set_item("maturity", self.maturity.clone().into_pyarray(py))?;
        result.set_item(
            "pending_serial",
            self.pending_serial.clone().into_pyarray(py),
        )?;
        result.set_item("birth_count", self.birth_count.clone().into_pyarray(py))?;
        Ok(result)
    }

    fn offers<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let result = PyDict::new(py);
        result
            .set_item(
                "resident_index",
                self.pending_serial
                    .iter()
                    .enumerate()
                    .filter_map(|(index, serial)| (*serial >= 0).then_some(index as i32))
                    .collect::<Vec<_>>()
                    .into_pyarray(py),
            )
            .unwrap();
        result
            .set_item(
                "serial",
                self.pending_serial
                    .iter()
                    .filter(|serial| **serial >= 0)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_pyarray(py),
            )
            .unwrap();
        result
    }

    fn commit(
        &mut self,
        resident_indices: PyReadonlyArray1<'_, i32>,
        serials: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        if resident_indices.shape() != serials.shape() {
            return Err(PyValueError::new_err("hatch commit arrays must align"));
        }
        let indices = resident_indices.as_slice()?;
        let serials = serials.as_slice()?;
        let mut seen = vec![false; self.residents];
        for (&raw_index, &serial) in indices.iter().zip(serials) {
            let index = usize::try_from(raw_index)
                .map_err(|_| PyValueError::new_err("hatch resident index is invalid"))?;
            if index >= self.residents
                || seen[index]
                || serial < 1
                || self.pending_serial[index] != serial
            {
                return Err(PyValueError::new_err(
                    "hatch offer is absent, stale, or duplicated",
                ));
            }
            seen[index] = true;
        }
        for (&raw_index, _) in indices.iter().zip(serials) {
            let index = raw_index as usize;
            self.pending_serial[index] = -1;
            self.maturity[index] = 0.0;
            self.birth_count[index] += 1;
        }
        Ok(())
    }

    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let result = PyDict::new(py);
        result
            .set_item("maturity", self.maturity.clone().into_pyarray(py))
            .unwrap();
        result
            .set_item(
                "pending_serial",
                self.pending_serial.clone().into_pyarray(py),
            )
            .unwrap();
        result
            .set_item("birth_count", self.birth_count.clone().into_pyarray(py))
            .unwrap();
        result
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut output = Vec::with_capacity(88 + self.residents * 24);
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(self.config_sha256.as_bytes());
        output.extend_from_slice(&(self.residents as u64).to_le_bytes());
        output.extend_from_slice(&self.next_serial.to_le_bytes());
        for value in &self.maturity {
            output.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.pending_serial {
            output.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.birth_count {
            output.extend_from_slice(&value.to_le_bytes());
        }
        PyBytes::new(py, &output)
    }

    fn restore(&mut self, snapshot: &[u8]) -> PyResult<()> {
        let expected = 8 + 64 + 16 + self.residents * 24;
        if snapshot.len() != expected
            || &snapshot[..8] != MAGIC
            || &snapshot[8..72] != self.config_sha256.as_bytes()
            || u64::from_le_bytes(snapshot[72..80].try_into().unwrap()) as usize != self.residents
        {
            return Err(PyValueError::new_err(
                "lifecycle snapshot identity or dimensions differ",
            ));
        }
        let next_serial = u64::from_le_bytes(snapshot[80..88].try_into().unwrap());
        if next_serial < 1 || next_serial > i64::MAX as u64 {
            return Err(PyValueError::new_err("lifecycle next serial is invalid"));
        }
        let mut cursor = 88;
        let mut maturity = Vec::with_capacity(self.residents);
        let mut pending = Vec::with_capacity(self.residents);
        let mut births = Vec::with_capacity(self.residents);
        for _ in 0..self.residents {
            let value = f64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(PyValueError::new_err("lifecycle maturity is invalid"));
            }
            maturity.push(value);
        }
        for _ in 0..self.residents {
            let value = i64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            if value < -1 || value >= next_serial as i64 {
                return Err(PyValueError::new_err("lifecycle pending serial is invalid"));
            }
            pending.push(value);
        }
        for _ in 0..self.residents {
            births.push(u64::from_le_bytes(
                snapshot[cursor..cursor + 8].try_into().unwrap(),
            ));
            cursor += 8;
        }
        if pending
            .iter()
            .enumerate()
            .any(|(index, serial)| *serial >= 0 && maturity[index] != 1.0)
        {
            return Err(PyValueError::new_err(
                "lifecycle offer exists without a mature embryo",
            ));
        }
        self.next_serial = next_serial;
        self.maturity = maturity;
        self.pending_serial = pending;
        self.birth_count = births;
        Ok(())
    }
}
