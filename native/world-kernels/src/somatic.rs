//! Recurring private physiology for the current twelve-action organism.

use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

const MAGIC: &[u8; 8] = b"CHSOM3\0\0";
const ACTIONS: usize = 12;
const TRAITS: usize = 18;
const EPS: f64 = 1e-12;

fn previous_nonnegative(value: f64) -> f64 {
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        0.0
    }
}

const MAINTENANCE_RATE: usize = 0;
const ACTIVATION_RATE: usize = 1;
const FATIGUE_RISE: usize = 2;
const FATIGUE_RECOVERY: usize = 3;
const ALLOCATION_RATE: usize = 4;
const SECRETION_RATE: usize = 5;
const RELEASE_RATE: usize = 6;
const STRUCTURAL_CAPACITY: usize = 7;
const GLAND_CAPACITY: usize = 8;
const BROOD_CAPACITY: usize = 9;
const RELEASE_RADIUS: usize = 10;
const WEIGHT_STRUCTURE: usize = 11;
const WEIGHT_GLAND: usize = 12;
const WEIGHT_BROOD: usize = 13;
const EXCHANGE_DECAY_RATE: usize = 14;
const EAT_ACTIVITY_COST: usize = 15;
const SECRETE_ACTIVITY_COST: usize = 16;
const ALLOCATE_ACTIVITY_COST: usize = 17;

struct PendingStep {
    dt: f64,
    actions: Vec<f64>,
    funded_scale: Vec<f64>,
    unmet_maintenance: Vec<f64>,
}

#[pyclass]
pub struct SomaticCohort {
    residents: usize,
    config_sha256: String,
    traits: Vec<f64>,
    fatigue: Vec<f64>,
    peak_structure_fraction: Vec<f64>,
    exchange_load: Vec<f64>,
    paid_maintenance: Vec<f64>,
    paid_activation: Vec<f64>,
    pending: Option<PendingStep>,
}

fn finite_unit(values: &[f64]) -> bool {
    values
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[pymethods]
impl SomaticCohort {
    #[new]
    fn new(
        config_sha256: String,
        traits: PyReadonlyArray2<'_, f64>,
        initial_fatigue: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let shape = traits.shape();
        if shape.len() != 2
            || !(1..=32).contains(&shape[0])
            || shape[1] != TRAITS
            || initial_fatigue.shape() != [shape[0]]
            || !valid_hash(&config_sha256)
        {
            return Err(PyValueError::new_err(
                "somatic configuration identity or dimensions differ",
            ));
        }
        let residents = shape[0];
        let traits = traits.as_slice()?.to_vec();
        let fatigue = initial_fatigue.as_slice()?.to_vec();
        if !traits
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            || !finite_unit(&fatigue)
        {
            return Err(PyValueError::new_err(
                "somatic traits and initial state must be finite and nonnegative",
            ));
        }
        for row in traits.chunks_exact(TRAITS) {
            if row[STRUCTURAL_CAPACITY] <= 0.0
                || row[GLAND_CAPACITY] <= 0.0
                || row[BROOD_CAPACITY] <= 0.0
                || row[RELEASE_RADIUS] <= 0.0
                || row[EXCHANGE_DECAY_RATE] <= 0.0
                || row[WEIGHT_STRUCTURE] + row[WEIGHT_GLAND] + row[WEIGHT_BROOD] <= 0.0
            {
                return Err(PyValueError::new_err(
                    "somatic capacities, release radius, decay, and allocation weights must be positive",
                ));
            }
        }
        Ok(Self {
            residents,
            config_sha256,
            traits,
            fatigue,
            peak_structure_fraction: vec![0.0; residents],
            exchange_load: vec![0.0; residents],
            paid_maintenance: vec![0.0; residents],
            paid_activation: vec![0.0; residents],
            pending: None,
        })
    }

    #[getter]
    fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    #[getter]
    fn resident_count(&self) -> usize {
        self.residents
    }

    fn expanded(
        &self,
        config_sha256: String,
        trait_row: PyReadonlyArray1<'_, f64>,
        initial_fatigue: f64,
    ) -> PyResult<Self> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot expand somatic state inside a physical step",
            ));
        }
        if self.residents >= 32
            || !valid_hash(&config_sha256)
            || trait_row.shape() != [TRAITS]
            || !initial_fatigue.is_finite()
            || !(0.0..=1.0).contains(&initial_fatigue)
        {
            return Err(PyValueError::new_err(
                "somatic expansion identity or dimensions differ",
            ));
        }
        let row = trait_row.as_slice()?;
        if row.iter().any(|value| !value.is_finite() || *value < 0.0)
            || row[STRUCTURAL_CAPACITY] <= 0.0
            || row[GLAND_CAPACITY] <= 0.0
            || row[BROOD_CAPACITY] <= 0.0
            || row[RELEASE_RADIUS] <= 0.0
            || row[EXCHANGE_DECAY_RATE] <= 0.0
            || row[WEIGHT_STRUCTURE] + row[WEIGHT_GLAND] + row[WEIGHT_BROOD] <= 0.0
        {
            return Err(PyValueError::new_err(
                "somatic expansion traits are invalid",
            ));
        }
        let mut traits = self.traits.clone();
        traits.extend_from_slice(row);
        let mut fatigue = self.fatigue.clone();
        fatigue.push(initial_fatigue);
        let mut peak = self.peak_structure_fraction.clone();
        peak.push(0.0);
        let mut exchange = self.exchange_load.clone();
        exchange.push(0.0);
        let mut maintenance = self.paid_maintenance.clone();
        maintenance.push(0.0);
        let mut activation = self.paid_activation.clone();
        activation.push(0.0);
        Ok(Self {
            residents: self.residents + 1,
            config_sha256,
            traits,
            fatigue,
            peak_structure_fraction: peak,
            exchange_load: exchange,
            paid_maintenance: maintenance,
            paid_activation: activation,
            pending: None,
        })
    }

    fn begin<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'_, f32>,
        available_atp: PyReadonlyArray1<'_, f64>,
        dt: f64,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f32>>)> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "somatic cohort has an unfinished physical step",
            ));
        }
        if actions.shape() != [self.residents, ACTIONS]
            || available_atp.shape() != [self.residents]
            || !dt.is_finite()
            || !(0.0..=1.0).contains(&dt)
            || dt == 0.0
        {
            return Err(PyValueError::new_err(
                "somatic action, ATP, or timestep dimensions differ",
            ));
        }
        let input = actions.as_slice()?;
        let available = available_atp.as_slice()?;
        if available
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "somatic available ATP must be finite and nonnegative",
            ));
        }
        let mut clean = Vec::with_capacity(input.len());
        let mut payments = vec![0.0; self.residents * 2];
        let mut scales = vec![0.0; self.residents];
        let mut unmet = vec![0.0; self.residents];
        for resident in 0..self.residents {
            let action = &input[resident * ACTIONS..(resident + 1) * ACTIONS];
            for (index, value) in action.iter().enumerate() {
                let value = *value as f64;
                let valid = value.is_finite()
                    && if index < 4 {
                        (-1.0..=1.0).contains(&value)
                    } else {
                        (0.0..=1.0).contains(&value)
                    };
                if !valid {
                    return Err(PyValueError::new_err(
                        "somatic actions violate the signed/rectified contract",
                    ));
                }
                clean.push(value);
            }
            let action = &clean[resident * ACTIONS..(resident + 1) * ACTIONS];
            let traits = &self.traits[resident * TRAITS..(resident + 1) * TRAITS];
            let activity = 0.08
                + 0.45 * action[0].abs()
                + 0.18 * action[1].abs()
                + 0.22 * action[3].abs()
                + 0.15 * action[4]
                + 0.12 * (action[5] + action[6] + action[7])
                + traits[EAT_ACTIVITY_COST] * action[8]
                + 0.06 * action[9]
                + traits[SECRETE_ACTIVITY_COST] * action[10]
                + traits[ALLOCATE_ACTIVITY_COST] * action[11];
            let maintenance_requested = dt * traits[MAINTENANCE_RATE];
            let maintenance = maintenance_requested.min(available[resident]);
            let remaining = (available[resident] - maintenance).max(0.0);
            let activation_requested = dt * traits[ACTIVATION_RATE] * activity;
            let mut activation = activation_requested.min(remaining);
            // The two independently meaningful ledger entries must also form
            // one representable debit no larger than the ATP snapshot. At the
            // depleted boundary, subtraction followed by addition can round a
            // single ULP above `available` even though each partition is valid.
            if maintenance + activation > available[resident] {
                activation = previous_nonnegative(activation);
            }
            payments[resident * 2] = maintenance;
            payments[resident * 2 + 1] = activation;
            scales[resident] = if activation_requested > EPS {
                activation / activation_requested
            } else {
                1.0
            };
            unmet[resident] = if maintenance_requested > EPS {
                1.0 - maintenance / maintenance_requested
            } else {
                0.0
            };
            self.paid_maintenance[resident] += maintenance;
            self.paid_activation[resident] += activation;
        }
        self.pending = Some(PendingStep {
            dt,
            actions: clean,
            funded_scale: scales.clone(),
            unmet_maintenance: unmet,
        });
        Ok((
            Array2::from_shape_vec((self.residents, 2), payments)
                .unwrap()
                .into_pyarray(py),
            scales
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        effort: PyReadonlyArray1<'_, f64>,
        structure_mass: PyReadonlyArray1<'_, f64>,
        gland_mass: PyReadonlyArray1<'_, f64>,
        brood_mass: PyReadonlyArray1<'_, f64>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pending = self.pending.take().ok_or_else(|| {
            PyRuntimeError::new_err("somatic finish requires one prepared physical step")
        })?;
        if pending.dt != dt
            || effort.shape() != [self.residents]
            || structure_mass.shape() != [self.residents]
            || gland_mass.shape() != [self.residents]
            || brood_mass.shape() != [self.residents]
        {
            self.pending = Some(pending);
            return Err(PyValueError::new_err(
                "somatic finish arrays or timestep differ from begin",
            ));
        }
        let effort = effort.as_slice()?;
        let structure = structure_mass.as_slice()?;
        let gland = gland_mass.as_slice()?;
        let brood = brood_mass.as_slice()?;
        if [effort, structure, gland, brood].iter().any(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        }) {
            self.pending = Some(pending);
            return Err(PyValueError::new_err(
                "somatic finish state must be finite and nonnegative",
            ));
        }
        let mut allocation = vec![0.0; self.residents * 3];
        let mut secretion = vec![0.0; self.residents];
        let mut release = vec![0.0; self.residents];
        let mut state = vec![0.0; self.residents * 6];
        for resident in 0..self.residents {
            let traits = &self.traits[resident * TRAITS..(resident + 1) * TRAITS];
            let actions = &pending.actions[resident * ACTIONS..(resident + 1) * ACTIONS];
            let scale = pending.funded_scale[resident];
            let drive = (effort[resident] * scale).max(pending.unmet_maintenance[resident]);
            self.fatigue[resident] = (self.fatigue[resident]
                + dt * (traits[FATIGUE_RISE] * drive
                    - traits[FATIGUE_RECOVERY] * (1.0 - drive.min(1.0))))
            .clamp(0.0, 1.0);
            let fills = [
                (structure[resident] / traits[STRUCTURAL_CAPACITY]).clamp(0.0, 1.0),
                (gland[resident] / traits[GLAND_CAPACITY]).clamp(0.0, 1.0),
                (brood[resident] / traits[BROOD_CAPACITY]).clamp(0.0, 1.0),
            ];
            self.peak_structure_fraction[resident] =
                self.peak_structure_fraction[resident].max(fills[0]);
            let integrity = if self.peak_structure_fraction[resident] > EPS {
                (fills[0] / self.peak_structure_fraction[resident]).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let total_weight =
                traits[WEIGHT_STRUCTURE] + traits[WEIGHT_GLAND] + traits[WEIGHT_BROOD];
            let total = dt * traits[ALLOCATION_RATE] * actions[11] * scale;
            for (index, (weight_index, capacity)) in [
                (WEIGHT_STRUCTURE, traits[STRUCTURAL_CAPACITY]),
                (WEIGHT_GLAND, traits[GLAND_CAPACITY]),
                (WEIGHT_BROOD, traits[BROOD_CAPACITY]),
            ]
            .iter()
            .enumerate()
            {
                let requested = total * traits[*weight_index] / total_weight;
                allocation[resident * 3 + index] = requested.min(capacity * (1.0 - fills[index]));
            }
            secretion[resident] =
                (dt * traits[SECRETION_RATE] * actions[10] * scale).min(gland[resident]);
            release[resident] = dt * traits[RELEASE_RATE] * actions[9] * scale;
            let offset = resident * 6;
            state[offset] = self.fatigue[resident];
            state[offset + 1] = integrity;
            state[offset + 2] = fills[0];
            state[offset + 3] = fills[1];
            state[offset + 4] = fills[2];
            state[offset + 5] = self.exchange_load[resident];
        }
        let result = PyDict::new(py);
        result.set_item(
            "allocation_mass",
            Array2::from_shape_vec((self.residents, 3), allocation)
                .unwrap()
                .into_pyarray(py),
        )?;
        result.set_item("secretion_mass", secretion.into_pyarray(py))?;
        result.set_item("release_mass", release.into_pyarray(py))?;
        result.set_item(
            "state",
            Array2::from_shape_vec((self.residents, 6), state)
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok(result)
    }

    fn record_exchange(
        &mut self,
        exchanged_mass: PyReadonlyArray1<'_, f64>,
        dt: f64,
    ) -> PyResult<()> {
        if exchanged_mass.shape() != [self.residents] || !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
            return Err(PyValueError::new_err(
                "exchange-load array or timestep differs",
            ));
        }
        for (resident, value) in exchanged_mass.as_slice()?.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 {
                return Err(PyValueError::new_err(
                    "exchange mass must be finite and nonnegative",
                ));
            }
            let traits = &self.traits[resident * TRAITS..(resident + 1) * TRAITS];
            let reference =
                traits[ALLOCATION_RATE] + traits[SECRETION_RATE] + traits[RELEASE_RATE] + EPS;
            let target = (*value / dt / reference).clamp(0.0, 1.0);
            let alpha = 1.0 - (-traits[EXCHANGE_DECAY_RATE] * dt).exp();
            self.exchange_load[resident] += alpha * (target - self.exchange_load[resident]);
        }
        Ok(())
    }

    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let result = PyDict::new(py);
        result
            .set_item("fatigue", self.fatigue.clone().into_pyarray(py))
            .unwrap();
        result
            .set_item(
                "peak_structure_fraction",
                self.peak_structure_fraction.clone().into_pyarray(py),
            )
            .unwrap();
        result
            .set_item("exchange_load", self.exchange_load.clone().into_pyarray(py))
            .unwrap();
        result
            .set_item(
                "paid_maintenance",
                self.paid_maintenance.clone().into_pyarray(py),
            )
            .unwrap();
        result
            .set_item(
                "paid_activation",
                self.paid_activation.clone().into_pyarray(py),
            )
            .unwrap();
        result
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot snapshot somatic state inside a physical step",
            ));
        }
        let mut output = Vec::with_capacity(80 + self.residents * 5 * 8);
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(self.config_sha256.as_bytes());
        output.extend_from_slice(&(self.residents as u64).to_le_bytes());
        for values in [
            &self.fatigue,
            &self.peak_structure_fraction,
            &self.exchange_load,
            &self.paid_maintenance,
            &self.paid_activation,
        ] {
            for value in values {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(PyBytes::new(py, &output))
    }

    fn restore(&mut self, snapshot: &[u8]) -> PyResult<()> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot restore somatic state inside a physical step",
            ));
        }
        let expected = 8 + 64 + 8 + self.residents * 5 * 8;
        if snapshot.len() != expected
            || &snapshot[..8] != MAGIC
            || &snapshot[8..72] != self.config_sha256.as_bytes()
            || u64::from_le_bytes(snapshot[72..80].try_into().unwrap()) as usize != self.residents
        {
            return Err(PyValueError::new_err(
                "somatic snapshot identity or dimensions differ",
            ));
        }
        let mut cursor = 80;
        let mut read = || -> PyResult<Vec<f64>> {
            let mut values = Vec::with_capacity(self.residents);
            for _ in 0..self.residents {
                let value = f64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap());
                cursor += 8;
                if !value.is_finite() || value < 0.0 {
                    return Err(PyValueError::new_err(
                        "somatic snapshot contains invalid state",
                    ));
                }
                values.push(value);
            }
            Ok(values)
        };
        let fatigue = read()?;
        let peak = read()?;
        let exchange = read()?;
        let maintenance = read()?;
        let activation = read()?;
        if !finite_unit(&fatigue) || !finite_unit(&peak) || !finite_unit(&exchange) {
            return Err(PyValueError::new_err(
                "somatic snapshot normalized state is outside [0,1]",
            ));
        }
        self.fatigue = fatigue;
        self.peak_structure_fraction = peak;
        self.exchange_load = exchange;
        self.paid_maintenance = maintenance;
        self.paid_activation = activation;
        Ok(())
    }
}
