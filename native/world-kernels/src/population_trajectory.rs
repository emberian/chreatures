//! Native complete-life physical descriptors for population campaigns.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyDict},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const FORMAT: &str = "chreatures-population-trajectory-v1";
const PHYS: usize = 12;
const ACTIONS: usize = 12;
const OUTCOMES: usize = 8;
const FLOWS: usize = 3;
const DT: f64 = 0.05;

#[derive(Serialize, Deserialize)]
struct Snapshot {
    format: String,
    residents: usize,
    world_size: [f64; 3],
    spatial_bin_width: f64,
    completed_ticks: u64,
    valid_ticks: Vec<u64>,
    phys_sum: Vec<f64>,
    phys_min: Vec<f64>,
    phys_max: Vec<f64>,
    action_sum: Vec<f64>,
    action_abs_sum: Vec<f64>,
    outcome_sum: Vec<f64>,
    flow_sum: Vec<f64>,
    first_energy: Vec<f64>,
    last_energy: Vec<f64>,
    has_energy: Vec<bool>,
    mouth_ticks: Vec<u64>,
    mouth_bouts: Vec<u64>,
    mouth_active: Vec<bool>,
    contact_ticks: Vec<u64>,
    contact_bouts: Vec<u64>,
    contact_active: Vec<bool>,
    quiet_ticks: Vec<u64>,
    height_sum: Vec<f64>,
    height_min: Vec<f64>,
    height_max: Vec<f64>,
    outside_world_ticks: Vec<u64>,
    outside_deviation_sum: Vec<f64>,
    outside_deviation_max: Vec<f64>,
    cells: Vec<Vec<u64>>,
}

#[pyclass]
pub struct PopulationTrajectory {
    state: Snapshot,
    cells: Vec<HashSet<u64>>,
}

fn finite(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite())
}

fn signed_cell_key(bins: [i64; 3]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in bins {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[pymethods]
impl PopulationTrajectory {
    #[new]
    fn new(
        resident_count: usize,
        world_size: [f64; 3],
        spatial_bin_width: f64,
        dt: f64,
    ) -> PyResult<Self> {
        if resident_count == 0
            || resident_count > 4096
            || !finite(&world_size)
            || world_size.iter().any(|x| *x <= 0.0)
            || !spatial_bin_width.is_finite()
            || spatial_bin_width <= 0.0
            || (dt - DT).abs() > 1e-12
        {
            return Err(PyValueError::new_err(
                "population trajectory dimensions or 50 ms sampling contract differ",
            ));
        }
        let n = resident_count;
        Ok(Self {
            state: Snapshot {
                format: FORMAT.into(),
                residents: n,
                world_size,
                spatial_bin_width,
                completed_ticks: 0,
                valid_ticks: vec![0; n],
                phys_sum: vec![0.0; n * PHYS],
                phys_min: vec![0.0; n * PHYS],
                phys_max: vec![0.0; n * PHYS],
                action_sum: vec![0.0; n * ACTIONS],
                action_abs_sum: vec![0.0; n * ACTIONS],
                outcome_sum: vec![0.0; n * OUTCOMES],
                flow_sum: vec![0.0; n * FLOWS],
                first_energy: vec![0.0; n],
                last_energy: vec![0.0; n],
                has_energy: vec![false; n],
                mouth_ticks: vec![0; n],
                mouth_bouts: vec![0; n],
                mouth_active: vec![false; n],
                contact_ticks: vec![0; n],
                contact_bouts: vec![0; n],
                contact_active: vec![false; n],
                quiet_ticks: vec![0; n],
                height_sum: vec![0.0; n],
                height_min: vec![0.0; n],
                height_max: vec![0.0; n],
                outside_world_ticks: vec![0; n],
                outside_deviation_sum: vec![0.0; n],
                outside_deviation_max: vec![0.0; n],
                cells: Vec::new(),
            },
            cells: (0..n).map(|_| HashSet::new()).collect(),
        })
    }

    fn advance(
        &mut self,
        position: PyReadonlyArray2<'_, f64>,
        physiology12: PyReadonlyArray2<'_, f64>,
        executed12: PyReadonlyArray2<'_, f64>,
        outcomes: PyReadonlyArray2<'_, f64>,
        organ_flows: PyReadonlyArray2<'_, f64>,
        valid: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<()> {
        let n = self.state.residents;
        if position.shape() != [n, 3]
            || physiology12.shape() != [n, PHYS]
            || executed12.shape() != [n, ACTIONS]
            || outcomes.shape() != [n, OUTCOMES]
            || organ_flows.shape() != [n, FLOWS]
            || valid.as_ref().is_some_and(|x| x.shape() != [n])
        {
            return Err(PyValueError::new_err(
                "population trajectory advance dimensions differ",
            ));
        }
        let pos = position.as_slice()?;
        let phys = physiology12.as_slice()?;
        let actions = executed12.as_slice()?;
        let outcome = outcomes.as_slice()?;
        let flows = organ_flows.as_slice()?;
        let mask = valid.as_ref().map(|x| x.as_slice()).transpose()?;
        if !finite(pos) || !finite(phys) || !finite(actions) || !finite(outcome) || !finite(flows) {
            return Err(PyValueError::new_err(
                "population trajectory inputs must be finite",
            ));
        }
        if flows.iter().any(|x| *x < 0.0)
            || outcome
                .chunks_exact(OUTCOMES)
                .any(|row| row[..7].iter().any(|x| *x < 0.0))
        {
            return Err(PyValueError::new_err(
                "physical outcomes and organ flows must be nonnegative",
            ));
        }
        for r in 0..n {
            if !mask.map_or(true, |x| x[r]) {
                self.state.mouth_active[r] = false;
                self.state.contact_active[r] = false;
                continue;
            }
            let first = self.state.valid_ticks[r] == 0;
            self.state.valid_ticks[r] += 1;
            for j in 0..PHYS {
                let x = phys[r * PHYS + j];
                let k = r * PHYS + j;
                self.state.phys_sum[k] += x;
                if first {
                    self.state.phys_min[k] = x;
                    self.state.phys_max[k] = x;
                } else {
                    self.state.phys_min[k] = self.state.phys_min[k].min(x);
                    self.state.phys_max[k] = self.state.phys_max[k].max(x);
                }
            }
            for j in 0..ACTIONS {
                let x = actions[r * ACTIONS + j];
                let k = r * ACTIONS + j;
                self.state.action_sum[k] += x;
                self.state.action_abs_sum[k] += x.abs();
            }
            for j in 0..OUTCOMES {
                self.state.outcome_sum[r * OUTCOMES + j] += outcome[r * OUTCOMES + j];
            }
            for j in 0..FLOWS {
                self.state.flow_sum[r * FLOWS + j] += flows[r * FLOWS + j];
            }
            if !self.state.has_energy[r] {
                self.state.first_energy[r] = phys[r * PHYS];
                self.state.has_energy[r] = true;
            }
            self.state.last_energy[r] = phys[r * PHYS];
            let mouth = outcome[r * OUTCOMES + 6] > 0.0;
            if mouth {
                self.state.mouth_ticks[r] += 1;
                if !self.state.mouth_active[r] {
                    self.state.mouth_bouts[r] += 1;
                }
            }
            self.state.mouth_active[r] = mouth;
            let contact = outcome[r * OUTCOMES + 1] > 0.0;
            if contact {
                self.state.contact_ticks[r] += 1;
                if !self.state.contact_active[r] {
                    self.state.contact_bouts[r] += 1;
                }
            }
            self.state.contact_active[r] = contact;
            if outcome[r * OUTCOMES + 3] <= 1e-6 {
                self.state.quiet_ticks[r] += 1;
            }
            let z = pos[r * 3 + 2];
            self.state.height_sum[r] += z;
            if first {
                self.state.height_min[r] = z;
                self.state.height_max[r] = z;
            } else {
                self.state.height_min[r] = self.state.height_min[r].min(z);
                self.state.height_max[r] = self.state.height_max[r].max(z);
            }
            let deviation = (0..3)
                .map(|axis| {
                    if pos[r * 3 + axis] < 0.0 {
                        -pos[r * 3 + axis]
                    } else {
                        (pos[r * 3 + axis] - self.state.world_size[axis]).max(0.0)
                    }
                })
                .sum::<f64>();
            if deviation > 0.0 {
                self.state.outside_world_ticks[r] += 1;
                self.state.outside_deviation_sum[r] += deviation;
                self.state.outside_deviation_max[r] =
                    self.state.outside_deviation_max[r].max(deviation);
            }
            let bins = [0, 1, 2]
                .map(|axis| (pos[r * 3 + axis] / self.state.spatial_bin_width).floor() as i64);
            self.cells[r].insert(signed_cell_key(bins));
        }
        self.state.completed_ticks += 1;
        Ok(())
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.state.residents;
        let result = PyDict::new(py);
        result.set_item("format", FORMAT)?;
        result.set_item("sampling_contract", "complete physical ticks at fixed 50 ms; valid mask excludes absent lives; raw tick counts are not independent samples")?;
        result.set_item("sampling_dt_seconds", DT)?;
        result.set_item(
            "physiology_order",
            [
                "energy",
                "gut",
                "fatigue",
                "speed",
                "turn",
                "neural_support",
                "structural_integrity",
                "development_fraction",
                "gland_fill",
                "brood_fill",
                "reproductive_maturity",
                "exchange_load",
            ],
        )?;
        result.set_item(
            "executed_action_order",
            [
                "thrust",
                "yaw",
                "gaze_pitch",
                "posture",
                "grip",
                "signal_low",
                "signal_mid",
                "signal_high",
                "eat",
                "release",
                "secrete",
                "allocate",
            ],
        )?;
        result.set_item(
            "outcome_order",
            [
                "nutrition",
                "contact",
                "distance",
                "effort",
                "mechanical_work",
                "ingested_mass",
                "mouth_material_contacts",
                "homeostatic_reward",
            ],
        )?;
        result.set_item(
            "organ_flow_order",
            ["release_mass", "secretion_mass", "allocation_mass"],
        )?;
        result.set_item("spatial_bin_width", self.state.spatial_bin_width)?;
        result.set_item("world_size", self.state.world_size)?;
        result.set_item(
            "resident_axis_keys",
            [
                "valid_ticks",
                "has_valid_observation",
                "valid_time_seconds",
                "visited_spatial_cells",
                "mouth_contact_ticks",
                "mouth_contact_bouts",
                "contact_ticks",
                "contact_bouts",
                "quiet_ticks",
                "outside_world_ticks",
                "outside_deviation_sum",
                "outside_deviation_max",
                "physiology_mean",
                "physiology_min",
                "physiology_max",
                "executed_action_mean",
                "executed_action_abs_mean",
                "outcome_sum",
                "organ_flow_sum",
                "contact_sum",
                "distance_sum",
                "effort_sum",
                "mechanical_work_sum",
                "ingested_mass_sum",
                "signal_activity_sum",
                "release_mass_sum",
                "secretion_mass_sum",
                "allocation_mass_sum",
                "energy_change",
                "mean_actual_speed",
                "height_mean",
                "height_range",
            ],
        )?;
        result.set_item("completed_batch_ticks", self.state.completed_ticks)?;
        result.set_item(
            "valid_ticks",
            self.state.valid_ticks.clone().into_pyarray(py),
        )?;
        result.set_item(
            "has_valid_observation",
            self.state
                .valid_ticks
                .iter()
                .map(|x| *x > 0)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "valid_time_seconds",
            self.state
                .valid_ticks
                .iter()
                .map(|x| *x as f64 * DT)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "visited_spatial_cells",
            self.cells
                .iter()
                .map(|x| x.len() as u64)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        for (name, values) in [
            ("mouth_contact_ticks", &self.state.mouth_ticks),
            ("mouth_contact_bouts", &self.state.mouth_bouts),
            ("contact_ticks", &self.state.contact_ticks),
            ("contact_bouts", &self.state.contact_bouts),
            ("quiet_ticks", &self.state.quiet_ticks),
        ] {
            result.set_item(name, values.clone().into_pyarray(py))?;
        }
        result.set_item(
            "outside_world_ticks",
            self.state.outside_world_ticks.clone().into_pyarray(py),
        )?;
        result.set_item(
            "outside_deviation_sum",
            self.state.outside_deviation_sum.clone().into_pyarray(py),
        )?;
        result.set_item(
            "outside_deviation_max",
            self.state.outside_deviation_max.clone().into_pyarray(py),
        )?;
        let mean = |sum: &[f64], width: usize| -> Vec<f64> {
            (0..n * width)
                .map(|k| {
                    let ticks = self.state.valid_ticks[k / width];
                    if ticks > 0 {
                        sum[k] / ticks as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        result.set_item(
            "physiology_mean",
            PyArray2::from_vec2(
                py,
                &mean(&self.state.phys_sum, PHYS)
                    .chunks(PHYS)
                    .map(|x| x.to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "physiology_min",
            PyArray2::from_vec2(
                py,
                &(0..n)
                    .map(|r| self.state.phys_min[r * PHYS..(r + 1) * PHYS].to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "physiology_max",
            PyArray2::from_vec2(
                py,
                &(0..n)
                    .map(|r| self.state.phys_max[r * PHYS..(r + 1) * PHYS].to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "executed_action_mean",
            PyArray2::from_vec2(
                py,
                &mean(&self.state.action_sum, ACTIONS)
                    .chunks(ACTIONS)
                    .map(|x| x.to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "executed_action_abs_mean",
            PyArray2::from_vec2(
                py,
                &mean(&self.state.action_abs_sum, ACTIONS)
                    .chunks(ACTIONS)
                    .map(|x| x.to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "outcome_sum",
            PyArray2::from_vec2(
                py,
                &self
                    .state
                    .outcome_sum
                    .chunks(OUTCOMES)
                    .map(|x| x.to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        result.set_item(
            "organ_flow_sum",
            PyArray2::from_vec2(
                py,
                &self
                    .state
                    .flow_sum
                    .chunks(FLOWS)
                    .map(|x| x.to_vec())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        for (name, column) in [
            ("contact_sum", 1),
            ("distance_sum", 2),
            ("effort_sum", 3),
            ("mechanical_work_sum", 4),
            ("ingested_mass_sum", 5),
        ] {
            result.set_item(
                name,
                (0..n)
                    .map(|r| self.state.outcome_sum[r * OUTCOMES + column])
                    .collect::<Vec<_>>()
                    .into_pyarray(py),
            )?;
        }
        result.set_item(
            "signal_activity_sum",
            (0..n)
                .map(|r| {
                    self.state.action_abs_sum[r * ACTIONS + 5]
                        + self.state.action_abs_sum[r * ACTIONS + 6]
                        + self.state.action_abs_sum[r * ACTIONS + 7]
                })
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        for (name, column) in [
            ("release_mass_sum", 0),
            ("secretion_mass_sum", 1),
            ("allocation_mass_sum", 2),
        ] {
            result.set_item(
                name,
                (0..n)
                    .map(|r| self.state.flow_sum[r * FLOWS + column])
                    .collect::<Vec<_>>()
                    .into_pyarray(py),
            )?;
        }
        result.set_item(
            "energy_change",
            (0..n)
                .map(|r| {
                    if self.state.has_energy[r] {
                        self.state.last_energy[r] - self.state.first_energy[r]
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "mean_actual_speed",
            (0..n)
                .map(|r| {
                    if self.state.valid_ticks[r] > 0 {
                        self.state.phys_sum[r * PHYS + 3] / self.state.valid_ticks[r] as f64
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "height_mean",
            (0..n)
                .map(|r| {
                    if self.state.valid_ticks[r] > 0 {
                        self.state.height_sum[r] / self.state.valid_ticks[r] as f64
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "height_range",
            (0..n)
                .map(|r| {
                    if self.state.valid_ticks[r] > 0 {
                        self.state.height_max[r] - self.state.height_min[r]
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        Ok(result)
    }

    fn snapshot<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.state.cells = self
            .cells
            .iter()
            .map(|set| {
                let mut v = set.iter().copied().collect::<Vec<_>>();
                v.sort_unstable();
                v
            })
            .collect();
        let bytes =
            serde_json::to_vec(&self.state).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn restore(&mut self, bytes: &[u8]) -> PyResult<()> {
        let restored: Snapshot =
            serde_json::from_slice(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if restored.format != FORMAT
            || restored.residents != self.state.residents
            || restored.world_size != self.state.world_size
            || restored.spatial_bin_width != self.state.spatial_bin_width
            || restored.cells.len() != restored.residents
            || restored.valid_ticks.len() != restored.residents
            || restored.phys_sum.len() != restored.residents * PHYS
            || restored.phys_min.len() != restored.residents * PHYS
            || restored.phys_max.len() != restored.residents * PHYS
            || restored.action_sum.len() != restored.residents * ACTIONS
            || restored.action_abs_sum.len() != restored.residents * ACTIONS
            || restored.outcome_sum.len() != restored.residents * OUTCOMES
            || restored.flow_sum.len() != restored.residents * FLOWS
            || restored.first_energy.len() != restored.residents
            || restored.last_energy.len() != restored.residents
            || restored.has_energy.len() != restored.residents
            || restored.mouth_ticks.len() != restored.residents
            || restored.mouth_bouts.len() != restored.residents
            || restored.mouth_active.len() != restored.residents
            || restored.contact_ticks.len() != restored.residents
            || restored.contact_bouts.len() != restored.residents
            || restored.contact_active.len() != restored.residents
            || restored.quiet_ticks.len() != restored.residents
            || restored.height_sum.len() != restored.residents
            || restored.height_min.len() != restored.residents
            || restored.height_max.len() != restored.residents
            || restored.outside_world_ticks.len() != restored.residents
            || restored.outside_deviation_sum.len() != restored.residents
            || restored.outside_deviation_max.len() != restored.residents
        {
            return Err(PyValueError::new_err(
                "population trajectory snapshot identity differs",
            ));
        }
        if !finite(&restored.phys_sum)
            || !finite(&restored.phys_min)
            || !finite(&restored.phys_max)
            || !finite(&restored.action_sum)
            || !finite(&restored.action_abs_sum)
            || !finite(&restored.outcome_sum)
            || !finite(&restored.flow_sum)
            || !finite(&restored.first_energy)
            || !finite(&restored.last_energy)
            || !finite(&restored.height_sum)
            || !finite(&restored.height_min)
            || !finite(&restored.height_max)
            || !finite(&restored.outside_deviation_sum)
            || !finite(&restored.outside_deviation_max)
        {
            return Err(PyValueError::new_err(
                "population trajectory snapshot contains non-finite state",
            ));
        }
        let cells = restored
            .cells
            .iter()
            .map(|x| x.iter().copied().collect())
            .collect();
        self.state = restored;
        self.cells = cells;
        Ok(())
    }
}
