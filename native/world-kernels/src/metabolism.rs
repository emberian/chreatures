//! Conservative batched metabolism with an immutable shared reaction program.

use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"CHMET2\0\0";
const EPS: f64 = 1e-12;

fn finite_nonnegative(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite() && *x >= 0.0)
}

fn hash_part(hash: &mut Sha256, name: &[u8], shape: &[usize], values: &[f64]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name);
    hash.update((shape.len() as u64).to_le_bytes());
    for &size in shape {
        hash.update((size as u64).to_le_bytes());
    }
    for value in values {
        hash.update(value.to_le_bytes());
    }
}

#[pyclass]
pub struct MetabolicCohort {
    n: usize,
    r: usize,
    k: usize,
    e: usize,
    stoich: Vec<f64>,
    elements: Vec<f64>,
    chemical_energy: Vec<f64>,
    atp_cost: Vec<f64>,
    atp_yield: Vec<f64>,
    photon_cost: Vec<f64>,
    half_saturation: Vec<f64>,
    base_rates: Vec<f64>,
    reaction_heat: Vec<f64>,
    enzyme_activity: Vec<f64>,
    pools: Vec<f64>,
    atp: Vec<f64>,
    atp_capacity: Vec<f64>,
    bulk_pool: Vec<f64>,
    bulk_atp: f64,
    time: f64,
    cumulative_ledger: Vec<f64>,
    regulation_baseline: Vec<f64>,
    regulation_substrate_response: Vec<f64>,
    regulation_atp_response: Vec<f64>,
    regulation_time_constant: Vec<f64>,
    regulation_change_cost: Vec<f64>,
    regulation_maximum: f64,
    regulation_total_budget: f64,
    cumulative_regulation_atp: Vec<f64>,
    program_sha256: String,
}

impl MetabolicCohort {
    fn clone_for_expansion(
        &self,
        added: usize,
        new_enzymes: &[f64],
        new_capacity: &[f64],
    ) -> PyResult<Self> {
        if !finite_nonnegative(new_enzymes) || !finite_nonnegative(new_capacity) {
            return Err(PyValueError::new_err(
                "metabolism expansion values are invalid",
            ));
        }
        for row in 0..added {
            let values = &new_enzymes[row * self.r..(row + 1) * self.r];
            if values.iter().any(|x| *x > self.regulation_maximum)
                || values.iter().sum::<f64>() > self.regulation_total_budget + EPS
            {
                return Err(PyValueError::new_err(
                    "expanded baseline exceeds enzyme budget",
                ));
            }
        }
        let mut enzymes = self.enzyme_activity.clone();
        enzymes.extend_from_slice(new_enzymes);
        let mut pools = self.pools.clone();
        pools.resize((self.n + added) * self.k, 0.0);
        let mut atp = self.atp.clone();
        atp.resize(self.n + added, 0.0);
        let mut capacities = self.atp_capacity.clone();
        capacities.extend_from_slice(new_capacity);
        let mut ledger = self.cumulative_ledger.clone();
        ledger.resize((self.n + added) * 6, 0.0);
        Ok(Self {
            n: self.n + added,
            r: self.r,
            k: self.k,
            e: self.e,
            stoich: self.stoich.clone(),
            elements: self.elements.clone(),
            chemical_energy: self.chemical_energy.clone(),
            atp_cost: self.atp_cost.clone(),
            atp_yield: self.atp_yield.clone(),
            photon_cost: self.photon_cost.clone(),
            half_saturation: self.half_saturation.clone(),
            base_rates: self.base_rates.clone(),
            reaction_heat: self.reaction_heat.clone(),
            enzyme_activity: enzymes,
            pools,
            atp,
            atp_capacity: capacities,
            bulk_pool: self.bulk_pool.clone(),
            bulk_atp: self.bulk_atp,
            time: self.time,
            cumulative_ledger: ledger,
            regulation_baseline: self.regulation_baseline.clone(),
            regulation_substrate_response: self.regulation_substrate_response.clone(),
            regulation_atp_response: self.regulation_atp_response.clone(),
            regulation_time_constant: self.regulation_time_constant.clone(),
            regulation_change_cost: self.regulation_change_cost.clone(),
            regulation_maximum: self.regulation_maximum,
            regulation_total_budget: self.regulation_total_budget,
            cumulative_regulation_atp: self.cumulative_regulation_atp.clone(),
            program_sha256: self.program_sha256.clone(),
        })
    }

    fn regulate(&mut self, dt: f64) -> Vec<f64> {
        let mut paid = vec![0.0; self.n];
        if self.regulation_baseline.is_empty() {
            return paid;
        }
        let mut target = vec![0.0; self.r];
        for row in 0..self.n {
            let atp_fraction = if self.atp_capacity[row] > 0.0 {
                (self.atp[row] / self.atp_capacity[row]).clamp(0.0, 1.0)
            } else {
                0.0
            };
            for reaction in 0..self.r {
                let mut availability = 1.0_f64;
                for species in 0..self.k {
                    if self.stoich[reaction * self.k + species] < 0.0 {
                        let amount = self.pools[row * self.k + species];
                        let km = self.half_saturation[reaction * self.k + species];
                        availability = availability.min(amount / (km + amount));
                    }
                }
                let index = row * self.r + reaction;
                target[reaction] = (self.regulation_baseline[index]
                    + self.regulation_substrate_response[index] * (availability - 0.5)
                    + self.regulation_atp_response[index] * (atp_fraction - 0.5))
                    .clamp(0.0, self.regulation_maximum);
            }
            let target_sum: f64 = target.iter().sum();
            if target_sum > self.regulation_total_budget {
                let scale = self.regulation_total_budget / target_sum;
                target.iter_mut().for_each(|x| *x *= scale);
            }
            let alpha = -(-dt / self.regulation_time_constant[row]).exp_m1();
            let mut requested_cost = 0.0;
            for (reaction, &goal) in target.iter().enumerate() {
                let index = row * self.r + reaction;
                requested_cost += (alpha * (goal - self.enzyme_activity[index])).abs()
                    * self.regulation_change_cost[row];
            }
            let limiter = if requested_cost > 0.0 {
                (self.atp[row] / requested_cost).min(1.0)
            } else {
                1.0
            };
            for (reaction, &goal) in target.iter().enumerate() {
                let index = row * self.r + reaction;
                self.enzyme_activity[index] +=
                    limiter * alpha * (goal - self.enzyme_activity[index]);
            }
            paid[row] = requested_cost * limiter;
            self.atp[row] -= paid[row];
            self.cumulative_regulation_atp[row] += paid[row];
        }
        paid
    }
}

#[pymethods]
impl MetabolicCohort {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        stoichiometry: PyReadonlyArray2<'_, f64>,
        elemental_composition: PyReadonlyArray2<'_, f64>,
        chemical_energy: PyReadonlyArray1<'_, f64>,
        atp_cost: PyReadonlyArray1<'_, f64>,
        atp_yield: PyReadonlyArray1<'_, f64>,
        photon_cost: PyReadonlyArray1<'_, f64>,
        half_saturation: PyReadonlyArray2<'_, f64>,
        base_rates: PyReadonlyArray1<'_, f64>,
        enzyme_activity: PyReadonlyArray2<'_, f64>,
        pools: PyReadonlyArray2<'_, f64>,
        atp: PyReadonlyArray1<'_, f64>,
        atp_capacity: PyReadonlyArray1<'_, f64>,
        bulk_pool: PyReadonlyArray1<'_, f64>,
        bulk_atp: f64,
    ) -> PyResult<Self> {
        let ss = stoichiometry.shape();
        let es = elemental_composition.shape();
        let gs = enzyme_activity.shape();
        let ps = pools.shape();
        if ss.len() != 2 || es.len() != 2 || gs.len() != 2 || ps.len() != 2 {
            return Err(PyValueError::new_err("metabolism arrays must be matrices"));
        }
        let (r, k, e, n) = (ss[0], ss[1], es[1], gs[0]);
        if !(1..=4096).contains(&n)
            || !(1..=256).contains(&r)
            || !(1..=64).contains(&k)
            || !(1..=16).contains(&e)
            || es[0] != k
            || gs[1] != r
            || ps != [n, k]
            || chemical_energy.shape() != [k]
            || atp_cost.shape() != [r]
            || atp_yield.shape() != [r]
            || photon_cost.shape() != [r]
            || half_saturation.shape() != [r, k]
            || base_rates.shape() != [r]
            || atp.shape() != [n]
            || atp_capacity.shape() != [n]
            || bulk_pool.shape() != [k]
        {
            return Err(PyValueError::new_err(
                "metabolism array shapes or dimensions differ",
            ));
        }
        let stoich = stoichiometry.as_slice()?.to_vec();
        let elements = elemental_composition.as_slice()?.to_vec();
        let chemical_energy = chemical_energy.as_slice()?.to_vec();
        let atp_cost = atp_cost.as_slice()?.to_vec();
        let atp_yield = atp_yield.as_slice()?.to_vec();
        let photon_cost = photon_cost.as_slice()?.to_vec();
        let half_saturation = half_saturation.as_slice()?.to_vec();
        let base_rates = base_rates.as_slice()?.to_vec();
        let enzyme_activity = enzyme_activity.as_slice()?.to_vec();
        let pools = pools.as_slice()?.to_vec();
        let atp = atp.as_slice()?.to_vec();
        let atp_capacity = atp_capacity.as_slice()?.to_vec();
        let bulk_pool = bulk_pool.as_slice()?.to_vec();
        if stoich.iter().any(|x| !x.is_finite())
            || !finite_nonnegative(&elements)
            || !finite_nonnegative(&chemical_energy)
            || !finite_nonnegative(&atp_cost)
            || !finite_nonnegative(&atp_yield)
            || !finite_nonnegative(&photon_cost)
            || !finite_nonnegative(&half_saturation)
            || !finite_nonnegative(&base_rates)
            || !finite_nonnegative(&enzyme_activity)
            || !finite_nonnegative(&pools)
            || !finite_nonnegative(&atp)
            || !finite_nonnegative(&atp_capacity)
            || !finite_nonnegative(&bulk_pool)
            || !bulk_atp.is_finite()
            || bulk_atp < 0.0
            || atp
                .iter()
                .zip(&atp_capacity)
                .any(|(x, cap)| x > &(cap + EPS))
        {
            return Err(PyValueError::new_err(
                "metabolism values must be finite, nonnegative, and within ATP capacity",
            ));
        }
        for reaction in 0..r {
            if !(0..k).any(|species| stoich[reaction * k + species] < 0.0) {
                return Err(PyValueError::new_err(
                    "each reaction must consume at least one resource",
                ));
            }
            for species in 0..k {
                let km = half_saturation[reaction * k + species];
                if stoich[reaction * k + species] < 0.0 && km <= 0.0 {
                    return Err(PyValueError::new_err(
                        "consumed resources require positive half saturation",
                    ));
                }
            }
            for element in 0..e {
                let balance: f64 = (0..k)
                    .map(|species| stoich[reaction * k + species] * elements[species * e + element])
                    .sum();
                let scale: f64 = (0..k)
                    .map(|species| {
                        (stoich[reaction * k + species] * elements[species * e + element]).abs()
                    })
                    .sum();
                if balance.abs() > 1e-10 * scale.max(1.0) {
                    return Err(PyValueError::new_err(format!(
                        "reaction {reaction} is not element balanced"
                    )));
                }
            }
        }
        let mut reaction_heat = vec![0.0; r];
        for reaction in 0..r {
            let delta: f64 = (0..k)
                .map(|species| stoich[reaction * k + species] * chemical_energy[species])
                .sum();
            let heat = atp_cost[reaction] + photon_cost[reaction] - atp_yield[reaction] - delta;
            if heat < -1e-10 || !heat.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "reaction {reaction} has negative heat"
                )));
            }
            reaction_heat[reaction] = heat.max(0.0);
        }
        let mut hash = Sha256::new();
        hash.update(b"chreatures-metabolism-program-v1");
        hash_part(&mut hash, b"stoichiometry", &[r, k], &stoich);
        hash_part(&mut hash, b"elemental_composition", &[k, e], &elements);
        hash_part(&mut hash, b"chemical_energy", &[k], &chemical_energy);
        hash_part(&mut hash, b"atp_cost", &[r], &atp_cost);
        hash_part(&mut hash, b"atp_yield", &[r], &atp_yield);
        hash_part(&mut hash, b"photon_cost", &[r], &photon_cost);
        hash_part(&mut hash, b"half_saturation", &[r, k], &half_saturation);
        hash_part(&mut hash, b"base_rates", &[r], &base_rates);
        let program_sha256 = format!("{:x}", hash.finalize());
        Ok(Self {
            n,
            r,
            k,
            e,
            stoich,
            elements,
            chemical_energy,
            atp_cost,
            atp_yield,
            photon_cost,
            half_saturation,
            base_rates,
            reaction_heat,
            enzyme_activity,
            pools,
            atp,
            atp_capacity,
            bulk_pool,
            bulk_atp,
            time: 0.0,
            cumulative_ledger: vec![0.0; n * 6],
            regulation_baseline: Vec::new(),
            regulation_substrate_response: Vec::new(),
            regulation_atp_response: Vec::new(),
            regulation_time_constant: Vec::new(),
            regulation_change_cost: Vec::new(),
            regulation_maximum: 0.0,
            regulation_total_budget: 0.0,
            cumulative_regulation_atp: Vec::new(),
            program_sha256,
        })
    }

    #[getter]
    fn program_sha256(&self) -> &str {
        &self.program_sha256
    }

    #[getter]
    fn pools<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_vec((self.n, self.k), self.pools.clone())
            .unwrap()
            .into_pyarray(py)
    }

    #[getter]
    fn atp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.atp.clone().into_pyarray(py)
    }

    #[getter]
    fn enzyme_activity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_vec((self.n, self.r), self.enzyme_activity.clone())
            .unwrap()
            .into_pyarray(py)
    }

    #[getter]
    fn atp_capacity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.atp_capacity.clone().into_pyarray(py)
    }

    #[getter]
    fn bulk_pool<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bulk_pool.clone().into_pyarray(py)
    }

    #[getter]
    fn bulk_atp(&self) -> f64 {
        self.bulk_atp
    }

    #[getter]
    fn time(&self) -> f64 {
        self.time
    }

    #[allow(clippy::too_many_arguments)]
    fn enable_regulation(
        &mut self,
        baseline: PyReadonlyArray2<'_, f64>,
        substrate_response: PyReadonlyArray2<'_, f64>,
        atp_response: PyReadonlyArray2<'_, f64>,
        time_constant: PyReadonlyArray1<'_, f64>,
        change_cost: PyReadonlyArray1<'_, f64>,
        maximum_expression: f64,
        total_budget: f64,
    ) -> PyResult<()> {
        if !self.regulation_baseline.is_empty()
            || baseline.shape() != [self.n, self.r]
            || substrate_response.shape() != [self.n, self.r]
            || atp_response.shape() != [self.n, self.r]
            || time_constant.shape() != [self.n]
            || change_cost.shape() != [self.n]
            || !maximum_expression.is_finite()
            || !total_budget.is_finite()
            || maximum_expression <= 0.0
            || total_budget < maximum_expression
        {
            return Err(PyValueError::new_err(
                "metabolic regulation dimensions or budgets differ",
            ));
        }
        let baseline = baseline.as_slice()?;
        let substrate = substrate_response.as_slice()?;
        let atp_response = atp_response.as_slice()?;
        let tau = time_constant.as_slice()?;
        let cost = change_cost.as_slice()?;
        if !finite_nonnegative(baseline)
            || substrate
                .iter()
                .any(|x| !x.is_finite() || x.abs() > maximum_expression)
            || atp_response
                .iter()
                .any(|x| !x.is_finite() || x.abs() > maximum_expression)
            || tau.iter().any(|x| !x.is_finite() || *x < 0.5 || *x > 120.0)
            || cost.iter().any(|x| !x.is_finite() || *x < 0.01 || *x > 2.0)
        {
            return Err(PyValueError::new_err(
                "metabolic regulation values are invalid",
            ));
        }
        for row in 0..self.n {
            let values = &baseline[row * self.r..(row + 1) * self.r];
            if values.iter().any(|x| *x > maximum_expression)
                || values.iter().sum::<f64>() > total_budget + EPS
            {
                return Err(PyValueError::new_err(
                    "regulation baseline exceeds enzyme budget",
                ));
            }
        }
        self.regulation_baseline = baseline.to_vec();
        self.regulation_substrate_response = substrate.to_vec();
        self.regulation_atp_response = atp_response.to_vec();
        self.regulation_time_constant = tau.to_vec();
        self.regulation_change_cost = cost.to_vec();
        self.regulation_maximum = maximum_expression;
        self.regulation_total_budget = total_budget;
        self.cumulative_regulation_atp = vec![0.0; self.n];
        self.enzyme_activity.copy_from_slice(baseline);
        Ok(())
    }

    #[getter]
    fn cumulative_regulation_atp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.cumulative_regulation_atp.clone().into_pyarray(py)
    }

    #[allow(clippy::too_many_arguments)]
    fn expanded(
        &self,
        baseline: PyReadonlyArray2<'_, f64>,
        atp_capacity: PyReadonlyArray1<'_, f64>,
        substrate_response: PyReadonlyArray2<'_, f64>,
        atp_response: PyReadonlyArray2<'_, f64>,
        time_constant: PyReadonlyArray1<'_, f64>,
        change_cost: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        if self.regulation_baseline.is_empty() {
            return Err(PyValueError::new_err(
                "metabolism regulation is not enabled",
            ));
        }
        let shape = baseline.shape();
        let added = shape.first().copied().unwrap_or(0);
        if shape != [added, self.r]
            || !(1..=5).contains(&added)
            || self.n + added > 4096
            || atp_capacity.shape() != [added]
            || substrate_response.shape() != [added, self.r]
            || atp_response.shape() != [added, self.r]
            || time_constant.shape() != [added]
            || change_cost.shape() != [added]
        {
            return Err(PyValueError::new_err(
                "regulated expansion dimensions differ",
            ));
        }
        let mut candidate =
            self.clone_for_expansion(added, baseline.as_slice()?, atp_capacity.as_slice()?)?;
        let substrate = substrate_response.as_slice()?;
        let atp_gain = atp_response.as_slice()?;
        let tau = time_constant.as_slice()?;
        let cost = change_cost.as_slice()?;
        if substrate
            .iter()
            .any(|x| !x.is_finite() || x.abs() > self.regulation_maximum)
            || atp_gain
                .iter()
                .any(|x| !x.is_finite() || x.abs() > self.regulation_maximum)
            || tau.iter().any(|x| !x.is_finite() || *x < 0.5 || *x > 120.0)
            || cost.iter().any(|x| !x.is_finite() || *x < 0.01 || *x > 2.0)
        {
            return Err(PyValueError::new_err(
                "regulated expansion values are invalid",
            ));
        }
        candidate
            .regulation_baseline
            .extend_from_slice(baseline.as_slice()?);
        candidate
            .regulation_substrate_response
            .extend_from_slice(substrate);
        candidate
            .regulation_atp_response
            .extend_from_slice(atp_gain);
        candidate.regulation_time_constant.extend_from_slice(tau);
        candidate.regulation_change_cost.extend_from_slice(cost);
        candidate
            .cumulative_regulation_atp
            .resize(self.n + added, 0.0);
        Ok(candidate)
    }

    #[getter]
    fn cumulative_ledger<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_vec((self.n, 6), self.cumulative_ledger.clone())
            .unwrap()
            .into_pyarray(py)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        light_energy_budget: PyReadonlyArray1<'_, f64>,
        mechanical_cost: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !dt.is_finite()
            || dt <= 0.0
            || dt > 10.0
            || light_energy_budget.shape() != [self.n]
            || mechanical_cost.shape() != [self.n]
        {
            return Err(PyValueError::new_err(
                "metabolism timestep or per-row shapes differ",
            ));
        }
        let light = light_energy_budget.as_slice()?;
        let mechanical = mechanical_cost.as_slice()?;
        if !finite_nonnegative(light) || !finite_nonnegative(mechanical) {
            return Err(PyValueError::new_err(
                "budgets and costs must be finite and nonnegative",
            ));
        }
        let regulation_atp_cost = self.regulate(dt);
        let mut extent = vec![0.0; self.n * self.r];
        let mut limiter = vec![1.0; self.n * self.r];
        let mut photon_used = vec![0.0; self.n];
        let mut atp_cost_used = vec![0.0; self.n];
        let mut atp_yielded = vec![0.0; self.n];
        let mut chemical_delta = vec![0.0; self.n];
        let mut reaction_heat = regulation_atp_cost.clone();
        let mut overflow = vec![0.0; self.n];
        let mut mechanical_paid = vec![0.0; self.n];
        let mut mechanical_unmet = vec![0.0; self.n];
        let mut elemental_residual = vec![0.0; self.n * self.e];
        let mut energy_residual = vec![0.0; self.n];
        let mut ratios = vec![1.0; self.k];
        for row in 0..self.n {
            let begin_atp = self.atp[row] + regulation_atp_cost[row];
            let begin_chemical: f64 = (0..self.k)
                .map(|species| self.pools[row * self.k + species] * self.chemical_energy[species])
                .sum();
            let begin_elements: Vec<f64> = (0..self.e)
                .map(|element| {
                    (0..self.k)
                        .map(|species| {
                            self.pools[row * self.k + species]
                                * self.elements[species * self.e + element]
                        })
                        .sum()
                })
                .collect();
            for reaction in 0..self.r {
                let mut rate =
                    dt * self.base_rates[reaction] * self.enzyme_activity[row * self.r + reaction];
                for species in 0..self.k {
                    if self.stoich[reaction * self.k + species] < 0.0 {
                        let x = self.pools[row * self.k + species];
                        let km = self.half_saturation[reaction * self.k + species];
                        rate *= x / (km + x);
                    }
                }
                extent[row * self.r + reaction] = rate;
            }
            for (species, ratio) in ratios.iter_mut().enumerate() {
                let demand: f64 = (0..self.r)
                    .map(|reaction| {
                        extent[row * self.r + reaction]
                            * (-self.stoich[reaction * self.k + species]).max(0.0)
                    })
                    .sum();
                *ratio = if demand > 0.0 {
                    (self.pools[row * self.k + species] / demand).min(1.0)
                } else {
                    1.0
                };
            }
            let atp_demand: f64 = (0..self.r)
                .map(|reaction| extent[row * self.r + reaction] * self.atp_cost[reaction])
                .sum();
            let photon_demand: f64 = (0..self.r)
                .map(|reaction| extent[row * self.r + reaction] * self.photon_cost[reaction])
                .sum();
            let atp_ratio = if atp_demand > 0.0 {
                (self.atp[row] / atp_demand).min(1.0)
            } else {
                1.0
            };
            let photon_ratio = if photon_demand > 0.0 {
                (light[row] / photon_demand).min(1.0)
            } else {
                1.0
            };
            for reaction in 0..self.r {
                let mut scale = 1.0_f64;
                for (species, &resource_ratio) in ratios.iter().enumerate() {
                    if self.stoich[reaction * self.k + species] < 0.0 {
                        scale = scale.min(resource_ratio);
                    }
                }
                if self.atp_cost[reaction] > 0.0 {
                    scale = scale.min(atp_ratio);
                }
                if self.photon_cost[reaction] > 0.0 {
                    scale = scale.min(photon_ratio);
                }
                limiter[row * self.r + reaction] = scale;
                extent[row * self.r + reaction] *= scale;
            }
            for reaction in 0..self.r {
                let x = extent[row * self.r + reaction];
                photon_used[row] += x * self.photon_cost[reaction];
                atp_cost_used[row] += x * self.atp_cost[reaction];
                atp_yielded[row] += x * self.atp_yield[reaction];
                reaction_heat[row] += x * self.reaction_heat[reaction];
                for species in 0..self.k {
                    let delta = x * self.stoich[reaction * self.k + species];
                    self.pools[row * self.k + species] += delta;
                    chemical_delta[row] += delta * self.chemical_energy[species];
                }
            }
            for species in 0..self.k {
                if self.pools[row * self.k + species] < 0.0
                    && self.pools[row * self.k + species] > -1e-10
                {
                    self.pools[row * self.k + species] = 0.0;
                }
            }
            self.atp[row] += atp_yielded[row] - atp_cost_used[row];
            if self.atp[row] > self.atp_capacity[row] {
                overflow[row] = self.atp[row] - self.atp_capacity[row];
                self.atp[row] = self.atp_capacity[row];
            }
            mechanical_paid[row] = mechanical[row].min(self.atp[row]);
            self.atp[row] -= mechanical_paid[row];
            mechanical_unmet[row] = mechanical[row] - mechanical_paid[row];
            let end_chemical: f64 = (0..self.k)
                .map(|species| self.pools[row * self.k + species] * self.chemical_energy[species])
                .sum();
            chemical_delta[row] = end_chemical - begin_chemical;
            for element in 0..self.e {
                let end: f64 = (0..self.k)
                    .map(|species| {
                        self.pools[row * self.k + species]
                            * self.elements[species * self.e + element]
                    })
                    .sum();
                elemental_residual[row * self.e + element] = end - begin_elements[element];
            }
            energy_residual[row] = photon_used[row]
                - (chemical_delta[row]
                    + (self.atp[row] - begin_atp)
                    + reaction_heat[row]
                    + overflow[row]
                    + mechanical_paid[row]);
            self.cumulative_ledger[row * 6] += photon_used[row];
            self.cumulative_ledger[row * 6 + 1] += chemical_delta[row];
            self.cumulative_ledger[row * 6 + 2] += reaction_heat[row];
            self.cumulative_ledger[row * 6 + 3] += overflow[row];
            self.cumulative_ledger[row * 6 + 4] += mechanical_paid[row];
            self.cumulative_ledger[row * 6 + 5] += mechanical_unmet[row];
        }
        self.time += dt;
        let total_heat: Vec<f64> = reaction_heat
            .iter()
            .zip(&overflow)
            .map(|(a, b)| a + b)
            .collect();
        let out = PyDict::new(py);
        macro_rules! put2 {
            ($name:literal,$value:expr,$cols:expr) => {
                out.set_item(
                    $name,
                    Array2::from_shape_vec((self.n, $cols), $value)
                        .unwrap()
                        .into_pyarray(py),
                )?
            };
        }
        macro_rules! put1 {
            ($name:literal,$value:expr) => {
                out.set_item($name, $value.into_pyarray(py))?
            };
        }
        put2!("extent", extent, self.r);
        put2!("limiter", limiter, self.r);
        put1!("photon_used", photon_used);
        put1!("reaction_atp_cost", atp_cost_used);
        put1!("reaction_atp_yield", atp_yielded);
        put1!("regulation_atp_cost", regulation_atp_cost);
        put1!("chemical_energy_delta", chemical_delta);
        put1!("reaction_heat", reaction_heat);
        put1!("atp_overflow_heat", overflow);
        put1!("mechanical_paid", mechanical_paid);
        put1!("mechanical_unmet", mechanical_unmet);
        put1!("total_heat", total_heat);
        put2!("elemental_residual", elemental_residual, self.e);
        put1!("energy_residual", energy_residual);
        Ok(out)
    }

    #[pyo3(signature=(donor,receiver,resources,atp=0.0))]
    fn transfer(
        &mut self,
        donor: Option<usize>,
        receiver: Option<usize>,
        resources: PyReadonlyArray1<'_, f64>,
        atp: f64,
    ) -> PyResult<()> {
        if donor == receiver
            || donor.is_some_and(|x| x >= self.n)
            || receiver.is_some_and(|x| x >= self.n)
            || resources.shape() != [self.k]
        {
            return Err(PyValueError::new_err(
                "transfer endpoints or resource shape differ",
            ));
        }
        let amount = resources.as_slice()?;
        if !finite_nonnegative(amount) || !atp.is_finite() || atp < 0.0 {
            return Err(PyValueError::new_err(
                "transfer amounts must be finite and nonnegative",
            ));
        }
        if let Some(d) = donor {
            if amount
                .iter()
                .enumerate()
                .any(|(s, x)| *x > self.pools[d * self.k + s] + EPS)
                || atp > self.atp[d] + EPS
            {
                return Err(PyValueError::new_err("donor has insufficient resources"));
            }
        } else if amount
            .iter()
            .enumerate()
            .any(|(s, x)| *x > self.bulk_pool[s] + EPS)
            || atp > self.bulk_atp + EPS
        {
            return Err(PyValueError::new_err(
                "bulk pool has insufficient resources",
            ));
        }
        if let Some(r) = receiver {
            if self.atp[r] + atp > self.atp_capacity[r] + EPS {
                return Err(PyValueError::new_err("receiver ATP capacity exceeded"));
            }
        }
        if let Some(d) = donor {
            for (s, &value) in amount.iter().enumerate() {
                self.pools[d * self.k + s] -= value;
            }
            self.atp[d] -= atp;
        } else {
            for (s, x) in amount.iter().enumerate() {
                self.bulk_pool[s] -= *x;
            }
            self.bulk_atp -= atp;
        }
        if let Some(r) = receiver {
            for (s, &value) in amount.iter().enumerate() {
                self.pools[r * self.k + s] += value;
            }
            self.atp[r] += atp;
        } else {
            for (s, x) in amount.iter().enumerate() {
                self.bulk_pool[s] += *x;
            }
            self.bulk_atp += atp;
        }
        Ok(())
    }

    fn transfer_batch<'py>(
        &mut self,
        py: Python<'py>,
        donors: PyReadonlyArray1<'_, i64>,
        receivers: PyReadonlyArray1<'_, i64>,
        resources: PyReadonlyArray2<'_, f64>,
        atp: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = donors.len();
        if receivers.shape() != [m] || resources.shape() != [m, self.k] || atp.shape() != [m] {
            return Err(PyValueError::new_err("transfer batch shapes differ"));
        }
        let donors = donors.as_slice()?;
        let receivers = receivers.as_slice()?;
        let requested = resources.as_slice()?;
        let requested_atp = atp.as_slice()?;
        let valid = |value: i64| value == -1 || (value >= 0 && value < self.n as i64);
        if !finite_nonnegative(requested)
            || !finite_nonnegative(requested_atp)
            || donors
                .iter()
                .zip(receivers)
                .any(|(&donor, &receiver)| !valid(donor) || !valid(receiver) || donor == receiver)
        {
            return Err(PyValueError::new_err(
                "transfer batch contains an invalid endpoint or amount",
            ));
        }

        // Endpoint n denotes the owned bulk pool. Every factor below is
        // computed before applying any transfer.
        let bulk = self.n;
        let endpoint_count = self.n + 1;
        let endpoint = |value: i64| if value == -1 { bulk } else { value as usize };
        let mut resource_demand = vec![0.0; endpoint_count * self.k];
        let mut atp_demand = vec![0.0; endpoint_count];
        let mut atp_incoming = vec![0.0; self.n];
        for edge in 0..m {
            let donor = endpoint(donors[edge]);
            let receiver = endpoint(receivers[edge]);
            for species in 0..self.k {
                resource_demand[donor * self.k + species] += requested[edge * self.k + species];
            }
            atp_demand[donor] += requested_atp[edge];
            if receiver < self.n {
                atp_incoming[receiver] += requested_atp[edge];
            }
        }
        if resource_demand.iter().any(|value| !value.is_finite())
            || atp_demand.iter().any(|value| !value.is_finite())
            || atp_incoming.iter().any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err(
                "transfer batch aggregate demand is not finite",
            ));
        }
        let mut resource_factor = vec![1.0; endpoint_count * self.k];
        let mut donor_atp_factor = vec![1.0; endpoint_count];
        let mut receiver_atp_factor = vec![1.0; self.n];
        for source in 0..endpoint_count {
            for species in 0..self.k {
                let demand = resource_demand[source * self.k + species];
                let available = if source == bulk {
                    self.bulk_pool[species]
                } else {
                    self.pools[source * self.k + species]
                };
                if demand > 0.0 {
                    resource_factor[source * self.k + species] = (available / demand).min(1.0);
                }
            }
            let available = if source == bulk {
                self.bulk_atp
            } else {
                self.atp[source]
            };
            if atp_demand[source] > 0.0 {
                donor_atp_factor[source] = (available / atp_demand[source]).min(1.0);
            }
        }
        for receiver in 0..self.n {
            if atp_incoming[receiver] > 0.0 {
                receiver_atp_factor[receiver] =
                    ((self.atp_capacity[receiver] - self.atp[receiver]) / atp_incoming[receiver])
                        .clamp(0.0, 1.0);
            }
        }

        let mut moved = vec![0.0; m * self.k];
        let mut resource_limiter = vec![1.0; m * self.k];
        let mut moved_atp = vec![0.0; m];
        let mut atp_limiter = vec![1.0; m];
        let mut pool_delta = vec![0.0; self.n * self.k];
        let mut bulk_delta = vec![0.0; self.k];
        let mut atp_delta = vec![0.0; self.n];
        let mut bulk_atp_delta = 0.0;
        for edge in 0..m {
            let donor = endpoint(donors[edge]);
            let receiver = endpoint(receivers[edge]);
            for species in 0..self.k {
                let factor = resource_factor[donor * self.k + species];
                let value = requested[edge * self.k + species] * factor;
                resource_limiter[edge * self.k + species] = factor;
                moved[edge * self.k + species] = value;
                if donor == bulk {
                    bulk_delta[species] -= value;
                } else {
                    pool_delta[donor * self.k + species] -= value;
                }
                if receiver == bulk {
                    bulk_delta[species] += value;
                } else {
                    pool_delta[receiver * self.k + species] += value;
                }
            }
            let receive_factor = if receiver == bulk {
                1.0
            } else {
                receiver_atp_factor[receiver]
            };
            let factor = donor_atp_factor[donor].min(receive_factor);
            let value = requested_atp[edge] * factor;
            atp_limiter[edge] = factor;
            moved_atp[edge] = value;
            if donor == bulk {
                bulk_atp_delta -= value;
            } else {
                atp_delta[donor] -= value;
            }
            if receiver == bulk {
                bulk_atp_delta += value;
            } else {
                atp_delta[receiver] += value;
            }
        }

        let mut next_pools = self.pools.clone();
        let mut next_atp = self.atp.clone();
        let mut next_bulk = self.bulk_pool.clone();
        for (value, delta) in next_pools.iter_mut().zip(pool_delta) {
            *value += delta;
        }
        for (value, delta) in next_atp.iter_mut().zip(atp_delta) {
            *value += delta;
        }
        for (value, delta) in next_bulk.iter_mut().zip(bulk_delta) {
            *value += delta;
        }
        let next_bulk_atp = self.bulk_atp + bulk_atp_delta;
        if next_pools
            .iter()
            .chain(&next_bulk)
            .any(|x| !x.is_finite() || *x < -1e-10)
            || next_atp.iter().any(|x| !x.is_finite() || *x < -1e-10)
            || !next_bulk_atp.is_finite()
            || next_bulk_atp < -1e-10
            || next_atp
                .iter()
                .zip(&self.atp_capacity)
                .any(|(x, cap)| *x > *cap + 1e-10)
        {
            return Err(PyValueError::new_err(
                "transfer batch produced invalid aggregate state",
            ));
        }
        for value in next_pools.iter_mut().chain(&mut next_bulk) {
            *value = value.max(0.0);
        }
        for value in &mut next_atp {
            *value = value.max(0.0);
        }
        self.pools = next_pools;
        self.atp = next_atp;
        self.bulk_pool = next_bulk;
        self.bulk_atp = next_bulk_atp.max(0.0);

        let out = PyDict::new(py);
        out.set_item(
            "moved_resources",
            Array2::from_shape_vec((m, self.k), moved)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "resource_limiter",
            Array2::from_shape_vec((m, self.k), resource_limiter)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("moved_atp", moved_atp.into_pyarray(py))?;
        out.set_item("atp_limiter", atp_limiter.into_pyarray(py))?;
        Ok(out)
    }

    /// Debit exported work from one compartment without creating a heat sink.
    fn pay_work(&mut self, row: usize, amount: f64) -> PyResult<()> {
        if row >= self.n || !amount.is_finite() || amount < 0.0 {
            return Err(PyValueError::new_err("work row or amount is invalid"));
        }
        if amount > self.atp[row] {
            return Err(PyValueError::new_err(
                "compartment has insufficient ATP for work",
            ));
        }
        self.atp[row] -= amount;
        self.cumulative_ledger[row * 6 + 4] += amount;
        Ok(())
    }

    /// Atomically debit a resident cohort after validating every row/payment.
    fn pay_work_batch(
        &mut self,
        rows: PyReadonlyArray1<'_, i64>,
        amounts: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        if rows.shape() != amounts.shape() {
            return Err(PyValueError::new_err("work rows and amounts differ"));
        }
        let rows = rows.as_slice()?;
        let amounts = amounts.as_slice()?;
        let mut debits = vec![0.0; self.n];
        for (&row, &amount) in rows.iter().zip(amounts) {
            if row < 0 || row as usize >= self.n || !amount.is_finite() || amount < 0.0 {
                return Err(PyValueError::new_err("work row or amount is invalid"));
            }
            let slot = row as usize;
            debits[slot] += amount;
            if !debits[slot].is_finite() || debits[slot] > self.atp[slot] {
                return Err(PyValueError::new_err(
                    "compartment has insufficient ATP for work",
                ));
            }
        }
        for (row, amount) in debits.into_iter().enumerate() {
            if amount == 0.0 {
                continue;
            }
            self.atp[row] -= amount;
            self.cumulative_ledger[row * 6 + 4] += amount;
        }
        Ok(())
    }

    fn split(&mut self, parent: usize, child: usize, fraction: f64) -> PyResult<()> {
        if parent >= self.n
            || child >= self.n
            || parent == child
            || !fraction.is_finite()
            || !(0.0..=1.0).contains(&fraction)
        {
            return Err(PyValueError::new_err("invalid split"));
        }
        if self.pools[child * self.k..(child + 1) * self.k]
            .iter()
            .any(|x| *x != 0.0)
            || self.atp[child] != 0.0
        {
            return Err(PyValueError::new_err("child must be empty before split"));
        }
        let atp_amount = self.atp[parent] * fraction;
        if atp_amount > self.atp_capacity[child] + EPS {
            return Err(PyValueError::new_err(
                "child ATP capacity is too small for split",
            ));
        }
        for s in 0..self.k {
            let amount = self.pools[parent * self.k + s] * fraction;
            self.pools[parent * self.k + s] -= amount;
            self.pools[child * self.k + s] = amount;
        }
        self.atp[parent] -= atp_amount;
        self.atp[child] = atp_amount;
        Ok(())
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut out = Vec::with_capacity(
            64 + (self.pools.len()
                + self.atp.len()
                + self.enzyme_activity.len()
                + self.atp_capacity.len()
                + self.bulk_pool.len()
                + self.cumulative_ledger.len()
                + self.cumulative_regulation_atp.len()
                + 2)
                * 8,
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(self.program_sha256.as_bytes());
        for x in [self.n, self.r, self.k, self.e] {
            out.extend_from_slice(&(x as u64).to_le_bytes());
        }
        out.push(u8::from(!self.regulation_baseline.is_empty()));
        for values in [
            &self.pools,
            &self.atp,
            &self.enzyme_activity,
            &self.atp_capacity,
            &self.bulk_pool,
            &self.cumulative_ledger,
            &self.cumulative_regulation_atp,
        ] {
            for x in values {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.bulk_atp.to_le_bytes());
        out.extend_from_slice(&self.time.to_le_bytes());
        PyBytes::new(py, &out)
    }

    fn restore(&mut self, snapshot: &[u8]) -> PyResult<()> {
        let expected = 8
            + 64
            + 32
            + 1
            + (self.pools.len()
                + self.atp.len()
                + self.enzyme_activity.len()
                + self.atp_capacity.len()
                + self.bulk_pool.len()
                + self.cumulative_ledger.len()
                + self.cumulative_regulation_atp.len()
                + 2)
                * 8;
        if snapshot.len() != expected
            || &snapshot[..8] != MAGIC
            || &snapshot[8..72] != self.program_sha256.as_bytes()
        {
            return Err(PyValueError::new_err(
                "metabolism snapshot identity or byte length differs",
            ));
        }
        let mut cursor = 72;
        for expected_dim in [self.n, self.r, self.k, self.e] {
            let got = u64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            if got != expected_dim {
                return Err(PyValueError::new_err(
                    "metabolism snapshot dimensions differ",
                ));
            }
        }
        let regulated = snapshot[cursor];
        cursor += 1;
        if regulated > 1 || (regulated == 1) != !self.regulation_baseline.is_empty() {
            return Err(PyValueError::new_err(
                "metabolism regulation identity differs",
            ));
        }
        let mut read = |target: &mut [f64], nonnegative: bool| -> PyResult<()> {
            for x in target {
                *x = f64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap());
                cursor += 8;
                if !x.is_finite() || (nonnegative && *x < 0.0) {
                    return Err(PyValueError::new_err(
                        "metabolism snapshot contains invalid state",
                    ));
                }
            }
            Ok(())
        };
        let mut pools = self.pools.clone();
        let mut atp = self.atp.clone();
        let mut enzyme = self.enzyme_activity.clone();
        let mut capacity = self.atp_capacity.clone();
        let mut bulk = self.bulk_pool.clone();
        let mut ledger = self.cumulative_ledger.clone();
        let mut regulation_ledger = self.cumulative_regulation_atp.clone();
        read(&mut pools, true)?;
        read(&mut atp, true)?;
        read(&mut enzyme, true)?;
        read(&mut capacity, true)?;
        read(&mut bulk, true)?;
        read(&mut ledger, false)?;
        read(&mut regulation_ledger, true)?;
        let bulk_atp = f64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let time = f64::from_le_bytes(snapshot[cursor..cursor + 8].try_into().unwrap());
        if !bulk_atp.is_finite()
            || bulk_atp < 0.0
            || !time.is_finite()
            || time < 0.0
            || atp.iter().zip(&capacity).any(|(x, c)| x > &(c + EPS))
        {
            return Err(PyValueError::new_err(
                "metabolism snapshot ATP state is invalid",
            ));
        }
        if !self.regulation_baseline.is_empty() {
            for row in 0..self.n {
                let values = &enzyme[row * self.r..(row + 1) * self.r];
                if values.iter().any(|x| *x > self.regulation_maximum)
                    || values.iter().sum::<f64>() > self.regulation_total_budget + EPS
                {
                    return Err(PyValueError::new_err(
                        "metabolism snapshot enzyme state exceeds regulation budget",
                    ));
                }
            }
        }
        self.pools = pools;
        self.atp = atp;
        self.enzyme_activity = enzyme;
        self.atp_capacity = capacity;
        self.bulk_pool = bulk;
        self.bulk_atp = bulk_atp;
        self.time = time;
        self.cumulative_ledger = ledger;
        self.cumulative_regulation_atp = regulation_ledger;
        Ok(())
    }
}
