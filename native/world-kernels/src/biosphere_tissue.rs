//! Persistent native bookkeeping for developed physical tissue.
//!
//! Python owns the canonical part records used by checkpoints and other
//! environment adapters. This kernel owns the transient dense numeric mirror,
//! preserves Python insertion order, and writes each completed turnover batch
//! back to those records. Rebinding is required only after topology changes.

use std::collections::{HashMap, HashSet};

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const MAX_PARTS: usize = 1_000_000;
const MAX_POOLS: usize = 64;
const MAX_REACTIONS: usize = 4096;
const MAX_COLONIES: usize = 4096;

#[pyclass]
pub struct BiosphereTissue {
    pool_names: Vec<String>,
    pool_indices: HashMap<String, usize>,
    colony_names: Vec<String>,
    colony_indices: HashMap<String, usize>,
    structure_rows: Vec<usize>,
    stoich: Vec<f64>,
    consumed: Vec<Vec<usize>>,
    reactions: usize,
    part_ids: Vec<String>,
    part_colonies: Vec<usize>,
    members: Vec<Vec<usize>>,
    resources: Vec<f64>,
    totals: Vec<f64>,
    generation: u64,
}

#[pymethods]
impl BiosphereTissue {
    #[new]
    fn new(
        pool_names: Vec<String>,
        colony_names: Vec<String>,
        structure_rows: Vec<usize>,
        stoich: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let shape = stoich.shape();
        let reactions = shape[0];
        let pools = shape[1];
        if pool_names.is_empty()
            || pool_names.len() != pools
            || pools > MAX_POOLS
            || reactions == 0
            || reactions > MAX_REACTIONS
            || colony_names.is_empty()
            || colony_names.len() > MAX_COLONIES
            || structure_rows.len() != colony_names.len()
        {
            return Err(PyValueError::new_err("invalid Biosphere tissue dimensions"));
        }
        let pool_indices = unique_names(&pool_names, "chemical pool")?;
        let colony_indices = unique_names(&colony_names, "colony")?;
        let stoich = stoich.as_slice()?.to_vec();
        if stoich.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("tissue stoichiometry must be finite"));
        }
        let consumed = (0..reactions)
            .map(|reaction| {
                (0..pools)
                    .filter(|pool| stoich[reaction * pools + pool] < 0.0)
                    .collect()
            })
            .collect();
        let colony_count = structure_rows.len();
        Ok(Self {
            pool_names,
            pool_indices,
            colony_names,
            colony_indices,
            structure_rows,
            stoich,
            consumed,
            reactions,
            part_ids: Vec::new(),
            part_colonies: Vec::new(),
            members: vec![Vec::new(); colony_count],
            resources: Vec::new(),
            totals: vec![0.0; colony_count * pools],
            generation: 0,
        })
    }

    /// Replace the transient dense mirror after a committed topology change.
    fn bind(
        &mut self,
        parts: &Bound<'_, PyDict>,
        web_pools: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<usize> {
        let (part_ids, part_colonies, members, resources, totals) = self.parse_parts(parts)?;
        self.validate_totals(&totals, &web_pools)?;
        self.part_ids = part_ids;
        self.part_colonies = part_colonies;
        self.members = members;
        self.resources = resources;
        self.totals = totals;
        self.generation = self.generation.checked_add(1).ok_or_else(|| {
            PyRuntimeError::new_err("physical tissue binding generation overflow")
        })?;
        Ok(self.part_ids.len())
    }

    /// Check the native tissue aggregate before any chemical mutation.
    fn validate(&self, web_pools: PyReadonlyArray2<'_, f64>) -> PyResult<usize> {
        self.validate_totals(&self.totals, &web_pools)?;
        Ok(self.part_ids.len())
    }

    /// Apply a whole reactor turnover ledger and publish all parts together.
    fn turnover<'py>(
        &mut self,
        py: Python<'py>,
        parts: &Bound<'py, PyDict>,
        extents: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<usize> {
        if parts.len() != self.part_ids.len() {
            return Err(PyRuntimeError::new_err(
                "physical tissue topology changed without native rebind",
            ));
        }
        let extent_shape = extents.shape();
        if extent_shape[1] != self.reactions
            || self
                .structure_rows
                .iter()
                .any(|row| *row >= extent_shape[0])
        {
            return Err(PyValueError::new_err(
                "structural reaction ledger shape differs",
            ));
        }
        let extents = extents.as_slice()?;
        if extents.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "structural reaction ledger must be finite",
            ));
        }

        // Resolve every target record before numeric work or Python mutation.
        let mut records = Vec::with_capacity(self.part_ids.len());
        for (index, part_id) in self.part_ids.iter().enumerate() {
            let value = parts.get_item(part_id)?.ok_or_else(|| {
                PyRuntimeError::new_err("physical tissue topology changed without native rebind")
            })?;
            let record = value
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("physical tissue record must be an object"))?;
            let colony: String = required(record, "colony")?.extract()?;
            if self.colony_indices.get(&colony) != Some(&self.part_colonies[index]) {
                return Err(PyRuntimeError::new_err(
                    "physical tissue colony changed without native rebind",
                ));
            }
            records.push(record.clone());
        }

        let pools = self.pool_names.len();
        let mut after = self.resources.clone();
        for colony in 0..self.colony_names.len() {
            let members = &self.members[colony];
            if members.is_empty() {
                continue;
            }
            let extent_offset = self.structure_rows[colony] * self.reactions;
            for reaction in 0..self.reactions {
                let extent = extents[extent_offset + reaction];
                if extent <= 0.0 {
                    continue;
                }
                if self.consumed[reaction].len() != 1 {
                    return Err(PyRuntimeError::new_err(
                        "structural reaction has unsupported geometric bookkeeping",
                    ));
                }
                let substrate = self.consumed[reaction][0];
                let total = pairwise_sum(&self.resources, pools, members, substrate);
                if total <= 0.0 {
                    return Err(PyRuntimeError::new_err(
                        "structural reaction consumed unallocated substrate",
                    ));
                }
                for &part in members {
                    let offset = part * pools;
                    // Match the NumPy reference's multiply, divide, multiply,
                    // and add evaluation order without reassociation.
                    let scaled = extent * self.resources[offset + substrate];
                    let fraction = scaled / total;
                    for pool in 0..pools {
                        let delta = fraction * self.stoich[reaction * pools + pool];
                        after[offset + pool] += delta;
                    }
                }
            }
        }
        if after.iter().any(|value| !value.is_finite()) {
            return Err(PyRuntimeError::new_err(
                "structural turnover produced nonfinite material",
            ));
        }
        if after.iter().any(|value| *value < -1.0e-12) {
            return Err(PyRuntimeError::new_err(
                "structural turnover produced negative material",
            ));
        }
        for value in &mut after {
            *value = value.max(0.0);
        }
        let totals = aggregate(&after, pools, &self.part_colonies, self.colony_names.len());

        // Build every replacement dictionary before changing canonical records.
        let mut replacements = Vec::with_capacity(records.len());
        for part in 0..records.len() {
            let resources = PyDict::new(py);
            for pool in 0..pools {
                resources.set_item(&self.pool_names[pool], after[part * pools + pool])?;
            }
            replacements.push(resources);
        }
        for (record, resources) in records.iter().zip(replacements) {
            record.set_item("resources", resources)?;
        }
        self.resources = after;
        self.totals = totals;
        Ok(self.part_ids.len())
    }

    fn part_count(&self) -> usize {
        self.part_ids.len()
    }
}

impl BiosphereTissue {
    pub(crate) fn capture_binding(
        &self,
        part_ids: &[String],
        pool_name: &str,
    ) -> PyResult<(u64, usize, Vec<usize>)> {
        let pool = *self
            .pool_indices
            .get(pool_name)
            .ok_or_else(|| PyValueError::new_err("unknown capture tissue pool"))?;
        let lookup: HashMap<&str, usize> = self
            .part_ids
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let mut indices = Vec::with_capacity(part_ids.len());
        for part_id in part_ids {
            indices.push(*lookup.get(part_id.as_str()).ok_or_else(|| {
                PyRuntimeError::new_err("light capture names unbound physical tissue")
            })?);
        }
        Ok((self.generation, pool, indices))
    }

    pub(crate) fn capture_fractions(
        &self,
        generation: u64,
        indices: &[usize],
        pool: usize,
        initial: &[f64],
        result: &mut Vec<f64>,
    ) -> PyResult<()> {
        if generation != self.generation || indices.len() != initial.len() {
            return Err(PyRuntimeError::new_err(
                "light capture cache differs from tissue topology",
            ));
        }
        let pools = self.pool_names.len();
        if pool >= pools || indices.iter().any(|index| *index >= self.part_ids.len()) {
            return Err(PyRuntimeError::new_err(
                "invalid cached light capture index",
            ));
        }
        result.clear();
        result.reserve(indices.len());
        for (&index, &original) in indices.iter().zip(initial) {
            if !original.is_finite() || original <= 0.0 {
                return Err(PyValueError::new_err(
                    "initial capture tissue must be finite and positive",
                ));
            }
            let fraction = self.resources[index * pools + pool] / original;
            result.push(fraction.clamp(0.0, 1.0));
        }
        Ok(())
    }

    fn parse_parts(
        &self,
        parts: &Bound<'_, PyDict>,
    ) -> PyResult<(Vec<String>, Vec<usize>, Vec<Vec<usize>>, Vec<f64>, Vec<f64>)> {
        if parts.len() > MAX_PARTS {
            return Err(PyValueError::new_err(
                "physical tissue exceeds native part capacity",
            ));
        }
        let pools = self.pool_names.len();
        let mut part_ids = Vec::with_capacity(parts.len());
        let mut part_colonies = Vec::with_capacity(parts.len());
        let mut members = vec![Vec::new(); self.colony_names.len()];
        let mut resources = Vec::with_capacity(parts.len() * pools);
        let mut seen = HashSet::with_capacity(parts.len());
        for (part_id, value) in parts.iter() {
            let part_id: String = part_id
                .extract()
                .map_err(|_| PyValueError::new_err("physical tissue identity must be a string"))?;
            if part_id.is_empty() || part_id.len() > 192 || !seen.insert(part_id.clone()) {
                return Err(PyValueError::new_err(
                    "physical tissue identities must be bounded and unique",
                ));
            }
            let record = value
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("physical tissue record must be an object"))?;
            let colony_name: String = required(record, "colony")?.extract()?;
            let colony = *self
                .colony_indices
                .get(&colony_name)
                .ok_or_else(|| PyValueError::new_err("physical tissue names an unknown colony"))?;
            let resource_value = required(record, "resources")?;
            let resource_dict = resource_value.cast::<PyDict>().map_err(|_| {
                PyValueError::new_err("physical tissue resources must be an object")
            })?;
            for (name, _) in resource_dict.iter() {
                let name: String = name.extract().map_err(|_| {
                    PyValueError::new_err("chemical resource name must be a string")
                })?;
                if !self.pool_indices.contains_key(&name) {
                    return Err(PyValueError::new_err("unknown chemical resource"));
                }
            }
            for name in &self.pool_names {
                let amount = match resource_dict.get_item(name)? {
                    Some(amount) => amount.extract::<f64>()?,
                    None => 0.0,
                };
                if !amount.is_finite() || amount < 0.0 {
                    return Err(PyValueError::new_err(
                        "resource quantities must be finite and nonnegative",
                    ));
                }
                resources.push(amount);
            }
            let index = part_ids.len();
            part_ids.push(part_id);
            part_colonies.push(colony);
            members[colony].push(index);
        }
        let totals = aggregate(&resources, pools, &part_colonies, self.colony_names.len());
        Ok((part_ids, part_colonies, members, resources, totals))
    }

    fn validate_totals(
        &self,
        totals: &[f64],
        web_pools: &PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let shape = web_pools.shape();
        let pools = self.pool_names.len();
        if shape[1] != pools || self.structure_rows.iter().any(|row| *row >= shape[0]) {
            return Err(PyValueError::new_err(
                "physical tissue and web pool dimensions differ",
            ));
        }
        let web_pools = web_pools.as_slice()?;
        if web_pools
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "web resource quantities must be finite and nonnegative",
            ));
        }
        for colony in 0..self.colony_names.len() {
            for pool in 0..pools {
                let actual = totals[colony * pools + pool];
                let expected = web_pools[self.structure_rows[colony] * pools + pool];
                if actual != expected
                    && (actual - expected).abs() > 1.0e-12 + 1.0e-11 * expected.abs()
                {
                    return Err(PyValueError::new_err(
                        "physical structures and allocated chemical tissue disagree",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn required<'py>(record: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    record
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("physical tissue lacks {name}")))
}

fn unique_names(values: &[String], kind: &str) -> PyResult<HashMap<String, usize>> {
    let mut result = HashMap::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        if value.is_empty() || value.len() > 96 || result.insert(value.clone(), index).is_some() {
            return Err(PyValueError::new_err(format!(
                "{kind} names must be bounded and unique"
            )));
        }
    }
    Ok(result)
}

fn aggregate(resources: &[f64], pools: usize, colonies: &[usize], colony_count: usize) -> Vec<f64> {
    let mut totals = vec![0.0; colony_count * pools];
    for (part, &colony) in colonies.iter().enumerate() {
        for pool in 0..pools {
            totals[colony * pools + pool] += resources[part * pools + pool];
        }
    }
    totals
}

/// NumPy's pairwise sum for floating arrays (PW_BLOCKSIZE = 128).
fn pairwise_sum(resources: &[f64], pools: usize, members: &[usize], substrate: usize) -> f64 {
    pairwise_range(resources, pools, members, substrate, 0, members.len())
}

fn pairwise_range(
    resources: &[f64],
    pools: usize,
    members: &[usize],
    substrate: usize,
    start: usize,
    count: usize,
) -> f64 {
    if count < 8 {
        if count == 0 {
            return -0.0;
        }
        let mut result = resources[members[start] * pools + substrate];
        for index in 1..count {
            result += resources[members[start + index] * pools + substrate];
        }
        return result;
    }
    if count <= 128 {
        let mut accumulators = [0.0; 8];
        for index in 0..8 {
            accumulators[index] = resources[members[start + index] * pools + substrate];
        }
        let mut index = 8;
        while index <= count - 8 {
            for lane in 0..8 {
                accumulators[lane] += resources[members[start + index + lane] * pools + substrate];
            }
            index += 8;
        }
        let mut result = ((accumulators[0] + accumulators[1])
            + (accumulators[2] + accumulators[3]))
            + ((accumulators[4] + accumulators[5]) + (accumulators[6] + accumulators[7]));
        while index < count {
            result += resources[members[start + index] * pools + substrate];
            index += 1;
        }
        return result;
    }
    let mut left = count / 2;
    left -= left % 8;
    pairwise_range(resources, pools, members, substrate, start, left)
        + pairwise_range(
            resources,
            pools,
            members,
            substrate,
            start + left,
            count - left,
        )
}
