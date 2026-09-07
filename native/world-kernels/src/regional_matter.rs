//! Conservative finite-material transport between physical habitat regions.
//!
//! Chemical inventory remains authoritative in MetabolicCohort rows. This
//! kernel owns only immutable network geometry, flow scheduling and ledgers.

use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const FORMAT: &str = "chreatures-regional-matter-v1";
const SNAPSHOT_FORMAT: &str = "chreatures-regional-matter-state-v1";
const MAX_REGIONS: usize = 256;
const MAX_ROUTES: usize = 512;
const MAX_OUTLETS: usize = 128;
const MAX_FACES: usize = 16;
const MAX_POOLS: usize = 64;
const MAX_SAMPLES_PER_ROUTE: usize = 8;

unsafe extern "C" {
    fn chreatures_regional_route_accessibility(
        model: *const std::ffi::c_void,
        data: *mut std::ffi::c_void,
        route_count: i32,
        sample_count: i32,
        sample_route: *const i32,
        starts: *const f64,
        ends: *const f64,
        radii: *const f64,
        excluded_bodies: *const i32,
        output: *mut f64,
    ) -> i32;
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    format: String,
    chemistry_sha256: String,
    world_size_m: [f64; 3],
    regions: Vec<RegionConfig>,
    routes: Vec<RouteConfig>,
    exit_faces: Vec<FaceConfig>,
    outlets: Vec<OutletConfig>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionConfig {
    id: String,
    row: usize,
    position: [f64; 3],
    capacities: HashMap<String, f64>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteConfig {
    id: String,
    a: String,
    b: String,
    conductance: HashMap<String, f64>,
    clearance_samples: Vec<ClearanceSample>,
    carrier_entity: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClearanceSample {
    from: [f64; 3],
    to: [f64; 3],
    radius: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FaceConfig {
    id: String,
    axis: String,
    side: String,
    coordinate_m: f64,
    region: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutletConfig {
    id: String,
    region: String,
    slots: Vec<String>,
    position: [f64; 3],
    interval_s: f64,
    maximum_release: HashMap<String, f64>,
}

#[derive(Clone)]
struct Region {
    id: String,
    row: usize,
    position: [f64; 3],
    capacity: Vec<f64>,
}

#[derive(Clone)]
struct Route {
    id: String,
    a: usize,
    b: usize,
    conductance: Vec<f64>,
    carrier_entity: Option<String>,
}

#[derive(Clone)]
struct Face {
    id: String,
    axis: usize,
    maximum: bool,
    coordinate: f64,
    region: usize,
}

#[derive(Clone)]
struct Outlet {
    id: String,
    region: usize,
    slots: Vec<String>,
    position: [f64; 3],
    interval: f64,
    maximum: Vec<f64>,
}

#[derive(Clone)]
struct Pending {
    token: String,
    dt: f64,
    next_credit: Vec<f64>,
    route_resources: Vec<f64>,
    outlet_resources: Vec<f64>,
    accessibility: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    format: String,
    config_sha256: String,
    time: f64,
    step_index: u64,
    outlet_credit: Vec<f64>,
    route_cumulative: Vec<f64>,
    outlet_cumulative: Vec<f64>,
    last_route: Vec<f64>,
    last_outlet: Vec<f64>,
    last_accessibility: Vec<f64>,
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphabetic()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn finite_vector(value: &[f64]) -> bool {
    value.iter().all(|item| item.is_finite())
}

fn sha256(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn same_f64(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn token_for(
    config_sha256: &str,
    step_index: u64,
    time: f64,
    dt: f64,
    pools: &[f64],
    accessibility: &[f64],
    slots: &[bool],
) -> String {
    let mut digest = Sha256::new();
    digest.update(config_sha256.as_bytes());
    digest.update(step_index.to_le_bytes());
    digest.update(time.to_bits().to_le_bytes());
    digest.update(dt.to_bits().to_le_bytes());
    for value in pools.iter().chain(accessibility) {
        digest.update(value.to_bits().to_le_bytes());
    }
    for value in slots {
        digest.update([u8::from(*value)]);
    }
    format!("{:x}", digest.finalize())
}

#[pyclass]
pub struct RegionalMatter {
    config_sha256: String,
    chemistry_sha256: String,
    pool_names: Vec<String>,
    regions: Vec<Region>,
    routes: Vec<Route>,
    sample_route: Vec<i32>,
    sample_starts: Vec<f64>,
    sample_ends: Vec<f64>,
    sample_radii: Vec<f64>,
    faces: Vec<Face>,
    outlets: Vec<Outlet>,
    time: f64,
    step_index: u64,
    outlet_credit: Vec<f64>,
    route_cumulative: Vec<f64>,
    outlet_cumulative: Vec<f64>,
    last_route: Vec<f64>,
    last_outlet: Vec<f64>,
    last_accessibility: Vec<f64>,
    pending: Option<Pending>,
}

#[pymethods]
impl RegionalMatter {
    #[new]
    fn new(config_json: String, config_sha256: String, pool_names: Vec<String>) -> PyResult<Self> {
        if sha256(config_json.as_bytes()) != config_sha256 || !valid_sha(&config_sha256) {
            return Err(PyValueError::new_err(
                "regional matter configuration hash differs",
            ));
        }
        if pool_names.is_empty()
            || pool_names.len() > MAX_POOLS
            || pool_names.iter().any(|name| !valid_id(name))
            || pool_names.iter().collect::<HashSet<_>>().len() != pool_names.len()
        {
            return Err(PyValueError::new_err(
                "regional matter pool identities differ",
            ));
        }
        let config: Config = serde_json::from_str(&config_json).map_err(|error| {
            PyValueError::new_err(format!("invalid regional matter config: {error}"))
        })?;
        if config.format != FORMAT
            || !valid_sha(&config.chemistry_sha256)
            || config.regions.is_empty()
            || config.regions.len() > MAX_REGIONS
            || config.routes.len() > MAX_ROUTES
            || config.exit_faces.is_empty()
            || config.exit_faces.len() > MAX_FACES
            || config.outlets.len() > MAX_OUTLETS
            || !finite_vector(&config.world_size_m)
            || config
                .world_size_m
                .iter()
                .any(|value| *value <= 0.0 || *value > 1e5)
        {
            return Err(PyValueError::new_err("regional matter dimensions differ"));
        }
        let pool_index: HashMap<&str, usize> = pool_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let mut ids = HashSet::new();
        let mut rows = HashSet::new();
        let mut regions = Vec::with_capacity(config.regions.len());
        for value in config.regions {
            if !valid_id(&value.id)
                || !ids.insert(value.id.clone())
                || !rows.insert(value.row)
                || !finite_vector(&value.position)
                || value
                    .position
                    .iter()
                    .zip(&config.world_size_m)
                    .any(|(position, limit)| *position < 0.0 || *position > *limit)
                || value.capacities.len() != pool_names.len()
                || value
                    .capacities
                    .keys()
                    .any(|name| !pool_index.contains_key(name.as_str()))
            {
                return Err(PyValueError::new_err("invalid regional matter node"));
            }
            let capacity: Vec<f64> = pool_names
                .iter()
                .map(|name| value.capacities[name])
                .collect();
            if capacity
                .iter()
                .any(|item| !item.is_finite() || *item <= 0.0 || *item > 1e12)
            {
                return Err(PyValueError::new_err(
                    "regional capacities must be finite and positive",
                ));
            }
            regions.push(Region {
                id: value.id,
                row: value.row,
                position: value.position,
                capacity,
            });
        }
        let region_index: HashMap<&str, usize> = regions
            .iter()
            .enumerate()
            .map(|(index, region)| (region.id.as_str(), index))
            .collect();
        let mut route_ids = HashSet::new();
        let mut routes = Vec::with_capacity(config.routes.len());
        let mut sample_route = Vec::new();
        let mut sample_starts = Vec::new();
        let mut sample_ends = Vec::new();
        let mut sample_radii = Vec::new();
        for value in config.routes {
            let Some(&a) = region_index.get(value.a.as_str()) else {
                return Err(PyValueError::new_err("route source region is absent"));
            };
            let Some(&b) = region_index.get(value.b.as_str()) else {
                return Err(PyValueError::new_err("route target region is absent"));
            };
            if !valid_id(&value.id)
                || !route_ids.insert(value.id.clone())
                || a == b
                || value.conductance.len() != pool_names.len()
                || value
                    .conductance
                    .keys()
                    .any(|name| !pool_index.contains_key(name.as_str()))
                || value.clearance_samples.is_empty()
                || value.clearance_samples.len() > MAX_SAMPLES_PER_ROUTE
                || value
                    .carrier_entity
                    .as_deref()
                    .is_some_and(|id| !valid_id(id))
            {
                return Err(PyValueError::new_err("invalid regional route"));
            }
            let conductance: Vec<f64> = pool_names
                .iter()
                .map(|name| value.conductance[name])
                .collect();
            if conductance
                .iter()
                .any(|item| !item.is_finite() || *item < 0.0 || *item > 1e9)
                || !conductance.iter().any(|item| *item > 0.0)
                || value.clearance_samples.iter().any(|sample| {
                    !finite_vector(&sample.from)
                        || !finite_vector(&sample.to)
                        || !sample.radius.is_finite()
                        || sample.radius < 0.0
                        || sample.radius > 10.0
                        || sample
                            .from
                            .iter()
                            .chain(&sample.to)
                            .any(|coordinate| coordinate.abs() > 1e5)
                        || sample
                            .from
                            .iter()
                            .zip(&sample.to)
                            .all(|(left, right)| left.to_bits() == right.to_bits())
                })
            {
                return Err(PyValueError::new_err("invalid regional route samples"));
            }
            let route_index = routes.len() as i32;
            for sample in value.clearance_samples {
                sample_route.push(route_index);
                sample_starts.extend_from_slice(&sample.from);
                sample_ends.extend_from_slice(&sample.to);
                sample_radii.push(sample.radius);
            }
            routes.push(Route {
                id: value.id,
                a,
                b,
                conductance,
                carrier_entity: value.carrier_entity,
            });
        }
        let mut face_ids = HashSet::new();
        let mut faces = Vec::with_capacity(config.exit_faces.len());
        for value in config.exit_faces {
            let axis = match value.axis.as_str() {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => return Err(PyValueError::new_err("invalid regional exit axis")),
            };
            let maximum = match value.side.as_str() {
                "min" => false,
                "max" => true,
                _ => return Err(PyValueError::new_err("invalid regional exit side")),
            };
            let Some(&region) = region_index.get(value.region.as_str()) else {
                return Err(PyValueError::new_err("exit receiver region is absent"));
            };
            if !valid_id(&value.id)
                || !face_ids.insert(value.id.clone())
                || !value.coordinate_m.is_finite()
                || value.coordinate_m < 0.0
                || value.coordinate_m > config.world_size_m[axis]
            {
                return Err(PyValueError::new_err("invalid regional exit face"));
            }
            faces.push(Face {
                id: value.id,
                axis,
                maximum,
                coordinate: value.coordinate_m,
                region,
            });
        }
        let mut outlet_ids = HashSet::new();
        let mut slot_ids = HashSet::new();
        let mut outlets = Vec::with_capacity(config.outlets.len());
        for value in config.outlets {
            let Some(&region) = region_index.get(value.region.as_str()) else {
                return Err(PyValueError::new_err("outlet source region is absent"));
            };
            if !valid_id(&value.id)
                || !outlet_ids.insert(value.id.clone())
                || value.slots.is_empty()
                || value.slots.len() > 64
                || value
                    .slots
                    .iter()
                    .any(|slot| !valid_id(slot) || !slot_ids.insert(slot.clone()))
                || !finite_vector(&value.position)
                || value
                    .position
                    .iter()
                    .zip(&config.world_size_m)
                    .any(|(position, limit)| *position < 0.0 || *position > *limit)
                || !value.interval_s.is_finite()
                || value.interval_s < 0.05
                || value.interval_s > 1e9
                || value.maximum_release.len() != pool_names.len()
                || value
                    .maximum_release
                    .keys()
                    .any(|name| !pool_index.contains_key(name.as_str()))
            {
                return Err(PyValueError::new_err("invalid regional outlet"));
            }
            let maximum: Vec<f64> = pool_names
                .iter()
                .map(|name| value.maximum_release[name])
                .collect();
            if maximum
                .iter()
                .any(|item| !item.is_finite() || *item < 0.0 || *item > 1e9)
                || !maximum.iter().any(|item| *item > 0.0)
            {
                return Err(PyValueError::new_err("invalid regional outlet release"));
            }
            outlets.push(Outlet {
                id: value.id,
                region,
                slots: value.slots,
                position: value.position,
                interval: value.interval_s,
                maximum,
            });
        }
        let k = pool_names.len();
        let route_count = routes.len();
        let outlet_count = outlets.len();
        Ok(Self {
            config_sha256,
            chemistry_sha256: config.chemistry_sha256,
            pool_names,
            regions,
            routes,
            sample_route,
            sample_starts,
            sample_ends,
            sample_radii,
            faces,
            outlets,
            time: 0.0,
            step_index: 0,
            outlet_credit: vec![0.0; outlet_count],
            route_cumulative: vec![0.0; route_count * k],
            outlet_cumulative: vec![0.0; outlet_count * k],
            last_route: vec![0.0; route_count * k],
            last_outlet: vec![0.0; outlet_count * k],
            last_accessibility: vec![1.0; route_count],
            pending: None,
        })
    }

    #[getter]
    fn config_sha256(&self) -> String {
        self.config_sha256.clone()
    }

    #[getter]
    fn chemistry_sha256(&self) -> String {
        self.chemistry_sha256.clone()
    }

    #[getter]
    fn time(&self) -> f64 {
        self.time
    }

    #[getter]
    fn step_index(&self) -> u64 {
        self.step_index
    }

    fn region_rows(&self) -> Vec<usize> {
        self.regions.iter().map(|region| region.row).collect()
    }

    fn route_carriers(&self) -> Vec<Option<String>> {
        self.routes
            .iter()
            .map(|route| route.carrier_entity.clone())
            .collect()
    }

    fn outlet_slots(&self) -> Vec<Vec<String>> {
        self.outlets
            .iter()
            .map(|outlet| outlet.slots.clone())
            .collect()
    }

    fn detect_exits<'py>(
        &self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let shape = positions.shape();
        if shape.len() != 2 || shape[1] != 3 || shape[0] > 4096 {
            return Err(PyValueError::new_err(
                "material exit positions must be [N,3]",
            ));
        }
        let values = positions.as_slice()?;
        if !finite_vector(values) {
            return Err(PyValueError::new_err(
                "material exit positions must be finite",
            ));
        }
        let mut result = vec![-1_i32; shape[0]];
        for packet in 0..shape[0] {
            let position = &values[packet * 3..packet * 3 + 3];
            let mut best = 0.0;
            for (face_index, face) in self.faces.iter().enumerate() {
                let distance = if face.maximum {
                    position[face.axis] - face.coordinate
                } else {
                    face.coordinate - position[face.axis]
                };
                if distance > best {
                    best = distance;
                    result[packet] = face_index as i32;
                }
            }
        }
        Ok(result.into_pyarray(py))
    }

    fn route_accessibility<'py>(
        &self,
        py: Python<'py>,
        model_address: usize,
        data_address: usize,
        excluded_bodies: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if model_address == 0 || data_address == 0 || excluded_bodies.shape() != [self.routes.len()]
        {
            return Err(PyValueError::new_err(
                "regional route physical bindings differ",
            ));
        }
        let excluded = excluded_bodies.as_slice()?;
        if excluded.iter().any(|body| *body < -1) {
            return Err(PyValueError::new_err("invalid regional route carrier body"));
        }
        let sample_count = self.sample_route.len();
        let mut output = vec![0.0; self.routes.len()];
        let got = py.detach(|| unsafe {
            chreatures_regional_route_accessibility(
                model_address as *const _,
                data_address as *mut _,
                self.routes.len() as i32,
                sample_count as i32,
                self.sample_route.as_ptr(),
                self.sample_starts.as_ptr(),
                self.sample_ends.as_ptr(),
                self.sample_radii.as_ptr(),
                excluded.as_ptr(),
                output.as_mut_ptr(),
            )
        });
        if got != self.routes.len() as i32
            || output
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
        {
            return Err(PyRuntimeError::new_err(
                "native regional route sampling failed",
            ));
        }
        Ok(output.into_pyarray(py))
    }

    fn propose<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        pools: PyReadonlyArray2<'_, f64>,
        accessibility: PyReadonlyArray1<'_, f64>,
        outlet_available: Vec<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "regional matter proposal already pending",
            ));
        }
        let n = self.regions.len();
        let k = self.pool_names.len();
        let edge_count = self.routes.len();
        let outlet_count = self.outlets.len();
        if !dt.is_finite()
            || dt <= 0.0
            || dt > 1.0
            || pools.shape() != [n, k]
            || accessibility.shape() != [edge_count]
            || outlet_available.len() != outlet_count
        {
            return Err(PyValueError::new_err(
                "regional matter proposal dimensions differ",
            ));
        }
        let pools = pools.as_slice()?;
        let access = accessibility.as_slice()?;
        if !finite_vector(pools)
            || pools.iter().any(|value| *value < 0.0)
            || !finite_vector(access)
            || access.iter().any(|value| *value < 0.0 || *value > 1.0)
        {
            return Err(PyValueError::new_err(
                "regional matter proposal values differ",
            ));
        }
        for region in 0..n {
            for pool in 0..k {
                if pools[region * k + pool] > self.regions[region].capacity[pool] + 1e-10 {
                    return Err(PyValueError::new_err("regional inventory exceeds capacity"));
                }
            }
        }

        // Different chemicals can have opposing gradients on one physical
        // route. Preserve one donor/receiver address per edge and pool.
        let mut source = vec![0_i64; edge_count * k];
        let mut target = vec![0_i64; edge_count * k];
        let mut route_raw = vec![0.0; edge_count * k];
        for (edge, route) in self.routes.iter().enumerate() {
            for pool in 0..k {
                let ca = pools[route.a * k + pool] / self.regions[route.a].capacity[pool];
                let cb = pools[route.b * k + pool] / self.regions[route.b].capacity[pool];
                let (donor, receiver, difference) = if ca >= cb {
                    (route.a, route.b, ca - cb)
                } else {
                    (route.b, route.a, cb - ca)
                };
                source[edge * k + pool] = donor as i64;
                target[edge * k + pool] = receiver as i64;
                route_raw[edge * k + pool] =
                    route.conductance[pool] * access[edge] * dt * difference;
            }
        }
        let mut next_credit = self.outlet_credit.clone();
        let mut outlet_raw = vec![0.0; outlet_count * k];
        for (outlet_index, outlet) in self.outlets.iter().enumerate() {
            next_credit[outlet_index] = (next_credit[outlet_index] + dt).min(outlet.interval);
            if next_credit[outlet_index] >= outlet.interval && outlet_available[outlet_index] {
                for pool in 0..k {
                    outlet_raw[outlet_index * k + pool] = outlet.maximum[pool];
                }
            }
        }

        // All route and outlet outflows compete against the same pre-state.
        let mut donor_demand = vec![0.0; n * k];
        let mut receiver_demand = vec![0.0; n * k];
        for edge in 0..edge_count {
            for pool in 0..k {
                donor_demand[source[edge * k + pool] as usize * k + pool] +=
                    route_raw[edge * k + pool];
                receiver_demand[target[edge * k + pool] as usize * k + pool] +=
                    route_raw[edge * k + pool];
            }
        }
        for (outlet_index, outlet) in self.outlets.iter().enumerate() {
            for pool in 0..k {
                donor_demand[outlet.region * k + pool] += outlet_raw[outlet_index * k + pool];
            }
        }
        if !finite_vector(&donor_demand) || !finite_vector(&receiver_demand) {
            return Err(PyValueError::new_err(
                "regional matter aggregate demand overflowed",
            ));
        }
        let mut donor_factor = vec![1.0; n * k];
        let mut receiver_factor = vec![1.0; n * k];
        for region in 0..n {
            for pool in 0..k {
                let index = region * k + pool;
                if donor_demand[index] > 0.0 {
                    donor_factor[index] = (pools[index] / donor_demand[index]).min(1.0);
                }
                if receiver_demand[index] > 0.0 {
                    receiver_factor[index] = ((self.regions[region].capacity[pool] - pools[index])
                        / receiver_demand[index])
                        .clamp(0.0, 1.0);
                }
            }
        }
        let mut route_resources = vec![0.0; edge_count * k];
        for edge in 0..edge_count {
            for pool in 0..k {
                route_resources[edge * k + pool] = route_raw[edge * k + pool]
                    * donor_factor[source[edge * k + pool] as usize * k + pool]
                    * receiver_factor[target[edge * k + pool] as usize * k + pool];
            }
        }
        let mut outlet_resources = vec![0.0; outlet_count * k];
        for (outlet_index, outlet) in self.outlets.iter().enumerate() {
            for pool in 0..k {
                outlet_resources[outlet_index * k + pool] =
                    outlet_raw[outlet_index * k + pool] * donor_factor[outlet.region * k + pool];
            }
        }
        let token = token_for(
            &self.config_sha256,
            self.step_index,
            self.time,
            dt,
            pools,
            access,
            &outlet_available,
        );
        self.pending = Some(Pending {
            token: token.clone(),
            dt,
            next_credit,
            route_resources: route_resources.clone(),
            outlet_resources: outlet_resources.clone(),
            accessibility: access.to_vec(),
        });
        let result = PyDict::new(py);
        result.set_item("token", token)?;
        result.set_item(
            "route_source",
            Array2::from_shape_vec((edge_count, k), source)
                .map_err(|_| PyRuntimeError::new_err("regional route source shape differs"))?
                .into_pyarray(py),
        )?;
        result.set_item(
            "route_target",
            Array2::from_shape_vec((edge_count, k), target)
                .map_err(|_| PyRuntimeError::new_err("regional route target shape differs"))?
                .into_pyarray(py),
        )?;
        result.set_item(
            "route_resources",
            Array2::from_shape_vec((edge_count, k), route_resources)
                .unwrap()
                .into_pyarray(py),
        )?;
        result.set_item(
            "outlet_resources",
            Array2::from_shape_vec((outlet_count, k), outlet_resources)
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok(result)
    }

    fn abort(&mut self, token: String) -> PyResult<()> {
        match &self.pending {
            Some(pending) if pending.token == token => {
                self.pending = None;
                Ok(())
            }
            _ => Err(PyRuntimeError::new_err(
                "regional matter proposal token differs",
            )),
        }
    }

    fn commit(
        &mut self,
        token: String,
        actual_route: PyReadonlyArray2<'_, f64>,
        actual_outlet: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let k = self.pool_names.len();
        let Some(pending) = self.pending.as_ref() else {
            return Err(PyRuntimeError::new_err(
                "regional matter has no pending proposal",
            ));
        };
        if pending.token != token
            || actual_route.shape() != [self.routes.len(), k]
            || actual_outlet.shape() != [self.outlets.len(), k]
        {
            return Err(PyRuntimeError::new_err(
                "regional matter commit identity differs",
            ));
        }
        let routes = actual_route.as_slice()?;
        let outlets = actual_outlet.as_slice()?;
        if !same_f64(routes, &pending.route_resources) || !finite_vector(outlets) {
            return Err(PyRuntimeError::new_err(
                "regional route application differs",
            ));
        }
        for outlet in 0..self.outlets.len() {
            let actual = &outlets[outlet * k..(outlet + 1) * k];
            let proposed = &pending.outlet_resources[outlet * k..(outlet + 1) * k];
            if actual
                .iter()
                .zip(proposed)
                .any(|(moved, requested)| !moved.is_finite() || *moved < 0.0 || *moved > *requested)
            {
                return Err(PyRuntimeError::new_err(
                    "regional outlet application exceeds its proposal",
                ));
            }
        }
        let pending = self.pending.take().unwrap();
        self.time += pending.dt;
        self.step_index = self.step_index.saturating_add(1);
        self.outlet_credit = pending.next_credit;
        for outlet in 0..self.outlets.len() {
            let actual = &outlets[outlet * k..(outlet + 1) * k];
            let moved: f64 = actual.iter().sum();
            let maximum: f64 = self.outlets[outlet].maximum.iter().sum();
            if moved > 0.0 && maximum > 0.0 {
                let consumed = self.outlets[outlet].interval * (moved / maximum).min(1.0);
                self.outlet_credit[outlet] = (self.outlet_credit[outlet] - consumed).max(0.0);
            }
        }
        for index in 0..self.route_cumulative.len() {
            self.route_cumulative[index] += routes[index];
        }
        for index in 0..self.outlet_cumulative.len() {
            self.outlet_cumulative[index] += outlets[index];
        }
        self.last_route.copy_from_slice(routes);
        self.last_outlet.copy_from_slice(outlets);
        self.last_accessibility = pending.accessibility;
        Ok(())
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let k = self.pool_names.len();
        let result = PyDict::new(py);
        result.set_item("time", self.time)?;
        result.set_item("step_index", self.step_index)?;
        result.set_item("outlet_credit", self.outlet_credit.clone().into_pyarray(py))?;
        result.set_item(
            "route_cumulative",
            Array2::from_shape_vec((self.routes.len(), k), self.route_cumulative.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        result.set_item(
            "outlet_cumulative",
            Array2::from_shape_vec((self.outlets.len(), k), self.outlet_cumulative.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        result.set_item(
            "last_route",
            Array2::from_shape_vec((self.routes.len(), k), self.last_route.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        result.set_item(
            "last_outlet",
            Array2::from_shape_vec((self.outlets.len(), k), self.last_outlet.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        result.set_item(
            "last_accessibility",
            self.last_accessibility.clone().into_pyarray(py),
        )?;
        Ok(result)
    }

    fn snapshot(&self) -> PyResult<String> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot snapshot regional matter during a pending transaction",
            ));
        }
        serde_json::to_string(&State {
            format: SNAPSHOT_FORMAT.into(),
            config_sha256: self.config_sha256.clone(),
            time: self.time,
            step_index: self.step_index,
            outlet_credit: self.outlet_credit.clone(),
            route_cumulative: self.route_cumulative.clone(),
            outlet_cumulative: self.outlet_cumulative.clone(),
            last_route: self.last_route.clone(),
            last_outlet: self.last_outlet.clone(),
            last_accessibility: self.last_accessibility.clone(),
        })
        .map_err(|error| PyRuntimeError::new_err(format!("regional snapshot failed: {error}")))
    }

    fn restore(&mut self, snapshot_json: String) -> PyResult<()> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot restore regional matter during a pending transaction",
            ));
        }
        let state: State = serde_json::from_str(&snapshot_json).map_err(|error| {
            PyValueError::new_err(format!("invalid regional snapshot: {error}"))
        })?;
        let k = self.pool_names.len();
        if state.format != SNAPSHOT_FORMAT
            || state.config_sha256 != self.config_sha256
            || !state.time.is_finite()
            || state.time < 0.0
            || state.outlet_credit.len() != self.outlets.len()
            || state.route_cumulative.len() != self.routes.len() * k
            || state.outlet_cumulative.len() != self.outlets.len() * k
            || state.last_route.len() != self.routes.len() * k
            || state.last_outlet.len() != self.outlets.len() * k
            || state.last_accessibility.len() != self.routes.len()
            || !finite_vector(&state.outlet_credit)
            || !finite_vector(&state.route_cumulative)
            || !finite_vector(&state.outlet_cumulative)
            || !finite_vector(&state.last_route)
            || !finite_vector(&state.last_outlet)
            || !finite_vector(&state.last_accessibility)
            || state
                .outlet_credit
                .iter()
                .zip(&self.outlets)
                .any(|(credit, outlet)| *credit < 0.0 || *credit > outlet.interval)
            || state
                .route_cumulative
                .iter()
                .chain(&state.outlet_cumulative)
                .chain(&state.last_route)
                .chain(&state.last_outlet)
                .any(|value| *value < 0.0)
            || state
                .last_accessibility
                .iter()
                .any(|value| *value < 0.0 || *value > 1.0)
        {
            return Err(PyValueError::new_err("regional snapshot state differs"));
        }
        self.time = state.time;
        self.step_index = state.step_index;
        self.outlet_credit = state.outlet_credit;
        self.route_cumulative = state.route_cumulative;
        self.outlet_cumulative = state.outlet_cumulative;
        self.last_route = state.last_route;
        self.last_outlet = state.last_outlet;
        self.last_accessibility = state.last_accessibility;
        Ok(())
    }

    fn metadata(&self) -> Vec<(String, usize, [f64; 3], Vec<f64>)> {
        self.regions
            .iter()
            .map(|region| {
                (
                    region.id.clone(),
                    region.row,
                    region.position,
                    region.capacity.clone(),
                )
            })
            .collect()
    }

    fn route_metadata(&self) -> Vec<(String, usize, usize, Option<String>)> {
        self.routes
            .iter()
            .map(|route| {
                (
                    route.id.clone(),
                    route.a,
                    route.b,
                    route.carrier_entity.clone(),
                )
            })
            .collect()
    }

    fn face_metadata(&self) -> Vec<(String, usize, Vec<f64>)> {
        self.faces
            .iter()
            .map(|face| {
                (
                    face.id.clone(),
                    face.region,
                    self.regions[face.region].capacity.clone(),
                )
            })
            .collect()
    }

    fn outlet_metadata(&self) -> Vec<(String, usize, Vec<String>, [f64; 3], f64)> {
        self.outlets
            .iter()
            .map(|outlet| {
                (
                    outlet.id.clone(),
                    outlet.region,
                    outlet.slots.clone(),
                    outlet.position,
                    outlet.interval,
                )
            })
            .collect()
    }
}
