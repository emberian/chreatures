//! Batched physical-light transduction for attached developmental tissue.
//!
//! MuJoCo remains authoritative for poses and ray occlusion.  This object only
//! caches immutable attachment topology and ray profiles; live tissue remains
//! owned by `BiosphereTissue` and is read through a generation-checked view.

use crate::biosphere_tissue::BiosphereTissue;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const MAX_SAMPLES: usize = 1_000_000;
const MAX_LIGHTS: usize = 4096;
const MAX_PROFILES: usize = 4096;
const MAX_RAYS: usize = 256;

unsafe extern "C" {
    fn chreatures_mujoco_header_version() -> i32;
    fn chreatures_mujoco_runtime_version() -> i32;
    fn chreatures_environment_batch(
        model: *const std::ffi::c_void,
        data: *mut std::ffi::c_void,
        samples: i32,
        sample_bodies: *const i32,
        sample_local: *const f64,
        sample_world_offset: *const f64,
        sample_profiles: *const i32,
        profiles: i32,
        profile_offsets: *const i32,
        ray_directions: *const f64,
        ray_weights: *const f64,
        blocked_transmission: *const f64,
        solar_direction: *const f64,
        solar_direct: f64,
        solar_diffuse: f64,
        lights: i32,
        light_bodies: *const i32,
        light_local_position: *const f64,
        light_local_direction: *const f64,
        light_intensity: *const f64,
        light_radius: *const f64,
        flash_position: *const f64,
        flash_intensity: f64,
        flash_active: i32,
        bounds: *const f64,
        illumination: *mut f64,
    ) -> i32;
}

#[pyclass]
pub struct LightEnvironment {
    colonies: usize,
    profile_offsets: Vec<i32>,
    ray_directions: Vec<f64>,
    ray_weights: Vec<f64>,
    blocked_transmission: Vec<f64>,
    capture_bodies: Vec<i32>,
    capture_local: Vec<f64>,
    capture_world_offset: Vec<f64>,
    capture_profiles: Vec<i32>,
    capture_colonies: Vec<usize>,
    capture_area: Vec<f64>,
    capture_leaf_slot: Vec<i32>,
    tissue_generation: u64,
    tissue_pool: usize,
    tissue_indices: Vec<usize>,
    initial_tissue: Vec<f64>,
    fractions: Vec<f64>,
    sample_bodies: Vec<i32>,
    sample_local: Vec<f64>,
    sample_world_offset: Vec<f64>,
    sample_profiles: Vec<i32>,
    illumination: Vec<f64>,
}

#[pymethods]
impl LightEnvironment {
    #[new]
    fn new(
        colonies: usize,
        directions: Vec<Vec<f64>>,
        weights: Vec<f64>,
        profile_offsets: Vec<i32>,
        blocked_transmission: Vec<f64>,
    ) -> PyResult<Self> {
        if colonies == 0
            || colonies > MAX_PROFILES
            || profile_offsets.len() < 2
            || profile_offsets.len() - 1 > MAX_PROFILES
            || blocked_transmission.len() + 1 != profile_offsets.len()
            || weights.len() != directions.len()
            || directions.is_empty()
            || directions.len() > MAX_RAYS
            || profile_offsets[0] != 0
            || profile_offsets.last().copied() != Some(directions.len() as i32)
            || profile_offsets.windows(2).any(|pair| pair[0] >= pair[1])
            || blocked_transmission
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(PyValueError::new_err("invalid physical light profiles"));
        }
        let mut ray_directions = Vec::with_capacity(directions.len() * 3);
        let mut ray_weights = Vec::with_capacity(weights.len());
        for profile in 0..blocked_transmission.len() {
            let start = profile_offsets[profile] as usize;
            let end = profile_offsets[profile + 1] as usize;
            let total: f64 = weights[start..end].iter().sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(PyValueError::new_err(
                    "physical light profile weights must have positive finite sum",
                ));
            }
            for ray in start..end {
                let direction = &directions[ray];
                if direction.len() != 3
                    || direction.iter().any(|value| !value.is_finite())
                    || direction[2] < 0.0
                    || !weights[ray].is_finite()
                    || weights[ray] <= 0.0
                {
                    return Err(PyValueError::new_err(
                        "physical light rays must be finite upward hemisphere vectors",
                    ));
                }
                let norm = (direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2])
                    .sqrt();
                if norm <= 1.0e-12 {
                    return Err(PyValueError::new_err(
                        "physical light ray direction cannot be zero",
                    ));
                }
                ray_directions.extend(direction.iter().map(|value| value / norm));
                ray_weights.push(weights[ray] / total);
            }
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
        Ok(Self {
            colonies,
            profile_offsets,
            ray_directions,
            ray_weights,
            blocked_transmission,
            capture_bodies: Vec::new(),
            capture_local: Vec::new(),
            capture_world_offset: Vec::new(),
            capture_profiles: Vec::new(),
            capture_colonies: Vec::new(),
            capture_area: Vec::new(),
            capture_leaf_slot: Vec::new(),
            tissue_generation: 0,
            tissue_pool: 0,
            tissue_indices: Vec::new(),
            initial_tissue: Vec::new(),
            fractions: Vec::new(),
            sample_bodies: Vec::new(),
            sample_local: Vec::new(),
            sample_world_offset: Vec::new(),
            sample_profiles: Vec::new(),
            illumination: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_capture(
        &mut self,
        tissue: PyRef<'_, BiosphereTissue>,
        bodies: PyReadonlyArray1<'_, i32>,
        local: PyReadonlyArray2<'_, f64>,
        world_offset: PyReadonlyArray2<'_, f64>,
        profiles: PyReadonlyArray1<'_, i32>,
        colonies: PyReadonlyArray1<'_, i32>,
        areas: PyReadonlyArray1<'_, f64>,
        part_ids: Vec<Option<String>>,
        initial_tissue: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<usize> {
        let bodies = bodies.as_slice()?;
        let local = local.as_slice()?;
        let world_offset = world_offset.as_slice()?;
        let profiles = profiles.as_slice()?;
        let colonies = colonies.as_slice()?;
        let areas = areas.as_slice()?;
        let initial_tissue = initial_tissue.as_slice()?;
        let count = bodies.len();
        if count == 0
            || count > MAX_SAMPLES
            || local.len() != count * 3
            || world_offset.len() != count * 3
            || profiles.len() != count
            || colonies.len() != count
            || areas.len() != count
            || part_ids.len() != count
            || initial_tissue.len() != count
            || local
                .iter()
                .chain(world_offset)
                .any(|value| !value.is_finite())
            || areas.iter().any(|value| !value.is_finite() || *value < 0.0)
            || profiles
                .iter()
                .any(|profile| *profile < 0 || *profile as usize >= self.blocked_transmission.len())
            || colonies
                .iter()
                .any(|colony| *colony < 0 || *colony as usize >= self.colonies)
        {
            return Err(PyValueError::new_err(
                "invalid physical light capture layout",
            ));
        }
        let mut leaf_names = Vec::new();
        let mut leaf_initial = Vec::new();
        let mut leaf_slot = Vec::with_capacity(count);
        for (part_id, &initial) in part_ids.iter().zip(initial_tissue) {
            if let Some(part_id) = part_id {
                if part_id.is_empty() || !initial.is_finite() || initial <= 0.0 {
                    return Err(PyValueError::new_err("invalid capture tissue binding"));
                }
                leaf_slot.push(leaf_names.len() as i32);
                leaf_names.push(part_id.clone());
                leaf_initial.push(initial);
            } else {
                if initial != 0.0 {
                    return Err(PyValueError::new_err(
                        "founder capture cannot name initial tissue",
                    ));
                }
                leaf_slot.push(-1);
            }
        }
        let (generation, pool, indices) = tissue.capture_binding(&leaf_names, "soft_tissue")?;
        self.capture_bodies = bodies.to_vec();
        self.capture_local = local.to_vec();
        self.capture_world_offset = world_offset.to_vec();
        self.capture_profiles = profiles.to_vec();
        self.capture_colonies = colonies.iter().map(|value| *value as usize).collect();
        self.capture_area = areas.to_vec();
        self.capture_leaf_slot = leaf_slot;
        self.tissue_generation = generation;
        self.tissue_pool = pool;
        self.tissue_indices = indices;
        self.initial_tissue = leaf_initial;
        Ok(count)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        tissue: PyRef<'_, BiosphereTissue>,
        model_address: usize,
        data_address: usize,
        bud_bodies: PyReadonlyArray1<'_, i32>,
        bud_local: PyReadonlyArray2<'_, f64>,
        bud_profiles: PyReadonlyArray1<'_, i32>,
        solar_direction: Vec<f64>,
        solar_direct: f64,
        solar_diffuse: f64,
        light_bodies: PyReadonlyArray1<'_, i32>,
        light_local_position: PyReadonlyArray2<'_, f64>,
        light_local_direction: PyReadonlyArray2<'_, f64>,
        light_intensity: PyReadonlyArray1<'_, f64>,
        light_radius: PyReadonlyArray1<'_, f64>,
        flash_position: Vec<f64>,
        flash_intensity: f64,
        flash_active: bool,
        bounds: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if model_address == 0 || data_address == 0 || self.capture_bodies.is_empty() {
            return Err(PyRuntimeError::new_err(
                "physical light environment is not bound",
            ));
        }
        tissue.capture_fractions(
            self.tissue_generation,
            &self.tissue_indices,
            self.tissue_pool,
            &self.initial_tissue,
            &mut self.fractions,
        )?;
        let bud_bodies = bud_bodies.as_slice()?;
        let bud_local = bud_local.as_slice()?;
        let bud_profiles = bud_profiles.as_slice()?;
        let buds = bud_bodies.len();
        let light_bodies = light_bodies.as_slice()?.to_vec();
        let light_local_position = light_local_position.as_slice()?.to_vec();
        let light_local_direction = light_local_direction.as_slice()?.to_vec();
        let light_intensity = light_intensity.as_slice()?.to_vec();
        let light_radius = light_radius.as_slice()?.to_vec();
        let lights = light_bodies.len();
        if self.capture_bodies.len() + buds > MAX_SAMPLES
            || bud_local.len() != buds * 3
            || bud_profiles.len() != buds
            || bud_profiles
                .iter()
                .any(|profile| *profile < 0 || *profile as usize >= self.blocked_transmission.len())
            || solar_direction.len() != 3
            || solar_direction.iter().any(|value| !value.is_finite())
            || !solar_direct.is_finite()
            || solar_direct < 0.0
            || !solar_diffuse.is_finite()
            || solar_diffuse < 0.0
            || lights > MAX_LIGHTS
            || light_local_position.len() != lights * 3
            || light_local_direction.len() != lights * 3
            || light_intensity.len() != lights
            || light_radius.len() != lights
            || light_local_position
                .iter()
                .chain(&light_local_direction)
                .chain(&light_intensity)
                .chain(&light_radius)
                .any(|value| !value.is_finite())
            || light_intensity.iter().any(|value| *value < 0.0)
            || light_radius.iter().any(|value| *value <= 0.0)
            || flash_position.len() != 3
            || flash_position.iter().any(|value| !value.is_finite())
            || !flash_intensity.is_finite()
            || flash_intensity < 0.0
            || bounds.len() != 3
            || bounds
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err("invalid physical light sample"));
        }
        let solar_norm = solar_direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if solar_norm <= 1.0e-12 || (solar_norm - 1.0).abs() > 1.0e-10 {
            return Err(PyValueError::new_err(
                "solar direction must be a normalized physical vector",
            ));
        }

        self.sample_bodies.clear();
        self.sample_bodies.extend_from_slice(&self.capture_bodies);
        self.sample_bodies.extend_from_slice(bud_bodies);
        self.sample_local.clear();
        self.sample_local.extend_from_slice(&self.capture_local);
        self.sample_local.extend_from_slice(bud_local);
        self.sample_world_offset.clear();
        self.sample_world_offset
            .extend_from_slice(&self.capture_world_offset);
        self.sample_world_offset
            .resize((self.capture_bodies.len() + buds) * 3, 0.0);
        self.sample_profiles.clear();
        self.sample_profiles
            .extend_from_slice(&self.capture_profiles);
        self.sample_profiles.extend_from_slice(bud_profiles);
        self.illumination.resize(self.sample_bodies.len(), 0.0);

        let count = self.sample_bodies.len();
        let profiles = self.blocked_transmission.len();
        let result = py.detach(|| unsafe {
            chreatures_environment_batch(
                model_address as *const std::ffi::c_void,
                data_address as *mut std::ffi::c_void,
                count as i32,
                self.sample_bodies.as_ptr(),
                self.sample_local.as_ptr(),
                self.sample_world_offset.as_ptr(),
                self.sample_profiles.as_ptr(),
                profiles as i32,
                self.profile_offsets.as_ptr(),
                self.ray_directions.as_ptr(),
                self.ray_weights.as_ptr(),
                self.blocked_transmission.as_ptr(),
                solar_direction.as_ptr(),
                solar_direct,
                solar_diffuse,
                lights as i32,
                light_bodies.as_ptr(),
                light_local_position.as_ptr(),
                light_local_direction.as_ptr(),
                light_intensity.as_ptr(),
                light_radius.as_ptr(),
                flash_position.as_ptr(),
                flash_intensity,
                i32::from(flash_active),
                bounds.as_ptr(),
                self.illumination.as_mut_ptr(),
            )
        });
        if result != count as i32 {
            return Err(PyRuntimeError::new_err(format!(
                "native physical light sampling failed ({result})"
            )));
        }

        let mut capture = vec![0.0; self.colonies];
        for sample in 0..self.capture_bodies.len() {
            let slot = self.capture_leaf_slot[sample];
            let fraction = if slot < 0 {
                1.0
            } else {
                self.fractions[slot as usize]
            };
            capture[self.capture_colonies[sample]] +=
                self.capture_area[sample] * fraction * self.illumination[sample];
        }
        let bud_light = &self.illumination[self.capture_bodies.len()..];
        Ok((
            PyArray1::from_vec(py, capture),
            PyArray1::from_slice(py, bud_light),
        ))
    }
}
