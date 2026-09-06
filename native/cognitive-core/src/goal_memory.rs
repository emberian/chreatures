use numpy::{
    ndarray::{Array2, Array3, Array4},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

const WINDOW: usize = 4;
const KEY: usize = 64;
const CAPACITY: usize = 128;
const HOLD_TICKS: u64 = 10;
const FORMAT: &str = "chreatures-achieved-goal-memory-v2";

pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn random_u64(state: &mut [u64]) -> u64 {
    debug_assert_eq!(state.len(), 4);
    let result = state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
    let t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = state[3].rotate_left(45);
    result
}

fn index_below(state: &mut [u64], bound: u64) -> u64 {
    debug_assert!(bound > 0);
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = random_u64(state);
        if value >= threshold {
            return value % bound;
        }
    }
}

pub(crate) fn unit_f64(state: &mut [u64]) -> f64 {
    ((random_u64(state) >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn required<'py>(value: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("goal-memory snapshot lacks {name}")))
}

fn finite_f32(value: &[f32]) -> bool {
    value.iter().all(|x| x.is_finite())
}

fn finite_f64(value: &[f64]) -> bool {
    value.iter().all(|x| x.is_finite() && *x >= 0.0)
}

/// Private achieved-history storage for a fixed resident cohort.
#[pyclass]
pub(crate) struct AchievedGoalMemoryCohort {
    batch: usize,
    observation_dim: usize,
    ring: Vec<f32>,
    ring_tick: Vec<u64>,
    ring_time: Vec<f64>,
    ring_count: Vec<u8>,
    ring_cursor: Vec<u8>,
    generation: Vec<u64>,
    consumed_generation: Vec<u64>,
    has_push: Vec<bool>,
    last_push_tick: Vec<u64>,
    windows: Vec<f32>,
    keys: Vec<f32>,
    recorded_tick: Vec<u64>,
    recorded_time: Vec<f64>,
    slot_generation: Vec<u64>,
    count: Vec<u16>,
    seen: Vec<u64>,
    rng: Vec<u64>,
    selected_valid: Vec<bool>,
    selected_slot: Vec<i32>,
    selected_window: Vec<f32>,
    selected_key: Vec<f32>,
    selected_recorded_tick: Vec<u64>,
    selected_recorded_time: Vec<f64>,
    selected_generation: Vec<u64>,
    selected_at_tick: Vec<u64>,
    hold_until_tick: Vec<u64>,
    has_choose_tick: Vec<bool>,
    last_choose_tick: Vec<u64>,
}

impl AchievedGoalMemoryCohort {
    pub(crate) fn recent_ring_state(&self) -> (&[u8], &[u8]) {
        (&self.ring_count, &self.ring_cursor)
    }
    pub(crate) fn key_rows(&self) -> (&[f32], &[u16]) {
        (&self.keys, &self.count)
    }

    pub(crate) fn counts(&self) -> &[u16] {
        &self.count
    }

    /// Flat `[resident, slot]` identity matrices for retrieval scoring. A
    /// generation of zero denotes an unoccupied slot.
    pub(crate) fn slot_identities(&self) -> (&[u64], &[u64]) {
        (&self.recorded_tick, &self.slot_generation)
    }

    pub(crate) fn current_selection(&self, ticks: &[u64]) -> SelectionArrays {
        self.selection_arrays(ticks, vec![false; self.batch])
    }

    fn window_offset(&self, row: usize, slot: usize) -> usize {
        (row * CAPACITY + slot) * WINDOW * self.observation_dim
    }

    fn key_offset(row: usize, slot: usize) -> usize {
        (row * CAPACITY + slot) * KEY
    }

    fn materialize_ring(&self, row: usize, target: &mut [f32]) {
        debug_assert_eq!(target.len(), WINDOW * self.observation_dim);
        target.fill(0.0);
        let count = self.ring_count[row] as usize;
        let start = if count == WINDOW {
            self.ring_cursor[row] as usize
        } else {
            0
        };
        for frame in 0..count {
            let source_frame = (start + frame) % WINDOW;
            let source = (row * WINDOW + source_frame) * self.observation_dim;
            let destination = frame * self.observation_dim;
            target[destination..destination + self.observation_dim]
                .copy_from_slice(&self.ring[source..source + self.observation_dim]);
        }
    }

    fn current_windows(&self) -> (Vec<f32>, Vec<bool>) {
        let mut windows = vec![0.0; self.batch * WINDOW * self.observation_dim];
        let valid = self
            .ring_count
            .iter()
            .map(|count| *count as usize == WINDOW)
            .collect();
        for row in 0..self.batch {
            self.materialize_ring(
                row,
                &mut windows[row * WINDOW * self.observation_dim
                    ..(row + 1) * WINDOW * self.observation_dim],
            );
        }
        (windows, valid)
    }

    fn clear_recent(&mut self, row: usize) {
        self.ring[row * WINDOW * self.observation_dim..(row + 1) * WINDOW * self.observation_dim]
            .fill(0.0);
        self.ring_tick[row * WINDOW..(row + 1) * WINDOW].fill(0);
        self.ring_time[row * WINDOW..(row + 1) * WINDOW].fill(0.0);
        self.ring_count[row] = 0;
        self.ring_cursor[row] = 0;
        self.clear_selection(row);
    }

    fn clear_selection(&mut self, row: usize) {
        self.selected_valid[row] = false;
        self.selected_slot[row] = -1;
        self.selected_window
            [row * WINDOW * self.observation_dim..(row + 1) * WINDOW * self.observation_dim]
            .fill(0.0);
        self.selected_key[row * KEY..(row + 1) * KEY].fill(0.0);
        self.selected_recorded_tick[row] = 0;
        self.selected_recorded_time[row] = 0.0;
        self.selected_generation[row] = 0;
        self.selected_at_tick[row] = 0;
        self.hold_until_tick[row] = 0;
        self.has_choose_tick[row] = false;
        self.last_choose_tick[row] = 0;
    }

    pub(crate) fn push_inner(
        &mut self,
        observations: &[f32],
        ticks: &[u64],
        times: &[f64],
        reset: &[bool],
    ) -> PyResult<(Vec<f32>, Vec<bool>)> {
        if !finite_f32(observations) || !finite_f64(times) {
            return Err(PyValueError::new_err(
                "goal-memory observations and times must be finite",
            ));
        }
        for (row, tick) in ticks.iter().enumerate() {
            if self.has_push[row] && *tick <= self.last_push_tick[row] {
                return Err(PyValueError::new_err(
                    "goal-memory recorded ticks must increase per resident",
                ));
            }
        }
        for row in 0..self.batch {
            if reset[row] {
                self.clear_recent(row);
            }
            let cursor = self.ring_cursor[row] as usize;
            let destination = (row * WINDOW + cursor) * self.observation_dim;
            self.ring[destination..destination + self.observation_dim].copy_from_slice(
                &observations[row * self.observation_dim..(row + 1) * self.observation_dim],
            );
            self.ring_tick[row * WINDOW + cursor] = ticks[row];
            self.ring_time[row * WINDOW + cursor] = times[row];
            self.ring_cursor[row] = ((cursor + 1) % WINDOW) as u8;
            self.ring_count[row] = (self.ring_count[row] + 1).min(WINDOW as u8);
            self.generation[row] = self.generation[row]
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("goal-memory generation exhausted"))?;
            self.has_push[row] = true;
            self.last_push_tick[row] = ticks[row];
        }
        Ok(self.current_windows())
    }

    pub(crate) fn remember_with_changes_inner(
        &mut self,
        encoded_keys: &[f32],
        include: &[bool],
    ) -> PyResult<RememberResult> {
        if !finite_f32(encoded_keys) {
            return Err(PyValueError::new_err("goal-memory keys must be finite"));
        }
        for (row, requested) in include.iter().enumerate() {
            if *requested && self.ring_count[row] as usize != WINDOW {
                return Err(PyValueError::new_err(
                    "cannot remember before four actual observations",
                ));
            }
            if self.generation[row] == self.consumed_generation[row] {
                return Err(PyValueError::new_err(
                    "current goal-memory window was already consumed",
                ));
            }
        }
        let mut slots = vec![-1; self.batch];
        let mut generations = vec![0; self.batch];
        for row in 0..self.batch {
            self.consumed_generation[row] = self.generation[row];
            if !include[row] {
                continue;
            }
            self.seen[row] = self.seen[row]
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("goal-memory reservoir count exhausted"))?;
            let slot = if (self.count[row] as usize) < CAPACITY {
                let slot = self.count[row] as usize;
                self.count[row] += 1;
                Some(slot)
            } else {
                let state = &mut self.rng[row * 4..(row + 1) * 4];
                let candidate = index_below(state, self.seen[row]);
                (candidate < CAPACITY as u64).then_some(candidate as usize)
            };
            let Some(slot) = slot else { continue };
            let window_offset = self.window_offset(row, slot);
            let oldest = self.ring_cursor[row] as usize;
            for frame in 0..WINDOW {
                let source_frame = (oldest + frame) % WINDOW;
                let source = (row * WINDOW + source_frame) * self.observation_dim;
                let destination = window_offset + frame * self.observation_dim;
                self.windows[destination..destination + self.observation_dim]
                    .copy_from_slice(&self.ring[source..source + self.observation_dim]);
            }
            let key_offset = Self::key_offset(row, slot);
            self.keys[key_offset..key_offset + KEY]
                .copy_from_slice(&encoded_keys[row * KEY..(row + 1) * KEY]);
            let latest = (self.ring_cursor[row] as usize + WINDOW - 1) % WINDOW;
            self.recorded_tick[row * CAPACITY + slot] = self.ring_tick[row * WINDOW + latest];
            self.recorded_time[row * CAPACITY + slot] = self.ring_time[row * WINDOW + latest];
            self.slot_generation[row * CAPACITY + slot] = self.generation[row];
            slots[row] = slot as i32;
            generations[row] = self.generation[row];
        }
        Ok(RememberResult { slots, generations })
    }

    fn selection_arrays(&self, ticks: &[u64], changed: Vec<bool>) -> SelectionArrays {
        SelectionArrays {
            slot: self.selected_slot.clone(),
            window: self.selected_window.clone(),
            key: self.selected_key.clone(),
            recorded_tick: self.selected_recorded_tick.clone(),
            recorded_time: self.selected_recorded_time.clone(),
            generation: self.selected_generation.clone(),
            remaining: ticks
                .iter()
                .enumerate()
                .map(|(row, tick)| {
                    if self.selected_valid[row] {
                        self.hold_until_tick[row].saturating_sub(*tick)
                    } else {
                        0
                    }
                })
                .collect(),
            valid: self.selected_valid.clone(),
            changed,
        }
    }

    pub(crate) fn choose_inner(
        &mut self,
        logits: &[f32],
        temperature: f32,
        ticks: &[u64],
    ) -> PyResult<SelectionArrays> {
        if !temperature.is_finite() || !(0.01..=100.0).contains(&temperature) {
            return Err(PyValueError::new_err(
                "goal-memory temperature must be finite in [0.01,100]",
            ));
        }
        if !finite_f32(logits) {
            return Err(PyValueError::new_err("goal-memory logits must be finite"));
        }
        for (row, tick) in ticks.iter().enumerate() {
            if self.has_choose_tick[row] && *tick < self.last_choose_tick[row] {
                return Err(PyValueError::new_err(
                    "goal-memory selection ticks cannot move backward",
                ));
            }
        }
        let mut changed = vec![false; self.batch];
        for row in 0..self.batch {
            if self.has_choose_tick[row] && ticks[row] == self.last_choose_tick[row] {
                continue;
            }
            self.has_choose_tick[row] = true;
            self.last_choose_tick[row] = ticks[row];
            if self.selected_valid[row] && ticks[row] < self.hold_until_tick[row] {
                continue;
            }
            let count = self.count[row] as usize;
            if count == 0 {
                if self.selected_valid[row] {
                    changed[row] = true;
                    self.clear_selection(row);
                }
                continue;
            }
            let values = &logits[row * CAPACITY..row * CAPACITY + count];
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f64 = values
                .iter()
                .map(|value| (((*value - maximum) / temperature) as f64).exp())
                .sum();
            let state = &mut self.rng[row * 4..(row + 1) * 4];
            let threshold = unit_f64(state) * total;
            let mut cumulative = 0.0;
            let mut slot = count - 1;
            for (index, value) in values.iter().enumerate() {
                cumulative += (((*value - maximum) / temperature) as f64).exp();
                if threshold < cumulative {
                    slot = index;
                    break;
                }
            }
            let window_offset = self.window_offset(row, slot);
            self.selected_window
                [row * WINDOW * self.observation_dim..(row + 1) * WINDOW * self.observation_dim]
                .copy_from_slice(
                    &self.windows[window_offset..window_offset + WINDOW * self.observation_dim],
                );
            let key_offset = Self::key_offset(row, slot);
            self.selected_key[row * KEY..(row + 1) * KEY]
                .copy_from_slice(&self.keys[key_offset..key_offset + KEY]);
            self.selected_slot[row] = slot as i32;
            self.selected_recorded_tick[row] = self.recorded_tick[row * CAPACITY + slot];
            self.selected_recorded_time[row] = self.recorded_time[row * CAPACITY + slot];
            self.selected_generation[row] = self.slot_generation[row * CAPACITY + slot];
            self.selected_at_tick[row] = ticks[row];
            self.hold_until_tick[row] = ticks[row]
                .checked_add(HOLD_TICKS)
                .ok_or_else(|| PyValueError::new_err("goal-memory hold tick exhausted"))?;
            self.selected_valid[row] = true;
            changed[row] = true;
        }
        Ok(self.selection_arrays(ticks, changed))
    }
}

pub(crate) struct RememberResult {
    pub(crate) slots: Vec<i32>,
    pub(crate) generations: Vec<u64>,
}

pub(crate) struct SelectionArrays {
    pub(crate) slot: Vec<i32>,
    pub(crate) window: Vec<f32>,
    pub(crate) key: Vec<f32>,
    pub(crate) recorded_tick: Vec<u64>,
    pub(crate) recorded_time: Vec<f64>,
    pub(crate) generation: Vec<u64>,
    pub(crate) remaining: Vec<u64>,
    pub(crate) valid: Vec<bool>,
    pub(crate) changed: Vec<bool>,
}

#[pymethods]
impl AchievedGoalMemoryCohort {
    #[new]
    pub(crate) fn new(batch: usize, observation_dim: usize, seed: u64) -> PyResult<Self> {
        if batch == 0 || batch > 256 || !(1..=8192).contains(&observation_dim) {
            return Err(PyValueError::new_err(
                "goal-memory batch must be 1..256 and observation_dim 1..8192",
            ));
        }
        let reservoir_values = batch
            .checked_mul(CAPACITY)
            .and_then(|value| value.checked_mul(WINDOW))
            .and_then(|value| value.checked_mul(observation_dim))
            .ok_or_else(|| PyValueError::new_err("goal-memory dimensions overflow"))?;
        if reservoir_values > (1_usize << 28) {
            return Err(PyValueError::new_err(
                "goal-memory raw reservoir exceeds the 1 GiB cohort bound",
            ));
        }
        let mut rng = vec![0; batch * 4];
        for row in 0..batch {
            let mut state = seed ^ (row as u64).wrapping_mul(0xd2b7_4407_b1ce_6e93);
            for word in &mut rng[row * 4..(row + 1) * 4] {
                *word = splitmix64(&mut state);
            }
        }
        Ok(Self {
            batch,
            observation_dim,
            ring: vec![0.0; batch * WINDOW * observation_dim],
            ring_tick: vec![0; batch * WINDOW],
            ring_time: vec![0.0; batch * WINDOW],
            ring_count: vec![0; batch],
            ring_cursor: vec![0; batch],
            generation: vec![0; batch],
            consumed_generation: vec![0; batch],
            has_push: vec![false; batch],
            last_push_tick: vec![0; batch],
            windows: vec![0.0; reservoir_values],
            keys: vec![0.0; batch * CAPACITY * KEY],
            recorded_tick: vec![0; batch * CAPACITY],
            recorded_time: vec![0.0; batch * CAPACITY],
            slot_generation: vec![0; batch * CAPACITY],
            count: vec![0; batch],
            seen: vec![0; batch],
            rng,
            selected_valid: vec![false; batch],
            selected_slot: vec![-1; batch],
            selected_window: vec![0.0; batch * WINDOW * observation_dim],
            selected_key: vec![0.0; batch * KEY],
            selected_recorded_tick: vec![0; batch],
            selected_recorded_time: vec![0.0; batch],
            selected_generation: vec![0; batch],
            selected_at_tick: vec![0; batch],
            hold_until_tick: vec![0; batch],
            has_choose_tick: vec![false; batch],
            last_choose_tick: vec![0; batch],
        })
    }

    /// Append one actual observation per resident and expose the owned current window.
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'_, f32>,
        recorded_ticks: PyReadonlyArray1<'_, u64>,
        recorded_times: PyReadonlyArray1<'_, f64>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if observations.shape() != [self.batch, self.observation_dim]
            || recorded_ticks.shape() != [self.batch]
            || recorded_times.shape() != [self.batch]
            || reset.shape() != [self.batch]
        {
            return Err(PyValueError::new_err("goal-memory push shapes differ"));
        }
        let (windows, valid) = self.push_inner(
            observations.as_slice()?,
            recorded_ticks.as_slice()?,
            recorded_times.as_slice()?,
            reset.as_slice()?,
        )?;
        let out = PyDict::new(py);
        out.set_item(
            "windows",
            Array3::from_shape_vec((self.batch, WINDOW, self.observation_dim), windows)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("valid", valid.into_pyarray(py))?;
        Ok(out)
    }

    /// Consume the just-pushed ring generation and optionally retain its encoded key.
    fn remember<'py>(
        &mut self,
        py: Python<'py>,
        encoded_keys: PyReadonlyArray2<'_, f32>,
        include: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if encoded_keys.shape() != [self.batch, KEY] || include.shape() != [self.batch] {
            return Err(PyValueError::new_err("goal-memory remember shapes differ"));
        }
        let result =
            self.remember_with_changes_inner(encoded_keys.as_slice()?, include.as_slice()?)?;
        let changed: Vec<bool> = result.slots.iter().map(|slot| *slot >= 0).collect();
        let out = PyDict::new(py);
        out.set_item("slot", result.slots.into_pyarray(py))?;
        out.set_item("generation", result.generations.into_pyarray(py))?;
        out.set_item("changed", changed.into_pyarray(py))?;
        Ok(out)
    }

    pub(crate) fn candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut valid = vec![false; self.batch * CAPACITY];
        for row in 0..self.batch {
            valid[row * CAPACITY..row * CAPACITY + self.count[row] as usize].fill(true);
        }
        let out = PyDict::new(py);
        out.set_item(
            "keys",
            Array3::from_shape_vec((self.batch, CAPACITY, KEY), self.keys.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "valid",
            Array2::from_shape_vec((self.batch, CAPACITY), valid)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "recorded_tick",
            Array2::from_shape_vec((self.batch, CAPACITY), self.recorded_tick.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "recorded_time",
            Array2::from_shape_vec((self.batch, CAPACITY), self.recorded_time.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "generation",
            Array2::from_shape_vec((self.batch, CAPACITY), self.slot_generation.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        Ok(out)
    }

    /// Sample manager logits only when the resident's ten-tick commitment has expired.
    fn choose<'py>(
        &mut self,
        py: Python<'py>,
        logits: PyReadonlyArray2<'_, f32>,
        temperature: f32,
        model_ticks: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if logits.shape() != [self.batch, CAPACITY] || model_ticks.shape() != [self.batch] {
            return Err(PyValueError::new_err("goal-memory choose shapes differ"));
        }
        let arrays = self.choose_inner(logits.as_slice()?, temperature, model_ticks.as_slice()?)?;
        let out = PyDict::new(py);
        out.set_item("slot", arrays.slot.into_pyarray(py))?;
        out.set_item(
            "window",
            Array3::from_shape_vec((self.batch, WINDOW, self.observation_dim), arrays.window)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "key",
            Array2::from_shape_vec((self.batch, KEY), arrays.key)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("recorded_tick", arrays.recorded_tick.into_pyarray(py))?;
        out.set_item("recorded_time", arrays.recorded_time.into_pyarray(py))?;
        out.set_item("generation", arrays.generation.into_pyarray(py))?;
        out.set_item("remaining_ticks", arrays.remaining.into_pyarray(py))?;
        out.set_item("valid", arrays.valid.into_pyarray(py))?;
        out.set_item("changed", arrays.changed.into_pyarray(py))?;
        Ok(out)
    }

    pub(crate) fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item("version", 2)?;
        out.set_item("batch", self.batch)?;
        out.set_item("capacity", CAPACITY)?;
        out.set_item("window", WINDOW)?;
        out.set_item("observation_dim", self.observation_dim)?;
        out.set_item("key_dim", KEY)?;
        out.set_item("hold_ticks", HOLD_TICKS)?;
        macro_rules! put1 {
            ($name:literal, $value:expr) => {
                out.set_item($name, $value.clone().into_pyarray(py))?
            };
        }
        out.set_item(
            "ring",
            Array3::from_shape_vec(
                (self.batch, WINDOW, self.observation_dim),
                self.ring.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "ring_tick",
            Array2::from_shape_vec((self.batch, WINDOW), self.ring_tick.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "ring_time",
            Array2::from_shape_vec((self.batch, WINDOW), self.ring_time.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        put1!("ring_count", self.ring_count);
        put1!("ring_cursor", self.ring_cursor);
        put1!("generation", self.generation);
        put1!("consumed_generation", self.consumed_generation);
        put1!("has_push", self.has_push);
        put1!("last_push_tick", self.last_push_tick);
        out.set_item(
            "windows",
            Array4::from_shape_vec(
                (self.batch, CAPACITY, WINDOW, self.observation_dim),
                self.windows.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "keys",
            Array3::from_shape_vec((self.batch, CAPACITY, KEY), self.keys.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "recorded_tick",
            Array2::from_shape_vec((self.batch, CAPACITY), self.recorded_tick.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "recorded_time",
            Array2::from_shape_vec((self.batch, CAPACITY), self.recorded_time.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "slot_generation",
            Array2::from_shape_vec((self.batch, CAPACITY), self.slot_generation.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        put1!("count", self.count);
        put1!("seen", self.seen);
        out.set_item(
            "rng",
            Array2::from_shape_vec((self.batch, 4), self.rng.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        put1!("selected_valid", self.selected_valid);
        put1!("selected_slot", self.selected_slot);
        out.set_item(
            "selected_window",
            Array3::from_shape_vec(
                (self.batch, WINDOW, self.observation_dim),
                self.selected_window.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "selected_key",
            Array2::from_shape_vec((self.batch, KEY), self.selected_key.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        put1!("selected_recorded_tick", self.selected_recorded_tick);
        put1!("selected_recorded_time", self.selected_recorded_time);
        put1!("selected_generation", self.selected_generation);
        put1!("selected_at_tick", self.selected_at_tick);
        put1!("hold_until_tick", self.hold_until_tick);
        put1!("has_choose_tick", self.has_choose_tick);
        put1!("last_choose_tick", self.last_choose_tick);
        Ok(out)
    }

    pub(crate) fn restore(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        if required(value, "format")?.extract::<String>()? != FORMAT
            || required(value, "version")?.extract::<u8>()? != 2
            || required(value, "batch")?.extract::<usize>()? != self.batch
            || required(value, "capacity")?.extract::<usize>()? != CAPACITY
            || required(value, "window")?.extract::<usize>()? != WINDOW
            || required(value, "observation_dim")?.extract::<usize>()? != self.observation_dim
            || required(value, "key_dim")?.extract::<usize>()? != KEY
            || required(value, "hold_ticks")?.extract::<u64>()? != HOLD_TICKS
        {
            return Err(PyValueError::new_err(
                "goal-memory snapshot identity differs",
            ));
        }
        macro_rules! array {
            ($name:literal, $type:ty, $shape:expr) => {{
                let value: $type = required(value, $name)?.extract()?;
                if value.shape() != $shape {
                    return Err(PyValueError::new_err(concat!(
                        "goal-memory snapshot shape differs: ",
                        $name
                    )));
                }
                value.as_slice()?.to_vec()
            }};
        }
        let ring = array!(
            "ring",
            PyReadonlyArray3<'_, f32>,
            [self.batch, WINDOW, self.observation_dim]
        );
        let ring_tick = array!("ring_tick", PyReadonlyArray2<'_, u64>, [self.batch, WINDOW]);
        let ring_time = array!("ring_time", PyReadonlyArray2<'_, f64>, [self.batch, WINDOW]);
        let ring_count = array!("ring_count", PyReadonlyArray1<'_, u8>, [self.batch]);
        let ring_cursor = array!("ring_cursor", PyReadonlyArray1<'_, u8>, [self.batch]);
        let generation = array!("generation", PyReadonlyArray1<'_, u64>, [self.batch]);
        let consumed_generation = array!(
            "consumed_generation",
            PyReadonlyArray1<'_, u64>,
            [self.batch]
        );
        let has_push = array!("has_push", PyReadonlyArray1<'_, bool>, [self.batch]);
        let last_push_tick = array!("last_push_tick", PyReadonlyArray1<'_, u64>, [self.batch]);
        let windows = array!(
            "windows",
            numpy::PyReadonlyArray4<'_, f32>,
            [self.batch, CAPACITY, WINDOW, self.observation_dim]
        );
        let keys = array!(
            "keys",
            PyReadonlyArray3<'_, f32>,
            [self.batch, CAPACITY, KEY]
        );
        let recorded_tick = array!(
            "recorded_tick",
            PyReadonlyArray2<'_, u64>,
            [self.batch, CAPACITY]
        );
        let recorded_time = array!(
            "recorded_time",
            PyReadonlyArray2<'_, f64>,
            [self.batch, CAPACITY]
        );
        let slot_generation = array!(
            "slot_generation",
            PyReadonlyArray2<'_, u64>,
            [self.batch, CAPACITY]
        );
        let count = array!("count", PyReadonlyArray1<'_, u16>, [self.batch]);
        let seen = array!("seen", PyReadonlyArray1<'_, u64>, [self.batch]);
        let rng = array!("rng", PyReadonlyArray2<'_, u64>, [self.batch, 4]);
        let selected_valid = array!("selected_valid", PyReadonlyArray1<'_, bool>, [self.batch]);
        let selected_slot = array!("selected_slot", PyReadonlyArray1<'_, i32>, [self.batch]);
        let selected_window = array!(
            "selected_window",
            PyReadonlyArray3<'_, f32>,
            [self.batch, WINDOW, self.observation_dim]
        );
        let selected_key = array!("selected_key", PyReadonlyArray2<'_, f32>, [self.batch, KEY]);
        let selected_recorded_tick = array!(
            "selected_recorded_tick",
            PyReadonlyArray1<'_, u64>,
            [self.batch]
        );
        let selected_recorded_time = array!(
            "selected_recorded_time",
            PyReadonlyArray1<'_, f64>,
            [self.batch]
        );
        let selected_generation = array!(
            "selected_generation",
            PyReadonlyArray1<'_, u64>,
            [self.batch]
        );
        let selected_at_tick = array!("selected_at_tick", PyReadonlyArray1<'_, u64>, [self.batch]);
        let hold_until_tick = array!("hold_until_tick", PyReadonlyArray1<'_, u64>, [self.batch]);
        let has_choose_tick = array!("has_choose_tick", PyReadonlyArray1<'_, bool>, [self.batch]);
        let last_choose_tick = array!("last_choose_tick", PyReadonlyArray1<'_, u64>, [self.batch]);
        if !finite_f32(&ring)
            || !finite_f64(&ring_time)
            || !finite_f32(&windows)
            || !finite_f32(&keys)
            || !finite_f64(&recorded_time)
            || !finite_f32(&selected_window)
            || !finite_f32(&selected_key)
            || !finite_f64(&selected_recorded_time)
        {
            return Err(PyValueError::new_err("goal-memory snapshot must be finite"));
        }
        for row in 0..self.batch {
            let occupied = count[row] as usize;
            let generations = &slot_generation[row * CAPACITY..(row + 1) * CAPACITY];
            let occupied_generations_valid = generations[..occupied]
                .iter()
                .all(|slot_generation| *slot_generation > 0 && *slot_generation <= generation[row]);
            let empty_generations_zero = generations[occupied..].iter().all(|value| *value == 0);
            let mut occupied_generations = generations[..occupied].to_vec();
            occupied_generations.sort_unstable();
            let occupied_generations_unique = occupied_generations
                .windows(2)
                .all(|pair| pair[0] != pair[1]);
            if ring_count[row] as usize > WINDOW
                || ring_cursor[row] as usize >= WINDOW
                || consumed_generation[row] > generation[row]
                || count[row] as usize > CAPACITY
                || seen[row] < count[row] as u64
                || rng[row * 4..(row + 1) * 4].iter().all(|word| *word == 0)
                || (has_push[row] && generation[row] == 0)
                || !occupied_generations_valid
                || !occupied_generations_unique
                || !empty_generations_zero
                || (selected_valid[row]
                    && (selected_slot[row] < 0
                        || selected_slot[row] as usize >= count[row] as usize
                        || selected_generation[row] == 0
                        || selected_generation[row] > generation[row]
                        || hold_until_tick[row]
                            != selected_at_tick[row].checked_add(HOLD_TICKS).unwrap_or(0)))
                || (!selected_valid[row]
                    && (selected_slot[row] != -1 || selected_generation[row] != 0))
                || (has_choose_tick[row]
                    && selected_valid[row]
                    && last_choose_tick[row] < selected_at_tick[row])
            {
                return Err(PyValueError::new_err(
                    "goal-memory snapshot invariants differ",
                ));
            }
        }
        self.ring = ring;
        self.ring_tick = ring_tick;
        self.ring_time = ring_time;
        self.ring_count = ring_count;
        self.ring_cursor = ring_cursor;
        self.generation = generation;
        self.consumed_generation = consumed_generation;
        self.has_push = has_push;
        self.last_push_tick = last_push_tick;
        self.windows = windows;
        self.keys = keys;
        self.recorded_tick = recorded_tick;
        self.recorded_time = recorded_time;
        self.slot_generation = slot_generation;
        self.count = count;
        self.seen = seen;
        self.rng = rng;
        self.selected_valid = selected_valid;
        self.selected_slot = selected_slot;
        self.selected_window = selected_window;
        self.selected_key = selected_key;
        self.selected_recorded_tick = selected_recorded_tick;
        self.selected_recorded_time = selected_recorded_time;
        self.selected_generation = selected_generation;
        self.selected_at_tick = selected_at_tick;
        self.hold_until_tick = hold_until_tick;
        self.has_choose_tick = has_choose_tick;
        self.last_choose_tick = last_choose_tick;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::personal_goals::{
        GoalSlotIdentity, GoalSlotReplacement, GoalStart, GoalTransition, PersonalGoalAssociations,
        PersonalGoalConfig,
    };

    #[test]
    fn copied_identity_survives_actual_replacement_and_blocks_credit() {
        let mut memory = AchievedGoalMemoryCohort::new(1, 1, 0x51a7).unwrap();
        let key = vec![0.0_f32; KEY];
        for tick in 1_u64..=CAPACITY as u64 + WINDOW as u64 - 1 {
            memory
                .push_inner(&[tick as f32], &[tick], &[tick as f64 * 0.05], &[false])
                .unwrap();
            let include = tick >= WINDOW as u64;
            memory
                .remember_with_changes_inner(&key, &[include])
                .unwrap();
        }
        assert_eq!(memory.counts(), &[CAPACITY as u16]);

        // Predict the next actual reservoir target without advancing the owned
        // RNG. Selection consumes one draw before remember performs its draw.
        let mut preview_rng = memory.rng[0..4].to_vec();
        let _selection_draw = random_u64(&mut preview_rng);
        let replacement_slot = index_below(&mut preview_rng, memory.seen[0] + 1) as usize;
        assert!(replacement_slot < CAPACITY);
        let identity = GoalSlotIdentity {
            recorded_tick: memory.recorded_tick[replacement_slot],
            generation: memory.slot_generation[replacement_slot],
        };
        let mut logits = vec![-100.0_f32; CAPACITY];
        logits[replacement_slot] = 100.0;
        let selected = memory.choose_inner(&logits, 0.01, &[132]).unwrap();
        assert_eq!(selected.slot, vec![replacement_slot as i32]);
        assert_eq!(selected.generation, vec![identity.generation]);

        let mut associations =
            PersonalGoalAssociations::new(1, PersonalGoalConfig::current(true)).unwrap();
        associations
            .replace_slots(&[GoalSlotReplacement {
                resident: 0,
                slot: replacement_slot,
                identity,
            }])
            .unwrap();
        associations
            .begin_goals(&[GoalStart {
                resident: 0,
                slot: replacement_slot,
                identity,
                selected_at_tick: 132,
                physiology: [0.35, 0.10, 0.20],
            }])
            .unwrap();

        memory
            .push_inner(&[132.0], &[132], &[6.6], &[false])
            .unwrap();
        let replacement = memory.remember_with_changes_inner(&key, &[true]).unwrap();
        assert_eq!(replacement.slots, vec![replacement_slot as i32]);
        let replacement_identity = GoalSlotIdentity {
            recorded_tick: memory.recorded_tick[replacement_slot],
            generation: replacement.generations[0],
        };
        assert!(replacement_identity.generation > identity.generation);
        associations
            .replace_slots(&[GoalSlotReplacement {
                resident: 0,
                slot: replacement_slot,
                identity: replacement_identity,
            }])
            .unwrap();

        let still_selected = memory.current_selection(&[132]);
        assert_eq!(still_selected.recorded_tick, vec![identity.recorded_tick]);
        assert_eq!(still_selected.generation, vec![identity.generation]);
        assert_ne!(
            memory.slot_generation[replacement_slot],
            still_selected.generation[0]
        );

        let mut physiology = [0.35, 0.10, 0.20];
        let mut final_receipt = None;
        let mut summed_reward = 0.0_f64;
        for offset in 0..10_u64 {
            let after = [
                physiology[0] + 0.002,
                physiology[1] + 0.001,
                physiology[2] - 0.001,
            ];
            let outcome = associations
                .observe_transitions(&[GoalTransition {
                    resident: 0,
                    transition_tick: 132 + offset,
                    before: physiology,
                    after,
                    effort: 0.20,
                    dt: 0.05,
                }])
                .unwrap()
                .pop()
                .unwrap();
            summed_reward += outcome.reward as f64;
            final_receipt = outcome.receipt;
            physiology = after;
        }
        let receipt = final_receipt.unwrap();
        assert!((summed_reward - 0.34104198589921).abs() < 1e-15);
        assert_eq!(receipt.summed_objective_return, summed_reward);
        assert!(receipt.slot_was_replaced);
        assert!(!receipt.attributed);
        assert!(!receipt.learned);
    }
}
