// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin NumPy/PyO3 host for the shared resident engine.
use super::*;
use numpy::{
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

impl DevelopmentalResidentCohort {
    fn output<'py>(&self, py: Python<'py>, proposed: Vec<f32>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item(
            "proposed_command",
            Array2::from_shape_vec((self.batch, ACTIONS), proposed)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "cns_recurrent_state",
            Array2::from_shape_vec((self.batch, HIDDEN), self.state.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "latent_goal",
            Array2::from_shape_vec((self.batch, GOAL), self.goal.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "current_cns_key",
            Array2::from_shape_vec((self.batch, GOAL), self.current_key.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "goal_origin_slot",
            Array1::from_vec(self.goal_origin_slot.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_origin_generation",
            Array1::from_vec(self.goal_origin_generation.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_origin_tick",
            Array1::from_vec(self.goal_origin_tick.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "goal_selected_tick",
            Array1::from_vec(self.goal_selected_tick.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "memory_inserted_slot",
            Array1::from_vec(self.memory_inserted_slot.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "memory_count",
            Array1::from_vec(
                self.goal_memory
                    .count
                    .iter()
                    .map(|count| *count as u16)
                    .collect::<Vec<_>>(),
            )
            .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_policy_version",
            self.sequence_control.policy_version,
        )?;
        out.set_item(
            "sequence_control_policy_sha256",
            self.sequence_control.policy_sha256.clone(),
        )?;
        out.set_item(
            "sequence_control_state",
            Array2::from_shape_vec((self.batch, CONTROL_STATE), self.control_state.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_proposal",
            Array3::from_shape_vec(
                (self.batch, CANDIDATES, CHOICE),
                self.control_proposal.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_active",
            Array2::from_shape_vec((self.batch, CHOICE), self.control_active.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_proposal_mask",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_mask.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_active_mask",
            Array1::from_vec(self.control_active_mask.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_logit",
            Array1::from_vec(self.decision.hazard_logits.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_logits",
            Array2::from_shape_vec(
                (self.batch, CANDIDATES),
                self.decision.selector_logits.clone(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_value",
            Array1::from_vec(self.decision.values.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_decision",
            Array1::from_vec(self.decision.terminate.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "selected_candidate",
            Array1::from_vec(self.decision.selected.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_mask",
            Array1::from_vec(self.decision.hazard_mask.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_mask",
            Array1::from_vec(self.decision.selector_mask.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_hazard_logp",
            Array1::from_vec(self.decision.hazard_logp.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_selector_logp",
            Array1::from_vec(self.decision.selector_logp.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_logp",
            Array1::from_vec(self.decision.logp.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "candidate_is_recalled_suffix",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_recalled.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_slot",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_slot.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_generation",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_generation.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "candidate_suffix_length",
            Array2::from_shape_vec((self.batch, CANDIDATES), self.candidate_length.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "active_source_slot",
            Array1::from_vec(self.active_source_slot.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "active_source_generation",
            Array1::from_vec(self.active_source_generation.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "active_phase",
            Array1::from_vec(self.active_phase.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "active_remaining",
            Array1::from_vec(self.active_remaining.clone()).into_pyarray(py),
        )?;
        let cancellation: Vec<_> = (0..self.batch)
            .flat_map(|row| self.suffixes.cancellation_counts(row))
            .collect();
        out.set_item(
            "motor_suffix_cancellation_totals",
            Array2::from_shape_vec((self.batch, 5), cancellation)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "motor_suffix_cancellation_reason",
            (0..self.batch)
                .map(|row| self.suffixes.cancellation_reason(row))
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "command_pending",
            Array1::from_vec(self.command_pending.clone()).into_pyarray(py),
        )?;
        Ok(out)
    }
}

#[pymethods]
impl DevelopmentalResidentCohort {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        batch: usize,
        action_mode: &str,
        action_seed: u64,
        suffix_seed: u64,
        core_packed: PyReadonlyArray1<'_, f32>,
        core_sha256: String,
        predictor_packed: PyReadonlyArray1<'_, f32>,
        predictor_sha256: String,
        sequence_control_packed: PyReadonlyArray1<'_, f32>,
        sequence_control_version: u64,
        sequence_control_sha256: String,
        research_training: bool,
    ) -> PyResult<Self> {
        Self::from_packed(
            batch,
            action_mode,
            action_seed,
            suffix_seed,
            core_packed.as_slice()?,
            core_sha256,
            predictor_packed.as_slice()?,
            predictor_sha256,
            sequence_control_packed.as_slice()?,
            sequence_control_version,
            sequence_control_sha256,
            research_training,
        )
        .map_err(PyValueError::new_err)
    }

    fn expanded(&self, additions: usize, action_seed: u64, suffix_seed: u64) -> PyResult<Self> {
        self.expanded_inner(additions, action_seed, suffix_seed)
            .map_err(PyValueError::new_err)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        cns_latent: PyReadonlyArray2<'_, f32>,
        previous_command: PyReadonlyArray2<'_, f32>,
        ticks: PyReadonlyArray1<'_, u64>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if cns_latent.shape() != [self.batch, Z]
            || previous_command.shape() != [self.batch, ACTIONS]
            || ticks.shape() != [self.batch]
            || reset.shape() != [self.batch]
        {
            return Err(PyValueError::new_err("CNS resident step shapes differ"));
        }
        let proposed = self
            .step_flat(
                cns_latent.as_slice()?,
                previous_command.as_slice()?,
                ticks.as_slice()?,
                reset.as_slice()?,
            )
            .map_err(PyValueError::new_err)?;
        self.output(py, proposed)
    }

    fn preview_sequence_control<'py>(
        &self,
        py: Python<'py>,
        cns_latent: PyReadonlyArray2<'_, f32>,
        previous_command: PyReadonlyArray2<'_, f32>,
        ticks: PyReadonlyArray1<'_, u64>,
        reset: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut fork = self.clone();
        fork.step(py, cns_latent, previous_command, ticks, reset)
    }

    fn acknowledge<'py>(
        &mut self,
        py: Python<'py>,
        ticks: PyReadonlyArray1<'_, u64>,
        delivered_command: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if ticks.shape() != [self.batch] || delivered_command.shape() != [self.batch, ACTIONS] {
            return Err(PyValueError::new_err("command receipt shapes differ"));
        }
        let exact = self
            .acknowledge_flat(ticks.as_slice()?, delivered_command.as_slice()?)
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item(
            "sequence_control_policy_version",
            self.sequence_control.policy_version,
        )?;
        out.set_item(
            "sequence_control_policy_sha256",
            self.sequence_control.policy_sha256.clone(),
        )?;
        out.set_item(
            "acknowledged",
            Array1::from_vec(vec![true; self.batch]).into_pyarray(py),
        )?;
        out.set_item("command_exact", Array1::from_vec(exact).into_pyarray(py))?;
        out.set_item(
            "command_pending",
            Array1::from_vec(self.command_pending.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "cns_outcome_pending",
            Array1::from_vec(self.acknowledged_valid.clone()).into_pyarray(py),
        )?;
        let cancellation: Vec<_> = (0..self.batch)
            .flat_map(|row| self.suffixes.cancellation_counts(row))
            .collect();
        out.set_item(
            "motor_suffix_cancellation_totals",
            Array2::from_shape_vec((self.batch, 5), cancellation)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "motor_suffix_cancellation_reason",
            (0..self.batch)
                .map(|row| self.suffixes.cancellation_reason(row))
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn sequence_control_likelihood<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray2<'_, f32>,
        proposal: PyReadonlyArray3<'_, f32>,
        active: PyReadonlyArray2<'_, f32>,
        proposal_mask: PyReadonlyArray2<'_, bool>,
        active_mask: PyReadonlyArray1<'_, bool>,
        hazard_decision: PyReadonlyArray1<'_, bool>,
        selected_candidate: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if state.shape() != [self.batch, CONTROL_STATE]
            || proposal.shape() != [self.batch, CANDIDATES, CHOICE]
            || active.shape() != [self.batch, CHOICE]
            || proposal_mask.shape() != [self.batch, CANDIDATES]
            || active_mask.shape() != [self.batch]
            || hazard_decision.shape() != [self.batch]
            || selected_candidate.shape() != [self.batch]
        {
            return Err(PyValueError::new_err(
                "sequence-control likelihood shapes differ",
            ));
        }
        let decision = self
            .sequence_control
            .likelihood(
                state.as_slice()?,
                proposal.as_slice()?,
                active.as_slice()?,
                proposal_mask.as_slice()?,
                active_mask.as_slice()?,
                hazard_decision.as_slice()?,
                selected_candidate.as_slice()?,
            )
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item(
            "sequence_control_policy_version",
            self.sequence_control.policy_version,
        )?;
        out.set_item(
            "sequence_control_policy_sha256",
            self.sequence_control.policy_sha256.clone(),
        )?;
        out.set_item(
            "sequence_control_hazard_logit",
            Array1::from_vec(decision.hazard_logits).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_logits",
            Array2::from_shape_vec((self.batch, CANDIDATES), decision.selector_logits)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_value",
            Array1::from_vec(decision.values).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_decision",
            Array1::from_vec(decision.terminate).into_pyarray(py),
        )?;
        out.set_item(
            "selected_candidate",
            Array1::from_vec(decision.selected).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_hazard_mask",
            Array1::from_vec(decision.hazard_mask).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_selector_mask",
            Array1::from_vec(decision.selector_mask).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_hazard_logp",
            Array1::from_vec(decision.hazard_logp).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_selector_logp",
            Array1::from_vec(decision.selector_logp).into_pyarray(py),
        )?;
        out.set_item(
            "sequence_control_behavior_logp",
            Array1::from_vec(decision.logp).into_pyarray(py),
        )?;
        Ok(out)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let value = self.snapshot_data().map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("format", value.format)?;
        out.set_item("version", value.version)?;
        out.set_item("private", value.private)?;
        out.set_item("motor_suffix_memory", value.motor_suffix_memory)?;
        out.set_item("goal_memory", value.goal_memory)?;
        out.set_item("sequence_control", value.sequence_control)?;
        Ok(out)
    }

    fn restore(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        if value.len() != 6 {
            return Err(PyValueError::new_err("CNS snapshot fields differ"));
        }
        let get = |name: &str| {
            value
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("CNS snapshot lacks {name}")))
        };
        if get("format")?.extract::<String>()? != FORMAT || get("version")?.extract::<u8>()? != 10 {
            return Err(PyValueError::new_err("CNS snapshot identity differs"));
        }
        self.restore_snapshot(&ResidentSnapshot {
            format: get("format")?.extract()?,
            version: get("version")?.extract()?,
            private: get("private")?.extract()?,
            motor_suffix_memory: get("motor_suffix_memory")?.extract()?,
            goal_memory: get("goal_memory")?.extract()?,
            sequence_control: get("sequence_control")?.extract()?,
        })
        .map_err(PyValueError::new_err)
    }

    fn replace_sequence_control<'py>(
        &mut self,
        py: Python<'py>,
        packed: PyReadonlyArray1<'_, f32>,
        new_version: u64,
        new_sha256: String,
        expected_version: u64,
        expected_sha256: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !self.research_training {
            return Err(PyValueError::new_err(
                "sequence-control replacement is restricted to research cohorts",
            ));
        }
        if self.command_pending.iter().any(|x| *x) || self.acknowledged_valid.iter().any(|x| *x) {
            return Err(PyValueError::new_err(
                "sequence-control replacement requires a fully coherent boundary",
            ));
        }
        let old_version = self.sequence_control.policy_version;
        let old_sha256 = self.sequence_control.policy_sha256.clone();
        self.sequence_control
            .replace(
                packed.as_slice()?,
                new_version,
                new_sha256,
                expected_version,
                &expected_sha256,
            )
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("old_version", old_version)?;
        out.set_item("old_sha256", old_sha256)?;
        out.set_item("new_version", self.sequence_control.policy_version)?;
        out.set_item("new_sha256", self.sequence_control.policy_sha256.clone())?;
        Ok(out)
    }
}
