"""Private inherited motor organ joined to experienced relational refinement.

This wrapper is a strict five-physics-tick state machine.  Its relational key
is the inherited artifact's frozen 64-dimensional projection of neural input;
it has no simulator geometry, identity, named-object, or goal interface.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .context_memory import ContextMemoryConfig, RelationalContextMemory
from .contextual_motor import (
    FINITE_ENERGY_UTILITY_PROFILE,
    MEMORY_OUTCOMES,
    MOTOR_ACTIONS,
    ContextualMotorConfig,
    ContextualMotorRefiner,
)
from .motor_inheritance import MotorArtifact, MotorOrgan
from .personal_plasticity import (
    FIXED_INHERITED_VARIANCE,
    STATE_LOG_STD_VARIANCE,
    FORMAT as PLASTICITY_FORMAT,
    PersonalMotorPlasticity,
    PersonalPlasticityConfig,
)


FORMAT = "chreatures-living-motor-organ-v1"
PHYSICS_DT = 0.05
MACRO_STEPS = 5
LEGACY_PROFILE = "projection-v1"
RESEARCH_PROFILE = "projection-v2"
CUSTOM_PROFILE = "custom"
RELATIONAL_PROFILES = frozenset((LEGACY_PROFILE, RESEARCH_PROFILE, CUSTOM_PROFILE))
VARIANCE_MATURATION_FORMAT = "chreatures-living-motor-variance-maturation-v1"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _config_sha256(config: PersonalPlasticityConfig) -> str:
    encoded = json.dumps(
        config.to_value(), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_memory_config(profile: str) -> ContextMemoryConfig:
    common = {"feature_dim": 64, "action_dim": len(MOTOR_ACTIONS),
              "outcome_dim": len(MEMORY_OUTCOMES)}
    if profile == LEGACY_PROFILE:
        return ContextMemoryConfig(**common)
    if profile == RESEARCH_PROFILE:
        # Live orchard measurements found useful projected trajectory changes
        # around 0.32--0.55 RMS, while the v1 allocation threshold was 0.95.
        # V2 retains the frozen projection and resolves those states instead of
        # amplifying or replacing the inherited representation.
        return ContextMemoryConfig(
            **common,
            observation_bandwidth=0.28,
            action_bandwidth=0.20,
            new_context_distance=0.34,
        )
    raise ValueError(f"unsupported built-in relational profile: {profile!r}")


class LivingMotorOrgan:
    """An inherited motor policy with private action-conditioned experience."""

    def __init__(
        self,
        artifact: MotorArtifact | str | Path,
        *,
        seed: int | None = None,
        deterministic: bool = False,
        contextual_enabled: bool = True,
        frozen: bool = False,
        plasticity: bool = False,
        relational_profile: str | None = None,
        memory_config: ContextMemoryConfig | None = None,
        refiner_config: ContextualMotorConfig | None = None,
        plasticity_config: PersonalPlasticityConfig | None = None,
    ) -> None:
        shared = artifact if isinstance(artifact, MotorArtifact) else MotorArtifact.load(artifact)
        projection_dim = int(shared.config["projection_dim"])
        if projection_dim != 64:
            raise ValueError("living motor requires the inherited 64-dimensional frozen projection")
        if int(shared.config["macro_steps"]) != MACRO_STEPS:
            raise ValueError("living motor requires a five-tick inherited motor macro")
        private_seed = int(shared.config["seed"]) + 311 if seed is None else int(seed)
        self.motor = MotorOrgan(shared, seed=private_seed, deterministic=deterministic)
        if relational_profile is not None and relational_profile not in RELATIONAL_PROFILES:
            raise ValueError(f"unsupported relational profile: {relational_profile!r}")
        if memory_config is None:
            # Projection-v2 remains an explicit research setting until its
            # increased state resolution has longitudinal evidence.
            self.relational_profile = relational_profile or LEGACY_PROFILE
            if self.relational_profile == CUSTOM_PROFILE:
                raise ValueError("custom relational profile requires memory_config")
            memory_config = _profile_memory_config(self.relational_profile)
        else:
            self.relational_profile = relational_profile or CUSTOM_PROFILE
            if self.relational_profile != CUSTOM_PROFILE:
                expected_profile = _profile_memory_config(self.relational_profile)
                if memory_config != expected_profile:
                    raise ValueError("memory configuration differs from its relational profile")
        if (
            memory_config.feature_dim != projection_dim
            or memory_config.action_dim != len(MOTOR_ACTIONS)
            or memory_config.outcome_dim != len(MEMORY_OUTCOMES)
        ):
            raise ValueError("memory configuration differs from living motor dimensions")
        if not plasticity and plasticity_config is not None:
            raise ValueError("plasticity_config requires plasticity=True")
        if plasticity:
            if refiner_config is None:
                refiner_config = ContextualMotorConfig(
                    utility_profile=FINITE_ENERGY_UTILITY_PROFILE,
                )
            elif refiner_config.utility_profile != FINITE_ENERGY_UTILITY_PROFILE:
                raise ValueError(
                    "personal plasticity requires contextual utility_profile='finite-energy-v1'"
                )
        self.memory = RelationalContextMemory(memory_config)
        self.refiner = ContextualMotorRefiner(
            projection_dim,
            seed=private_seed + 1709,
            config=refiner_config,
            enabled=contextual_enabled,
        )
        self.refiner.freeze(frozen)
        self.personal_plasticity = (
            PersonalMotorPlasticity(
                (
                    PersonalPlasticityConfig(seed=private_seed + 3301)
                    if plasticity_config is None else plasticity_config
                ),
                objective_config=refiner_config.finite_energy_config,
                learning=not frozen,
            )
            if plasticity else None
        )
        self.pending: dict[str, Any] | None = None
        self.last_record: dict[str, Any] | None = None
        self.last_plasticity_update: dict[str, Any] | None = None
        self.last_actual_correction: float | None = None
        self.variance_maturation: dict[str, Any] | None = None
        self.metrics: dict[str, int | float] = {
            "ticks": 0,
            "macro_decisions": 0,
            "recorded_transitions": 0,
            "context_changed_choices": 0,
            "experienced_nutrition": 0.0,
            "experienced_effort": 0.0,
            "contacted_transitions": 0,
            "episodes": 0,
        }

    @property
    def artifact(self) -> MotorArtifact:
        return self.motor.artifact

    @property
    def last_prediction_error(self) -> float | None:
        return self.motor.last_prediction_error

    @staticmethod
    def _local_physiology(value: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("local physiology must be a mapping")
        # MotorOrgan performs the strict no-extra-fields check and validates all
        # six local channels. The relational consequence model retains only
        # homeostatic state, not kinematic or environment fields.
        MotorOrgan.physiology_vector(value)
        return ContextualMotorRefiner.physiology(value)

    @staticmethod
    def _physics_outcome(value: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("previous physics outcome must be a mapping")
        nutrition = _finite(value.get("nutrition", 0.0), "nutrition")
        effort = _finite(value.get("effort", 0.0), "effort")
        contact = _finite(value.get("contact", 0.0), "contact")
        if nutrition < 0 or effort < 0 or not 0 <= contact <= 1:
            raise ValueError("nutrition/effort must be nonnegative and contact must be in [0, 1]")
        return {"nutrition": nutrition, "effort": effort, "contact": contact}

    @staticmethod
    def _validate_dt(dt: Any) -> float:
        step = _finite(dt, "dt")
        if not math.isclose(step, PHYSICS_DT, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"living motor requires fixed dt={PHYSICS_DT}")
        return step

    def _feature(self, normalized: np.ndarray) -> np.ndarray:
        projected = self.motor.projected(normalized)
        return _vector(projected, 64, "projected relational feature")

    def _new_pending(
        self,
        feature: np.ndarray,
        physiology: Mapping[str, float],
        decision: Mapping[str, Any],
    ) -> None:
        action = _vector(decision["action_vector"], len(MOTOR_ACTIONS), "committed action")
        selected = int(decision["selected_index"])
        candidate = decision["candidates"][selected]
        self.pending = {
            "feature": feature.copy(),
            "before_physiology": dict(physiology),
            "action": action.copy(),
            "outcome": {"nutrition": 0.0, "effort": 0.0, "contact": 0.0},
            "outcomes_seen": 0,
        }
        if self.personal_plasticity is not None:
            self.pending.update({
                "tick_before_physiology": dict(physiology),
                "plasticity_transitions": [],
            })
        self.metrics["macro_decisions"] += 1
        self.metrics["context_changed_choices"] += int(decision["context_changed_choice"])
        self.last_actual_correction = float(candidate["contextual_correction"])

    def _accumulate(
        self, value: Mapping[str, Any], dt: float,
        after_physiology: Mapping[str, float],
    ) -> None:
        if self.pending is None:
            raise RuntimeError("physics outcome arrived without a pending motor transition")
        outcome = self._physics_outcome(value)
        total = self.pending["outcome"]
        total["nutrition"] += outcome["nutrition"]
        total["effort"] += outcome["effort"] * dt
        total["contact"] = max(total["contact"], outcome["contact"])
        if self.personal_plasticity is not None:
            if outcome["effort"] > 1.0:
                raise ValueError("personal plasticity requires physical effort in [0, 1]")
            tick_before = dict(self.pending["tick_before_physiology"])
            tick_after = dict(after_physiology)
            self.pending["plasticity_transitions"].append({
                "before": tick_before,
                "after": tick_after,
                "nutrition": outcome["nutrition"],
                "effort": outcome["effort"],
                "dt": dt,
            })
            self.pending["tick_before_physiology"] = tick_after
        self.pending["outcomes_seen"] += 1

    def _refine_and_commit(
        self,
        projected: np.ndarray,
        normalized: np.ndarray,
        raw_neural_features: Any,
        local_physiology: Mapping[str, Any],
        step: float,
        candidate_evidence: Callable[
            [tuple[tuple[float, ...], ...]], Mapping[str, Any]
        ] | None,
    ) -> dict[str, Any]:
        """Select one macro action, optionally adapting its ancestral mean."""
        if self.personal_plasticity is None:
            # This is the original path, including the exact MotorOrgan RNG
            # draw performed by ContextualMotorRefiner.
            return self.refiner.refine_and_commit(
                self.motor, self.memory, projected, raw_neural_features,
                local_physiology, step, candidate_evidence=candidate_evidence,
            )

        motor_physiology = self.motor.physiology_vector(local_physiology)
        inherited_mean, _inherited_value, hidden = self.motor.forward(
            normalized, motor_physiology
        )
        log_std = self.motor.distribution_log_std(hidden)
        inherited_noise = None if self.motor.deterministic else self.motor.rng.standard_normal(
            len(MOTOR_ACTIONS), dtype=np.float32
        )
        proposal = self.personal_plasticity.propose(
            projected,
            inherited_mean,
            log_std,
            inherited_noise=inherited_noise,
            local_physiology=motor_physiology,
            deterministic=self.motor.deterministic,
        )
        decision = self.refiner.refine_and_commit(
            self.motor, self.memory, projected, raw_neural_features,
            local_physiology, step,
            policy_mean_override=proposal.adapted_mean,
            baseline_latent=proposal.latent_action,
            # Alternative perturbations remain tied to the immutable inherited
            # scale. Private variance changes only baseline proposal z0.
            alternative_log_std=(
                log_std
                if self.personal_plasticity.config.variance_adaptation
                == STATE_LOG_STD_VARIANCE
                else None
            ),
            proposal_credit_contract="latent-proposal-downstream-selector-v2",
            candidate_evidence=candidate_evidence,
        )
        selected = int(decision["selected_index"])
        selected_action = _vector(
            decision["action_vector"], len(MOTOR_ACTIONS), "selected action"
        )
        reranking_active = any(
            abs(float(candidate["contextual_correction"])) > 0.0
            or abs(float(candidate.get("external_correction", 0.0))) > 0.0
            for candidate in decision["candidates"]
        )
        generated_pipeline = bool(
            decision["provenance"].get("candidate_pipeline")
            == "generated-around-baseline-v1"
            and decision["provenance"].get("proposal_credit_contract")
            == "latent-proposal-downstream-selector-v2"
        )
        external = decision["provenance"].get("external_evidence")
        proposal_independent_selector = bool(
            generated_pipeline
            and (
                external is None
                or bool(external.get("proposal_independent", False))
            )
        )
        provenance = (
            "latent_proposal_composite_selector"
            if proposal_independent_selector else "unknown_selection_actor_skipped"
        )
        self.personal_plasticity.commit(
            proposal, executed_action=selected_action, provenance=provenance,
            selection_pipeline=(
                "generated-candidate-selector-v1"
                if proposal_independent_selector else "unknown-selection-v1"
            ),
            proposal_independent=proposal_independent_selector,
        )
        personal = self.personal_plasticity.pending
        if personal is None:
            raise RuntimeError("personal plasticity did not retain its committed proposal")
        decision["provenance"]["personal_plasticity"] = {
            "format": PLASTICITY_FORMAT,
            "decision_sequence": personal.sequence,
            "selected_candidate": selected,
            "reranking_active": reranking_active,
            "external_evidence_declared_independent": (
                None if external is None else bool(external.get("proposal_independent", False))
            ),
            "actor_eligible": personal.actor_eligible,
            "selection": personal.provenance,
            "credit_assignment": personal.credit_assignment,
            "selection_pipeline": personal.selection_pipeline,
            "proposal_independent_selection": personal.proposal_independent_selection,
            "proposal_log_probability": personal.proposal_log_probability,
            "raw_standard_noise": personal.standard_noise.astype(float).tolist(),
            "raw_baseline_latent": personal.latent_action.astype(float).tolist(),
        }
        if personal.variance_feature_profile is not None:
            decision["provenance"]["personal_plasticity"].update({
                "variance_adaptation": self.personal_plasticity.config.variance_adaptation,
                "variance_feature_profile": personal.variance_feature_profile,
                "inherited_log_std": personal.inherited_log_std.astype(float).tolist(),
                "effective_log_std": personal.log_std.astype(float).tolist(),
                "log_std_offset": personal.log_std_offset.astype(float).tolist(),
                "physiology_vector": personal.local_physiology.astype(float).tolist(),
                "alternative_noise_scale": "immutable-inherited-log-std",
            })
        self.refiner.last_decision = copy.deepcopy(decision)
        return decision

    def tick(
        self,
        raw_neural_features: Any,
        local_physiology: Mapping[str, Any],
        previous_physics_outcome: Mapping[str, Any] | None,
        dt: float = PHYSICS_DT,
        *,
        candidate_evidence: Callable[
            [tuple[tuple[float, ...], ...]], Mapping[str, Any]
        ] | None = None,
    ) -> dict[str, float]:
        """Consume the preceding outcome and return this tick's physical action.

        The first call of an episode requires ``previous_physics_outcome=None``.
        Every later call consumes exactly one body's outcome from the physics
        step driven by the action returned from the preceding call.
        """
        step = self._validate_dt(dt)
        normalized = self.motor.normalize(raw_neural_features)
        projected = self._feature(normalized)
        physiology = self._local_physiology(local_physiology)

        if self.motor.held_ticks == 0:
            if self.pending is not None:
                raise RuntimeError("pending transition exists at an open motor boundary")
            if previous_physics_outcome is not None:
                raise ValueError("initial episode tick cannot consume a prior physics outcome")
            self._apply_queued_variance_maturation(completed_old_pending=False)
            decision = self._refine_and_commit(
                projected, normalized, raw_neural_features,
                local_physiology, step, candidate_evidence,
            )
            self._new_pending(projected, physiology, decision)
            self.metrics["ticks"] += 1
            return dict(decision["action"])

        if self.pending is None:
            raise RuntimeError("held motor action has no pending transition")
        if previous_physics_outcome is None:
            raise ValueError("each held tick requires the preceding physics outcome")
        expected_seen = self.motor.held_ticks - 1
        if int(self.pending["outcomes_seen"]) != expected_seen:
            raise RuntimeError("pending outcome count differs from inherited hold state")
        self._accumulate(previous_physics_outcome, step, physiology)

        if self.motor.held_ticks < MACRO_STEPS:
            action = self.motor.continue_macro_action(step)
            self.metrics["ticks"] += 1
            return action

        if int(self.pending["outcomes_seen"]) != MACRO_STEPS:
            raise RuntimeError("completed macro does not contain five physics outcomes")
        completed = self.pending
        record = self.refiner.record(
            self.memory,
            completed["feature"],
            completed["action"],
            projected,
            completed["before_physiology"],
            physiology,
            completed["outcome"],
        )
        self.last_record = copy.deepcopy(record)
        if self.personal_plasticity is not None:
            transitions = completed.get("plasticity_transitions")
            if not isinstance(transitions, list) or len(transitions) != MACRO_STEPS:
                raise RuntimeError("personal plasticity requires five physical tick records")
            self.last_plasticity_update = self.personal_plasticity.observe(
                projected, transitions=transitions
            )
            # A queued anatomy capability begins only after the old proposal's
            # complete measured consequence and update have been consumed.
            self._apply_queued_variance_maturation(completed_old_pending=True)
        self.metrics["recorded_transitions"] += 1
        self.metrics["experienced_nutrition"] += float(completed["outcome"]["nutrition"])
        self.metrics["experienced_effort"] += float(completed["outcome"]["effort"])
        self.metrics["contacted_transitions"] += int(completed["outcome"]["contact"] > 0)
        self.pending = None

        # The relational graph sees the completed transition before inherited
        # context advances and the next candidate set is evaluated.
        self.motor.open_macro_boundary(normalized)
        decision = self._refine_and_commit(
            projected, normalized, raw_neural_features,
            local_physiology, step, candidate_evidence,
        )
        self._new_pending(projected, physiology, decision)
        self.metrics["ticks"] += 1
        return dict(decision["action"])

    def freeze(self, frozen: bool = True) -> None:
        """Freeze learning while retaining already learned refinement."""
        self.refiner.freeze(frozen)
        if self.personal_plasticity is not None:
            self.personal_plasticity.set_learning(not frozen)

    def set_refinement_enabled(self, enabled: bool) -> None:
        """Enable contextual choice changes without changing learning state."""
        self.refiner.enabled = bool(enabled)

    def clear_memory(self) -> None:
        """Explicitly clear learned context while preserving inherited state."""
        self.refiner.clear(self.memory)

    def queue_state_log_std_v2(self) -> dict[str, Any]:
        """Queue explicit variance maturation at the next quiescent boundary."""
        personal = self.personal_plasticity
        if personal is None:
            raise RuntimeError("variance maturation requires personal plasticity")
        if self.variance_maturation is not None:
            raise RuntimeError("variance maturation was already requested")
        if personal.config.variance_adaptation != FIXED_INHERITED_VARIANCE:
            raise RuntimeError("personal variance adaptation is already enabled")
        target = replace(
            personal.config, variance_adaptation=STATE_LOG_STD_VARIANCE,
        )
        self.variance_maturation = {
            "format": VARIANCE_MATURATION_FORMAT,
            "status": "queued",
            "source": "explicit-runtime-request-v1",
            "source_config_sha256": _config_sha256(personal.config),
            "target_config_sha256": _config_sha256(target),
            "queued_at_macro_decision": int(self.metrics["macro_decisions"]),
            "old_pending_decision_sequence": (
                None if personal.pending is None else int(personal.pending.sequence)
            ),
            "old_pending_variance_adaptation": personal.config.variance_adaptation,
        }
        return copy.deepcopy(self.variance_maturation)

    def _apply_queued_variance_maturation(
        self, *, completed_old_pending: bool,
    ) -> None:
        record = self.variance_maturation
        if record is None or record["status"] != "queued":
            return
        personal = self.personal_plasticity
        if personal is None or personal.pending is not None:
            raise RuntimeError("variance maturation requires a quiescent personal motor boundary")
        if (
            record["old_pending_decision_sequence"] is not None
            and not completed_old_pending
        ):
            raise RuntimeError(
                "queued variance maturation cannot bypass its old pending outcome"
            )
        if _config_sha256(personal.config) != record["source_config_sha256"]:
            raise RuntimeError("variance maturation source configuration changed while queued")
        personal.enable_state_log_std_v2()
        if _config_sha256(personal.config) != record["target_config_sha256"]:
            raise RuntimeError("variance maturation target configuration identity differs")
        record.update({
            "status": "applied",
            "applied_at_macro_decision": int(self.metrics["macro_decisions"]),
            "first_variance_decision_sequence": int(personal.decision_count),
            "new_head_nonzero_count": int(np.count_nonzero(personal.variance_actor)),
            "new_trace_nonzero_count": int(np.count_nonzero(personal.variance_trace)),
            "credit_boundary": "old-pending-observed-before-zero-head-maturation-v1",
        })

    def reset_episode(self) -> None:
        """Discard an incomplete transition and reset inference, not learning."""
        self.motor.reset_episode()
        self.memory.reset()
        if self.personal_plasticity is not None:
            self.personal_plasticity.discard_pending()
        self.pending = None
        self.metrics["episodes"] += 1

    def view(self) -> dict[str, Any]:
        state = self.memory.state()
        return {
            "controller": "living-contextual-motor",
            "artifact_sha256": self.artifact.sha256,
            "motor": self.motor.view(),
            "memory": {
                "profile": self.relational_profile,
                "contexts": state["contexts"],
                "transitions": state["transitions"],
                "active_context": state["active_context"],
                "revision": self.refiner.memory_revision,
            },
            "learning": {
                "frozen": not self.refiner.learning,
                "refinement_enabled": self.refiner.enabled,
                "plasticity_enabled": self.personal_plasticity is not None,
                "plasticity_frozen": (
                    None if self.personal_plasticity is None
                    else not self.personal_plasticity.learning
                ),
            },
            "utility": {
                "profile": self.refiner.config.utility_profile,
                "finite_energy_sha256": self.refiner.config.finite_energy_sha256,
            },
            "plasticity": (
                None if self.personal_plasticity is None
                else self.personal_plasticity.view()
            ),
            "variance_maturation": copy.deepcopy(self.variance_maturation),
            "last_plasticity_update": copy.deepcopy(self.last_plasticity_update),
            "last_contextual_correction": self.last_actual_correction,
            "last_record": copy.deepcopy(self.last_record),
            "metrics": copy.deepcopy(self.metrics),
            "pending": None if self.pending is None else {
                "outcomes_seen": int(self.pending["outcomes_seen"]),
                "nutrition": float(self.pending["outcome"]["nutrition"]),
                "effort": float(self.pending["outcome"]["effort"]),
                "contact": float(self.pending["outcome"]["contact"]),
            },
        }

    def snapshot(self) -> dict[str, Any]:
        pending = None
        if self.pending is not None:
            pending = {
                "feature": self.pending["feature"].tolist(),
                "before_physiology": copy.deepcopy(self.pending["before_physiology"]),
                "action": self.pending["action"].tolist(),
                "outcome": copy.deepcopy(self.pending["outcome"]),
                "outcomes_seen": int(self.pending["outcomes_seen"]),
            }
            if self.personal_plasticity is not None:
                pending.update({
                    "tick_before_physiology": copy.deepcopy(
                        self.pending["tick_before_physiology"]
                    ),
                    "plasticity_transitions": copy.deepcopy(
                        self.pending["plasticity_transitions"]
                    ),
                })
        result = {
            "format": FORMAT,
            "version": 1,
            "artifact_sha256": self.artifact.sha256,
            "physics_dt": PHYSICS_DT,
            "macro_steps": MACRO_STEPS,
            "relational_profile": self.relational_profile,
            "motor": self.motor.snapshot_value(include_artifact=False),
            "contextual": self.refiner.snapshot(self.memory),
            "pending": pending,
            "last_actual_correction": self.last_actual_correction,
            "last_record": copy.deepcopy(self.last_record),
            "metrics": copy.deepcopy(self.metrics),
        }
        if self.personal_plasticity is not None:
            result.update({
                "plasticity": self.personal_plasticity.snapshot_value(),
                "last_plasticity_update": copy.deepcopy(self.last_plasticity_update),
            })
        if self.variance_maturation is not None:
            result["variance_maturation"] = copy.deepcopy(self.variance_maturation)
        return result

    def snapshot_value(self, *, include_artifact: bool = False) -> dict[str, Any]:
        """Return private JSON state with a shared artifact reference by default."""
        value = self.snapshot()
        if include_artifact:
            value["artifact"] = self.artifact.to_value()
        return value

    @classmethod
    def restore(
        cls,
        value: Any,
        artifact: MotorArtifact | str | Path,
    ) -> "LivingMotorOrgan":
        if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("version") != 1:
            raise ValueError("unsupported living motor snapshot")
        if value.get("physics_dt") != PHYSICS_DT or value.get("macro_steps") != MACRO_STEPS:
            raise ValueError("living motor timing contract differs")
        shared = artifact if isinstance(artifact, MotorArtifact) else MotorArtifact.load(artifact)
        if value.get("artifact_sha256") != shared.sha256:
            raise ValueError("living motor artifact identity differs")
        profile = value.get("relational_profile", LEGACY_PROFILE)
        if profile not in RELATIONAL_PROFILES:
            raise ValueError("unsupported saved relational profile")
        try:
            saved_config = ContextMemoryConfig(**value["contextual"]["memory"]["config"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid saved relational memory configuration") from error
        instance = cls(
            shared,
            relational_profile=profile,
            memory_config=saved_config if profile == CUSTOM_PROFILE else None,
            plasticity=value.get("plasticity") is not None,
        )
        instance.motor = MotorOrgan.restore_value(value["motor"], shared)
        instance.refiner, instance.memory = ContextualMotorRefiner.restore(value["contextual"])
        if value.get("plasticity") is not None:
            instance.personal_plasticity = PersonalMotorPlasticity.restore_value(
                value["plasticity"]
            )
            last_plasticity_update = value.get("last_plasticity_update")
            if last_plasticity_update is not None and not isinstance(last_plasticity_update, dict):
                raise ValueError("invalid last personal plasticity update")
            instance.last_plasticity_update = copy.deepcopy(last_plasticity_update)
            if instance.refiner.config.utility_profile == FINITE_ENERGY_UTILITY_PROFILE:
                personal_identity = str(
                    instance.personal_plasticity.objective.config.to_value()["sha256"]
                )
                if personal_identity != instance.refiner.config.finite_energy_sha256:
                    raise ValueError(
                        "personal and contextual finite-energy configuration identities differ"
                    )
        if instance.refiner.feature_dim != 64:
            raise ValueError("living motor contextual feature dimension differs")
        if profile != CUSTOM_PROFILE and instance.memory.config != _profile_memory_config(profile):
            raise ValueError("saved memory configuration differs from its relational profile")
        instance.relational_profile = profile
        instance.pending = instance._restore_pending(
            value.get("pending"), instance.personal_plasticity is not None
        )
        correction = value.get("last_actual_correction")
        instance.last_actual_correction = None if correction is None else _finite(
            correction, "last actual correction"
        )
        last_record = value.get("last_record")
        if last_record is not None and not isinstance(last_record, dict):
            raise ValueError("invalid last living motor record")
        instance.last_record = copy.deepcopy(last_record)
        instance.metrics = instance._restore_metrics(value.get("metrics"))
        instance.variance_maturation = instance._restore_variance_maturation(
            value.get("variance_maturation"), instance.personal_plasticity,
        )
        instance._validate_pending_state()
        if instance.last_actual_correction is not None and abs(instance.last_actual_correction) > (
            instance.refiner.config.max_logit_correction + 1e-7
        ):
            raise ValueError("last contextual correction exceeds configured bound")
        if not (
            instance.motor.decision_count == instance.refiner.decision_count
            == int(instance.metrics["macro_decisions"])
        ):
            raise ValueError("living motor decision counters differ")
        if (
            instance.personal_plasticity is not None
            and instance.personal_plasticity.decision_count
            != int(instance.metrics["macro_decisions"])
        ):
            raise ValueError("personal plasticity decision counter differs")
        if int(instance.metrics["recorded_transitions"]) > int(instance.metrics["macro_decisions"]):
            raise ValueError("living motor transition count exceeds decisions")
        return instance

    @staticmethod
    def _restore_variance_maturation(
        value: Any,
        personal: PersonalMotorPlasticity | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict) or personal is None:
            raise ValueError("invalid variance maturation record")
        common = {
            "format", "status", "source", "source_config_sha256",
            "target_config_sha256", "queued_at_macro_decision",
            "old_pending_decision_sequence", "old_pending_variance_adaptation",
        }
        applied = {
            "applied_at_macro_decision", "first_variance_decision_sequence",
            "new_head_nonzero_count", "new_trace_nonzero_count", "credit_boundary",
        }
        status = value.get("status")
        expected = common if status == "queued" else common | applied
        if (
            status not in {"queued", "applied"}
            or set(value) != expected
            or value.get("format") != VARIANCE_MATURATION_FORMAT
            or value.get("source") != "explicit-runtime-request-v1"
            or value.get("old_pending_variance_adaptation") != FIXED_INHERITED_VARIANCE
        ):
            raise ValueError("invalid variance maturation record")
        for name in ("source_config_sha256", "target_config_sha256"):
            digest = value.get(name)
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("invalid variance maturation configuration identity")
        integer_names = ["queued_at_macro_decision"]
        if status == "applied":
            integer_names.extend((
                "applied_at_macro_decision", "first_variance_decision_sequence",
                "new_head_nonzero_count", "new_trace_nonzero_count",
            ))
        if any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), int)
            or value[name] < 0
            for name in integer_names
        ):
            raise ValueError("invalid variance maturation counters")
        pending_sequence = value.get("old_pending_decision_sequence")
        if pending_sequence is not None and (
            isinstance(pending_sequence, bool)
            or not isinstance(pending_sequence, int)
            or pending_sequence < 0
        ):
            raise ValueError("invalid pre-maturation pending decision sequence")
        active_identity = _config_sha256(personal.config)
        expected_identity = (
            value["source_config_sha256"]
            if status == "queued" else value["target_config_sha256"]
        )
        expected_mode = (
            FIXED_INHERITED_VARIANCE
            if status == "queued" else STATE_LOG_STD_VARIANCE
        )
        if (
            active_identity != expected_identity
            or personal.config.variance_adaptation != expected_mode
        ):
            raise ValueError("variance maturation record differs from personal state")
        if status == "applied" and (
            value.get("credit_boundary")
            != "old-pending-observed-before-zero-head-maturation-v1"
            or value.get("new_head_nonzero_count") != 0
            or value.get("new_trace_nonzero_count") != 0
        ):
            raise ValueError("variance maturation did not begin from an exact zero head")
        return copy.deepcopy(value)

    @classmethod
    def restore_value(
        cls,
        value: Any,
        artifact: MotorArtifact | str | Path | None = None,
    ) -> "LivingMotorOrgan":
        """Restore private state against a shared or embedded immutable artifact."""
        if artifact is None:
            if not isinstance(value, dict) or "artifact" not in value:
                raise ValueError("living motor restore requires its shared artifact")
            artifact = MotorArtifact.from_value(value["artifact"])
        return cls.restore(value, artifact)

    @staticmethod
    def _restore_pending(value: Any, plasticity: bool = False) -> dict[str, Any] | None:
        if value is None:
            return None
        basic = {
            "feature", "before_physiology", "action", "outcome", "outcomes_seen",
        }
        expected = basic | (
            {"tick_before_physiology", "plasticity_transitions"} if plasticity else set()
        )
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid pending living motor transition")
        physiology = ContextualMotorRefiner.physiology(value["before_physiology"])
        action = _vector(value["action"], len(MOTOR_ACTIONS), "pending action")
        if np.any(np.abs(action) > 1) or np.any(action[[3, 4, 5, 6]] < 0):
            raise ValueError("invalid pending physical action")
        outcome = LivingMotorOrgan._physics_outcome(value["outcome"])
        seen = int(value["outcomes_seen"])
        if isinstance(value["outcomes_seen"], bool) or not 0 <= seen <= MACRO_STEPS:
            raise ValueError("invalid pending outcome count")
        result = {
            "feature": _vector(value["feature"], 64, "pending feature"),
            "before_physiology": physiology,
            "action": action,
            "outcome": outcome,
            "outcomes_seen": seen,
        }
        if plasticity:
            tick_before = ContextualMotorRefiner.physiology(value["tick_before_physiology"])
            transitions = value["plasticity_transitions"]
            if not isinstance(transitions, list) or len(transitions) != seen:
                raise ValueError("pending physical tick records differ from outcome count")
            restored_transitions = []
            previous = physiology
            nutrition_total = 0.0
            effort_total = 0.0
            for index, transition in enumerate(transitions):
                if not isinstance(transition, dict) or set(transition) != {
                    "before", "after", "nutrition", "effort", "dt",
                }:
                    raise ValueError("invalid pending personal physical tick")
                before = ContextualMotorRefiner.physiology(transition["before"])
                after = ContextualMotorRefiner.physiology(transition["after"])
                if any(abs(before[key] - previous[key]) > 2e-6 for key in before):
                    raise ValueError("pending physical tick physiology is not contiguous")
                nutrition = _finite(transition["nutrition"], "pending tick nutrition")
                effort = _finite(transition["effort"], "pending tick effort")
                tick_dt = _finite(transition["dt"], "pending tick dt")
                if nutrition < 0 or not 0 <= effort <= 1 or not math.isclose(
                    tick_dt, PHYSICS_DT, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("invalid pending personal physical outcome")
                restored_transitions.append({
                    "before": before, "after": after, "nutrition": nutrition,
                    "effort": effort, "dt": tick_dt,
                })
                nutrition_total += nutrition
                effort_total += effort * tick_dt
                previous = after
            if any(abs(tick_before[key] - previous[key]) > 2e-6 for key in tick_before):
                raise ValueError("pending current physiology differs from physical tick history")
            if (
                abs(nutrition_total - outcome["nutrition"]) > 2e-6
                or abs(effort_total - outcome["effort"]) > 2e-6
            ):
                raise ValueError("pending physical tick totals differ from contextual outcome")
            result.update({
                "tick_before_physiology": tick_before,
                "plasticity_transitions": restored_transitions,
            })
        return result

    @staticmethod
    def _restore_metrics(value: Any) -> dict[str, int | float]:
        expected = {
            "ticks", "macro_decisions", "recorded_transitions",
            "context_changed_choices", "experienced_nutrition",
            "experienced_effort", "contacted_transitions", "episodes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid living motor metrics")
        result: dict[str, int | float] = {}
        float_fields = {"experienced_nutrition", "experienced_effort"}
        for name in expected:
            number = _finite(value[name], f"metric {name}")
            if number < 0:
                raise ValueError("living motor metrics cannot be negative")
            if name in float_fields:
                result[name] = number
            else:
                integer = int(number)
                if isinstance(value[name], bool) or number != integer:
                    raise ValueError(f"metric {name} must be an integer")
                result[name] = integer
        return result

    def _validate_pending_state(self) -> None:
        held = self.motor.held_ticks
        if held == 0:
            if self.pending is not None:
                raise ValueError("open motor boundary cannot have a pending transition")
            if self.personal_plasticity is not None and self.personal_plasticity.pending is not None:
                raise ValueError("open motor boundary cannot have pending personal plasticity")
            return
        if self.pending is None:
            raise ValueError("held motor action requires a pending transition")
        if int(self.pending["outcomes_seen"]) != held - 1:
            raise ValueError("pending outcome count differs from held motor ticks")
        if not np.array_equal(self.pending["action"], self.motor.held_action):
            raise ValueError("pending action differs from inherited held action")
        if self.personal_plasticity is not None:
            personal = self.personal_plasticity.pending
            if personal is None:
                raise ValueError("held motor action requires pending personal plasticity")
            if not np.array_equal(personal.physical_action, self.pending["action"]):
                raise ValueError("personal plasticity action differs from inherited held action")
            if len(self.pending.get("plasticity_transitions", ())) != held - 1:
                raise ValueError("personal physical tick count differs from held motor ticks")


__all__ = [
    "FORMAT", "PHYSICS_DT", "MACRO_STEPS", "LEGACY_PROFILE",
    "RESEARCH_PROFILE", "CUSTOM_PROFILE", "RELATIONAL_PROFILES",
    "LivingMotorOrgan",
]
