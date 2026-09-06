"""Validated Torch-free boundary for the native sensorimotor worker."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .organism_interface import ACTION_DIM, OBSERVATION_DIM, PHYSIOLOGY_DIM, identity
from .resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)

DEVELOPMENTAL_FORMAT = NATIVE_POPULATION_FORMAT
PERSONAL_GOAL_CONTRACT = {
    "format": "chreatures-private-goal-associations-v1",
    "objective": {
        "format": "chreatures-finite-energy-homeostasis-v1",
        "sha256": "01ae937a153a056c8cc5fa5be4d55cdfb38dbfcede4dbceb16ec33e19c5f4d00",
        "config": {
            "version": 1,
            "assimilation_efficiency": 0.84,
            "reserve_target": 0.85,
            "reserve_temperature": 0.08,
            "fatigue_energy_weight": 0.08,
            "gut_comfort": 0.55,
            "gut_overload_energy_weight": 0.08,
            "effort_energy_rate": 0.0042,
            "effort_extra_weight": 0.25,
            "reward_per_energy": 12.0,
            "max_interval_seconds": 2.0,
        },
    },
    "features": [
        "bias",
        "two_energy_minus_one",
        "two_gut_minus_one",
        "two_fatigue_minus_one",
    ],
    "slots": 128,
    "horizon_ticks": 10,
    "return_scale": 0.01,
    "learning_rate": 0.05,
    "weight_norm_limit": 4.0,
    "logit_gain": 0.35,
    "learning_default_enabled": True,
}
PREDICTOR_ORDER = (
    "context.mean", "context.scale", "action.mean", "action.scale",
    "target.mean", "target.scale",
) + tuple(
    f"member.{member}.{part}" for member in range(3)
    for part in ("context.weight", "context.bias", "transition.weight_ih",
                 "transition.weight_hh", "transition.bias_ih", "transition.bias_hh",
                 "output.weight", "output.bias")
)
POPULATION_ADAPTER_ORDER = (
    "population_adapter.down",
    "population_adapter.up",
    "population_adapter.bias",
)
PHYSIOLOGY_ADAPTER_ORDER = ("model.physiology_adapter.weight",)
NEW_ACTUATOR_ORDER = (
    "model.new_actuator_active.weight",
    "model.new_actuator_active.bias",
    "model.new_actuator_positive.weight",
    "model.new_actuator_positive.bias",
)
PREDICTOR_RUNTIME_ORDER = tuple("predictor." + name for name in PREDICTOR_ORDER)
PREDICTOR_CONTEXT_SEGMENTS = {
    "frame_codes_t_minus_3_through_t": [0, 1024],
    "private_effective_worker_context_t": [1024, 1152],
    "neural_readouts_t": [1152, 1536],
    "raw_physiology_t": [1536, 1548],
    "previous_delivered_action": [1548, 1560],
}
PREDICTOR_PHYSIOLOGY_LINK = {
    "proposal": "target-denormalized output head",
    "formula": "up*tanh(qplus/max(up,1e-4))-down*tanh(qminus/max(down,1e-4))",
    "signed_split": "stable qplus/qminus from sqrt(q*q+1e-8), epsilon=1e-4",
    "anchor": "actual physiology at t, then member-private predicted state",
    "epsilon": 1e-4,
    "lower": [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    "upper": [1] * 12,
    "clipping": False,
}
PREDICTOR_SHAPES = {
    "context.mean": (1560,), "context.scale": (1560,),
    "action.mean": (12,), "action.scale": (12,),
    "target.mean": (268,), "target.scale": (268,),
}
for _member in range(3):
    PREDICTOR_SHAPES.update({
        f"member.{_member}.{part}": shape for part, shape in {
            "context.weight": (256, 1560), "context.bias": (256,),
            "transition.weight_ih": (768, 12), "transition.weight_hh": (768, 256),
            "transition.bias_ih": (768,), "transition.bias_hh": (768,),
            "output.weight": (268, 256), "output.bias": (268,),
        }.items()
    })
MANAGER_ORDER = (
    "manager.query.0.weight",
    "manager.query.0.bias",
    "manager.query.2.weight",
    "manager.query.2.bias",
    "manager.query_gain",
)
RICH_DEVELOPMENTAL_ORDER = (
    "normalizer.mean",
    "normalizer.scale",
    "model.visual.peripheral.first.weight",
    "model.visual.peripheral.first.bias",
    "model.visual.peripheral.second.weight",
    "model.visual.peripheral.second.bias",
    "model.visual.foveal.first.weight",
    "model.visual.foveal.first.bias",
    "model.visual.foveal.second.weight",
    "model.visual.foveal.second.bias",
    "model.visual.peripheral_projection.0.weight",
    "model.visual.peripheral_projection.0.bias",
    "model.visual.foveal_projection.0.weight",
    "model.visual.foveal_projection.0.bias",
    "model.body.0.weight",
    "model.body.0.bias",
    "model.goal_encoder.0.weight",
    "model.goal_encoder.0.bias",
    "model.goal_encoder.2.weight",
    "model.goal_encoder.2.bias",
    "model.observation_projection.0.weight",
    "model.observation_projection.0.bias",
    "model.history.weight_ih_l0",
    "model.history.weight_hh_l0",
    "model.history.bias_ih_l0",
    "model.history.bias_hh_l0",
    "model.policy_trunk.0.weight",
    "model.policy_trunk.0.bias",
    "model.signed_head.weight",
    "model.signed_head.bias",
    "model.active_head.weight",
    "model.active_head.bias",
    "model.positive_head.weight",
    "model.positive_head.bias",
) + MANAGER_ORDER
RICH_DEVELOPMENTAL_SHAPES = {
    "normalizer.mean": (4459,),
    "normalizer.scale": (4459,),
    "model.visual.peripheral.first.weight": (16, 4, 3, 3),
    "model.visual.peripheral.first.bias": (16,),
    "model.visual.peripheral.second.weight": (24, 16, 3, 3),
    "model.visual.peripheral.second.bias": (24,),
    "model.visual.foveal.first.weight": (16, 4, 3, 3),
    "model.visual.foveal.first.bias": (16,),
    "model.visual.foveal.second.weight": (24, 16, 3, 3),
    "model.visual.foveal.second.bias": (24,),
    "model.visual.peripheral_projection.0.weight": (64, 768),
    "model.visual.peripheral_projection.0.bias": (64,),
    "model.visual.foveal_projection.0.weight": (64, 2304),
    "model.visual.foveal_projection.0.bias": (64,),
    "model.body.0.weight": (128, 363),
    "model.body.0.bias": (128,),
    "model.goal_encoder.0.weight": (256, 1024),
    "model.goal_encoder.0.bias": (256,),
    "model.goal_encoder.2.weight": (64, 256),
    "model.goal_encoder.2.bias": (64,),
    "model.observation_projection.0.weight": (128, 268),
    "model.observation_projection.0.bias": (128,),
    "model.history.weight_ih_l0": (384, 128),
    "model.history.weight_hh_l0": (384, 128),
    "model.history.bias_ih_l0": (384,),
    "model.history.bias_hh_l0": (384,),
    "model.policy_trunk.0.weight": (256, 205),
    "model.policy_trunk.0.bias": (256,),
    "model.signed_head.weight": (260, 256),
    "model.signed_head.bias": (260,),
    "model.active_head.weight": (8, 256),
    "model.active_head.bias": (8,),
    "model.positive_head.weight": (256, 256),
    "model.positive_head.bias": (256,),
    "manager.query.0.weight": (128, 524),
    "manager.query.0.bias": (128,),
    "manager.query.2.weight": (64, 128),
    "manager.query.2.bias": (64,),
    "manager.query_gain": (1,),
}
SUCCESSOR_SHAPES = {
    **RICH_DEVELOPMENTAL_SHAPES,
    "model.physiology_adapter.weight": (128, 12),
    "model.new_actuator_active.weight": (4, 256),
    "model.new_actuator_active.bias": (4,),
    "model.new_actuator_positive.weight": (128, 256),
    "model.new_actuator_positive.bias": (128,),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _artifact_identity(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    clean = copy.deepcopy(metadata)
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(_canonical(clean))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(_canonical(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _predictor_identity(metadata, arrays):
    clean = copy.deepcopy(metadata)
    clean.pop("artifact_identity", None)
    receipts = {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    return hashlib.sha256(
        _canonical({"metadata": clean, "arrays": receipts})
    ).hexdigest()


def _extension():
    try:
        return importlib.import_module("_cognitive_core")
    except ImportError as exc:
        raise RuntimeError("native cognitive core is unavailable") from exc


def _candidate_arrays(adapters, count, population, organism_sha256):
    from .population import canonical_bytes as population_canonical

    required = {
        "candidate_sha256",
        "loci_sha256",
        "policy_adapter_index",
        "population_adapter_bank_sha256",
        "policy_adapter_count",
        "policy_adapter_rank",
        "organism_interface_sha256",
        "recurrent_gain",
        "learning_rate_gain",
        "action_gain",
        "action_logit_temperature_offset",
    }
    if not isinstance(adapters, list) or len(adapters) != count:
        raise ValueError("candidate adapters must contain one row per resident")
    candidate_sha256 = []
    loci_sha256 = []
    recurrent_gain = np.empty(count, dtype=np.float32)
    learning_rate_gain = np.empty(count, dtype=np.float32)
    action_gain = np.empty((count, ACTION_DIM), dtype=np.float32)
    temperature = np.empty((count, ACTION_DIM), dtype=np.float32)
    indices = np.empty(count, dtype=np.uint16)
    for row, adapter in enumerate(adapters):
        if not isinstance(adapter, dict) or set(adapter) != required:
            raise ValueError("candidate adapter fields differ")
        for name in ("candidate_sha256", "loci_sha256"):
            digest = adapter[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"invalid candidate adapter {name}")
        if (
            adapter["population_adapter_bank_sha256"] != population["identity"]
            or adapter["policy_adapter_count"] != population["count"]
            or adapter["policy_adapter_rank"] != population["rank"]
            or adapter["organism_interface_sha256"] != organism_sha256
        ):
            raise ValueError("candidate adapter immutable contract differs")
        identity_body = {
            name: adapter[name]
            for name in required - {"candidate_sha256", "loci_sha256"}
        }
        if (
            hashlib.sha256(population_canonical(identity_body)).hexdigest()
            != adapter["loci_sha256"]
        ):
            raise ValueError("candidate controller loci identity differs")
        adapter_index = adapter["policy_adapter_index"]
        if (
            type(adapter_index) is not int
            or not 0 <= adapter_index < population["count"]
        ):
            raise ValueError("candidate policy adapter index differs")
        candidate_sha256.append(adapter["candidate_sha256"])
        loci_sha256.append(adapter["loci_sha256"])
        indices[row] = adapter_index
        recurrent_gain[row] = adapter["recurrent_gain"]
        learning_rate_gain[row] = adapter["learning_rate_gain"]
        action_gain[row] = adapter["action_gain"]
        temperature[row] = adapter["action_logit_temperature_offset"]
    return (
        candidate_sha256,
        loci_sha256,
        indices,
        recurrent_gain,
        learning_rate_gain,
        action_gain,
        temperature,
    )


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "base64": base64.b64encode(array.tobytes()).decode(),
        }
    if isinstance(value, dict):
        return {name: _encode(item) for name, item in value.items()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"dtype", "shape", "base64"}:
        dtype = np.dtype(value["dtype"])
        shape = tuple(value["shape"])
        array = np.frombuffer(
            base64.b64decode(value["base64"], validate=True), dtype=dtype
        ).copy()
        if array.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError("encoded developmental array shape differs")
        return array.reshape(shape)
    if isinstance(value, dict):
        return {name: _decode(item) for name, item in value.items()}
    return value


class DevelopmentalResidentCohort:
    """Complete recurring resident controller with native private state."""

    def __init__(
        self,
        artifact: str | Path,
        batch_size: int,
        *,
        action_mode: str,
        goal_seed: int,
        action_seed: int,
        candidate_adapters: list[dict[str, Any]],
        population_response_artifact: str | Path | None = None,
    ) -> None:
        path = Path(artifact).expanduser().resolve()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        population_response_json = None
        population_response_identity = None
        population_feature_contract_identity = None
        self.population_response_artifact = None
        if population_response_artifact is not None:
            response_path = Path(population_response_artifact).expanduser().resolve()
            response_bytes = response_path.read_bytes()
            population_response_json = response_bytes.decode("utf-8")
            response_value = json.loads(population_response_json)
            population_response_identity = hashlib.sha256(response_bytes).hexdigest()
            population_feature_contract_identity = response_value.get(
                "feature_contract_sha256"
            )
            if (
                response_value.get("schema") != "chreatures-population-response-bank-v1"
                or not isinstance(population_feature_contract_identity, str)
                or len(population_feature_contract_identity) != 64
            ):
                raise ValueError("population response artifact contract differs")
            self.population_response_artifact = {
                "path": str(response_path),
                "sha256": population_response_identity,
                "feature_contract_sha256": population_feature_contract_identity,
            }
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            order = (
                RICH_DEVELOPMENTAL_ORDER
                + PHYSIOLOGY_ADAPTER_ORDER
                + POPULATION_ADAPTER_ORDER
                + NEW_ACTUATOR_ORDER
                + PREDICTOR_RUNTIME_ORDER
            )
            if (
                metadata.get("format") != DEVELOPMENTAL_FORMAT
                or metadata.get("version") != NATIVE_POPULATION_VERSION
                or metadata.get("execution") != NATIVE_EXECUTION
                or metadata.get("pack_order") != list(order)
                or set(archive.files) != set(order) | {"metadata"}
            ):
                raise ValueError("developmental resident artifact metadata differs")
            arrays = {}
            for name in order:
                value = np.asarray(archive[name])
                receipt = metadata.get("tensors", {}).get(name, {})
                if (
                    value.dtype != np.float32
                    or (
                        name in SUCCESSOR_SHAPES
                        and value.shape != SUCCESSOR_SHAPES[name]
                    )
                    or (
                        name in PREDICTOR_RUNTIME_ORDER
                        and value.shape != PREDICTOR_SHAPES[name[10:]]
                    )
                    or list(value.shape) != receipt.get("shape")
                    or receipt.get("dtype") != "float32"
                    or hashlib.sha256(value.tobytes()).hexdigest()
                    != receipt.get("sha256")
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"invalid developmental resident tensor: {name}")
                arrays[name] = value
        if _artifact_identity(metadata, arrays) != metadata.get("artifact_sha256"):
            raise ValueError("developmental resident artifact identity differs")
        runtime = metadata.get("runtime_contract", {})
        if runtime.get("personal_goal_associations") != PERSONAL_GOAL_CONTRACT:
            raise ValueError("personal goal association contract differs")
        temporal = runtime.get("temporal", {})
        if (
            temporal.get("observation_interval_seconds") != 0.05
            or temporal.get("manager_commit_ticks") != 10
        ):
            raise ValueError("developmental resident timing differs")
        refinement = runtime.get("consequence_refinement", {})
        laws = metadata.get("consequence_laws", {})
        law_bank = laws.get("value")
        if (
            not isinstance(law_bank, dict)
            or laws.get("content_sha256")
            != hashlib.sha256(_canonical(law_bank)).hexdigest()
            or [
                refinement.get(x)
                for x in (
                    "candidates",
                    "tilt",
                    "learning_rate",
                    "error_decay",
                    "innovation_limit",
                )
            ]
            != [4, 0.5, 0.05, 0.99, 4.0]
        ):
            raise ValueError("developmental consequence refinement differs")
        predictor = metadata.get("recurrent_predictor", {})
        population_adapters = metadata.get("population_adapters", {})
        adapter_count = population_adapters.get("count")
        adapter_rank = population_adapters.get("rank")
        if (
            type(adapter_count) is not int
            or type(adapter_rank) is not int
            or not 1 <= adapter_count <= 256
            or not 1 <= adapter_rank <= 256
            or arrays["population_adapter.down"].shape
            != (adapter_count, adapter_rank, 256)
            or arrays["population_adapter.up"].shape
            != (adapter_count, 256, adapter_rank)
            or arrays["population_adapter.bias"].shape != (adapter_count, 256)
        ):
            raise ValueError("population policy adapter contract differs")
        predictor_metadata = predictor.get("metadata", {})
        predictor_order = tuple(predictor_metadata.get("pack_order", ()))
        predictor_identity = predictor_metadata.get("artifact_identity")
        scoring = predictor.get("runtime_scoring", {})
        calibration = predictor_metadata.get("validation", {}).get("goal_calibration", {}).get("empirical_goal_rms_by_horizon", [])
        goal_rms = scoring.get("goal_error_rms")
        predictor_arrays = {
            name.removeprefix("predictor."): arrays[name]
            for name in PREDICTOR_RUNTIME_ORDER
        }
        source_predictor_metadata = copy.deepcopy(predictor_metadata)
        source_predictor_metadata["pack_order"] = source_predictor_metadata.pop("source_pack_order", None)
        representation = predictor_metadata.get("representation", {})
        temporal = predictor_metadata.get("temporal_contract", {})
        input_contract = predictor_metadata.get("input_contract", {})
        from .organism_interface import ACTION_NAMES, PHYSIOLOGY_NAMES
        model_arrays = {name.removeprefix("model."): value for name, value in arrays.items() if name.startswith("model.")}
        model_hash = hashlib.sha256()
        for name in sorted(model_arrays):
            value = np.ascontiguousarray(model_arrays[name], dtype="<f4")
            model_hash.update(name.encode())
            model_hash.update(b"\0")
            model_hash.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
            model_hash.update(b"\0<f4\0")
            model_hash.update(value.tobytes())
        if (
            predictor_order != PREDICTOR_RUNTIME_ORDER
            or source_predictor_metadata["pack_order"] != list(PREDICTOR_ORDER)
            or predictor_metadata.get("format") != "chreatures-rich-recurrent-consequence-ensemble-v3"
            or predictor_metadata.get("version") != 3
            or not isinstance(predictor_identity, str) or len(predictor_identity) != 64
            or predictor.get("artifact_identity") != predictor_identity
            or _predictor_identity(source_predictor_metadata, predictor_arrays) != predictor_identity
            or representation.get("file_sha256") != metadata.get("checkpoint", {}).get("sha256")
            or representation.get("model_tensor_sha256") != model_hash.hexdigest()
            or representation.get("identity", {}).get("normalizer") != metadata.get("training_identity", {}).get("normalizer")
            or temporal.get("observation_interval_seconds") != 0.05
            or temporal.get("horizons_ticks") != list(range(1, 9))
            or input_contract.get("context_dim") != 1560
            or input_contract.get("context_segments") != PREDICTOR_CONTEXT_SEGMENTS
            or input_contract.get("worker_recurrent_context") != "native state plus recurrent_adapter after current observation and policy adapter update"
            or predictor_metadata.get("output_contract", {}).get("physiology_link") != PREDICTOR_PHYSIOLOGY_LINK
            or input_contract.get("action_names") != list(ACTION_NAMES)
            or input_contract.get("physiology_names") != list(PHYSIOLOGY_NAMES)
            or scoring.get("horizon_ticks") != 4 or scoring.get("horizon_seconds") != 0.2
            or scoring.get("proposal_suffix") != "hold the proposed delivered action constant for four ticks"
            or len(calibration) != 8 or goal_rms != calibration[3]
            or not isinstance(goal_rms, (int, float)) or not np.isfinite(goal_rms) or goal_rms < 1e-4
        ):
            raise ValueError("recurrent predictive consequence contract differs")
        if action_mode not in {"sample", "map"}:
            raise ValueError("action_mode must be sample or map")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 4096
        ):
            raise ValueError("batch_size must be an integer in 1..4096")
        for name, seed in (("goal_seed", goal_seed), ("action_seed", action_seed)):
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed < 2**64
            ):
                raise ValueError(f"{name} must be an unsigned 64-bit integer")
        organism_sha256 = hashlib.sha256(_canonical(identity())).hexdigest()
        (
            candidate_sha256,
            loci_sha256,
            policy_adapter_index,
            recurrent_gain,
            learning_rate_gain,
            action_gain,
            temperature,
        ) = _candidate_arrays(
            candidate_adapters,
            batch_size,
            population_adapters,
            organism_sha256,
        )
        packed = np.ascontiguousarray(
            np.concatenate(
                [arrays[name].reshape(-1) for name in RICH_DEVELOPMENTAL_ORDER]
            ),
            dtype=np.float32,
        )
        predictor_packed = np.ascontiguousarray(
            np.concatenate([arrays[name].reshape(-1) for name in PREDICTOR_RUNTIME_ORDER]),
            dtype=np.float32,
        )
        policy_adapter_packed = np.ascontiguousarray(
            np.concatenate(
                [arrays[name].reshape(-1) for name in POPULATION_ADAPTER_ORDER]
            ),
            dtype=np.float32,
        )
        physiology_adapter_packed = np.ascontiguousarray(
            arrays[PHYSIOLOGY_ADAPTER_ORDER[0]].reshape(-1), dtype=np.float32
        )
        new_actuator_packed = np.ascontiguousarray(
            np.concatenate([arrays[name].reshape(-1) for name in NEW_ACTUATOR_ORDER]),
            dtype=np.float32,
        )
        self.artifact_path = path
        self.batch_size = batch_size
        self.action_mode = action_mode
        self._policy_adapter_count = adapter_count
        self._population_adapters = copy.deepcopy(population_adapters)
        self._organism_interface_sha256 = organism_sha256
        self.candidate_adapters = copy.deepcopy(candidate_adapters)
        self.model_identity = {
            "format": DEVELOPMENTAL_FORMAT,
            "artifact_sha256": metadata["artifact_sha256"],
            "file_sha256": file_hash,
            "mode": "rich-achieved-goal",
            "execution": metadata["execution"],
            "personal_goal_contract_sha256": hashlib.sha256(
                _canonical(PERSONAL_GOAL_CONTRACT)
            ).hexdigest(),
            "finite_energy_objective_sha256": PERSONAL_GOAL_CONTRACT["objective"][
                "sha256"
            ],
            "population_response": copy.deepcopy(self.population_response_artifact),
            "recurrent_predictor": {"artifact_identity": predictor_identity, "horizon_ticks": 4,
                                    "proposal_suffix": scoring["proposal_suffix"]},
        }
        trained = metadata.get("training_identity", {})
        self.neural_contract = {
            name: trained.get(name)
            for name in ("graph_sha256", "port_spec_sha256", "port_bundle_sha256")
        }
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in self.neural_contract.values()
        ):
            raise ValueError(
                "developmental artifact lacks its trained neural substrate"
            )
        self.observation_contract = copy.deepcopy(metadata.get("organism_interface"))
        if self.observation_contract != identity():
            raise ValueError("developmental resident observation contract differs")
        self._native = _extension().DevelopmentalResidentCohort(
            batch_size,
            "rich-achieved-goal",
            action_mode,
            goal_seed,
            action_seed,
            packed,
            _canonical(law_bank).decode(),
            laws["file_sha256"],
            refinement["learning_rate"],
            refinement["error_decay"],
            refinement["innovation_limit"],
            population_response_json,
            population_response_identity,
            population_feature_contract_identity,
            predictor_packed,
            goal_rms,
            physiology_adapter_packed,
            policy_adapter_packed,
            adapter_count,
            adapter_rank,
            new_actuator_packed,
            policy_adapter_index,
            candidate_sha256,
            loci_sha256,
            recurrent_gain,
            learning_rate_gain,
            action_gain,
            temperature,
        )

    def hatch_slots(
        self,
        rows,
        candidate_adapters: list[dict[str, Any]],
        *,
        goal_seeds,
        action_seeds,
    ) -> None:
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or rows.size == 0
            or not np.issubdtype(rows.dtype, np.integer)
            or np.issubdtype(rows.dtype, np.bool_)
        ):
            raise ValueError("hatch rows must be a nonempty integer vector")
        row_values = [int(value) for value in rows]
        if (
            len(set(row_values)) != len(row_values)
            or any(not 0 <= row < self.batch_size for row in row_values)
            or len(candidate_adapters) != len(row_values)
        ):
            raise ValueError("hatch rows or candidate adapters differ")
        count = len(row_values)
        goal_seeds = np.asarray(goal_seeds)
        action_seeds = np.asarray(action_seeds)
        if goal_seeds.shape != (count,) or action_seeds.shape != (count,):
            raise ValueError("hatch seed shapes differ")
        if not all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and 0 <= int(value) < 2**64
            for value in (*goal_seeds.tolist(), *action_seeds.tolist())
        ):
            raise ValueError("hatch seeds must be unsigned 64-bit integers")
        (
            candidate_hashes,
            loci_hashes,
            indices,
            recurrent,
            learning,
            gains,
            temperatures,
        ) = _candidate_arrays(
            candidate_adapters,
            count,
            self._population_adapters,
            self._organism_interface_sha256,
        )
        self._native.hatch_slots(
            np.asarray(row_values, dtype=np.uint16),
            np.ascontiguousarray(goal_seeds, dtype=np.uint64),
            np.ascontiguousarray(action_seeds, dtype=np.uint64),
            candidate_hashes,
            loci_hashes,
            indices,
            recurrent,
            learning,
            gains,
            temperatures,
        )
        for row, adapter in zip(row_values, candidate_adapters, strict=True):
            self.candidate_adapters[row] = copy.deepcopy(adapter)

    def expanded(
        self,
        new_candidate_adapters: list[dict[str, Any]],
        *,
        goal_seed: int,
        action_seed: int,
    ) -> DevelopmentalResidentCohort:
        if not isinstance(new_candidate_adapters, list) or not new_candidate_adapters:
            raise ValueError("new candidate adapters must be a nonempty list")
        if self.batch_size + len(new_candidate_adapters) > 4096:
            raise ValueError("expanded cohort exceeds 4096 residents")
        for name, seed in (("goal_seed", goal_seed), ("action_seed", action_seed)):
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed < 2**64
            ):
                raise ValueError(f"{name} must be an unsigned 64-bit integer")
        (
            candidate_hashes,
            loci_hashes,
            indices,
            recurrent,
            learning,
            gains,
            temperatures,
        ) = _candidate_arrays(
            new_candidate_adapters,
            len(new_candidate_adapters),
            self._population_adapters,
            self._organism_interface_sha256,
        )
        expanded_native = self._native.expanded(
            goal_seed,
            action_seed,
            candidate_hashes,
            loci_hashes,
            indices,
            recurrent,
            learning,
            gains,
            temperatures,
        )
        result = object.__new__(type(self))
        result.artifact_path = self.artifact_path
        result.batch_size = self.batch_size + len(new_candidate_adapters)
        result.action_mode = self.action_mode
        result._policy_adapter_count = self._policy_adapter_count
        result._population_adapters = copy.deepcopy(self._population_adapters)
        result._organism_interface_sha256 = self._organism_interface_sha256
        result.population_response_artifact = copy.deepcopy(
            self.population_response_artifact
        )
        result.candidate_adapters = copy.deepcopy(
            self.candidate_adapters + new_candidate_adapters
        )
        result.model_identity = copy.deepcopy(self.model_identity)
        result.neural_contract = copy.deepcopy(self.neural_contract)
        result.observation_contract = copy.deepcopy(self.observation_contract)
        result._native = expanded_native
        return result

    def step(
        self,
        observations,
        neural,
        physiology,
        actual_previous_actions,
        ticks,
        times,
        reset,
    ):
        result = self._native.step(
            np.ascontiguousarray(observations, dtype=np.float32),
            np.ascontiguousarray(neural, dtype=np.float32),
            np.ascontiguousarray(physiology, dtype=np.float32),
            np.ascontiguousarray(actual_previous_actions, dtype=np.float32),
            np.ascontiguousarray(ticks, dtype=np.uint64),
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(reset, dtype=np.bool_),
        )
        result = {name: np.asarray(value) for name, value in result.items()}
        expected = {
            "proposed_action": (self.batch_size, ACTION_DIM),
            "candidate_scores": (self.batch_size, 4),
            "candidate_out_of_domain": (self.batch_size, 4),
            "selected_candidate": (self.batch_size,),
            "selected_consequence_correction": (self.batch_size, 3),
            "personal_consequence_updates": (self.batch_size,),
            "forecast_progress": (self.batch_size, 4),
            "worker_recurrent_context": (self.batch_size, 128),
            "forecast_horizon_ticks": (),
            "forecast_physiology": (self.batch_size, 4, 12),
            "forecast_disagreement": (self.batch_size, 4),
            "forecast_invalid": (self.batch_size, 4),
            "forecast_tilt": (self.batch_size, 4),
            "forecast_goal_rms": (),
            "actual_previous_action": (self.batch_size, ACTION_DIM),
            "hidden": (self.batch_size, 128),
            "physiology": (self.batch_size, PHYSIOLOGY_DIM),
            "memory_inserted_slot": (self.batch_size,),
            "memory_count": (self.batch_size,),
            "goal_slot": (self.batch_size,),
            "goal_valid": (self.batch_size,),
            "goal_changed": (self.batch_size,),
            "goal_recorded_tick": (self.batch_size,),
            "goal_recorded_time": (self.batch_size,),
            "goal_generation": (self.batch_size,),
            "goal_remaining_ticks": (self.batch_size,),
            "goal_key": (self.batch_size, 64),
            "goal_window": (self.batch_size, 4, OBSERVATION_DIM),
            "personal_goal_selected_bias": (self.batch_size,),
            "personal_goal_prediction": (self.batch_size,),
            "personal_goal_last_reward": (self.batch_size,),
            "personal_goal_last_return": (self.batch_size,),
            "personal_goal_completed": (self.batch_size,),
            "personal_goal_attributed": (self.batch_size,),
            "personal_goal_learned": (self.batch_size,),
            "personal_goal_completed_total": (self.batch_size,),
            "personal_goal_learned_total": (self.batch_size,),
            "personal_goal_frozen_total": (self.batch_size,),
            "personal_goal_skipped_total": (self.batch_size,),
            "personal_goal_cancelled_total": (self.batch_size,),
            "personal_goal_learning_enabled": (),
            "contextual_retrieval_bias": (self.batch_size,),
            "goal_attainment_rms_threshold": (),
            "goal_sequence_selected_bias": (self.batch_size,),
            "goal_sequence_experienced_path_depth": (self.batch_size,),
            "goal_sequence_selected_confidence": (self.batch_size,),
            "goal_sequence_learned_transitions_total": (self.batch_size,),
            "goal_sequence_failed_attempts_total": (self.batch_size,),
            "contextual_episodic_updates": (self.batch_size,),
        }
        if set(result) != set(expected):
            raise RuntimeError("native developmental result fields differ")
        for name, shape in expected.items():
            if result[name].shape != shape:
                raise RuntimeError(f"native developmental result shape differs: {name}")
            if (
                np.issubdtype(result[name].dtype, np.floating)
                and not np.isfinite(result[name]).all()
            ):
                raise RuntimeError(f"native developmental result is nonfinite: {name}")
        return result

    def observe_consequences(
        self,
        ticks,
        before_physiology,
        after_physiology,
        executed_actions,
        effort,
        *,
        dt: float,
    ):
        result = self._native.observe_consequences(
            np.ascontiguousarray(ticks, dtype=np.uint64),
            np.ascontiguousarray(before_physiology, dtype=np.float32),
            np.ascontiguousarray(after_physiology, dtype=np.float32),
            np.ascontiguousarray(executed_actions, dtype=np.float32),
            np.ascontiguousarray(effort, dtype=np.float32),
            float(dt),
        )
        result = {name: np.asarray(value) for name, value in result.items()}
        expected = {
            "reward": (self.batch_size,),
            "completed": (self.batch_size,),
            "summed_return": (self.batch_size,),
            "attributed": (self.batch_size,),
            "learned": (self.batch_size,),
            "completed_total": (self.batch_size,),
            "learned_total": (self.batch_size,),
            "frozen_total": (self.batch_size,),
            "skipped_total": (self.batch_size,),
            "cancelled_total": (self.batch_size,),
            "actual_attained": (self.batch_size,),
            "observed_normalized_progress": (self.batch_size,),
            "measurement_start_rms": (self.batch_size,),
            "measurement_min_rms": (self.batch_size,),
            "measurement_latest_rms": (self.batch_size,),
            "measurement_samples": (self.batch_size,),
            "measurement_window_ending_last_observed_tick": (self.batch_size,),
        }
        if self.population_response_artifact is not None:
            expected.update({
                "population_response_identity": (),
                "population_feature_contract_identity": (),
                "population_response_in_domain": (self.batch_size,),
                "population_response_in_domain_total": (self.batch_size,),
                "population_response_out_of_domain_total": (self.batch_size,),
            })
        if set(result) != set(expected):
            raise RuntimeError("native personal goal receipt fields differ")
        for name, shape in expected.items():
            if result[name].shape != shape:
                raise RuntimeError(
                    f"native personal goal receipt shape differs: {name}"
                )
            if (
                np.issubdtype(result[name].dtype, np.floating)
                and not np.isfinite(result[name]).all()
            ):
                raise RuntimeError(f"native personal goal receipt is nonfinite: {name}")
        if self.population_response_artifact is not None and (
            str(result["population_response_identity"])
            != self.population_response_artifact["sha256"]
            or str(result["population_feature_contract_identity"])
            != self.population_response_artifact["feature_contract_sha256"]
        ):
            raise RuntimeError("native population response receipt identity differs")
        return result

    def set_personal_goal_learning(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        self._native.set_personal_goal_learning(enabled)

    def snapshot_value(self) -> dict[str, Any]:
        return {
            "format": "chreatures-developmental-resident-population-snapshot-v5",
            "version": 5,
            "model_identity": copy.deepcopy(self.model_identity),
            "batch_size": self.batch_size,
            "observation_contract": copy.deepcopy(self.observation_contract),
            "action_mode": self.action_mode,
            "candidate_adapters": copy.deepcopy(self.candidate_adapters),
            "population_response_artifact": copy.deepcopy(
                self.population_response_artifact
            ),
            "native": _encode(dict(self._native.snapshot())),
        }

    @classmethod
    def restore_value(
        cls, value: dict[str, Any], artifact: str | Path,
        population_response_artifact: str | Path | None = None,
    ) -> DevelopmentalResidentCohort:
        if (
            not isinstance(value, dict)
            or value.get("format")
            != "chreatures-developmental-resident-population-snapshot-v5"
            or value.get("version") != 5
        ):
            raise ValueError("unsupported developmental resident snapshot")
        instance = cls(
            artifact,
            int(value["batch_size"]),
            action_mode=value["action_mode"],
            goal_seed=0,
            action_seed=0,
            candidate_adapters=value["candidate_adapters"],
            population_response_artifact=population_response_artifact,
        )
        if (
            value.get("model_identity") != instance.model_identity
            or value.get("observation_contract") != instance.observation_contract
            or value.get("population_response_artifact")
            != instance.population_response_artifact
        ):
            raise ValueError("developmental resident snapshot model identity differs")
        native = _decode(value["native"])
        instance._native.restore(native)
        return instance


__all__ = ["DevelopmentalResidentCohort"]
