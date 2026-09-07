"""Torch side of the learned native sequence-control contract.

The native runtime owns feature construction, proposal sampling, control draws,
and physical execution.  This module replays those recorded decisions in batch,
computes their hierarchical likelihood, and exports the small immutable head
bundle consumed by Rust.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from chreatures.sequence_control import (
    CANDIDATES as PROPOSAL_COUNT,
    CANDIDATE_DIM,
    CONTRACT_SHA256,
    ORDER as PARAMETER_ORDER,
    SHAPES as PARAMETER_SHAPES,
    STATE_DIM,
    valid_sha256,
)

CONTRACT_VERSION = "learned-sequence-control-v1"
ROLLOUT_FORMAT = "chreatures-native-sequence-control-rollout-v1"
# Root must replace this with the approved CNS-derived contract identity before
# any artifact initialization or rollout optimization is permitted.
ACTIVE_ROLLOUT_CONTRACT: str | None = None
STATE_CODE_DIM = 256
CANDIDATE_CODE_DIM = 128
HIDDEN_DIM = 128
ACTION_DIM = 12
INITIAL_HAZARD = 1.0 / 8.0
INITIAL_HAZARD_LOGIT = -math.log(7.0)
CANCELLATION_REASONS = (
    "none",
    "voluntary_termination",
    "host_override",
    "tick_gap",
    "reset",
    "invalid_source",
)

PARAMETER_COUNT = sum(math.prod(shape) for shape in PARAMETER_SHAPES.values())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class SequenceControlHeads(nn.Module):
    """Shared selector, learned termination hazard, and pre-decision critic."""

    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = nn.Linear(STATE_DIM, STATE_CODE_DIM)
        self.candidate_encoder = nn.Linear(CANDIDATE_DIM, CANDIDATE_CODE_DIM)
        self.selector_hidden = nn.Linear(
            STATE_CODE_DIM + CANDIDATE_CODE_DIM, HIDDEN_DIM
        )
        self.selector_out = nn.Linear(HIDDEN_DIM, 1)
        joint_dim = STATE_CODE_DIM + 2 * CANDIDATE_CODE_DIM
        self.hazard_hidden = nn.Linear(joint_dim, HIDDEN_DIM)
        self.hazard_out = nn.Linear(HIDDEN_DIM, 1)
        self.value_hidden = nn.Linear(joint_dim, HIDDEN_DIM)
        self.value_out = nn.Linear(HIDDEN_DIM, 1)

    def initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            for layer in (
                self.state_encoder,
                self.candidate_encoder,
                self.selector_hidden,
                self.hazard_hidden,
                self.value_hidden,
            ):
                nn.init.xavier_uniform_(layer.weight, generator=generator)
                layer.bias.zero_()
            for layer in (self.selector_out, self.hazard_out, self.value_out):
                layer.weight.zero_()
                layer.bias.zero_()
            self.hazard_out.bias.fill_(INITIAL_HAZARD_LOGIT)

    def forward(
        self,
        state: torch.Tensor,
        proposal: torch.Tensor,
        active: torch.Tensor,
        proposal_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_head_inputs(state, proposal, active, proposal_mask, active_mask)
        state_code = torch.tanh(self.state_encoder(state))
        candidate_code = torch.tanh(self.candidate_encoder(proposal))
        mask = proposal_mask.unsqueeze(-1)
        count = mask.sum(dim=-2).clamp_min(1)
        candidate_pool = (candidate_code * mask).sum(dim=-2) / count
        active_code = torch.tanh(self.candidate_encoder(active))
        active_code = torch.where(
            active_mask.unsqueeze(-1), active_code, torch.zeros_like(active_code)
        )
        selector_state = state_code.unsqueeze(-2).expand(
            *candidate_code.shape[:-1], STATE_CODE_DIM
        )
        selector_hidden = torch.tanh(
            self.selector_hidden(torch.cat((selector_state, candidate_code), dim=-1))
        )
        selector_logits = self.selector_out(selector_hidden).squeeze(-1)
        joint = torch.cat((state_code, active_code, candidate_pool), dim=-1)
        hazard_logit = self.hazard_out(
            torch.tanh(self.hazard_hidden(joint))
        ).squeeze(-1)
        value = self.value_out(torch.tanh(self.value_hidden(joint))).squeeze(-1)
        return {
            "hazard_logit": hazard_logit,
            "selector_logits": selector_logits,
            "value": value,
        }


def _validate_head_inputs(state, proposal, active, proposal_mask, active_mask) -> None:
    batch_shape = state.shape[:-1]
    if state.shape[-1:] != (STATE_DIM,):
            raise ValueError(f"state must end with {STATE_DIM}")
    if proposal.shape != (*batch_shape, PROPOSAL_COUNT, CANDIDATE_DIM):
        raise ValueError(
            f"proposal must be [...,{PROPOSAL_COUNT},{CANDIDATE_DIM}]"
        )
    if active.shape != (*batch_shape, CANDIDATE_DIM):
        raise ValueError(f"active must be [...,{CANDIDATE_DIM}]")
    if proposal_mask.shape != (*batch_shape, PROPOSAL_COUNT):
        raise ValueError(f"proposal_mask must be [...,{PROPOSAL_COUNT}]")
    if active_mask.shape != batch_shape:
        raise ValueError("active_mask must match the leading batch shape")
    if proposal_mask.dtype != torch.bool or active_mask.dtype != torch.bool:
        raise ValueError("sequence-control masks must be bool")
    if not torch.all(proposal_mask.any(dim=-1)):
        raise ValueError("every decision must have an available proposal")
    for name, value in (("state", state), ("proposal", proposal), ("active", active)):
        if not value.is_floating_point() or not torch.all(torch.isfinite(value)):
            raise ValueError(f"{name} must be finite floating point")


def hierarchical_terms(
    outputs: Mapping[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    active_mask: torch.Tensor,
    hazard_decision: torch.Tensor,
    selected_candidate: torch.Tensor,
    hazard_mask: torch.Tensor,
    selector_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return branched Bernoulli and conditional categorical terms.

    Sentinels never enter a gather and masked ``-inf`` values are never
    multiplied by zero.
    """

    hazard_logit = outputs["hazard_logit"]
    selector_logits = outputs["selector_logits"]
    shape = hazard_logit.shape
    for name, value in (
        ("active_mask", active_mask),
        ("hazard_decision", hazard_decision),
        ("hazard_mask", hazard_mask),
        ("selector_mask", selector_mask),
    ):
        if value.shape != shape or value.dtype != torch.bool:
            raise ValueError(f"{name} must be bool with the decision batch shape")
    if selected_candidate.shape != shape or selected_candidate.dtype != torch.long:
        raise ValueError("selected_candidate must be int64 with the decision shape")
    if proposal_mask.shape != (*shape, PROPOSAL_COUNT):
        raise ValueError("proposal_mask shape differs from logits")
    if not torch.equal(hazard_mask, active_mask):
        raise ValueError("hazard_mask must equal active_mask")
    expected_selector = (~active_mask) | hazard_decision
    if not torch.equal(selector_mask, expected_selector):
        raise ValueError("selector_mask differs from the sampled hierarchy")
    if torch.any((~hazard_mask) & hazard_decision):
        raise ValueError("inactive hazard decision must use the false sentinel")
    if torch.any((~selector_mask) & (selected_candidate != -1)):
        raise ValueError("continuation must use selected candidate -1")
    if torch.any(selector_mask):
        selected = selected_candidate[selector_mask]
        if torch.any((selected < 0) | (selected >= PROPOSAL_COUNT)):
            raise ValueError("selected candidate index is out of bounds")
        available = proposal_mask[selector_mask].gather(
            -1, selected.unsqueeze(-1)
        ).squeeze(-1)
        if not torch.all(available):
            raise ValueError("selected candidate is unavailable")

    hazard_logp = torch.zeros_like(hazard_logit)
    hazard_entropy = torch.zeros_like(hazard_logit)
    if torch.any(hazard_mask):
        logits = hazard_logit[hazard_mask]
        decisions = hazard_decision[hazard_mask]
        hazard_logp[hazard_mask] = torch.where(
            decisions, F.logsigmoid(logits), F.logsigmoid(-logits)
        )
        probability = torch.sigmoid(logits)
        hazard_entropy[hazard_mask] = -(
            probability * F.logsigmoid(logits)
            + (1.0 - probability) * F.logsigmoid(-logits)
        )

    selector_logp = torch.zeros_like(hazard_logit)
    selector_entropy = torch.zeros_like(hazard_logit)
    if torch.any(selector_mask):
        logits = selector_logits[selector_mask]
        available = proposal_mask[selector_mask]
        masked_logits = logits.masked_fill(~available, -torch.inf)
        log_probs = masked_logits - torch.logsumexp(
            masked_logits, dim=-1, keepdim=True
        )
        selected = selected_candidate[selector_mask]
        selector_logp[selector_mask] = log_probs.gather(
            -1, selected.unsqueeze(-1)
        ).squeeze(-1)
        probabilities = torch.softmax(masked_logits, dim=-1)
        safe_log_probs = torch.where(
            available, log_probs, torch.zeros_like(log_probs)
        )
        selector_entropy[selector_mask] = -(
            probabilities * safe_log_probs
        ).sum(dim=-1)
    return {
        "hazard_logp": hazard_logp,
        "selector_logp": selector_logp,
        "logp": hazard_logp + selector_logp,
        "hazard_entropy": hazard_entropy,
        "selector_entropy": selector_entropy,
    }


def generalized_advantage_estimate(
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    terminal: torch.Tensor,
    truncated: torch.Tensor,
    discount: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-tick GAE over [time,resident] without crossing boundaries."""

    if reward.ndim != 2 or any(
        item.shape != reward.shape
        for item in (value, next_value, terminal, truncated)
    ):
        raise ValueError("GAE tensors must share [time,resident] shape")
    if terminal.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("terminal and truncated must be bool")
    if torch.any(terminal & truncated):
        raise ValueError("a transition cannot be terminal and truncated")
    if not 0.0 <= discount <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("discount and GAE lambda must be in [0,1]")
    bootstrap = (~terminal).to(reward.dtype)
    delta = reward + discount * next_value * bootstrap - value
    advantage = torch.empty_like(delta)
    carry = torch.zeros_like(delta[-1])
    for index in range(delta.shape[0] - 1, -1, -1):
        continues = (~(terminal[index] | truncated[index])).to(reward.dtype)
        carry = delta[index] + discount * gae_lambda * continues * carry
        advantage[index] = carry
    return advantage, advantage + value


def ppo_loss(
    outputs: Mapping[str, torch.Tensor],
    terms: Mapping[str, torch.Tensor],
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    returns: torch.Tensor,
    actor_valid: torch.Tensor,
    hazard_mask: torch.Tensor,
    selector_mask: torch.Tensor,
    *,
    clip_ratio: float,
    value_coefficient: float,
    hazard_entropy_coefficient: float,
    selector_entropy_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 <= clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be in [0,1]")
    if actor_valid.dtype != torch.bool or actor_valid.shape != old_logp.shape:
        raise ValueError("actor_valid shape or dtype differs")
    if torch.any(actor_valid):
        ratio = torch.exp(terms["logp"][actor_valid] - old_logp[actor_valid])
        valid_advantage = advantage[actor_valid]
        unclipped = ratio * valid_advantage
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * valid_advantage
        actor = -torch.minimum(unclipped, clipped).mean()
    else:
        ratio = outputs["value"].new_empty((0,))
        actor = outputs["value"].new_zeros(())
    critic = F.mse_loss(outputs["value"], returns)

    valid_hazard = actor_valid & hazard_mask
    hazard_entropy = (
        terms["hazard_entropy"][valid_hazard].mean()
        if torch.any(valid_hazard)
        else outputs["value"].new_zeros(())
    )
    valid_selector = actor_valid & selector_mask
    selector_entropy = (
        terms["selector_entropy"][valid_selector].mean()
        if torch.any(valid_selector)
        else outputs["value"].new_zeros(())
    )
    total = (
        actor
        + value_coefficient * critic
        - hazard_entropy_coefficient * hazard_entropy
        - selector_entropy_coefficient * selector_entropy
    )
    with torch.no_grad():
        log_ratio = terms["logp"][actor_valid] - old_logp[actor_valid]
        metrics = {
            "loss": total.detach(),
            "actor_loss": actor.detach(),
            "value_loss": critic.detach(),
            "hazard_entropy": hazard_entropy.detach(),
            "selector_entropy": selector_entropy.detach(),
            "approximate_kl": (
                ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                if log_ratio.numel()
                else outputs["value"].new_zeros(())
            ),
            "clip_fraction": (
                (torch.abs(ratio - 1.0) > clip_ratio).to(torch.float32).mean()
                if ratio.numel()
                else outputs["value"].new_zeros(())
            ),
            "actor_rows": actor_valid.sum(),
            "hazard_rows": valid_hazard.sum(),
            "selector_rows": valid_selector.sum(),
        }
    return total, metrics


def parameter_arrays(model: SequenceControlHeads) -> dict[str, np.ndarray]:
    state = model.state_dict()
    if tuple(state) != PARAMETER_ORDER:
        raise RuntimeError("Torch sequence-control parameter order differs")
    return {
        name: np.ascontiguousarray(state[name].detach().cpu().numpy(), dtype="<f4")
        for name in PARAMETER_ORDER
    }


def initial_parameter_arrays(seed: int) -> dict[str, np.ndarray]:
    model = SequenceControlHeads()
    model.initialize(seed)
    return parameter_arrays(model)


def packed_parameters(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = []
    for name in PARAMETER_ORDER:
        value = np.ascontiguousarray(arrays[name], dtype="<f4")
        if value.shape != PARAMETER_SHAPES[name]:
            raise ValueError(f"sequence-control tensor shape differs: {name}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"sequence-control tensor is nonfinite: {name}")
        values.append(value.reshape(-1))
    result = np.concatenate(values).astype("<f4", copy=False)
    if result.shape != (PARAMETER_COUNT,):
        raise RuntimeError("packed sequence-control parameter count differs")
    return result


def load_parameter_arrays(
    model: SequenceControlHeads, arrays: Mapping[str, np.ndarray]
) -> None:
    expected = set(PARAMETER_ORDER)
    if set(arrays) != expected:
        raise ValueError("sequence-control parameter names differ")
    state = {}
    for name in PARAMETER_ORDER:
        value = np.ascontiguousarray(arrays[name], dtype="<f4")
        if value.shape != PARAMETER_SHAPES[name] or not np.all(np.isfinite(value)):
            raise ValueError(f"sequence-control tensor differs: {name}")
        state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)


@dataclass(frozen=True)
class NativeRollout:
    manifest: dict[str, Any]
    manifest_file_sha256: str
    arrays: dict[str, np.ndarray]

    @property
    def time_steps(self) -> int:
        return int(self.arrays["state"].shape[0])

    @property
    def residents(self) -> int:
        return int(self.arrays["state"].shape[1])


ROLLOUT_ARRAY_SPECS = {
    "state": (np.dtype("<f4"), (STATE_DIM,)),
    "proposal": (np.dtype("<f4"), (PROPOSAL_COUNT, CANDIDATE_DIM)),
    "active": (np.dtype("<f4"), (CANDIDATE_DIM,)),
    "proposal_mask": (np.dtype("|b1"), (PROPOSAL_COUNT,)),
    "active_mask": (np.dtype("|b1"), ()),
    "hazard_logit": (np.dtype("<f4"), ()),
    "selector_logits": (np.dtype("<f4"), (PROPOSAL_COUNT,)),
    "value": (np.dtype("<f4"), ()),
    "hazard_decision": (np.dtype("|b1"), ()),
    "selected_candidate": (np.dtype("<i4"), ()),
    "hazard_mask": (np.dtype("|b1"), ()),
    "selector_mask": (np.dtype("|b1"), ()),
    "behavior_hazard_logp": (np.dtype("<f4"), ()),
    "behavior_selector_logp": (np.dtype("<f4"), ()),
    "behavior_logp": (np.dtype("<f4"), ()),
    "proposed_action": (np.dtype("<f4"), (ACTION_DIM,)),
    "acknowledged_action": (np.dtype("<f4"), (ACTION_DIM,)),
    "before_physiology": (np.dtype("<f4"), (12,)),
    "after_physiology": (np.dtype("<f4"), (12,)),
    "reward_components": (np.dtype("<f4"), None),
    "reward": (np.dtype("<f4"), ()),
    "next_state": (np.dtype("<f4"), (STATE_DIM,)),
    "next_proposal": (np.dtype("<f4"), (PROPOSAL_COUNT, CANDIDATE_DIM)),
    "next_active": (np.dtype("<f4"), (CANDIDATE_DIM,)),
    "next_proposal_mask": (np.dtype("|b1"), (PROPOSAL_COUNT,)),
    "next_active_mask": (np.dtype("|b1"), ()),
    "next_value": (np.dtype("<f4"), ()),
    "terminal": (np.dtype("|b1"), ()),
    "truncated": (np.dtype("|b1"), ()),
    "actor_valid": (np.dtype("|b1"), ()),
    "acknowledged": (np.dtype("|b1"), ()),
    "cancellation_reason": (np.dtype("|u1"), ()),
    "active_source_slot": (np.dtype("<i4"), ()),
    "active_source_generation": (np.dtype("<u8"), ()),
    "active_phase": (np.dtype("|u1"), ()),
    "active_remaining": (np.dtype("|u1"), ()),
    "policy_version": (np.dtype("<u8"), ()),
    "policy_sha256": (np.dtype("|u1"), (32,)),
    "resident": (np.dtype("<i4"), ()),
    "episode": (np.dtype("<i4"), ()),
    "tick": (np.dtype("<i8"), ()),
}


def load_rollout(path: Path) -> NativeRollout:
    if ACTIVE_ROLLOUT_CONTRACT is None:
        raise RuntimeError(
            "native sequence-control rollout is disabled pending the CNS-derived contract"
        )
    path = path.resolve()
    manifest_path = path / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    body = dict(manifest)
    content_sha256 = body.pop("content_sha256", None)
    if (
        manifest.get("format") != ROLLOUT_FORMAT
        or manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("contract_sha256") != CONTRACT_SHA256
        or manifest.get("completed") is not True
        or content_sha256 != canonical_sha256(body)
    ):
        raise ValueError("sequence-control rollout manifest identity differs")
    if tuple(manifest.get("cancellation_reason_order", ())) != CANCELLATION_REASONS:
        raise ValueError("rollout cancellation reason order differs")
    reward_contract = manifest.get("reward", {})
    component_order = reward_contract.get("component_order")
    weights = reward_contract.get("weights")
    if (
        not isinstance(component_order, list)
        or not component_order
        or not isinstance(weights, list)
        or len(weights) != len(component_order)
        or not all(isinstance(name, str) and name for name in component_order)
        or not np.all(np.isfinite(np.asarray(weights, dtype=np.float64)))
        or not 0.0 <= float(reward_contract.get("discount", -1)) <= 1.0
        or not 0.0 <= float(reward_contract.get("gae_lambda", -1)) <= 1.0
    ):
        raise ValueError("rollout reward contract differs")
    boundary = manifest.get("boundary", {})
    if (
        boundary.get("coherent") is not True
        or boundary.get("pending_unacknowledged_action") is not False
        or boundary.get("policy_version_acknowledged") is not True
        or not valid_sha256(boundary.get("before_checkpoint_sha256"))
        or not valid_sha256(boundary.get("after_checkpoint_sha256"))
    ):
        raise ValueError("rollout is not enclosed by coherent acknowledged boundaries")
    packets = manifest.get("packets")
    if not isinstance(packets, list) or not packets:
        raise ValueError("sequence-control rollout packets are missing")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ROLLOUT_ARRAY_SPECS}
    expected_start = 0
    residents = int(manifest.get("scope", {}).get("residents", 0))
    for receipt in packets:
        start = int(receipt.get("start_step", -1))
        stop = int(receipt.get("stop_step", -1))
        packet_path = (path / str(receipt.get("path"))).resolve()
        if (
            start != expected_start
            or stop <= start
            or not packet_path.is_relative_to(path)
            or not packet_path.is_file()
            or packet_path.stat().st_size != int(receipt.get("bytes", -1))
            or file_sha256(packet_path) != receipt.get("sha256")
        ):
            raise ValueError("sequence-control rollout packet receipt differs")
        with np.load(packet_path, allow_pickle=False) as bundle:
            if set(bundle.files) != set(ROLLOUT_ARRAY_SPECS):
                raise ValueError("sequence-control rollout packet fields differ")
            for name, (dtype, tail) in ROLLOUT_ARRAY_SPECS.items():
                value = np.ascontiguousarray(bundle[name])
                expected_tail = (
                    (len(component_order),) if name == "reward_components" else tail
                )
                if (
                    value.dtype != dtype
                    or value.shape[:2] != (stop - start, residents)
                    or value.shape[2:] != expected_tail
                    or (
                        value.dtype.kind == "f" and not np.all(np.isfinite(value))
                    )
                ):
                    raise ValueError(f"rollout packet tensor differs: {name}")
                chunks[name].append(value)
        expected_start = stop
    scope = manifest["scope"]
    if expected_start != int(scope.get("steps", -1)) or residents < 1:
        raise ValueError("rollout packet coverage differs")
    arrays = {
        name: np.concatenate(values, axis=0) for name, values in chunks.items()
    }
    _validate_rollout_semantics(
        arrays,
        np.asarray(weights, dtype=np.float32),
        manifest.get("policy", {}),
    )
    return NativeRollout(
        manifest=manifest,
        manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        arrays=arrays,
    )


def _validate_rollout_semantics(
    arrays: Mapping[str, np.ndarray],
    reward_weights: np.ndarray,
    policy: Mapping[str, Any],
) -> None:
    if not np.all(arrays["acknowledged"]):
        raise ValueError("pending native decisions cannot enter a rollout")
    if np.any(arrays["terminal"] & arrays["truncated"]):
        raise ValueError("terminal and truncation flags overlap")
    if not np.array_equal(arrays["hazard_mask"], arrays["active_mask"]):
        raise ValueError("recorded hazard mask differs from active availability")
    expected_selector = (~arrays["active_mask"]) | arrays["hazard_decision"]
    if not np.array_equal(arrays["selector_mask"], expected_selector):
        raise ValueError("recorded selector mask differs from hierarchy")
    if np.any((~arrays["hazard_mask"]) & arrays["hazard_decision"]):
        raise ValueError("inactive hazard uses a non-sentinel decision")
    selected = arrays["selected_candidate"]
    if np.any((~arrays["selector_mask"]) & (selected != -1)):
        raise ValueError("continuation candidate sentinel differs")
    select_rows = arrays["selector_mask"]
    chosen = selected[select_rows]
    if np.any((chosen < 0) | (chosen >= PROPOSAL_COUNT)):
        raise ValueError("recorded selector choice is out of bounds")
    masks = arrays["proposal_mask"][select_rows]
    if chosen.size and not np.all(masks[np.arange(chosen.size), chosen]):
        raise ValueError("recorded selector chose an unavailable proposal")
    if not np.all(arrays["proposal_mask"].any(axis=-1)) or not np.all(
        arrays["next_proposal_mask"].any(axis=-1)
    ):
        raise ValueError("rollout contains an empty proposal set")
    if not np.all(arrays["proposal_mask"][..., :4]) or not np.all(
        arrays["next_proposal_mask"][..., :4]
    ):
        raise ValueError("four local proposals must always be available")
    for prefix in ("", "next_"):
        state = arrays[f"{prefix}state"]
        proposal = arrays[f"{prefix}proposal"]
        active = arrays[f"{prefix}active"]
        proposal_mask = arrays[f"{prefix}proposal_mask"]
        active_mask = arrays[f"{prefix}active_mask"]
        if not np.array_equal(state[..., -1], active_mask.astype(np.float32)):
            raise ValueError(f"{prefix}state active feature differs from active_mask")
        if np.any(proposal[~proposal_mask] != 0.0):
            raise ValueError(f"{prefix}unavailable proposal features must be zero")
        if np.any(active[~active_mask] != 0.0):
            raise ValueError(f"{prefix}absent active features must be zero")
        flags = proposal[..., (99, 103)]
        if not np.all(np.isin(flags[proposal_mask], (0.0, 1.0))):
            raise ValueError(f"{prefix}proposal validity features differ")
        bounded = proposal[..., (96, 97, 98)]
        if np.any((bounded[proposal_mask] < 0.0) | (bounded[proposal_mask] > 1.0)):
            raise ValueError(f"{prefix}proposal duration features are out of bounds")
        cosine = proposal[..., 104]
        disagreement = proposal[..., 105]
        if np.any((cosine[proposal_mask] < -1.0) | (cosine[proposal_mask] > 1.0)) or np.any(
            disagreement[proposal_mask] < 0.0
        ):
            raise ValueError(f"{prefix}proposal goal features are out of bounds")
    components = arrays["reward_components"]
    reconstructed = components @ reward_weights
    if not np.allclose(reconstructed, arrays["reward"], rtol=2e-6, atol=2e-6):
        raise ValueError("recorded scalar reward differs from declared components")
    component_sum = (
        arrays["behavior_hazard_logp"] + arrays["behavior_selector_logp"]
    )
    if not np.allclose(component_sum, arrays["behavior_logp"], rtol=2e-6, atol=2e-6):
        raise ValueError("recorded behavior log-probability components differ")
    if np.any((arrays["cancellation_reason"] >= len(CANCELLATION_REASONS))):
        raise ValueError("rollout cancellation reason is unknown")
    if np.any((arrays["cancellation_reason"] >= 2) & arrays["actor_valid"]):
        raise ValueError("override/censor rows cannot be actor-valid")
    policy_identity = policy.get("artifact_sha256")
    policy_version = policy.get("version")
    if not valid_sha256(policy_identity) or type(policy_version) is not int:
        raise ValueError("rollout policy identity differs")
    expected_digest = np.frombuffer(bytes.fromhex(policy_identity), dtype=np.uint8)
    if not np.all(arrays["policy_version"] == policy_version) or not np.all(
        arrays["policy_sha256"] == expected_digest
    ):
        raise ValueError("per-decision policy identity differs from manifest")
    active_mask = arrays["active_mask"]
    if (
        np.any(arrays["active_source_slot"][~active_mask] != -1)
        or np.any(arrays["active_source_generation"][~active_mask] != 0)
        or np.any(arrays["active_phase"][~active_mask] != 0)
        or np.any(arrays["active_remaining"][~active_mask] != 0)
        or np.any(arrays["active_source_slot"][active_mask] < 0)
        or np.any(arrays["active_remaining"][active_mask] < 1)
        or np.any(arrays["active_remaining"][active_mask] > 7)
    ):
        raise ValueError("active execution attribution metadata differs")
    if np.any(
        np.abs(
            arrays["active"][..., 96]
            - arrays["active_remaining"].astype(np.float32) / 8.0
        )
        > 1e-6
    ) or np.any(
        np.abs(
            arrays["active"][..., 97]
            - arrays["active_phase"].astype(np.float32) / 8.0
        )
        > 1e-6
    ):
        raise ValueError("active execution metadata differs from policy features")
    resident = arrays["resident"]
    if np.any(resident < 0) or not np.all(resident == resident[0:1]):
        raise ValueError("rollout resident columns are not stable")
    if arrays["tick"].shape[0] > 1:
        same_episode = arrays["episode"][1:] == arrays["episode"][:-1]
        tick_delta = arrays["tick"][1:] - arrays["tick"][:-1]
        if np.any(same_episode & (tick_delta != 1)):
            raise ValueError("rollout has a gap within an episode")
        episode_advance = arrays["episode"][1:] - arrays["episode"][:-1]
        if np.any((~same_episode) & (episode_advance != 1)) or np.any(
            (~same_episode) & ~(arrays["terminal"][:-1] | arrays["truncated"][:-1])
        ):
            raise ValueError("rollout episode boundary differs")


def tensor_rollout(
    rollout: NativeRollout, device: torch.device
) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in rollout.arrays.items():
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.int32:
            tensor = tensor.to(torch.long)
        result[name] = tensor.to(device)
    return result


def replay_error(
    model: SequenceControlHeads,
    tensors: Mapping[str, torch.Tensor],
    batch_size: int = 4096,
) -> dict[str, float]:
    """Compare an archived native rollout against its immutable parent heads."""

    flat = {name: value.flatten(0, 1) for name, value in tensors.items()}
    maxima = {
        "hazard_logit": 0.0,
        "selector_logits": 0.0,
        "value": 0.0,
        "hazard_logp": 0.0,
        "selector_logp": 0.0,
        "logp": 0.0,
        "next_value": 0.0,
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, flat["state"].shape[0], batch_size):
            rows = slice(start, start + batch_size)
            outputs = model(
                flat["state"][rows],
                flat["proposal"][rows],
                flat["active"][rows],
                flat["proposal_mask"][rows],
                flat["active_mask"][rows],
            )
            terms = hierarchical_terms(
                outputs,
                flat["proposal_mask"][rows],
                flat["active_mask"][rows],
                flat["hazard_decision"][rows],
                flat["selected_candidate"][rows],
                flat["hazard_mask"][rows],
                flat["selector_mask"][rows],
            )
            expected = {
                "hazard_logit": flat["hazard_logit"][rows],
                "selector_logits": flat["selector_logits"][rows],
                "value": flat["value"][rows],
                "hazard_logp": flat["behavior_hazard_logp"][rows],
                "selector_logp": flat["behavior_selector_logp"][rows],
                "logp": flat["behavior_logp"][rows],
            }
            next_outputs = model(
                flat["next_state"][rows],
                flat["next_proposal"][rows],
                flat["next_active"][rows],
                flat["next_proposal_mask"][rows],
                flat["next_active_mask"][rows],
            )
            for name in expected:
                error = torch.max(torch.abs((outputs | terms)[name] - expected[name]))
                maxima[name] = max(maxima[name], float(error.cpu()))
            next_error = torch.max(
                torch.abs(next_outputs["value"] - flat["next_value"][rows])
            )
            maxima["next_value"] = max(
                maxima["next_value"], float(next_error.cpu())
            )
    return maxima
