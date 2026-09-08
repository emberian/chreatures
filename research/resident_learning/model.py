"""Differentiable mirror of the canonical packed native resident tensors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from chreatures.sequence_control import CORE_ORDER, PREDICTOR_ORDER
from research.sensorimotor_skills.sequence_control import SequenceControlHeads

Z, ACTIONS, HIDDEN, GOAL = 512, 12, 256, 128  # ACTIONS is legacy tensor-axis naming: context12.
LOCAL, CANDIDATES, MAX_HORIZON = 4, 8, 8
CONTEXT, STATE, CHOICE = 908, 909, 234


class PredictorMember(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = nn.Linear(CONTEXT, HIDDEN)
        self.transition = nn.GRUCell(ACTIONS, HIDDEN)
        self.delta = nn.Linear(HIDDEN, Z)


class CnsResidentModel(nn.Module):
    """The actual Rust resident parameterization, with no extra actor head."""

    def __init__(self) -> None:
        super().__init__()
        self.core = nn.GRUCell(Z + ACTIONS, HIDDEN)
        self.goal_encoder = nn.Linear(Z + HIDDEN, GOAL)
        self.proposal_hidden = nn.Linear(HIDDEN + GOAL + ACTIONS, HIDDEN)
        self.proposal_out = nn.Linear(HIDDEN, LOCAL * ACTIONS)
        self.predictor = nn.ModuleList(PredictorMember() for _ in range(3))
        self.sequence_control = SequenceControlHeads()

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], device: torch.device) -> "CnsResidentModel":
        model = cls().to(device)
        mapped: dict[str, torch.Tensor] = {}
        for name in CORE_ORDER:
            target = "core." + name.removeprefix("core.") if name.startswith("core.") else name
            mapped[target] = torch.as_tensor(arrays[name], device=device)
        for name in PREDICTOR_ORDER:
            _, member, rest = name.split(".", 2)
            mapped[f"predictor.{member}.{rest}"] = torch.as_tensor(arrays[name], device=device)
        prefix = "sequence_control."
        for name, parameter in model.sequence_control.state_dict().items():
            mapped[prefix + name] = torch.as_tensor(arrays[prefix + name], device=device)
        model.load_state_dict(mapped, strict=True)
        return model

    def arrays(self) -> dict[str, np.ndarray]:
        state = self.state_dict()
        result: dict[str, np.ndarray] = {}
        for name in CORE_ORDER:
            source = "core." + name.removeprefix("core.") if name.startswith("core.") else name
            result[name] = np.ascontiguousarray(state[source].detach().cpu().numpy(), dtype=np.float32)
        for name in PREDICTOR_ORDER:
            _, member, rest = name.split(".", 2)
            result[name] = np.ascontiguousarray(state[f"predictor.{member}.{rest}"].detach().cpu().numpy(), dtype=np.float32)
        for name, value in self.sequence_control.state_dict().items():
            result["sequence_control." + name] = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        return result

    @staticmethod
    def bounded_actions(raw: torch.Tensor) -> torch.Tensor:
        """Bound every neural context-current axis symmetrically."""
        return raw.tanh()

    def observe(
        self,
        latent: torch.Tensor,
        previous: torch.Tensor,
        reset: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.where(reset[:, None], torch.zeros_like(state), state)
        previous = torch.where(reset[:, None], torch.zeros_like(previous), previous)
        state = self.core(torch.cat((latent, previous), -1), state)
        key = F.normalize(torch.tanh(self.goal_encoder(torch.cat((latent, state), -1))), dim=-1, eps=1e-8)
        return state, key

    def local_actions(self, state: torch.Tensor, goal: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.proposal_hidden(torch.cat((state, goal, previous), -1)))
        raw = self.proposal_out(hidden).reshape(*state.shape[:-1], LOCAL, ACTIONS)
        return self.bounded_actions(raw)

    def unroll(
        self,
        latent: torch.Tensor,
        previous: torch.Tensor,
        reset: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if latent.ndim != 3 or latent.shape[-1] != Z or previous.shape != (*latent.shape[:-1], ACTIONS):
            raise ValueError("resident unroll requires [time,batch,Z512] and previous context12")
        state = latent.new_zeros((latent.shape[1], HIDDEN)) if initial_state is None else initial_state
        if state.shape != (latent.shape[1], HIDDEN):
            raise ValueError("resident initial recurrent state differs")
        states, keys, proposals = [], [], []
        for t in range(latent.shape[0]):
            state, key = self.observe(latent[t], previous[t], reset[t], state)
            states.append(state)
            keys.append(key)
            proposals.append(self.local_actions(state, key, previous[t]))
        return {"state": torch.stack(states), "key": torch.stack(keys), "local": torch.stack(proposals)}

    def predict(self, context: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rust-matching ensemble forecast for action suffixes [N,H,12]."""
        base_z, base_h = context[:, :Z], context[:, Z : Z + HIDDEN]
        goals, final_z = [], []
        for member in self.predictor:
            pred_state = torch.tanh(member.context(context))
            core_state, pred_z = base_h, base_z
            for step in range(actions.shape[1]):
                pred_state = member.transition(actions[:, step], pred_state)
                pred_z = base_z + member.delta(pred_state)
                core_state = self.core(torch.cat((pred_z, actions[:, step]), -1), core_state)
            goals.append(F.normalize(torch.tanh(self.goal_encoder(torch.cat((pred_z, core_state), -1))), dim=-1, eps=1e-8))
            final_z.append(pred_z)
        return torch.stack(final_z, 1), torch.stack(goals, 1)

    def control_features(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
        previous: torch.Tensor,
        local: torch.Tensor,
        demonstrated_suffix: torch.Tensor,
        suffix_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build exact native head shapes with one experienced suffix candidate."""
        n = latent.shape[0]
        context = torch.cat((latent, state, goal, previous), -1)
        proposal = latent.new_zeros((n, CANDIDATES, CHOICE))
        mask = torch.zeros((n, CANDIDATES), dtype=torch.bool, device=latent.device)
        for k in range(LOCAL):
            sequence = local[:, k, None, :].expand(n, MAX_HORIZON, ACTIONS)
            _, forecast_goal = self.predict(context, sequence)
            mean = F.normalize(forecast_goal.mean(1), dim=-1, eps=1e-8)
            proposal[:, k, :96] = sequence.reshape(n, 96)
            proposal[:, k, 96] = 1.0 / 8.0
            proposal[:, k, 98] = 1.0 / 8.0
            proposal[:, k, 103] = 1.0
            proposal[:, k, 104] = (mean * goal).sum(-1)
            proposal[:, k, 105] = ((forecast_goal - mean[:, None]) ** 2).mean((1, 2)).sqrt()
            proposal[:, k, 106:] = mean
            mask[:, k] = True
        _, suffix_goal = self.predict(context, demonstrated_suffix)
        mean = F.normalize(suffix_goal.mean(1), dim=-1, eps=1e-8)
        if demonstrated_suffix.shape[1] < MAX_HORIZON:
            padding = demonstrated_suffix[:, -1:].expand(
                n, MAX_HORIZON - demonstrated_suffix.shape[1], ACTIONS
            )
            packed_suffix = torch.cat((demonstrated_suffix, padding), dim=1)
        else:
            packed_suffix = demonstrated_suffix
        proposal[:, LOCAL, :96] = packed_suffix.reshape(n, 96)
        proposal[:, LOCAL, 96] = suffix_length.float() / 8.0
        proposal[:, LOCAL, 98] = suffix_length.float() / 8.0
        proposal[:, LOCAL, 99] = 1.0
        proposal[:, LOCAL, 100] = math.log(2.0)
        # Physical return is a target for selector/value optimization, never a
        # controller feature. Rust will populate this position only from the
        # resident's own CNS-goal progress after executing a remembered suffix.
        proposal[:, LOCAL, 102] = 0.0
        proposal[:, LOCAL, 103] = 1.0
        proposal[:, LOCAL, 104] = (mean * goal).sum(-1)
        proposal[:, LOCAL, 105] = ((suffix_goal - mean[:, None]) ** 2).mean((1, 2)).sqrt()
        proposal[:, LOCAL, 106:] = mean
        mask[:, LOCAL] = suffix_length > 1
        active = proposal[:, LOCAL].clone()
        active_mask = suffix_length > 1
        control_state = torch.cat((context, active_mask[:, None].to(latent.dtype)), -1)
        return control_state, proposal, mask, active, active_mask


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    action: torch.Tensor
    prediction: torch.Tensor
    memory: torch.Tensor
    selector: torch.Tensor
    hazard: torch.Tensor
    value: torch.Tensor


def training_loss(model: CnsResidentModel, batch: Mapping[str, torch.Tensor], discount: float = 0.97) -> LossTerms:
    latent, previous, reset = batch["cns_latent"], batch["previous_delivered_context"], batch["reset"]
    delivered, reward = batch["delivered_context"], batch["physical_reward"]
    burn_in = int(batch.get("burn_in", 0))
    if burn_in < 0 or burn_in >= delivered.shape[0]:
        raise ValueError("resident burn-in leaves no optimized transitions")
    initial_state = None
    if burn_in:
        # Reconstruct recent private recurrent context from the actual CNS and
        # delivered-action chronology.  The state is detached at the target
        # window boundary: burn-in supplies context, never an extra loss path.
        with torch.no_grad():
            prefix = model.unroll(latent[:burn_in], previous[:burn_in], reset[:burn_in])
            initial_state = prefix["state"][-1].detach()
        latent = latent[burn_in:]
        previous = previous[burn_in:]
        reset = reset[burn_in:]
        delivered = delivered[burn_in:]
        reward = reward[burn_in:]
    result = model.unroll(latent, previous, reset, initial_state)
    states, keys = result["state"][:-1], result["key"]
    t, b = delivered.shape[:2]
    goal_horizon = min(MAX_HORIZON, t)
    # The motor inverse is conditioned on a CNS key actually achieved later in
    # this physical trajectory. Deployment can retrieve that same kind of key
    # only from the resident's own achieved-history reservoir.
    achieved_goal = torch.cat(
        (result["key"][goal_horizon : t + 1], result["key"][-1:].expand(goal_horizon - 1, -1, -1)),
        dim=0,
    )
    local = model.local_actions(states, achieved_goal, previous[:-1])
    action_error = ((local - delivered[:, :, None]) ** 2).mean(-1)
    action_loss = ((math.log(LOCAL) - torch.logsumexp(-12.0 * action_error, -1)) / 12.0).mean()

    horizon = min(MAX_HORIZON, t)
    starts = t - horizon + 1
    suffixes = []
    utilities = []
    lengths = []
    for step in range(starts):
        seq = delivered[step : step + horizon].permute(1, 0, 2)
        suffixes.append(seq)
        weights = reward.new_tensor([discount**i for i in range(horizon)])[:, None]
        utilities.append((reward[step : step + horizon] * weights).sum(0))
        lengths.append(torch.full((b,), horizon, dtype=torch.long, device=latent.device))
    suffix = torch.cat(suffixes)
    utility = torch.cat(utilities)
    length = torch.cat(lengths)
    # Fit every rollout depth.  Training only the eighth step allowed the
    # recurrent transition to trade short-horizon accuracy for one endpoint,
    # which made the first campaign's held-out predictor worse.  Candidate
    # construction below is detached so selector gradients cannot rewrite the
    # dynamics model to create an easier classification feature.
    prediction_terms = []
    for depth in range(1, horizon + 1):
        depth_starts = t - depth + 1
        depth_suffix = torch.cat([
            delivered[step : step + depth].permute(1, 0, 2)
            for step in range(depth_starts)
        ])
        # Runtime prediction receives a goal retrieved from the resident's past
        # achieved-key reservoir.  Use the current experienced key here; the
        # future achieved key remains a valid hindsight target only for inverse
        # action fitting above and must not shortcut its own latent forecast.
        depth_context = torch.cat([
            torch.cat((latent[step], states[step], keys[step], previous[step]), -1)
            for step in range(depth_starts)
        ]).detach()
        depth_target = torch.cat([latent[step + depth] for step in range(depth_starts)]).detach()
        depth_predicted, _ = model.predict(depth_context, depth_suffix)
        prediction_terms.append(F.smooth_l1_loss(
            depth_predicted, depth_target[:, None].expand_as(depth_predicted), beta=0.02
        ))
    prediction_loss = torch.stack(prediction_terms).mean()

    # Future keys from the same embodied stream are positives; other residents
    # in the batch are negatives. Rewarded changes strengthen the association.
    query = keys[:starts].reshape(starts * b, GOAL)
    positive = keys[horizon : horizon + starts].reshape(starts * b, GOAL)
    logits = query @ positive.T / 0.12
    labels = torch.arange(logits.shape[0], device=latent.device)
    weights = (1.0 + utility.detach().abs()).clamp_max(4.0)
    memory_loss = (F.cross_entropy(logits, labels, reduction="none") * weights).mean()

    start_local = local[:starts].reshape(starts * b, LOCAL, ACTIONS)
    start_latent = latent[:starts].reshape(starts * b, Z)
    start_state = states[:starts].reshape(starts * b, HIDDEN)
    start_goal = achieved_goal[:starts].reshape(starts * b, GOAL)
    start_previous = previous[:starts].reshape(starts * b, ACTIONS)
    control_state, proposals, proposal_mask, active, active_mask = model.control_features(
        start_latent, start_state, start_goal, start_previous, start_local,
        suffix, length,
    )
    outputs = model.sequence_control(
        control_state.detach(), proposals.detach(), active.detach(), proposal_mask, active_mask
    )
    nearest = action_error[:starts].reshape(starts * b, LOCAL).argmin(-1)
    selector_target = torch.where(utility > 0.02, torch.full_like(nearest, LOCAL), nearest)
    selector_logits = outputs["selector_logits"].masked_fill(~proposal_mask, -torch.inf)
    selector_weight = (1.0 + utility.detach().abs()).clamp_max(4.0)
    selector_loss = (
        F.cross_entropy(selector_logits, selector_target, reduction="none") * selector_weight
    ).sum() / selector_weight.sum()

    # Positive active examples replay the suffix that was actually useful in
    # this context.  Rolled examples represent recall of an experienced suffix
    # in another resident/time context; disagreement with the current teacher
    # delivered context current is the termination target. This supplies both sides of the
    # hazard decision instead of the old self-comparison, which was always zero.
    rolled_suffix = torch.roll(suffix, shifts=max(1, b), dims=0)
    _, rolled_proposals, rolled_mask, rolled_active, rolled_active_mask = model.control_features(
        start_latent, start_state, start_goal, start_previous, start_local,
        rolled_suffix, length,
    )
    rolled_outputs = model.sequence_control(
        control_state.detach(), rolled_proposals.detach(), rolled_active.detach(),
        rolled_mask, rolled_active_mask,
    )
    own_hazard_target = utility < -0.02
    break_error = ((rolled_suffix[:, 0] - delivered[:starts].reshape(starts * b, ACTIONS)) ** 2).mean(-1)
    rolled_hazard_target = (break_error > 0.02) | (reward[:starts].reshape(starts * b) < -0.02)
    hazard_logits = torch.cat((outputs["hazard_logit"], rolled_outputs["hazard_logit"]))
    hazard_target = torch.cat((own_hazard_target, rolled_hazard_target)).float()
    positive = hazard_target.sum().clamp_min(1.0)
    negative = (hazard_target.numel() - hazard_target.sum()).clamp_min(1.0)
    hazard_weight = torch.where(hazard_target.bool(), 0.5 / positive, 0.5 / negative)
    hazard_loss = (
        F.binary_cross_entropy_with_logits(hazard_logits, hazard_target, reduction="none")
        * hazard_weight
    ).sum()
    value_target = utility.detach().clamp(-4.0, 4.0)
    value_loss = F.smooth_l1_loss(outputs["value"], value_target)
    total = action_loss + prediction_loss * 2.0 + memory_loss * 0.05 + selector_loss * 0.25 + hazard_loss * 0.15 + value_loss * 0.5
    return LossTerms(total, action_loss, prediction_loss, memory_loss, selector_loss, hazard_loss, value_loss)
