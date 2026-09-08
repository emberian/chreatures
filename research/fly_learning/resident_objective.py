"""Fly-timing objective for the canonical CNS-only private resident.

The runtime model remains :class:`CnsResidentModel`.  This module changes only
the offline chronology: achieved goals are 0.4 seconds (40 control ticks) in
the future while an acquired context suffix remains at most eight ticks.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from research.resident_learning.model import (
    ACTIONS,
    GOAL,
    HIDDEN,
    LOCAL,
    MAX_HORIZON,
    Z,
    CnsResidentModel,
    LossTerms,
)

CONTROL_DT_S = 0.01
GOAL_HORIZON_S = 0.4
GOAL_HORIZON = 40


def training_loss(
    model: CnsResidentModel,
    batch: Mapping[str, torch.Tensor],
    *,
    discount: float = 0.97,
) -> LossTerms:
    """Fit inverse context, dynamics, memory and selection from CNS histories.

    Privileged physical reward and consequence summaries appear only as loss
    targets/private experienced statistics.  Predictor inputs contain the
    current achieved key, never the future key they are asked to predict.
    """
    latent = batch["cns_latent"]
    previous = batch["previous_delivered_context"]
    reset = batch["reset"]
    delivered = batch["delivered_context"]
    reward = batch["physical_reward"]
    burn_in = int(batch.get("burn_in", 0))
    if latent.ndim != 3 or latent.shape[-1] != Z or delivered.shape[-1] != ACTIONS:
        raise ValueError("fly resident objective requires Z512/context12 chronologies")
    if burn_in < 0 or delivered.shape[0] - burn_in < GOAL_HORIZON:
        raise ValueError("optimized chronology must cover the 0.4-second achieved-goal horizon")

    initial_state = None
    if burn_in:
        with torch.no_grad():
            prefix = model.unroll(latent[:burn_in], previous[:burn_in], reset[:burn_in])
            initial_state = prefix["state"][-1].detach()
        latent = latent[burn_in:]
        previous = previous[burn_in:]
        reset = reset[burn_in:]
        delivered = delivered[burn_in:]
        reward = reward[burn_in:]
        consequence_count = batch["consequence_count"][burn_in:]
        consequence_utility = batch["consequence_utility"][burn_in:]
        consequence_uncertainty = batch["consequence_uncertainty"][burn_in:]
    else:
        consequence_count = batch["consequence_count"]
        consequence_utility = batch["consequence_utility"]
        consequence_uncertainty = batch["consequence_uncertainty"]

    result = model.unroll(latent, previous, reset, initial_state)
    states = result["state"][:-1]
    keys = result["key"]
    ticks, residents = delivered.shape[:2]

    goal_index = torch.arange(ticks, device=latent.device) + GOAL_HORIZON
    goal_index.clamp_(max=ticks)
    achieved_goal = keys[goal_index]
    local = model.local_actions(states, achieved_goal, previous[:-1])
    action_error = ((local - delivered[:, :, None]) ** 2).mean(-1)
    action_loss = (
        (math.log(LOCAL) - torch.logsumexp(-12.0 * action_error, -1)) / 12.0
    ).mean()

    prediction_terms = []
    for depth in range(1, MAX_HORIZON + 1):
        starts = ticks - depth + 1
        suffix = torch.cat(
            [delivered[step : step + depth].permute(1, 0, 2) for step in range(starts)]
        )
        # This key is current/past experience.  The future key remains solely a
        # prediction target, preventing hindsight leakage into dynamics.
        context = torch.cat(
            [
                torch.cat((latent[step], states[step], keys[step], previous[step]), -1)
                for step in range(starts)
            ]
        ).detach()
        target = torch.cat([latent[step + depth] for step in range(starts)]).detach()
        predicted, _ = model.predict(context, suffix)
        prediction_terms.append(
            F.smooth_l1_loss(
                predicted, target[:, None].expand_as(predicted), beta=0.02
            )
        )
    prediction_loss = torch.stack(prediction_terms).mean()

    memory_starts = ticks - GOAL_HORIZON + 1
    query = keys[:memory_starts].reshape(memory_starts * residents, GOAL)
    positive_goal = keys[GOAL_HORIZON:].reshape(memory_starts * residents, GOAL)
    logits = query @ positive_goal.T / 0.12
    labels = torch.arange(logits.shape[0], device=latent.device)
    goal_return = []
    goal_weights = reward.new_tensor([discount**i for i in range(GOAL_HORIZON)])[:, None]
    for step in range(memory_starts):
        goal_return.append((reward[step : step + GOAL_HORIZON] * goal_weights).sum(0))
    goal_return = torch.cat(goal_return)
    memory_weight = (1.0 + goal_return.detach().abs()).clamp_max(4.0)
    memory_loss = (
        F.cross_entropy(logits, labels, reduction="none") * memory_weight
    ).sum() / memory_weight.sum()

    suffix_starts = ticks - MAX_HORIZON + 1
    suffix = torch.cat(
        [delivered[step : step + MAX_HORIZON].permute(1, 0, 2) for step in range(suffix_starts)]
    )
    suffix_weights = reward.new_tensor([discount**i for i in range(MAX_HORIZON)])[:, None]
    observed_utility = torch.cat(
        [
            (reward[step : step + MAX_HORIZON] * suffix_weights).sum(0)
            for step in range(suffix_starts)
        ]
    )
    length = torch.full(
        (suffix_starts * residents,), MAX_HORIZON,
        dtype=torch.long, device=latent.device,
    )
    start_local = local[:suffix_starts].reshape(-1, LOCAL, ACTIONS)
    start_latent = latent[:suffix_starts].reshape(-1, Z)
    start_state = states[:suffix_starts].reshape(-1, HIDDEN)
    start_key = keys[:suffix_starts].reshape(-1, GOAL)
    start_goal = achieved_goal[:suffix_starts].reshape(-1, GOAL)
    start_previous = previous[:suffix_starts].reshape(-1, ACTIONS)
    endpoint = keys[MAX_HORIZON:].reshape(-1, GOAL)
    count = consequence_count[:suffix_starts].reshape(-1).clamp_min(1)
    lcb = consequence_utility[:suffix_starts].reshape(-1).clamp(-4, 4).tanh()
    empirical_uncertainty = consequence_uncertainty[:suffix_starts].reshape(-1).clamp_min(0)
    control_state, proposals, proposal_mask, active, active_mask = model.control_features(
        start_latent,
        start_state,
        start_key,
        start_goal,
        start_previous,
        start_local,
        suffix,
        length,
        endpoint,
        count,
        lcb,
        empirical_uncertainty,
        tick_seconds=CONTROL_DT_S,
    )
    one_tick = CONTROL_DT_S / GOAL_HORIZON_S
    suffix_duration = MAX_HORIZON * one_tick
    proposals[:, :LOCAL, 97] = one_tick
    proposals[:, LOCAL, 96] = suffix_duration
    proposals[:, LOCAL, 97] = one_tick
    proposals[:, LOCAL, 98] = suffix_duration
    proposals[:, LOCAL, 101] = 1.0
    active = proposals[:, LOCAL].clone()
    outputs = model.sequence_control(
        control_state.detach(), proposals.detach(), active.detach(), proposal_mask, active_mask
    )
    nearest = action_error[:suffix_starts].reshape(-1, LOCAL).argmin(-1)
    selector_target = torch.where(observed_utility > 0.02, torch.full_like(nearest, LOCAL), nearest)
    selector_logits = outputs["selector_logits"].masked_fill(~proposal_mask, -torch.inf)
    selector_weight = (1.0 + observed_utility.detach().abs()).clamp_max(4.0)
    selector_loss = (
        F.cross_entropy(selector_logits, selector_target, reduction="none") * selector_weight
    ).sum() / selector_weight.sum()

    rolled_suffix = torch.roll(suffix, shifts=max(1, residents), dims=0)
    _, rolled_proposals, rolled_mask, rolled_active, rolled_active_mask = model.control_features(
        start_latent,
        start_state,
        start_key,
        start_goal,
        start_previous,
        start_local,
        rolled_suffix,
        length,
        torch.roll(endpoint, shifts=max(1, residents), dims=0),
        count,
        torch.roll(lcb, shifts=max(1, residents), dims=0),
        torch.roll(empirical_uncertainty, shifts=max(1, residents), dims=0),
        tick_seconds=CONTROL_DT_S,
    )
    rolled_proposals[:, :LOCAL, 97] = one_tick
    rolled_proposals[:, LOCAL, 96] = suffix_duration
    rolled_proposals[:, LOCAL, 97] = one_tick
    rolled_proposals[:, LOCAL, 98] = suffix_duration
    rolled_start_key = torch.roll(start_key, shifts=max(1, residents), dims=0)
    rolled_proposals[:, LOCAL, 101] = (start_key * rolled_start_key).sum(-1).clamp(-1, 1)
    rolled_active = rolled_proposals[:, LOCAL].clone()
    rolled_outputs = model.sequence_control(
        control_state.detach(), rolled_proposals.detach(), rolled_active.detach(),
        rolled_mask, rolled_active_mask,
    )
    own_hazard = (observed_utility < -0.02)
    current_context = delivered[:suffix_starts].reshape(-1, ACTIONS)
    break_error = ((rolled_suffix[:, 0] - current_context) ** 2).mean(-1)
    rolled_hazard = (break_error > 0.02) | (reward[:suffix_starts].reshape(-1) < -0.02)
    hazard_logits = torch.cat((outputs["hazard_logit"], rolled_outputs["hazard_logit"]))
    hazard_target = torch.cat((own_hazard, rolled_hazard)).float()
    positives = hazard_target.sum().clamp_min(1)
    negatives = (hazard_target.numel() - hazard_target.sum()).clamp_min(1)
    hazard_weight = torch.where(hazard_target.bool(), 0.5 / positives, 0.5 / negatives)
    hazard_loss = (
        F.binary_cross_entropy_with_logits(hazard_logits, hazard_target, reduction="none")
        * hazard_weight
    ).sum()
    value_loss = F.smooth_l1_loss(outputs["value"], observed_utility.detach().clamp(-4, 4))
    total = (
        action_loss
        + prediction_loss * 2.0
        + memory_loss * 0.05
        + selector_loss * 0.25
        + hazard_loss * 0.15
        + value_loss * 0.5
    )
    return LossTerms(
        total, action_loss, prediction_loss, memory_loss,
        selector_loss, hazard_loss, value_loss,
    )
