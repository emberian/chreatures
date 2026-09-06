//! Resident-private sparse memory of experienced goal succession.
//!
//! Nodes are exact achieved-memory `(slot, recorded_tick, generation)` identities.
//! Edges record measured attempts from the last attained episode toward a later
//! selected episode. Past succession is only an exploration bias: it is never
//! represented as current controllability or executed as an automatic policy.

use crate::contextual_episodic::CONTEXT;
use crate::personal_goals::{GoalSlotIdentity, GoalSlotReplacement};
use serde::{Deserialize, Serialize};

const FORMAT: &str = "chreatures-private-goal-sequence-v1";
const MAX_EDGES: usize = 256;
const MAX_DEPTH: usize = 3;
const MAX_BIAS: f64 = 0.24;
const DISCOUNT: f64 = 0.68;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SequenceNode {
    pub(crate) slot: usize,
    pub(crate) identity: GoalSlotIdentity,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SequenceEpisode {
    pub(crate) resident: usize,
    pub(crate) node: SequenceNode,
    pub(crate) completed_tick: u64,
    pub(crate) context: [f64; CONTEXT],
    pub(crate) normalized_return: f64,
    pub(crate) normalized_progress: f64,
    pub(crate) attained: bool,
    pub(crate) attributed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequenceUpdate {
    pub(crate) edge_updated: bool,
    pub(crate) edge_replaced: bool,
    pub(crate) attained: bool,
    pub(crate) chain_advanced: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SequencePlan {
    pub(crate) biases: Vec<f64>,
    pub(crate) experienced_path_depth: Vec<u8>,
    pub(crate) confidence: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SequenceStats {
    pub(crate) attempts: u64,
    pub(crate) attainments: u64,
    pub(crate) failed_attempts: u64,
    pub(crate) learned_transitions: u64,
    pub(crate) replaced_edges: u64,
    pub(crate) invalidated_edges: u64,
    pub(crate) consolidation_steps: u64,
    pub(crate) cancelled_chains: u64,
}

impl SequenceStats {
    fn empty() -> Self {
        Self {
            attempts: 0,
            attainments: 0,
            failed_attempts: 0,
            learned_transitions: 0,
            replaced_edges: 0,
            invalidated_edges: 0,
            consolidation_steps: 0,
            cancelled_chains: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Edge {
    from: SequenceNode,
    to: SequenceNode,
    context: [f64; CONTEXT],
    attempts: u32,
    attainments: u32,
    last_tick: u64,
    mean_return: f64,
    mean_progress: f64,
    consolidated_return: f64,
    consolidated_progress: f64,
    consolidated_confidence: f64,
    consolidation_passes: u32,
}

impl Edge {
    fn attainment_rate(&self) -> f64 {
        self.attainments as f64 / self.attempts as f64
    }

    fn online_confidence(&self) -> f64 {
        self.attainment_rate() * self.attempts as f64 / (self.attempts as f64 + 4.0)
    }

    fn planning_confidence(&self) -> f64 {
        if self.consolidation_passes == 0 {
            self.online_confidence()
        } else {
            self.consolidated_confidence
        }
    }

    fn utility(&self) -> f64 {
        let (return_value, progress) = if self.consolidation_passes == 0 {
            (self.mean_return, self.mean_progress)
        } else {
            (self.consolidated_return, self.consolidated_progress)
        };
        (0.45 * return_value + 0.55 * progress).clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Individual {
    slot_identity: Vec<Option<GoalSlotIdentity>>,
    edges: Vec<Edge>,
    last_attained: Option<SequenceNode>,
    consolidation_cursor: usize,
    stats: SequenceStats,
}

impl Individual {
    fn empty(slots: usize) -> Self {
        Self {
            slot_identity: vec![None; slots],
            edges: Vec::new(),
            last_attained: None,
            consolidation_cursor: 0,
            stats: SequenceStats::empty(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct GoalSequenceMemory {
    schema: String,
    residents: usize,
    slots: usize,
    individuals: Vec<Individual>,
}

impl GoalSequenceMemory {
    pub(crate) fn new(residents: usize, slots: usize) -> Result<Self, String> {
        if !(1..=4096).contains(&residents) || slots == 0 || slots > 4096 {
            return Err("goal sequence dimensions are invalid".into());
        }
        Ok(Self {
            schema: FORMAT.into(),
            residents,
            slots,
            individuals: vec![Individual::empty(slots); residents],
        })
    }

    pub(crate) fn grow(&mut self, new_residents: usize) -> Result<(), String> {
        if new_residents <= self.residents || new_residents > 4096 {
            return Err("goal sequence growth must append residents".into());
        }
        self.individuals
            .resize_with(new_residents, || Individual::empty(self.slots));
        self.residents = new_residents;
        Ok(())
    }

    pub(crate) fn clear_resident(&mut self, resident: usize) -> Result<(), String> {
        let individual = self
            .individuals
            .get_mut(resident)
            .ok_or("unknown goal sequence resident")?;
        *individual = Individual::empty(self.slots);
        Ok(())
    }

    fn validate_identity(&self, node: SequenceNode) -> Result<(), String> {
        if node.slot >= self.slots || node.identity.generation == 0 {
            return Err("goal sequence node identity is invalid".into());
        }
        Ok(())
    }

    fn validate_context(context: &[f64; CONTEXT]) -> Result<(), String> {
        if context.iter().any(|value| !value.is_finite()) {
            return Err("goal sequence context must be finite".into());
        }
        Ok(())
    }

    pub(crate) fn replace_slots(
        &mut self,
        replacements: &[GoalSlotReplacement],
    ) -> Result<Vec<usize>, String> {
        if replacements.len() > self.residents.saturating_mul(self.slots) {
            return Err("goal sequence replacement batch exceeds capacity".into());
        }
        let mut keys = Vec::with_capacity(replacements.len());
        for replacement in replacements {
            self.validate_identity(SequenceNode {
                slot: replacement.slot,
                identity: replacement.identity,
            })?;
            if replacement.resident >= self.residents {
                return Err("unknown goal sequence resident".into());
            }
            keys.push((replacement.resident, replacement.slot));
        }
        keys.sort_unstable();
        if keys.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("goal sequence replacement batch contains duplicates".into());
        }
        let mut invalidated = Vec::with_capacity(replacements.len());
        for replacement in replacements {
            let individual = &mut self.individuals[replacement.resident];
            if individual.slot_identity[replacement.slot] == Some(replacement.identity) {
                invalidated.push(0);
                continue;
            }
            individual.slot_identity[replacement.slot] = Some(replacement.identity);
            let before = individual.edges.len();
            individual.edges.retain(|edge| {
                (edge.from.slot != replacement.slot || edge.from.identity == replacement.identity)
                    && (edge.to.slot != replacement.slot
                        || edge.to.identity == replacement.identity)
            });
            let removed = before - individual.edges.len();
            individual.stats.invalidated_edges = individual
                .stats
                .invalidated_edges
                .saturating_add(removed as u64);
            if individual.last_attained.is_some_and(|node| {
                node.slot == replacement.slot && node.identity != replacement.identity
            }) {
                individual.last_attained = None;
                individual.stats.cancelled_chains =
                    individual.stats.cancelled_chains.saturating_add(1);
            }
            individual.consolidation_cursor = individual
                .consolidation_cursor
                .min(individual.edges.len().saturating_sub(1));
            invalidated.push(removed);
        }
        Ok(invalidated)
    }

    fn edge_priority(edge: &Edge) -> f64 {
        edge.online_confidence() + 0.002 * (edge.attempts.min(100) as f64)
    }

    fn update_edge(edge: &mut Edge, episode: &SequenceEpisode) {
        edge.attempts = edge.attempts.saturating_add(1);
        if episode.attained {
            edge.attainments = edge.attainments.saturating_add(1);
        }
        edge.last_tick = episode.completed_tick;
        let rate = 1.0 / edge.attempts as f64;
        edge.mean_return += rate * (episode.normalized_return - edge.mean_return);
        edge.mean_progress += rate * (episode.normalized_progress - edge.mean_progress);
        for (stored, observed) in edge.context.iter_mut().zip(episode.context) {
            *stored += rate * (observed - *stored);
        }
    }

    pub(crate) fn observe_episode(
        &mut self,
        episode: SequenceEpisode,
    ) -> Result<SequenceUpdate, String> {
        self.validate_identity(episode.node)?;
        Self::validate_context(&episode.context)?;
        if !episode.normalized_return.is_finite()
            || episode.normalized_return.abs() > 1.0
            || !episode.normalized_progress.is_finite()
            || episode.normalized_progress.abs() > 1.0
        {
            return Err("goal sequence outcome must be finite in [-1,1]".into());
        }
        let individual = self
            .individuals
            .get_mut(episode.resident)
            .ok_or("unknown goal sequence resident")?;
        if individual.slot_identity[episode.node.slot] != Some(episode.node.identity) {
            return Err("goal sequence episode identity differs from the current slot".into());
        }
        individual.stats.attempts = individual.stats.attempts.saturating_add(1);
        if episode.attained {
            individual.stats.attainments = individual.stats.attainments.saturating_add(1);
        } else {
            individual.stats.failed_attempts = individual.stats.failed_attempts.saturating_add(1);
        }
        if !episode.attributed {
            individual.last_attained = None;
            individual.stats.cancelled_chains = individual.stats.cancelled_chains.saturating_add(1);
            return Ok(SequenceUpdate {
                edge_updated: false,
                edge_replaced: false,
                attained: episode.attained,
                chain_advanced: false,
            });
        }
        let from = individual.last_attained;
        let mut edge_updated = false;
        let mut edge_replaced = false;
        if let Some(from) = from.filter(|from| *from != episode.node) {
            if let Some(edge) = individual
                .edges
                .iter_mut()
                .find(|edge| edge.from == from && edge.to == episode.node)
            {
                Self::update_edge(edge, &episode);
                edge_updated = true;
            } else {
                let edge = Edge {
                    from,
                    to: episode.node,
                    context: episode.context,
                    attempts: 1,
                    attainments: u32::from(episode.attained),
                    last_tick: episode.completed_tick,
                    mean_return: episode.normalized_return,
                    mean_progress: episode.normalized_progress,
                    consolidated_return: 0.0,
                    consolidated_progress: 0.0,
                    consolidated_confidence: 0.0,
                    consolidation_passes: 0,
                };
                if individual.edges.len() < MAX_EDGES {
                    individual.edges.push(edge);
                } else {
                    let replace = individual
                        .edges
                        .iter()
                        .enumerate()
                        .min_by(|(_, left), (_, right)| {
                            Self::edge_priority(left)
                                .total_cmp(&Self::edge_priority(right))
                                .then_with(|| left.last_tick.cmp(&right.last_tick))
                        })
                        .map(|(index, _)| index)
                        .unwrap();
                    individual.edges[replace] = edge;
                    individual.stats.replaced_edges =
                        individual.stats.replaced_edges.saturating_add(1);
                    edge_replaced = true;
                }
                individual.stats.learned_transitions =
                    individual.stats.learned_transitions.saturating_add(1);
                edge_updated = true;
            }
        }
        if episode.attained {
            individual.last_attained = Some(episode.node);
        }
        Ok(SequenceUpdate {
            edge_updated,
            edge_replaced,
            attained: episode.attained,
            chain_advanced: episode.attained,
        })
    }

    pub(crate) fn cancel_chain(&mut self, resident: usize) -> Result<bool, String> {
        let individual = self
            .individuals
            .get_mut(resident)
            .ok_or("unknown goal sequence resident")?;
        let cancelled = individual.last_attained.take().is_some();
        if cancelled {
            individual.stats.cancelled_chains = individual.stats.cancelled_chains.saturating_add(1);
        }
        Ok(cancelled)
    }

    fn context_similarity(left: &[f64; CONTEXT], right: &[f64; CONTEXT]) -> f64 {
        let squared = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / CONTEXT as f64;
        (-0.5 * squared).exp()
    }

    pub(crate) fn selection_biases(
        &self,
        resident: usize,
        recorded_ticks: &[u64],
        generations: &[u64],
        current_context: &[f64; CONTEXT],
    ) -> Result<SequencePlan, String> {
        Self::validate_context(current_context)?;
        if recorded_ticks.len() != self.slots || generations.len() != self.slots {
            return Err("goal sequence identity rows have the wrong length".into());
        }
        let individual = self
            .individuals
            .get(resident)
            .ok_or("unknown goal sequence resident")?;
        let current: Vec<_> = recorded_ticks
            .iter()
            .zip(generations)
            .enumerate()
            .map(|(slot, (&recorded_tick, &generation))| {
                (generation != 0).then_some(SequenceNode {
                    slot,
                    identity: GoalSlotIdentity {
                        recorded_tick,
                        generation,
                    },
                })
            })
            .collect();
        let mut biases = vec![0.0; self.slots];
        let mut depths = vec![0; self.slots];
        let mut confidence = vec![0.0_f64; self.slots];
        let Some(start) = individual.last_attained.filter(|node| {
            current
                .get(node.slot)
                .is_some_and(|value| *value == Some(*node))
        }) else {
            return Ok(SequencePlan {
                biases,
                experienced_path_depth: depths,
                confidence,
            });
        };
        let mut frontier = vec![0.0_f64; self.slots];
        frontier[start.slot] = 1.0;
        for depth in 1..=MAX_DEPTH {
            let mut next = vec![0.0_f64; self.slots];
            for edge in &individual.edges {
                if current.get(edge.from.slot).copied().flatten() != Some(edge.from)
                    || current.get(edge.to.slot).copied().flatten() != Some(edge.to)
                    || frontier[edge.from.slot] <= 0.0
                {
                    continue;
                }
                let contextual = Self::context_similarity(&edge.context, current_context);
                let edge_confidence = edge.planning_confidence() * contextual;
                let path_confidence = frontier[edge.from.slot] * edge_confidence;
                if path_confidence <= next[edge.to.slot] {
                    continue;
                }
                next[edge.to.slot] = path_confidence;
                let discounted = DISCOUNT.powi((depth - 1) as i32);
                let candidate = MAX_BIAS * discounted * path_confidence * edge.utility();
                if candidate.abs() > biases[edge.to.slot].abs() {
                    biases[edge.to.slot] = candidate.clamp(-MAX_BIAS, MAX_BIAS);
                    depths[edge.to.slot] = depth as u8;
                    confidence[edge.to.slot] = path_confidence.clamp(0.0, 1.0);
                }
            }
            frontier = next;
        }
        Ok(SequencePlan {
            biases,
            experienced_path_depth: depths,
            confidence,
        })
    }

    /// Consolidate at most `budget` already experienced edges. This operation
    /// neither invents edges nor samples replay; its cursor is private state.
    pub(crate) fn consolidate(&mut self, resident: usize, budget: usize) -> Result<usize, String> {
        if budget > MAX_EDGES {
            return Err("goal sequence consolidation budget exceeds edge capacity".into());
        }
        let individual = self
            .individuals
            .get_mut(resident)
            .ok_or("unknown goal sequence resident")?;
        let work = budget.min(individual.edges.len());
        for _ in 0..work {
            let index = individual.consolidation_cursor % individual.edges.len();
            let edge = &mut individual.edges[index];
            let rate = (edge.attempts as f64 / 32.0).clamp(0.05, 0.25);
            edge.consolidated_return += rate * (edge.mean_return - edge.consolidated_return);
            edge.consolidated_progress += rate * (edge.mean_progress - edge.consolidated_progress);
            edge.consolidated_confidence +=
                rate * (edge.online_confidence() - edge.consolidated_confidence);
            edge.consolidation_passes = edge.consolidation_passes.saturating_add(1);
            individual.consolidation_cursor =
                (individual.consolidation_cursor + 1) % individual.edges.len();
        }
        individual.stats.consolidation_steps = individual
            .stats
            .consolidation_steps
            .saturating_add(work as u64);
        Ok(work)
    }

    pub(crate) fn stats(&self, resident: usize) -> Result<SequenceStats, String> {
        self.individuals
            .get(resident)
            .map(|individual| individual.stats)
            .ok_or("unknown goal sequence resident".into())
    }

    pub(crate) fn snapshot(&self) -> Result<String, String> {
        self.validate_state()?;
        serde_json::to_string(self).map_err(|error| error.to_string())
    }

    pub(crate) fn restore(value: &str, residents: usize, slots: usize) -> Result<Self, String> {
        let state: Self = serde_json::from_str(value).map_err(|error| error.to_string())?;
        if state.schema != FORMAT || state.residents != residents || state.slots != slots {
            return Err("goal sequence snapshot belongs to another contract".into());
        }
        state.validate_state()?;
        Ok(state)
    }

    fn validate_state(&self) -> Result<(), String> {
        if self.schema != FORMAT
            || !(1..=4096).contains(&self.residents)
            || self.slots == 0
            || self.slots > 4096
            || self.individuals.len() != self.residents
        {
            return Err("goal sequence state dimensions differ".into());
        }
        for individual in &self.individuals {
            if individual.slot_identity.len() != self.slots
                || individual.edges.len() > MAX_EDGES
                || (!individual.edges.is_empty()
                    && individual.consolidation_cursor >= individual.edges.len())
            {
                return Err("goal sequence resident state dimensions differ".into());
            }
            if let Some(last) = individual.last_attained {
                self.validate_identity(last)?;
                if individual.slot_identity[last.slot] != Some(last.identity) {
                    return Err("goal sequence last episode identity is stale".into());
                }
            }
            for (index, edge) in individual.edges.iter().enumerate() {
                self.validate_identity(edge.from)?;
                self.validate_identity(edge.to)?;
                Self::validate_context(&edge.context)?;
                if edge.from == edge.to
                    || edge.attempts == 0
                    || edge.attainments > edge.attempts
                    || individual.slot_identity[edge.from.slot] != Some(edge.from.identity)
                    || individual.slot_identity[edge.to.slot] != Some(edge.to.identity)
                    || [
                        edge.mean_return,
                        edge.mean_progress,
                        edge.consolidated_return,
                        edge.consolidated_progress,
                    ]
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > 1.0)
                    || !edge.consolidated_confidence.is_finite()
                    || !(0.0..=1.0).contains(&edge.consolidated_confidence)
                    || individual.edges[index + 1..]
                        .iter()
                        .any(|other| other.from == edge.from && other.to == edge.to)
                {
                    return Err("goal sequence edge state differs".into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(slot: usize, generation: u64) -> SequenceNode {
        SequenceNode {
            slot,
            identity: GoalSlotIdentity {
                recorded_tick: 10 + slot as u64,
                generation,
            },
        }
    }

    fn episode(node: SequenceNode, tick: u64, attained: bool) -> SequenceEpisode {
        SequenceEpisode {
            resident: 0,
            node,
            completed_tick: tick,
            context: [0.25; CONTEXT],
            normalized_return: if attained { 0.8 } else { -0.4 },
            normalized_progress: if attained { 0.7 } else { -0.2 },
            attained,
            attributed: true,
        }
    }

    #[test]
    fn exact_identity_paths_failures_consolidation_and_restore() {
        let mut memory = GoalSequenceMemory::new(1, 4).unwrap();
        let nodes = [identity(0, 1), identity(1, 1), identity(2, 1)];
        memory
            .replace_slots(
                &nodes
                    .iter()
                    .map(|node| GoalSlotReplacement {
                        resident: 0,
                        slot: node.slot,
                        identity: node.identity,
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(
            memory
                .observe_episode(episode(nodes[0], 20, true))
                .unwrap()
                .attained
        );
        assert!(
            memory
                .observe_episode(episode(nodes[1], 30, true))
                .unwrap()
                .edge_updated
        );
        memory
            .observe_episode(episode(nodes[2], 40, false))
            .unwrap();
        assert_eq!(memory.stats(0).unwrap().failed_attempts, 1);
        memory.observe_episode(episode(nodes[2], 50, true)).unwrap();
        memory.observe_episode(episode(nodes[0], 60, true)).unwrap();
        assert_eq!(memory.consolidate(0, MAX_EDGES).unwrap(), 3);
        let recorded = [10, 11, 12, 0];
        let generations = [1, 1, 1, 0];
        let plan = memory
            .selection_biases(0, &recorded, &generations, &[0.25; CONTEXT])
            .unwrap();
        assert!(plan.biases[1] > 0.0);
        assert_eq!(plan.experienced_path_depth[1], 1);
        assert!(plan.biases[2] > 0.0);
        assert_eq!(plan.experienced_path_depth[2], 2);
        assert!(plan.confidence[2] > 0.0);

        let snapshot = memory.snapshot().unwrap();
        let restored = GoalSequenceMemory::restore(&snapshot, 1, 4).unwrap();
        assert_eq!(memory, restored);

        let replacement = GoalSlotReplacement {
            resident: 0,
            slot: 1,
            identity: identity(1, 2).identity,
        };
        assert_eq!(memory.replace_slots(&[replacement]).unwrap(), vec![2]);
        let plan = memory
            .selection_biases(0, &[10, 11, 12, 0], &[1, 2, 1, 0], &[0.25; CONTEXT])
            .unwrap();
        assert_eq!(plan.biases, vec![0.0; 4]);
        memory.grow(2).unwrap();
        memory.clear_resident(1).unwrap();
        assert_eq!(memory.stats(1).unwrap(), SequenceStats::empty());
    }
}
