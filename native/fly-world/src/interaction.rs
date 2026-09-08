// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded, snapshot-safe physical encounter programs on the 100 Hz model clock.
//!
//! Relative offsets are compiled once against the earliest control tick that has
//! not already been prepared. Only `prepare_tick` releases events; observing
//! status never changes the queue or exposes program identity to a controller.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::f64::consts::{PI, TAU};

pub const CONTROL_HZ: u64 = 100;
pub const MAX_PENDING_EVENTS: usize = 4096;
pub const MAX_HORIZON_TICKS: u64 = 360_000;
pub const PATTERN_WIDTH: usize = 160;
pub const PATTERN_HEIGHT: usize = 120;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionProgram {
    pub events: Vec<RelativeInteractionEvent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RelativeInteractionEvent {
    pub offset_ticks: u64,
    pub event: InteractionEvent,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InteractionEvent {
    Tone {
        position_mm: [f64; 3],
        frequency_hz: f64,
        amplitude: f64,
        duration_s: f64,
    },
    Pattern {
        pattern: ScreenPattern,
    },
    ToyForce {
        entity_id: String,
        force: [f64; 3],
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScreenPattern {
    Blank,
    Uniform {
        rgb: [f32; 3],
    },
    Grating {
        low_rgb: [f32; 3],
        high_rgb: [f32; 3],
        phase_radians: f64,
        orientation_radians: f64,
        spatial_cycles: f64,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScheduledEvent {
    tick: u64,
    sequence_id: u64,
    program_id: u64,
    event: InteractionEvent,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionScheduler {
    pending: Vec<ScheduledEvent>,
    next_sequence_id: u64,
    next_program_id: u64,
    last_dispatch_tick: Option<u64>,
    completed_events: u64,
}

impl Default for InteractionScheduler {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            next_sequence_id: 1,
            next_program_id: 1,
            last_dispatch_tick: None,
            completed_events: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct InteractionScheduleReceipt {
    pub program_id: u64,
    pub scheduled_at_tick: u64,
    pub starts_at_tick: u64,
    pub event_count: usize,
    pub first_event_tick: Option<u64>,
    pub last_event_tick: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DispatchedInteractionEvent {
    pub program_id: u64,
    pub sequence_id: u64,
    pub scheduled_tick: u64,
    pub kind: &'static str,
    #[serde(skip)]
    pub event: InteractionEvent,
}

#[derive(Clone, Debug, Serialize)]
pub struct InteractionTickReceipt {
    pub tick: u64,
    pub repeated: bool,
    pub dispatched: Vec<DispatchedInteractionEvent>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ActiveInteractionProgram {
    pub program_id: u64,
    pub pending_events: usize,
    pub first_event_tick: u64,
    pub last_event_tick: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct InteractionStatus {
    pub current_tick: u64,
    pub screen_revision: u64,
    pub pending_events: usize,
    pub completed_events: u64,
    pub last_dispatch_tick: Option<u64>,
    pub next_event_tick: Option<u64>,
    pub active_programs: Vec<ActiveInteractionProgram>,
}

impl InteractionEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Tone { .. } => "tone",
            Self::Pattern { .. } => "pattern",
            Self::ToyForce { .. } => "toy_force",
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Tone {
                position_mm,
                frequency_hz,
                amplitude,
                duration_s,
            } => {
                if position_mm
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > 10_000.0)
                    || !frequency_hz.is_finite()
                    || !(40.0..=1600.0).contains(frequency_hz)
                    || !amplitude.is_finite()
                    || !(0.0..=1.0).contains(amplitude)
                    || !duration_s.is_finite()
                    || !(0.01..=5.0).contains(duration_s)
                {
                    return Err("interaction tone is outside physical bounds".into());
                }
            }
            Self::Pattern { pattern } => pattern.validate()?,
            Self::ToyForce { entity_id, force } => {
                let squared_norm = force.iter().map(|value| value * value).sum::<f64>();
                if entity_id.is_empty()
                    || entity_id.len() > 256
                    || entity_id.chars().any(char::is_control)
                    || force.iter().any(|value| !value.is_finite())
                    || squared_norm > 60.0 * 60.0
                {
                    return Err("interaction toy force is outside physical bounds".into());
                }
            }
        }
        Ok(())
    }
}

impl ScreenPattern {
    fn validate(&self) -> Result<(), String> {
        fn rgb_valid(rgb: &[f32; 3]) -> bool {
            rgb.iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        }
        let valid = match self {
            Self::Blank => true,
            Self::Uniform { rgb } => rgb_valid(rgb),
            Self::Grating {
                low_rgb,
                high_rgb,
                phase_radians,
                orientation_radians,
                spatial_cycles,
            } => {
                rgb_valid(low_rgb)
                    && rgb_valid(high_rgb)
                    && phase_radians.is_finite()
                    && (-TAU..=TAU).contains(phase_radians)
                    && orientation_radians.is_finite()
                    && (-PI..=PI).contains(orientation_radians)
                    && spatial_cycles.is_finite()
                    && (0.1..=64.0).contains(spatial_cycles)
            }
        };
        if valid {
            Ok(())
        } else {
            Err("interaction screen pattern is outside physical bounds".into())
        }
    }

    pub fn render_rgb(&self) -> Vec<f32> {
        let mut frame = vec![0.0; PATTERN_WIDTH * PATTERN_HEIGHT * 3];
        match self {
            Self::Blank => {}
            Self::Uniform { rgb } => {
                for pixel in frame.chunks_exact_mut(3) {
                    pixel.copy_from_slice(rgb);
                }
            }
            Self::Grating {
                low_rgb,
                high_rgb,
                phase_radians,
                orientation_radians,
                spatial_cycles,
            } => {
                let direction = [orientation_radians.cos(), orientation_radians.sin()];
                for y in 0..PATTERN_HEIGHT {
                    let ny = (y as f64 + 0.5) / PATTERN_HEIGHT as f64 - 0.5;
                    for x in 0..PATTERN_WIDTH {
                        let nx = (x as f64 + 0.5) / PATTERN_WIDTH as f64 - 0.5;
                        let coordinate = nx * direction[0] + ny * direction[1];
                        let blend =
                            0.5 + 0.5 * (TAU * spatial_cycles * coordinate + phase_radians).sin();
                        let index = (y * PATTERN_WIDTH + x) * 3;
                        for channel in 0..3 {
                            frame[index + channel] = (f64::from(low_rgb[channel])
                                + (f64::from(high_rgb[channel]) - f64::from(low_rgb[channel]))
                                    * blend)
                                as f32;
                        }
                    }
                }
            }
        }
        frame
    }
}

impl InteractionScheduler {
    pub fn schedule_relative(
        &mut self,
        current_tick: u64,
        program: InteractionProgram,
        mut validate_force_target: impl FnMut(&str) -> Result<(), String>,
    ) -> Result<InteractionScheduleReceipt, String> {
        if program.events.len() > MAX_PENDING_EVENTS
            || self.pending.len() > MAX_PENDING_EVENTS - program.events.len()
        {
            return Err(format!(
                "interaction queue exceeds {MAX_PENDING_EVENTS} events"
            ));
        }
        let program_id = self
            .next_program_id
            .checked_add(0)
            .ok_or("interaction program identity space exhausted")?;
        let next_program_id = self
            .next_program_id
            .checked_add(1)
            .ok_or("interaction program identity space exhausted")?;
        let next_sequence_id = self
            .next_sequence_id
            .checked_add(program.events.len() as u64)
            .ok_or("interaction sequence identity space exhausted")?;
        let starts_at_tick = match self.last_dispatch_tick {
            Some(last_dispatch_tick) if last_dispatch_tick >= current_tick => last_dispatch_tick
                .checked_add(1)
                .ok_or("interaction start tick overflows")?,
            _ => current_tick,
        };

        // Construct and validate every event before changing scheduler state.
        let mut additions = Vec::with_capacity(program.events.len());
        for (index, relative) in program.events.into_iter().enumerate() {
            if relative.offset_ticks > MAX_HORIZON_TICKS {
                return Err(format!(
                    "interaction event {index} exceeds {MAX_HORIZON_TICKS}-tick horizon"
                ));
            }
            let tick = starts_at_tick
                .checked_add(relative.offset_ticks)
                .ok_or("interaction event tick overflows")?;
            relative.event.validate()?;
            if let InteractionEvent::ToyForce { entity_id, .. } = &relative.event {
                validate_force_target(entity_id)?;
            }
            additions.push(ScheduledEvent {
                tick,
                sequence_id: self.next_sequence_id + index as u64,
                program_id,
                event: relative.event,
            });
        }
        additions.sort_by_key(|event| (event.tick, event.sequence_id));
        let first_event_tick = additions.first().map(|event| event.tick);
        let last_event_tick = additions.last().map(|event| event.tick);

        self.pending.extend(additions);
        self.pending
            .sort_by_key(|event| (event.tick, event.sequence_id));
        self.next_sequence_id = next_sequence_id;
        self.next_program_id = next_program_id;
        Ok(InteractionScheduleReceipt {
            program_id,
            scheduled_at_tick: current_tick,
            starts_at_tick,
            event_count: self
                .pending
                .iter()
                .filter(|event| event.program_id == program_id)
                .count(),
            first_event_tick,
            last_event_tick,
        })
    }

    pub fn prepare_tick(&mut self, current_tick: u64) -> Result<InteractionTickReceipt, String> {
        if self
            .last_dispatch_tick
            .is_some_and(|last_dispatch_tick| current_tick < last_dispatch_tick)
        {
            return Err("interaction model clock moved backward".into());
        }
        if self.last_dispatch_tick == Some(current_tick) {
            return Ok(InteractionTickReceipt {
                tick: current_tick,
                repeated: true,
                dispatched: Vec::new(),
            });
        }
        let due = self
            .pending
            .partition_point(|event| event.tick <= current_tick);
        let completed_events = self
            .completed_events
            .checked_add(due as u64)
            .ok_or("interaction completion count overflows")?;
        let events = self.pending.drain(..due).collect::<Vec<_>>();
        self.completed_events = completed_events;
        self.last_dispatch_tick = Some(current_tick);
        Ok(InteractionTickReceipt {
            tick: current_tick,
            repeated: false,
            dispatched: events
                .into_iter()
                .map(|event| DispatchedInteractionEvent {
                    program_id: event.program_id,
                    sequence_id: event.sequence_id,
                    scheduled_tick: event.tick,
                    kind: event.event.kind(),
                    event: event.event,
                })
                .collect(),
        })
    }

    pub fn status(&self, current_tick: u64, screen_revision: u64) -> InteractionStatus {
        let mut active = BTreeMap::<u64, (usize, u64, u64)>::new();
        for event in &self.pending {
            let progress = active
                .entry(event.program_id)
                .or_insert((0, event.tick, event.tick));
            progress.0 += 1;
            progress.1 = progress.1.min(event.tick);
            progress.2 = progress.2.max(event.tick);
        }
        InteractionStatus {
            current_tick,
            screen_revision,
            pending_events: self.pending.len(),
            completed_events: self.completed_events,
            last_dispatch_tick: self.last_dispatch_tick,
            next_event_tick: self.pending.first().map(|event| event.tick),
            active_programs: active
                .into_iter()
                .map(
                    |(program_id, (pending_events, first_event_tick, last_event_tick))| {
                        ActiveInteractionProgram {
                            program_id,
                            pending_events,
                            first_event_tick,
                            last_event_tick,
                        }
                    },
                )
                .collect(),
        }
    }

    pub fn validate_restore(
        &self,
        current_tick: u64,
        mut validate_force_target: impl FnMut(&str) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.pending.len() > MAX_PENDING_EVENTS
            || self.next_sequence_id == 0
            || self.next_program_id == 0
            || self
                .last_dispatch_tick
                .is_some_and(|tick| tick > current_tick)
        {
            return Err("interaction snapshot counters differ".into());
        }
        if self.completed_events.checked_add(self.pending.len() as u64)
            != self.next_sequence_id.checked_sub(1)
        {
            return Err("interaction snapshot completion accounting differs".into());
        }
        let starts_at_tick = match self.last_dispatch_tick {
            Some(last_dispatch_tick) if last_dispatch_tick >= current_tick => last_dispatch_tick
                .checked_add(1)
                .ok_or("interaction restore start tick overflows")?,
            _ => current_tick,
        };
        let horizon = starts_at_tick
            .checked_add(MAX_HORIZON_TICKS)
            .ok_or("interaction restore horizon overflows")?;
        let mut previous = None;
        let mut sequence_ids = HashSet::with_capacity(self.pending.len());
        for event in &self.pending {
            if event.tick < current_tick
                || self
                    .last_dispatch_tick
                    .is_some_and(|last_dispatch_tick| event.tick <= last_dispatch_tick)
                || event.tick > horizon
                || event.sequence_id == 0
                || event.sequence_id >= self.next_sequence_id
                || !sequence_ids.insert(event.sequence_id)
                || event.program_id == 0
                || event.program_id >= self.next_program_id
                || previous.is_some_and(|key| key >= (event.tick, event.sequence_id))
            {
                return Err("interaction snapshot queue order or identity differs".into());
            }
            event.event.validate()?;
            if let InteractionEvent::ToyForce { entity_id, .. } = &event.event {
                validate_force_target(entity_id)?;
            }
            previous = Some((event.tick, event.sequence_id));
        }
        Ok(())
    }
}

pub fn control_tick_from_seconds(seconds: f64) -> Result<u64, String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("interaction model clock is invalid".into());
    }
    let scaled = seconds * CONTROL_HZ as f64;
    let rounded = scaled.round();
    if rounded > u64::MAX as f64 || (scaled - rounded).abs() > 1e-4 {
        return Err("interaction model clock is not on a control tick".into());
    }
    Ok(rounded as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(offset_ticks: u64) -> RelativeInteractionEvent {
        RelativeInteractionEvent {
            offset_ticks,
            event: InteractionEvent::Tone {
                position_mm: [1.0, 2.0, 3.0],
                frequency_hz: 200.0,
                amplitude: 0.7,
                duration_s: 0.1,
            },
        }
    }

    #[test]
    fn stable_same_tick_and_exactly_once() {
        let mut scheduler = InteractionScheduler::default();
        scheduler
            .schedule_relative(
                12,
                InteractionProgram {
                    events: vec![tone(2), tone(0), tone(2)],
                },
                |_| Ok(()),
            )
            .unwrap();
        let first = scheduler.prepare_tick(12).unwrap();
        assert_eq!(first.dispatched.len(), 1);
        assert_eq!(first.dispatched[0].sequence_id, 2);
        assert!(scheduler.prepare_tick(12).unwrap().repeated);
        let second = scheduler.prepare_tick(14).unwrap();
        assert_eq!(
            second
                .dispatched
                .iter()
                .map(|event| event.sequence_id)
                .collect::<Vec<_>>(),
            [1, 3]
        );
    }

    #[test]
    fn rejected_program_does_not_mutate_queue() {
        let mut scheduler = InteractionScheduler::default();
        let before = serde_json::to_vec(&scheduler).unwrap();
        let error = scheduler
            .schedule_relative(
                2,
                InteractionProgram {
                    events: vec![
                        tone(1),
                        RelativeInteractionEvent {
                            offset_ticks: 2,
                            event: InteractionEvent::ToyForce {
                                entity_id: "missing".into(),
                                force: [1.0, 0.0, 0.0],
                            },
                        },
                    ],
                },
                |_| Err("movable physical entity missing".into()),
            )
            .unwrap_err();
        assert_eq!(error, "movable physical entity missing");
        assert_eq!(serde_json::to_vec(&scheduler).unwrap(), before);
    }

    #[test]
    fn zero_offset_after_prepare_targets_next_tick() {
        let mut scheduler = InteractionScheduler::default();
        scheduler.prepare_tick(8).unwrap();
        let receipt = scheduler
            .schedule_relative(
                8,
                InteractionProgram {
                    events: vec![tone(0)],
                },
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(receipt.scheduled_at_tick, 8);
        assert_eq!(receipt.starts_at_tick, 9);
        assert!(scheduler.prepare_tick(8).unwrap().repeated);
        assert_eq!(scheduler.prepare_tick(9).unwrap().dispatched.len(), 1);
    }

    #[test]
    fn grating_is_finite_and_bounded() {
        let pattern = ScreenPattern::Grating {
            low_rgb: [0.1, 0.2, 0.3],
            high_rgb: [0.8, 0.9, 1.0],
            phase_radians: 0.7,
            orientation_radians: -0.4,
            spatial_cycles: 3.0,
        };
        pattern.validate().unwrap();
        let frame = pattern.render_rgb();
        assert_eq!(frame.len(), PATTERN_WIDTH * PATTERN_HEIGHT * 3);
        assert!(frame
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    }

    #[test]
    fn snapshot_round_trip_preserves_pending_identity_and_dispatch() {
        let mut scheduler = InteractionScheduler::default();
        scheduler
            .schedule_relative(
                20,
                InteractionProgram {
                    events: vec![tone(0), tone(7)],
                },
                |_| Ok(()),
            )
            .unwrap();
        scheduler.prepare_tick(20).unwrap();
        let encoded = serde_json::to_vec(&scheduler).unwrap();
        let mut restored: InteractionScheduler = serde_json::from_slice(&encoded).unwrap();
        restored.validate_restore(20, |_| Ok(())).unwrap();
        let receipt = restored.prepare_tick(27).unwrap();
        assert_eq!(receipt.dispatched.len(), 1);
        assert_eq!(receipt.dispatched[0].sequence_id, 2);
        assert_eq!(restored.status(27, 4).completed_events, 2);
    }
}
