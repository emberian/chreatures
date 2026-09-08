//! A host-neutral, conservative ecology kernel.
//!
//! Physics owns contacts, poses, collision queries and topology application.
//! This crate owns finite material, transport, private physiology, inheritance,
//! and the prepare/commit boundary joining those mechanisms. Chemical samples
//! returned to a host are ground truth for an afferent transducer; they are not
//! controller observations. In the current Chreatures contract they may reach a
//! policy only after CNS recurrence.

mod growth;
mod model;
mod world;

pub use model::*;
pub use world::{EcologyBatch, EcologyWorld};
