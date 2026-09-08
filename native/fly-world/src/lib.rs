// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native MuJoCo host for the same portable fly/ecology core used by the browser world.

mod ffi;
mod host;
mod protocol;

pub use host::{NativeFlyWorld, ResearchSample, SensorySample};
pub use protocol::serve_stdio;
