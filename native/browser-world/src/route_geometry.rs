//! Thin Wasm/native boundary over the shared ecological aperture mechanism.
use chreatures_ecology_core::{EcologyConfig, RouteGeometryPlan, RouteGeometryState};
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct RouteGeometry {
    plan: RouteGeometryPlan,
    state: RouteGeometryState,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl RouteGeometry {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(fixture: &str) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Config {
            ecology: EcologyConfig,
        }
        let config: Config = serde_json::from_str(fixture).map_err(|e| e.to_string())?;
        let plan = RouteGeometryPlan::new(&config.ecology.regions, &config.ecology.routes)
            .map_err(|e| e.to_string())?;
        let state = RouteGeometryState::new(&plan);
        Ok(Self { plan, state })
    }
    pub fn plan(&self) -> String {
        serde_json::to_string(&self.plan).unwrap()
    }
    pub fn refresh_due(&self, tick: u32, topology: u32) -> Result<bool, String> {
        self.state
            .refresh_due(&self.plan, tick.into(), topology.into())
            .map_err(|e| e.to_string())
    }
    pub fn measure(
        &mut self,
        tick: u32,
        topology: u32,
        plan_sha256: &str,
        distances_m: &[f64],
        inside: &[u8],
    ) -> Result<Vec<f64>, String> {
        if inside.iter().any(|x| *x > 1) {
            return Err("endpoint containment must be binary".into());
        }
        let inside = inside.iter().map(|x| *x != 0).collect::<Vec<_>>();
        self.state
            .refresh(
                &self.plan,
                tick.into(),
                topology.into(),
                plan_sha256,
                distances_m,
                &inside,
            )
            .map_err(|e| e.to_string())?;
        self.openness(tick, topology)
    }
    pub fn openness(&self, tick: u32, topology: u32) -> Result<Vec<f64>, String> {
        self.state
            .openness(&self.plan, tick.into(), topology.into())
            .map(|x| x.to_vec())
            .map_err(|e| e.to_string())
    }
    pub fn snapshot(&self) -> String {
        serde_json::to_string(&self.state).unwrap()
    }
    pub fn validate_snapshot(&self, snapshot: &str) -> Result<(), String> {
        let state: RouteGeometryState =
            serde_json::from_str(snapshot).map_err(|e| e.to_string())?;
        state.validate(&self.plan).map_err(|e| e.to_string())
    }
    pub fn restore(&mut self, snapshot: &str) -> Result<(), String> {
        let state: RouteGeometryState =
            serde_json::from_str(snapshot).map_err(|e| e.to_string())?;
        state.validate(&self.plan).map_err(|e| e.to_string())?;
        self.state = state;
        Ok(())
    }
}
