//! Physical aperture quadrature for the existing conservative route transport.
//!
//! Hosts raycast actual solids and measure endpoint containment. This module
//! neither invents airflow nor changes material transport equations. Its area
//! fraction multiplies the declared route conductance in `EcologyWorld`, where
//! `base_open_fraction` is applied once. This is a straight-path aperture model,
//! not a resolved molecular diffusion or tortuosity solver.
use crate::{EcologyError, EcologyResult, RegionSpec, RouteSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const ROUTE_GEOMETRY_FORMAT: &str = "chreatures-route-aperture-plan-v1";
pub const ROUTE_GEOMETRY_STATE_FORMAT: &str = "chreatures-route-aperture-state-v1";
pub const ROUTE_APERTURE_SAMPLES: usize = 32;
pub const ROUTE_GEOMETRY_REFRESH_TICKS: u64 = 10;
pub const ROUTE_GEOMETRY_CONTROL_DT_S: f64 = 0.01;
/// Hosts may use this tiny sphere for actual signed geom-distance containment.
/// Its overlap conservatively inflates a solid by ten nanometres.
pub const ROUTE_ENDPOINT_PROBE_RADIUS_M: f64 = 1e-8;
pub const ROUTE_DISTANCE_TOLERANCE_M: f64 = 1e-9;

// Fixed [squared radial fraction, cos(angle), sin(angle)] nodes. One
// central disk cell plus 7/12/12 annular sectors, all area A/32. Constants
// avoid backend-specific runtime trigonometric evaluation.
const APERTURE_NODES: [[f64; 3]; 32] = [
    [0.0, 1.0, 0.0],
    [0.140625, 1.0, 0.0],
    [0.140625, 0.6234898018587336, 0.7818314824680298],
    [0.140625, -0.22252093395631434, 0.9749279121818236],
    [0.140625, -0.900968867902419, 0.43388373911755823],
    [0.140625, -0.9009688679024191, -0.433883739117558],
    [0.140625, -0.2225209339563146, -0.9749279121818236],
    [0.140625, 0.6234898018587334, -0.7818314824680299],
    [0.4375, 0.9659258262890683, 0.25881904510252074],
    [0.4375, 0.7071067811865476, 0.7071067811865475],
    [0.4375, 0.25881904510252096, 0.9659258262890682],
    [0.4375, -0.25881904510252063, 0.9659258262890683],
    [0.4375, -0.7071067811865475, 0.7071067811865476],
    [0.4375, -0.9659258262890683, 0.2588190451025206],
    [0.4375, -0.9659258262890683, -0.2588190451025208],
    [0.4375, -0.7071067811865477, -0.7071067811865475],
    [0.4375, -0.2588190451025215, -0.9659258262890681],
    [0.4375, 0.2588190451025203, -0.9659258262890684],
    [0.4375, 0.7071067811865474, -0.7071067811865477],
    [0.4375, 0.9659258262890681, -0.25881904510252157],
    [0.8125, 1.0, 0.0],
    [0.8125, 0.8660254037844387, 0.49999999999999994],
    [0.8125, 0.5000000000000001, 0.8660254037844386],
    [0.8125, 0.0, 1.0],
    [0.8125, -0.49999999999999983, 0.8660254037844387],
    [0.8125, -0.8660254037844387, 0.49999999999999994],
    [0.8125, -1.0, 0.0],
    [0.8125, -0.8660254037844388, -0.4999999999999998],
    [0.8125, -0.5000000000000004, -0.8660254037844384],
    [0.8125, 0.0, -1.0],
    [0.8125, 0.5000000000000001, -0.8660254037844386],
    [0.8125, 0.8660254037844384, -0.5000000000000004],
];

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RouteGeometryRay {
    pub route_index: usize,
    pub sample_index: usize,
    pub reverse: bool,
    pub origin_endpoint: usize,
    pub destination_endpoint: usize,
    pub direction: [f64; 3],
    pub max_distance_m: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RouteAperture {
    pub route_id: String,
    pub cross_section_m2: f64,
    pub aperture_radius_m: f64,
    /// Used by transport; it can encode a declared longer effective path.
    pub declared_transport_length_m: f64,
    /// Actual geometric segment between the region centers, used by raycasts.
    pub center_distance_m: f64,
    pub first_ray: usize,
    pub ray_count: usize,
}

/// Immutable, deterministic geometry. Rebuild from the same config on restore;
/// do not accept a serialized plan as a replacement for authenticated config.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RouteGeometryPlan {
    pub format: String,
    pub sha256: String,
    pub samples_per_route: usize,
    pub refresh_ticks: u64,
    pub control_dt_s: f64,
    pub endpoint_probe_radius_m: f64,
    pub distance_tolerance_m: f64,
    pub routes: Vec<RouteAperture>,
    pub endpoints_m: Vec<[f64; 3]>,
    /// Opposed forward/reverse rays are consecutive for each aperture sample.
    pub rays: Vec<RouteGeometryRay>,
}

fn fail(message: impl Into<String>) -> EcologyError {
    EcologyError(message.into())
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    v.map(|x| x / n)
}
fn endpoint(
    point: [f64; 3],
    points: &mut Vec<[f64; 3]>,
    index: &mut BTreeMap<[u64; 3], usize>,
) -> usize {
    let point = point.map(|x| if x == 0.0 { 0.0 } else { x });
    let key = point.map(f64::to_bits);
    *index.entry(key).or_insert_with(|| {
        let i = points.len();
        points.push(point);
        i
    })
}

impl RouteGeometryPlan {
    pub fn new(regions: &[RegionSpec], routes: &[RouteSpec]) -> EcologyResult<Self> {
        let mut centers = BTreeMap::new();
        for region in regions {
            if region.id.is_empty()
                || !region.center_m.iter().all(|x| x.is_finite())
                || centers
                    .insert(region.id.as_str(), region.center_m)
                    .is_some()
            {
                return Err(fail("invalid or duplicate region in route geometry"));
            }
        }
        let capacity = routes
            .len()
            .checked_mul(ROUTE_APERTURE_SAMPLES * 2)
            .ok_or_else(|| fail("too many aperture rays"))?;
        let mut plan = Self {
            format: ROUTE_GEOMETRY_FORMAT.into(),
            sha256: String::new(),
            samples_per_route: ROUTE_APERTURE_SAMPLES,
            refresh_ticks: ROUTE_GEOMETRY_REFRESH_TICKS,
            control_dt_s: ROUTE_GEOMETRY_CONTROL_DT_S,
            endpoint_probe_radius_m: ROUTE_ENDPOINT_PROBE_RADIUS_M,
            distance_tolerance_m: ROUTE_DISTANCE_TOLERANCE_M,
            routes: Vec::new(),
            endpoints_m: Vec::new(),
            rays: Vec::new(),
        };
        plan.rays
            .try_reserve_exact(capacity)
            .map_err(|e| fail(e.to_string()))?;
        let mut endpoint_index = BTreeMap::new();
        let mut route_ids = BTreeSet::new();
        for (route_index, route) in routes.iter().enumerate() {
            if route.id.is_empty()
                || !route_ids.insert(route.id.as_str())
                || !route.length_m.is_finite()
                || route.length_m <= 0.0
                || !route.cross_section_m2.is_finite()
                || route.cross_section_m2 <= 0.0
            {
                return Err(fail("invalid or duplicate route aperture specification"));
            }
            let a = *centers
                .get(route.a.as_str())
                .ok_or_else(|| fail("route origin region missing"))?;
            let b = *centers
                .get(route.b.as_str())
                .ok_or_else(|| fail("route destination region missing"))?;
            let axis = sub(b, a);
            let length = norm(axis);
            if !length.is_finite() || length <= 2.0 * ROUTE_DISTANCE_TOLERANCE_M {
                return Err(fail(
                    "route centers coincide or are below geometry resolution",
                ));
            }
            let direction = axis.map(|x| x / length);
            let reference_axis = (0..3)
                .min_by(|&i, &j| direction[i].abs().total_cmp(&direction[j].abs()))
                .unwrap();
            let mut reference = [0.0; 3];
            reference[reference_axis] = 1.0;
            let u = normalize(cross(direction, reference));
            let v = cross(direction, u);
            let radius = (route.cross_section_m2 / std::f64::consts::PI).sqrt();
            if !radius.is_finite() || radius <= 0.0 {
                return Err(fail("invalid aperture radius"));
            }
            let first_ray = plan.rays.len();
            // Equal-area cells include the center rather than leaving a
            // central blind region. Finite sampling can still miss small solids.
            for (sample_index, [squared_radius, x, y]) in APERTURE_NODES.into_iter().enumerate() {
                let r = radius * squared_radius.sqrt();
                let offset: [f64; 3] = std::array::from_fn(|i| r * (u[i] * x + v[i] * y));
                let pa: [f64; 3] = std::array::from_fn(|i| a[i] + offset[i]);
                let pb: [f64; 3] = std::array::from_fn(|i| b[i] + offset[i]);
                if !pa.iter().chain(pb.iter()).all(|x| x.is_finite()) {
                    return Err(fail("aperture endpoint overflow"));
                }
                let ia = endpoint(pa, &mut plan.endpoints_m, &mut endpoint_index);
                let ib = endpoint(pb, &mut plan.endpoints_m, &mut endpoint_index);
                plan.rays.push(RouteGeometryRay {
                    route_index,
                    sample_index,
                    reverse: false,
                    origin_endpoint: ia,
                    destination_endpoint: ib,
                    direction,
                    max_distance_m: length,
                });
                plan.rays.push(RouteGeometryRay {
                    route_index,
                    sample_index,
                    reverse: true,
                    origin_endpoint: ib,
                    destination_endpoint: ia,
                    direction: direction.map(|x| -x),
                    max_distance_m: length,
                });
            }
            plan.routes.push(RouteAperture {
                route_id: route.id.clone(),
                cross_section_m2: route.cross_section_m2,
                aperture_radius_m: radius,
                declared_transport_length_m: route.length_m,
                center_distance_m: length,
                first_ray,
                ray_count: ROUTE_APERTURE_SAMPLES * 2,
            });
        }
        // Hash the complete canonical plan with the identity field empty.
        plan.sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&plan)?));
        Ok(plan)
    }

    /// Hosts clip no-hit or beyond-end distances to max_distance_m. Missing,
    /// nonfinite, negative, out-of-range, or reordered-plan data is rejected.
    /// True containment flags mean the probe overlaps an actual solid. Both
    /// endpoint flags and both directed traversals must be clear for a sample.
    pub fn evaluate(
        &self,
        plan_sha256: &str,
        free_distance_m: &[f64],
        endpoint_inside_solid: &[bool],
    ) -> EcologyResult<Vec<f64>> {
        if plan_sha256 != self.sha256
            || free_distance_m.len() != self.rays.len()
            || endpoint_inside_solid.len() != self.endpoints_m.len()
        {
            return Err(fail(
                "route geometry measurement identity or dimensions differ",
            ));
        }
        for (ray, distance) in self.rays.iter().zip(free_distance_m) {
            if !distance.is_finite()
                || *distance < 0.0
                || *distance > ray.max_distance_m + ROUTE_DISTANCE_TOLERANCE_M
            {
                return Err(fail("invalid measured route free distance"));
            }
        }
        Ok(self
            .routes
            .iter()
            .map(|route| {
                let clear = (route.first_ray..route.first_ray + route.ray_count)
                    .step_by(2)
                    .filter(|&i| {
                        let forward = &self.rays[i];
                        !endpoint_inside_solid[forward.origin_endpoint]
                            && !endpoint_inside_solid[forward.destination_endpoint]
                            && free_distance_m[i] + ROUTE_DISTANCE_TOLERANCE_M
                                >= forward.max_distance_m
                            && free_distance_m[i + 1] + ROUTE_DISTANCE_TOLERANCE_M
                                >= forward.max_distance_m
                    })
                    .count();
                clear as f64 / ROUTE_APERTURE_SAMPLES as f64
            })
            .collect())
    }
}

/// Private cached physical measurements. Save this with the physical world.
/// A topology revision changes when solids are constructed/removed; ordinary
/// rigid-body movement is observed at the fixed 0.1-second cadence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RouteGeometryState {
    format: String,
    plan_sha256: String,
    measured_at_tick: Option<u64>,
    topology_revision: Option<u64>,
    route_open_fraction: Vec<f64>,
}

impl RouteGeometryState {
    pub fn new(plan: &RouteGeometryPlan) -> Self {
        Self {
            format: ROUTE_GEOMETRY_STATE_FORMAT.into(),
            plan_sha256: plan.sha256.clone(),
            measured_at_tick: None,
            topology_revision: None,
            route_open_fraction: vec![0.0; plan.routes.len()],
        }
    }
    pub fn validate(&self, plan: &RouteGeometryPlan) -> EcologyResult<()> {
        if self.format != ROUTE_GEOMETRY_STATE_FORMAT
            || self.plan_sha256 != plan.sha256
            || self.route_open_fraction.len() != plan.routes.len()
            || self
                .route_open_fraction
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || self.measured_at_tick.is_some() != self.topology_revision.is_some()
            || (self.measured_at_tick.is_none()
                && self.route_open_fraction.iter().any(|x| *x != 0.0))
        {
            return Err(fail("invalid route geometry checkpoint or plan identity"));
        }
        Ok(())
    }
    pub fn refresh_due(
        &self,
        plan: &RouteGeometryPlan,
        tick: u64,
        topology_revision: u64,
    ) -> EcologyResult<bool> {
        self.validate(plan)?;
        match self.measured_at_tick {
            None => Ok(true),
            Some(previous) if tick < previous => {
                Err(fail("route measurement tick moved backwards"))
            }
            Some(previous) => Ok(self.topology_revision != Some(topology_revision)
                || tick - previous >= ROUTE_GEOMETRY_REFRESH_TICKS),
        }
    }
    pub fn refresh(
        &mut self,
        plan: &RouteGeometryPlan,
        tick: u64,
        topology_revision: u64,
        plan_sha256: &str,
        free_distance_m: &[f64],
        endpoint_inside_solid: &[bool],
    ) -> EcologyResult<()> {
        self.refresh_due(plan, tick, topology_revision)?;
        let measured = plan.evaluate(plan_sha256, free_distance_m, endpoint_inside_solid)?;
        self.route_open_fraction = measured;
        self.measured_at_tick = Some(tick);
        self.topology_revision = Some(topology_revision);
        Ok(())
    }
    pub fn openness<'a>(
        &'a self,
        plan: &RouteGeometryPlan,
        tick: u64,
        topology_revision: u64,
    ) -> EcologyResult<&'a [f64]> {
        if self.refresh_due(plan, tick, topology_revision)? {
            return Err(fail("fresh physical route measurements are required"));
        }
        Ok(&self.route_open_fraction)
    }
}
