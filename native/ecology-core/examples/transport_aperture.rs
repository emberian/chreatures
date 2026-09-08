//! One native joined aperture/transport/cache scenario with analytic sphere rays.
//! It validates the shared kernel, not a MuJoCo host integration.
use chreatures_ecology_core::*;
use serde_json::json;

fn measurements(
    plan: &RouteGeometryPlan,
    sphere: Option<([f64; 3], f64)>,
) -> (Vec<f64>, Vec<bool>) {
    let inside = plan
        .endpoints_m
        .iter()
        .map(|p| {
            sphere.is_some_and(|(c, r)| (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f64>() <= r * r)
        })
        .collect();
    let distances = plan
        .rays
        .iter()
        .map(|ray| {
            let Some((center, radius)) = sphere else {
                return ray.max_distance_m;
            };
            let p = plan.endpoints_m[ray.origin_endpoint];
            let offset: [f64; 3] = std::array::from_fn(|i| p[i] - center[i]);
            let b = (0..3).map(|i| offset[i] * ray.direction[i]).sum::<f64>();
            let c = offset.iter().map(|x| x * x).sum::<f64>() - radius * radius;
            let d = b * b - c;
            if d < 0.0 {
                return ray.max_distance_m;
            }
            [-b - d.sqrt(), -b + d.sqrt()]
                .into_iter()
                .filter(|t| *t >= 0.0)
                .fold(ray.max_distance_m, f64::min)
        })
        .collect();
    (distances, inside)
}
fn step(world: &mut EcologyWorld, openness: f64) -> EcologyResult<()> {
    let input = TickInput {
        growth_token: None,
        dt_s: 0.01,
        route_open_fraction: vec![openness],
        route_advection_m3_s: vec![0.0],
        contacts: vec![],
        directed_exchanges: vec![],
        photon_exposures: vec![],
        metabolic_work_demands: vec![],
        construction_sites: vec![],
        birth_sites: vec![],
        packet_retirements: vec![],
        afferent_sample_sites: vec![],
    };
    let delta = world.prepare_step(&input)?;
    world.commit_step(&CommitReceipt::accept_all(&delta))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/finite-garden-v2.json"))?;
    let mut config: EcologyConfig = serde_json::from_value(fixture["ecology"].clone())?;
    config.organisms.clear();
    config.packets.clear();
    config.regions.truncate(2);
    config.routes.truncate(1);
    config.routes[0].base_open_fraction = 0.37;
    for pool in &mut config.pools {
        pool.diffusivity_m2_s = 1e-6;
    }
    let pools = config.pools.len();
    for (i, region) in config.regions.iter_mut().enumerate() {
        region.capacity = vec![2.0; pools];
        region.initial = vec![if i == 0 { 0.1 } else { 0.0 }; pools];
    }
    let plan = RouteGeometryPlan::new(&config.regions, &config.routes)?;
    assert_eq!(
        plan,
        RouteGeometryPlan::new(&config.regions, &config.routes)?
    );
    assert_eq!(plan.rays.len(), 64);
    assert_eq!(plan.endpoints_m.len(), 64);
    let mut cache = RouteGeometryState::new(&plan);
    assert!(cache.openness(&plan, 0, 0).is_err());
    let (clear, outside) = measurements(&plan, None);
    cache.refresh(&plan, 0, 0, &plan.sha256, &clear, &outside)?;
    assert_eq!(cache.openness(&plan, 9, 0)?, &[1.0]);
    assert!(cache.refresh_due(&plan, 10, 0)?);
    let state_json = serde_json::to_string(&cache)?;
    let mut restored: RouteGeometryState = serde_json::from_str(&state_json)?;
    restored.validate(&plan)?;
    assert_eq!(restored, cache);
    assert!(restored
        .refresh(&plan, 9, 0, "wrong-plan", &clear, &outside)
        .is_err());
    assert_eq!(state_json, serde_json::to_string(&restored)?);
    let p = plan.endpoints_m[plan.rays[2].origin_endpoint];
    let q = plan.endpoints_m[plan.rays[2].destination_endpoint];
    let sphere_center = std::array::from_fn(|i| (p[i] + q[i]) * 0.5);
    let sphere_radius = plan.routes[0].aperture_radius_m * 0.1;
    // Off-axis grain: the center ray misses it, but one area cell is blocked.
    let (blocked, inside) = measurements(&plan, Some((sphere_center, sphere_radius)));
    cache.refresh(&plan, 10, 0, &plan.sha256, &blocked, &inside)?;
    let partial = cache.openness(&plan, 10, 0)?[0];
    assert_eq!(partial, 31.0 / 32.0);
    assert!(cache.openness(&plan, 10, 1).is_err());
    // A thick solid encloses both endpoints. Ray exits lie beyond the route;
    // the mandatory endpoint flags still close it rather than reporting clear.
    let center = std::array::from_fn(|i| {
        (config.regions[0].center_m[i] + config.regions[1].center_m[i]) * 0.5
    });
    let (thick, inside_thick) =
        measurements(&plan, Some((center, plan.routes[0].center_distance_m)));
    assert!(thick
        .iter()
        .zip(&plan.rays)
        .all(|(d, r)| *d == r.max_distance_m));
    assert!(inside_thick.iter().all(|x| *x));
    restored.refresh(&plan, 10, 1, &plan.sha256, &thick, &inside_thick)?;
    assert_eq!(restored.openness(&plan, 10, 1)?, &[0.0]);
    restored.refresh(&plan, 10, 2, &plan.sha256, &clear, &outside)?;
    assert_eq!(restored.openness(&plan, 10, 2)?, &[1.0]);
    let mut flux = vec![];
    let mut residual: f64 = 0.0;
    for open in [1.0, partial, 0.0] {
        let mut world = EcologyWorld::new(config.clone())?;
        step(&mut world, open)?;
        flux.push(world.state().regions[1].quantity[0]);
        residual = residual.max(world.state().accounting.maximum_absolute_residual);
        let mut replay = EcologyWorld::from_snapshot_json(&world.snapshot_json()?)?;
        step(&mut world, open)?;
        step(&mut replay, open)?;
        assert_eq!(world.snapshot_json()?, replay.snapshot_json()?);
    }
    let expected = config.pools[0].diffusivity_m2_s * config.routes[0].cross_section_m2
        / config.routes[0].length_m
        * config.routes[0].base_open_fraction
        * (0.1 / config.regions[0].volume_m3)
        * 0.01;
    assert!((flux[0] - expected).abs() < 1e-15);
    assert!((flux[1] / flux[0] - partial).abs() < 1e-12);
    assert_eq!(flux[2], 0.0);
    assert!(residual < 1e-12);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "format":"chreatures-route-aperture-native-check-v1",
            "scope":"Shared Rust kernel with analytical solid-sphere ray/containment inputs; not a MuJoCo integration claim",
            "plan_sha256":plan.sha256,"area_samples":32,"opposed_rays":64,"unique_endpoints":plan.endpoints_m.len(),
            "openness_clear_partial_thick_removed":[1.0,partial,0.0,1.0],
            "center_ray_misses_partial_obstacle":true,"central_thin_obstacle_detected":true,"inside_solid_without_near_ray_hit_rejected":true,
            "transferred_quantity_clear_partial_thick":flux,"base_open_fraction_applied_once":0.37,
            "advection_m3_s":0.0,"maximum_elemental_residual":residual,
            "cache_and_ecology_checkpoint_replay":"exact","cadence_and_topology_staleness":"enforced"
        }))?
    );
    Ok(())
}
