//! Joined native scenario with an explicit analytical capsule-clearance host.
//! This is not a MuJoCo integration receipt; the production host performs that
//! separate final geometry check against its actual compiled world.
#[allow(dead_code)]
#[path = "closed_ecology.rs"]
mod garden;
use chreatures_ecology_core::*;
use serde_json::json;

fn local(id: &str) -> ColonyGrowthInput {
    ColonyGrowthInput {
        organism_id: id.into(),
        position_m: [0.003, 0.004, 0.0002],
        orientation_xyzw: [0., 0., 0., 1.],
        surface_normal: [0., 0., 1.],
        light_direction: [1., 0., 0.4],
        light_intensity: 0.8,
        nearby_surfaces: vec![GrowthSurface {
            surface_id: "ground-nearby".into(),
            region_id: "west".into(),
            point_m: [0.0044, 0.004, 0.],
            normal: [0., 0., 1.],
            attachable: true,
        }],
        clearance_samples: Vec::new(),
        host_template_id: "capsule-colony-segment".into(),
        child_template_id: Some("anchored-colony".into()),
    }
}
fn input(world: &EcologyWorld, proposal: Option<&GrowthProposals>, accept: bool) -> TickInput {
    let openness = world
        .config()
        .routes
        .iter()
        .map(|route| {
            let a = world
                .config()
                .regions
                .iter()
                .find(|r| r.id == route.a)
                .unwrap()
                .center_m;
            let b = world
                .config()
                .regions
                .iter()
                .find(|r| r.id == route.b)
                .unwrap()
                .center_m;
            // Conservative host route clearance from the actual installed capsule
            // bounding spheres. This deliberately reports its analytical scope.
            if world
                .state()
                .structures
                .iter()
                .filter(|s| s.active)
                .any(|s| {
                    let ab = std::array::from_fn::<_, 3, _>(|i| b[i] - a[i]);
                    let ap = std::array::from_fn::<_, 3, _>(|i| s.position_m[i] - a[i]);
                    let t = (ap.iter().zip(ab).map(|(x, y)| x * y).sum::<f64>()
                        / ab.iter().map(|v| v * v).sum::<f64>())
                    .clamp(0., 1.);
                    let distance = (0..3)
                        .map(|i| (s.position_m[i] - a[i] - t * ab[i]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    distance
                        < (route.cross_section_m2 / std::f64::consts::PI).sqrt()
                            + s.nominal_length_m * 0.5
                            + s.nominal_radius_m
                })
            {
                0.05
            } else {
                1.0
            }
        })
        .collect();
    TickInput {
        growth_token: proposal.map(|p| p.token.clone()),
        dt_s: 0.05,
        route_open_fraction: openness,
        route_advection_m3_s: vec![1e-10, 0.],
        contacts: Vec::new(),
        directed_exchanges: Vec::new(),
        photon_exposures: vec![PhotonExposure {
            organism_id: "resident-a".into(),
            available_energy_per_s: 0.25,
        }],
        metabolic_work_demands: Vec::new(),
        construction_sites: if accept {
            proposal
                .map(|p| p.construction_sites.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        },
        birth_sites: if accept {
            proposal.map(|p| p.birth_sites.clone()).unwrap_or_default()
        } else {
            Vec::new()
        },
        packet_retirements: Vec::new(),
        afferent_sample_sites: Vec::new(),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = garden::fixture_config(20260909);
    let development = config.organisms[0].genotype.development.as_mut().unwrap();
    development.interval_s = 0.05;
    development.maximum_structures = 12;
    development.nominal_length_m = 0.0008;
    development.nominal_radius_m = 0.00006;
    development.lateral_probability = 1.;
    development.decay_time_constant_s = 30.;
    let reproduction = config.organisms[0].genotype.reproduction.as_mut().unwrap();
    reproduction.interval_s = 0.15;
    reproduction.dispersal_distance_m = 0.003;
    let mut world = EcologyWorld::new(config.clone())?;
    let growth = GrowthInput {
        dt_s: 0.05,
        colonies: vec![local("resident-a"), local("resident-b")],
    };
    let original_private = world.state().organisms[0].development_state.clone();
    let proposal = world.propose_growth(&growth)?;
    assert_eq!(proposal.construction_sites.len(), 1);
    assert_eq!(proposal.clearance_queries[0].kind, GrowthKind::Apical);
    assert!(proposal.birth_sites.is_empty());
    assert!(proposal
        .clearance_queries
        .iter()
        .all(|q| q.organism_id == "resident-a"));
    let outstanding = world.snapshot_json()?;
    let mut restored = EcologyWorld::from_snapshot_json(&outstanding)?;
    assert_eq!(restored.propose_growth(&growth)?, proposal);
    let accepted = input(&world, Some(&proposal), true);
    let prepared = world.prepare_step(&accepted)?;
    assert_eq!(prepared.physical_creations.len(), 1);
    let pending = world.snapshot_json()?;
    assert!(world
        .commit_step(&CommitReceipt {
            token: prepared.token.clone(),
            created: vec![],
            removed: vec![]
        })
        .is_err());
    assert_eq!(world.snapshot_json()?, pending);
    world.abort_step(&prepared.token)?;
    assert_eq!(world.snapshot_json()?, outstanding);
    // A committed rejection records an attempted developmental choice. It
    // advances private RNG, but cannot install an apical node/material allocation.
    let rejected = world.prepare_step(&input(&world, Some(&proposal), false))?;
    assert!(rejected.physical_creations.is_empty());
    world.commit_step(&CommitReceipt::accept_all(&rejected))?;
    let rejected_private = &world.state().organisms[0].development_state;
    assert_ne!(rejected_private.rng, original_private.rng);
    assert_eq!(rejected_private.attempt_index, original_private.attempt_index + 1);
    assert_eq!(rejected_private.apical_binding, original_private.apical_binding);
    assert_eq!(rejected_private.lateral_cursor, original_private.lateral_cursor);
    assert!(world.state().structures.is_empty());
    // Restore the exact pre-rejection world and execute its accepted topology.
    world = restored;
    let delta = world.prepare_step(&accepted)?;
    world.commit_step(&CommitReceipt::accept_all(&delta))?;
    assert_ne!(
        world.state().organisms[0].development_state,
        original_private
    );
    assert_eq!(world.state().structures.len(), 1);
    let first = &world.state().structures[0];
    assert!(first.position_m[0] > growth.colonies[0].position_m[0]); // measured light response
    assert_eq!(
        world.state().organisms[0]
            .development_state
            .apical_binding
            .as_ref(),
        Some(&first.physics_binding)
    );
    // The same measured tip with a short clearance ray bends away locally.
    let base = world.snapshot_json()?;
    let plain = world.propose_growth(&growth)?;
    let query = plain
        .clearance_queries
        .iter()
        .find(|q| q.kind == GrowthKind::Lateral)
        .unwrap();
    let mut obstructed_growth = growth.clone();
    let direction = std::array::from_fn(|i| {
        (query.to_m[i] - query.from_m[i])
            / config.organisms[0]
                .genotype
                .development
                .as_ref()
                .unwrap()
                .nominal_length_m
    });
    obstructed_growth.colonies[0]
        .clearance_samples
        .push(GrowthClearanceSample {
            origin_m: query.from_m,
            direction,
            free_distance_m: 0.,
        });
    let mut obstructed = EcologyWorld::from_snapshot_json(&base)?;
    let avoided = obstructed.propose_growth(&obstructed_growth)?;
    assert_ne!(plain.construction_sites, avoided.construction_sites);
    world.discard_growth(&plain.token)?;
    let mut apical = 1;
    let mut lateral = 0;
    let mut colony_births = 0;
    let mut route_changed = false;
    for _ in 0..8 {
        let proposal = world.propose_growth(&growth)?;
        let tick = input(&world, Some(&proposal), true);
        route_changed |= tick.route_open_fraction.iter().any(|v| *v < 1.);
        let delta = world.prepare_step(&tick)?;
        for creation in &delta.physical_creations {
            if creation.kind == PhysicalCreationKind::OffspringBody {
                colony_births += 1;
            } else {
                let site = proposal
                    .construction_sites
                    .iter()
                    .find(|s| s.physics_binding == creation.physics_binding)
                    .unwrap();
                match proposal
                    .clearance_queries
                    .iter()
                    .find(|q| q.site_id == site.site_id)
                    .unwrap()
                    .kind
                {
                    GrowthKind::Apical => apical += 1,
                    GrowthKind::Lateral => lateral += 1,
                    _ => unreachable!(),
                }
            }
        }
        let mut replay = EcologyWorld::from_snapshot_json(&world.snapshot_json()?)?;
        let receipt = CommitReceipt::accept_all(&delta);
        world.commit_step(&receipt)?;
        replay.commit_step(&receipt)?;
        assert_eq!(world.snapshot_json()?, replay.snapshot_json()?);
    }
    assert!(lateral > 0 && colony_births == 1 && route_changed);
    let child = world
        .state()
        .organisms
        .iter()
        .find(|o| o.generation == 1)
        .unwrap();
    assert!(child.anchored_region.is_some());
    assert_ne!(
        child.genotype.development,
        config.organisms[0].genotype.development
    );
    assert_eq!(
        world
            .state()
            .organisms
            .iter()
            .filter(|o| o.anchored_region.is_none())
            .count(),
        1
    );
    // Same committed material + distinct geometry-derived openness changes
    // transport, while each branch retains its elemental ledger.
    let snapshot = world.snapshot_json()?;
    let mut open = EcologyWorld::from_snapshot_json(&snapshot)?;
    let mut blocked = EcologyWorld::from_snapshot_json(&snapshot)?;
    let blocked_input = input(&blocked, None, false);
    let mut open_input = blocked_input.clone();
    open_input.route_open_fraction.fill(1.);
    let a = open.prepare_step(&open_input)?;
    let b = blocked.prepare_step(&blocked_input)?;
    let flux = |d: &WorldDelta| {
        d.transfers
            .iter()
            .filter(|t| t.cause == TransferCause::RouteDiffusionAdvection)
            .flat_map(|t| t.quantity.iter())
            .sum::<f64>()
    };
    assert!((flux(&a) - flux(&b)).abs() > 1e-12);
    // A due but unfunded proposal records the attempted draw, while installing
    // neither a body nor a structural pointer and spending no construction cost.
    config.organisms[0]
        .genotype
        .development
        .as_mut()
        .unwrap()
        .material_cost[6] = 1.9;
    config.organisms[0]
        .genotype
        .development
        .as_mut()
        .unwrap()
        .decay_return = vec![1.9, 1.9, 1.9, 0.19, 0., 0., 0., 0.];
    let mut hungry = EcologyWorld::new(config)?;
    let old = hungry.state().organisms[0].development_state.clone();
    let p = hungry.propose_growth(&growth)?;
    let d = hungry.prepare_step(&input(&hungry, Some(&p), true))?;
    assert!(d.physical_creations.is_empty());
    assert!(d
        .blocked_events
        .iter()
        .any(|e| e.reason == "development-resource-or-atp-shortfall"));
    hungry.commit_step(&CommitReceipt::accept_all(&d))?;
    let unfunded = &hungry.state().organisms[0].development_state;
    assert_ne!(old.rng, unfunded.rng);
    assert_eq!(unfunded.attempt_index, old.attempt_index + 1);
    assert_eq!(unfunded.apical_binding, old.apical_binding);
    assert_eq!(unfunded.lateral_cursor, old.lateral_cursor);
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"format":"chreatures-ecology-growth-joined-v2","passed":true,
        "scope":"native inherited colony growth plus analytical capsule-clearance/route host; not a MuJoCo physical integration claim",
        "apical_creations":apical,"lateral_creations":lateral,"anchored_colony_births":colony_births,"mobile_births":0,
        "geometry_changed_route_clearance":route_changed,"open_route_flux":flux(&a),"blocked_route_flux":flux(&b),
        "maximum_elemental_residual":world.state().accounting.maximum_absolute_residual,
        "pending_proposal_restore_exact":true,"pending_transaction_restore_exact":true,"abort_restores_exact":true,
        "committed_rejection_records_attempt":true,"unfunded_attempt_allocates_no_structure":true,"light_and_contact_change_growth":true})
        )?
    );
    Ok(())
}
