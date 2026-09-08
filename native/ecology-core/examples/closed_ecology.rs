use chreatures_ecology_core::*;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

const W: usize = 0;
const C: usize = 1;
const A: usize = 2;
const M: usize = 3;
const O: usize = 4;
const CO: usize = 5;
const B: usize = 6;
const V: usize = 7;

fn reaction(
    id: &str,
    stoichiometry: [f64; 8],
    base_rate_quantity_s: f64,
    atp_cost_per_flux: f64,
    atp_yield_per_flux: f64,
    photon_cost_per_flux: f64,
) -> ReactionSpec {
    ReactionSpec {
        id: id.into(),
        stoichiometry: stoichiometry.to_vec(),
        base_rate_quantity_s,
        half_saturation_quantity: stoichiometry
            .iter()
            .map(|value| if *value < 0.0 { 0.04 } else { 0.0 })
            .collect(),
        atp_cost_per_flux,
        atp_yield_per_flux,
        photon_cost_per_flux,
    }
}

fn genotype(
    id: &str,
    enzymes: [f64; 6],
    permeability: [f64; 8],
    development: bool,
    reproduction: bool,
) -> Genotype {
    Genotype {
        lineage_id: id.into(),
        enzyme_baseline: enzymes.to_vec(),
        enzyme_substrate_response: vec![0.15, 0.2, 0.15, 0.2, 0.2, 0.1],
        enzyme_atp_response: vec![-0.1, 0.2, -0.1, -0.1, 0.0, 0.0],
        enzyme_time_constant_s: 1.5,
        enzyme_change_atp_cost: 0.004,
        enzyme_maximum: 1.6,
        enzyme_total_budget: 4.0,
        membrane_permeability: permeability.to_vec(),
        maintenance_atp_s: 0.002,
        development: development.then_some(DevelopmentProgram {
            branch_angle_rad: 0.65,
            lateral_probability: 0.3,
            phototropism: 0.7,
            contact_avoidance: 1.2,
            directional_persistence: 0.8,
            interval_s: 1.0,
            material_cost: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0].to_vec(),
            decay_return: [0.08, 0.08, 0.08, 0.008, 0.0, 0.0, 0.0, 0.0].to_vec(),
            atp_cost: 0.008,
            maximum_structures: 1,
            nominal_radius_m: 0.00018,
            nominal_length_m: 0.0014,
            nominal_volume_m3: 1.4e-10,
            decay_time_constant_s: 3.0,
        }),
        reproduction: reproduction.then_some(ReproductionProgram {
            dispersal_distance_m: 0.003,
            interval_s: 4.0,
            material_endowment: [0.08, 0.05, 0.03, 0.01, 0.02, 0.02, 0.08, 0.01].to_vec(),
            atp_cost: 0.03,
            atp_endowment: 0.06,
            maximum_descendants: 1,
            mutation_fraction: 0.12,
        }),
    }
}

pub fn fixture_config(seed: u64) -> EcologyConfig {
    let pools = vec![
        PoolSpec {
            id: "water".into(),
            elemental_composition: vec![0.0, 0.0, 1.0, 0.0, 0.0],
            chemical_energy_per_quantity: 0.0,
            diffusivity_m2_s: 8.0e-8,
        },
        PoolSpec {
            id: "soluble-carbon".into(),
            elemental_composition: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            chemical_energy_per_quantity: 1.0,
            diffusivity_m2_s: 4.0e-8,
        },
        PoolSpec {
            id: "amino-nutrient".into(),
            elemental_composition: vec![1.0, 1.0, 0.0, 0.0, 0.0],
            chemical_energy_per_quantity: 1.2,
            diffusivity_m2_s: 3.0e-8,
        },
        PoolSpec {
            id: "mineral".into(),
            elemental_composition: vec![0.0, 0.0, 0.0, 1.0, 0.0],
            chemical_energy_per_quantity: 0.0,
            diffusivity_m2_s: 1.0e-8,
        },
        PoolSpec {
            id: "oxygen".into(),
            elemental_composition: vec![0.0, 0.0, 0.0, 0.0, 1.0],
            chemical_energy_per_quantity: 0.0,
            diffusivity_m2_s: 9.0e-7,
        },
        PoolSpec {
            id: "oxidized-carbon".into(),
            elemental_composition: vec![1.0, 0.0, 0.0, 0.0, 1.0],
            chemical_energy_per_quantity: 0.0,
            diffusivity_m2_s: 6.0e-7,
        },
        PoolSpec {
            id: "biomass".into(),
            elemental_composition: vec![2.0, 1.0, 1.0, 0.1, 0.0],
            chemical_energy_per_quantity: 2.0,
            diffusivity_m2_s: 1.0e-10,
        },
        PoolSpec {
            id: "volatile-carbon".into(),
            elemental_composition: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            chemical_energy_per_quantity: 0.8,
            diffusivity_m2_s: 1.1e-6,
        },
    ];
    let reactions = vec![
        reaction(
            "catabolism",
            [0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
            0.035,
            0.0,
            0.72,
            0.0,
        ),
        reaction(
            "assimilation",
            [-1.0, -1.0, -1.0, -0.1, 0.0, 0.0, 1.0, 0.0],
            0.012,
            0.2,
            0.0,
            0.0,
        ),
        reaction(
            "mineralization",
            [1.0, 1.0, 1.0, 0.1, 0.0, 0.0, -1.0, 0.0],
            0.022,
            0.3,
            0.0,
            0.0,
        ),
        reaction(
            "photo-reduction",
            [0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            0.04,
            0.0,
            0.0,
            1.25,
        ),
        reaction(
            "volatile-synthesis",
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            0.018,
            0.0,
            0.0,
            0.0,
        ),
        reaction(
            "volatile-decay",
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0],
            0.018,
            0.0,
            0.0,
            0.0,
        ),
    ];
    let region_capacity = vec![20.0, 10.0, 10.0, 10.0, 10.0, 12.0, 8.0, 8.0];
    EcologyConfig {
        format: CONFIG_FORMAT.into(),
        coordinate_contract: CoordinateContract {
            length_unit: "meter".into(),
            time_unit: "second".into(),
            material_unit: "abstract_conserved_quantity".into(),
            world_min_m: [0.0, 0.0, 0.0],
            world_max_m: [0.012, 0.008, 0.004],
        },
        elements: vec![
            "carbon".into(),
            "nitrogen".into(),
            "water".into(),
            "mineral".into(),
            "oxygen".into(),
        ],
        pools,
        reactions,
        regions: vec![
            RegionSpec {
                id: "west".into(),
                center_m: [0.002, 0.004, 0.001],
                volume_m3: 1.5e-7,
                capacity: region_capacity.clone(),
                initial: vec![6.0, 3.0, 1.4, 1.2, 3.0, 0.4, 0.0, 0.3],
            },
            RegionSpec {
                id: "middle".into(),
                center_m: [0.006, 0.004, 0.001],
                volume_m3: 1.5e-7,
                capacity: region_capacity.clone(),
                initial: vec![5.0, 1.0, 0.4, 0.4, 2.0, 0.2, 0.0, 0.0],
            },
            RegionSpec {
                id: "east".into(),
                center_m: [0.010, 0.004, 0.001],
                volume_m3: 1.5e-7,
                capacity: region_capacity,
                initial: vec![4.0, 0.3, 0.2, 0.2, 1.0, 0.2, 0.0, 0.0],
            },
        ],
        routes: vec![
            RouteSpec {
                id: "west-middle".into(),
                a: "west".into(),
                b: "middle".into(),
                length_m: 0.004,
                cross_section_m2: 2.0e-6,
                hydraulic_capacity_m3_s: 2.0e-9,
                base_open_fraction: 1.0,
            },
            RouteSpec {
                id: "middle-east".into(),
                a: "middle".into(),
                b: "east".into(),
                length_m: 0.004,
                cross_section_m2: 2.0e-6,
                hydraulic_capacity_m3_s: 2.0e-9,
                base_open_fraction: 1.0,
            },
        ],
        organisms: vec![
            OrganismSeed {
                id: "resident-a".into(),
                physics_binding: "body-a".into(),
                anchored_region: Some("west".into()),
                internal_volume_m3: 4.0e-9,
                capacity: vec![3.0, 2.0, 3.0, 2.0, 4.0, 3.0, 2.0, 2.0],
                initial: vec![0.8, 1.5, 0.3, 0.25, 0.2, 0.1, 0.9, 0.1],
                atp: 0.8,
                atp_capacity: 1.0,
                genotype: genotype(
                    "lineage-a",
                    [0.3, 0.8, 0.08, 1.25, 0.3, 0.15],
                    [0.8, 0.8, 0.9, 0.7, 0.4, 0.05, 0.8, 0.7],
                    true,
                    true,
                ),
            },
            OrganismSeed {
                id: "resident-b".into(),
                physics_binding: "body-b".into(),
                anchored_region: None,
                internal_volume_m3: 3.0e-9,
                capacity: vec![2.0, 1.5, 2.5, 1.5, 3.0, 2.5, 1.5, 1.5],
                initial: vec![0.5, 0.35, 0.1, 0.1, 0.15, 0.08, 0.25, 0.08],
                atp: 0.35,
                atp_capacity: 0.7,
                genotype: genotype(
                    "lineage-b",
                    [1.25, 0.08, 1.25, 0.03, 0.65, 0.35],
                    [0.3, 0.5, 0.8, 0.3, 0.7, 1.0, 0.9, 0.9],
                    false,
                    false,
                ),
            },
        ],
        packets: vec![PacketSeed {
            id: "packet-food".into(),
            physics_binding: "food-body".into(),
            volume_m3: 8.0e-10,
            capacity: vec![0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 1.4, 0.3],
            initial: vec![0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.0],
        }],
        seed,
    }
}

fn fly_interoception_spec() -> FlyInteroceptionSpec {
    FlyInteroceptionSpec {
        gut_pool_ids: vec![
            "biomass".into(),
            "soluble-carbon".into(),
            "amino-nutrient".into(),
        ],
        carbon_reserve_pool_ids: vec!["soluble-carbon".into(), "amino-nutrient".into()],
        nitrogen_reserve_pool_ids: vec!["amino-nutrient".into()],
        water_pool_id: "water".into(),
        structural_pool_ids: vec!["biomass".into()],
        salivary_pool_id: "water".into(),
        oxygen_pool_id: "oxygen".into(),
        osmotic_pool_ids: vec![
            "soluble-carbon".into(),
            "amino-nutrient".into(),
            "water".into(),
        ],
        osmotic_target_quantity_m3: 3.0e8,
        osmotic_scale_quantity_m3: 1.0e8,
    }
}

fn fly_chemical_sense_spec() -> FlyChemicalSenseSpec {
    FlyChemicalSenseSpec {
        // water, soluble C, amino nutrient, mineral, oxygen, oxidized C,
        // biomass, volatile C
        odor_permeability: vec![0.01, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        half_saturation_quantity_m3: vec![1.0e7; 8],
    }
}

fn export_recipe(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let payload = json!({
        "format": "chreatures-finite-garden-export-v1",
        "ecology": fixture_config(0x5eed_2026_0908),
        "fly_interoception": fly_interoception_spec(),
        "fly_chemical_sense": fly_chemical_sense_spec(),
        "pool_order": [
            "water",
            "soluble-carbon",
            "amino-nutrient",
            "mineral",
            "oxygen",
            "oxidized-carbon",
            "biomass",
            "volatile-carbon"
        ],
        "host_exchange_contract": {
            "respiratory_surface_conductance_m3_s_by_pool": [
                2.0e-14, 0.0, 0.0, 0.0, 1.0e-12, 1.0e-12, 0.0, 1.0e-12
            ],
            "respiratory_exchange": "passive ContactEvent between measured local region and organism; inherited membrane_permeability is applied by ecology-core",
            "mouth_intake": "DirectedExchange only after measured mouth contact and pump actuation",
            "saliva": "DirectedExchange of finite organism water to the measured contacted store",
            "controller_visibility": "chemical and interoceptive values enter only masked CNS afferents"
        }
    });
    std::fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

#[derive(Default)]
struct MockPhysicalHost {
    blockers: BTreeMap<String, String>,
    created: usize,
    removed: usize,
}

impl MockPhysicalHost {
    fn openness(&self) -> Vec<f64> {
        ["west-middle", "middle-east"]
            .iter()
            .map(|route| {
                if self.blockers.values().any(|value| value == route) {
                    0.18
                } else {
                    1.0
                }
            })
            .collect()
    }

    fn commit(&mut self, delta: &WorldDelta) -> CommitReceipt {
        for creation in &delta.physical_creations {
            self.created += 1;
            if let Some(route) = &creation.route_hint {
                self.blockers
                    .insert(creation.physics_binding.clone(), route.clone());
            }
        }
        for removal in &delta.physical_removals {
            self.removed += 1;
            self.blockers.remove(&removal.physics_binding);
        }
        CommitReceipt::accept_all(delta)
    }
}

fn tick_input(world: &EcologyWorld, host: &MockPhysicalHost, step: usize) -> TickInput {
    let packet_active = world
        .state()
        .packets
        .iter()
        .any(|value| value.id == "packet-food" && value.active);
    let structure_has_been_built = !world.state().structures.is_empty();
    let child_exists = world
        .state()
        .organisms
        .iter()
        .any(|value| value.id == "resident-a-child");
    let structure_serial = world.state().structures.len();
    let contacts = vec![
        ContactEvent {
            a: StoreId::Region("west".into()),
            b: StoreId::Organism("resident-a".into()),
            conductance_m3_s: vec![2e-10, 2e-10, 2e-10, 2e-10, 2e-11, 0.0, 2e-10, 1e-10],
        },
        ContactEvent {
            a: StoreId::Region("middle".into()),
            b: StoreId::Organism("resident-b".into()),
            conductance_m3_s: vec![1e-10, 1e-10, 1e-10, 5e-11, 1e-10, 0.0, 2e-10, 2e-10],
        },
    ];
    let mut directed_exchanges = Vec::new();
    if packet_active && step < 100 {
        let mut intake = vec![0.0; 8];
        intake[B] = 0.008;
        directed_exchanges.push(DirectedExchange {
            event_id: format!("pump-{step:04}"),
            from: StoreId::Packet("packet-food".into()),
            to: StoreId::Organism("resident-b".into()),
            maximum_quantity: intake,
        });
        let mut saliva = vec![0.0; 8];
        saliva[W] = 0.001;
        directed_exchanges.push(DirectedExchange {
            event_id: format!("saliva-{step:04}"),
            from: StoreId::Organism("resident-b".into()),
            to: StoreId::Packet("packet-food".into()),
            maximum_quantity: saliva,
        });
    }
    TickInput {
        growth_token: None,
        dt_s: 0.25,
        route_open_fraction: host.openness(),
        route_advection_m3_s: vec![7.0e-10, 4.0e-10],
        contacts,
        directed_exchanges,
        photon_exposures: vec![
            PhotonExposure {
                organism_id: "resident-a".into(),
                available_energy_per_s: 0.3,
            },
            PhotonExposure {
                organism_id: "resident-b".into(),
                available_energy_per_s: 0.0,
            },
        ],
        metabolic_work_demands: vec![MetabolicWorkDemand {
            organism_id: "resident-b".into(),
            atp_quantity: 0.0015,
        }],
        construction_sites: if !structure_has_been_built {
            vec![ConstructionSite {
                site_id: format!("build-site-{structure_serial:03}"),
                organism_id: "resident-a".into(),
                region_id: "west".into(),
                host_template_id: "fiber-capsule".into(),
                physics_binding: format!("fiber-body-{structure_serial:03}"),
                position_m: [0.0039, 0.004, 0.0008],
                orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
                route_hint: Some("west-middle".into()),
            }]
        } else {
            Vec::new()
        },
        birth_sites: if child_exists {
            Vec::new()
        } else {
            vec![BirthSite {
                site_id: "birth-site-a".into(),
                parent_id: "resident-a".into(),
                child_id: "resident-a-child".into(),
                host_template_id: "fly-body-mm-v1".into(),
                physics_binding: "body-a-child".into(),
                anchored_region: None,
                position_m: [0.0025, 0.0036, 0.0012],
                orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
                internal_volume_m3: 2.5e-9,
                capacity: vec![1.0, 0.8, 1.2, 0.8, 1.5, 1.0, 0.8, 0.8],
                atp_capacity: 0.5,
                nominal_radius_m: 0.00055,
                nominal_length_m: 0.0028,
            }]
        },
        packet_retirements: if packet_active && step == 100 {
            vec![PacketRetirement {
                packet_id: "packet-food".into(),
                receiver_region_id: "middle".into(),
                cause: "measured-east-exit".into(),
            }]
        } else {
            Vec::new()
        },
        afferent_sample_sites: vec![
            AfferentSampleSite {
                site_id: "antenna-a".into(),
                region_id: "west".into(),
            },
            AfferentSampleSite {
                site_id: "antenna-b".into(),
                region_id: "middle".into(),
            },
        ],
    }
}

fn quiet_tick() -> TickInput {
    TickInput {
        growth_token: None,
        dt_s: 0.05,
        route_open_fraction: vec![1.0, 1.0],
        route_advection_m3_s: vec![0.0, 0.0],
        contacts: Vec::new(),
        directed_exchanges: Vec::new(),
        photon_exposures: Vec::new(),
        metabolic_work_demands: Vec::new(),
        construction_sites: Vec::new(),
        birth_sites: Vec::new(),
        packet_retirements: Vec::new(),
        afferent_sample_sites: Vec::new(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() == 3 && args[1] == "--export-config" {
        let path = Path::new(&args[2]);
        export_recipe(path)?;
        println!("{}", path.display());
        return Ok(());
    }
    if args.len() != 1 {
        return Err("usage: closed_ecology [--export-config PATH]".into());
    }
    let config = fixture_config(0x5eed_2026_0908);
    let mut world = EcologyWorld::new(config.clone())?;
    let mut host = MockPhysicalHost::default();
    let initial = world.state().accounting.initial_elements.clone();
    let mut route_transfers = 0usize;
    let mut active_exchanges = 0usize;
    let mut construction_creations = 0usize;
    let mut offspring_creations = 0usize;
    let mut removals = 0usize;
    let mut structure_removals = 0usize;
    let mut minimum_open = 1.0_f64;
    let mut route_was_blocked = false;
    let mut route_reopened = false;
    let mut last_medium_sample = None;

    for step in 0..220 {
        let input = tick_input(&world, &host, step);
        minimum_open = minimum_open.min(input.route_open_fraction[0]);
        if input.route_open_fraction[0] < 0.2 {
            route_was_blocked = true;
        } else if route_was_blocked {
            route_reopened = true;
        }
        if step == 36 {
            let before_abort = world.snapshot_json()?;
            let aborted = world.prepare_step(&input)?;
            world.abort_step(&aborted.token)?;
            assert_eq!(before_abort, world.snapshot_json()?);
        }
        let delta = world.prepare_step(&input)?;
        route_transfers += delta
            .transfers
            .iter()
            .filter(|value| value.cause == TransferCause::RouteDiffusionAdvection)
            .count();
        active_exchanges += delta
            .transfers
            .iter()
            .filter(|value| value.cause == TransferCause::MeasuredDirectedExchange)
            .count();
        construction_creations += delta
            .physical_creations
            .iter()
            .filter(|value| value.kind == PhysicalCreationKind::ConstructedGeometry)
            .count();
        offspring_creations += delta
            .physical_creations
            .iter()
            .filter(|value| value.kind == PhysicalCreationKind::OffspringBody)
            .count();
        removals += delta.physical_removals.len();
        structure_removals += delta
            .physical_removals
            .iter()
            .filter(|value| value.cause == "resource-returned-decay")
            .count();
        last_medium_sample = delta
            .afferent_transduction_samples
            .iter()
            .find(|value| value.site_id == "antenna-b")
            .cloned();
        assert!(delta
            .elemental_residual
            .iter()
            .all(|value| value.abs() < 1e-8));
        let receipt = host.commit(&delta);
        if step == 64 {
            // A pending snapshot contains the uncommitted successor, private
            // enzyme/RNG state and exact physical delta. Both restored worlds
            // continue identically after the same host receipt.
            let pending_snapshot = world.snapshot_json()?;
            let mut restored = EcologyWorld::from_snapshot_json(&pending_snapshot)?;
            world.commit_step(&receipt)?;
            restored.commit_step(&receipt)?;
            assert_eq!(world.snapshot_json()?, restored.snapshot_json()?);
        } else {
            world.commit_step(&receipt)?;
        }
    }

    let interoception = world.fly_interoception12("resident-b", &fly_interoception_spec())?;
    assert!(interoception[..11]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    assert!(interoception[11].is_finite() && (-1.0..=1.0).contains(&interoception[11]));
    let chemical =
        world.fly_chemical_projection(&last_medium_sample.unwrap(), &fly_chemical_sense_spec())?;
    assert_eq!(chemical.odor.len(), 8);
    assert_eq!(chemical.taste.len(), 8);
    assert_eq!(chemical.odor[C], 0.0);
    assert_eq!(chemical.odor[A], 0.0);
    assert_eq!(chemical.odor[M], 0.0);
    assert_eq!(chemical.odor[B], 0.0);
    assert!(chemical.odor[W] <= chemical.taste[W] * 0.0100000001);
    assert_eq!(chemical.odor[O], chemical.taste[O]);
    assert_eq!(chemical.odor[CO], chemical.taste[CO]);
    assert_eq!(chemical.odor[V], chemical.taste[V]);
    assert!(route_transfers > 0);
    assert!(active_exchanges > 0);
    assert!(construction_creations > 0);
    assert_eq!(offspring_creations, 1);
    assert!(removals > 0);
    assert_eq!(structure_removals, 1);
    assert!(minimum_open < 0.2);
    assert!(route_reopened);
    let child = world
        .state()
        .organisms
        .iter()
        .find(|value| value.id == "resident-a-child")
        .unwrap();
    let parent = world
        .state()
        .organisms
        .iter()
        .find(|value| value.id == "resident-a")
        .unwrap();
    assert_ne!(child.genotype.lineage_id, parent.genotype.lineage_id);
    assert_ne!(
        child.genotype.enzyme_baseline,
        parent.genotype.enzyme_baseline
    );
    assert!(world
        .state()
        .accounting
        .last_elements
        .iter()
        .zip(&initial)
        .all(|(a, b)| (a - b).abs() < 1e-8));

    // Exercise the multi-world host path and exact batch restoration separately
    // from the longer physical mock-host continuation above.
    let mut batch = EcologyBatch::new(vec![config.clone(), fixture_config(0x5eed_2026_0909)])?;
    let deltas = batch.prepare_batch(&[quiet_tick(), quiet_tick()])?;
    batch.commit_batch(
        &deltas
            .iter()
            .map(CommitReceipt::accept_all)
            .collect::<Vec<_>>(),
    )?;
    let batch_snapshot = batch.snapshot_json()?;
    let restored_batch = EcologyBatch::from_snapshot_json(&batch_snapshot)?;
    assert_eq!(batch_snapshot, restored_batch.snapshot_json()?);

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "format": "chreatures-closed-ecology-example-v1",
            "coordinate_units": {"length": "meter", "time": "second"},
            "ticks": world.state().step_index,
            "simulated_seconds": world.state().time_s,
            "regions": world.state().regions.len(),
            "organisms": world.state().organisms.len(),
            "structures_total": world.state().structures.len(),
            "route_transfer_records": route_transfers,
            "directed_contact_transfer_records": active_exchanges,
            "physical_creations": host.created,
            "physical_removals": host.removed,
            "structure_decay_removals": structure_removals,
            "route_reopened_after_decay": route_reopened,
            "minimum_measured_route_open_fraction": minimum_open,
            "initial_elements": initial,
            "final_elements": world.state().accounting.last_elements,
            "maximum_absolute_elemental_residual": world.state().accounting.maximum_absolute_residual,
            "resident_b_interoception12": interoception,
            "resident_b_odor8": chemical.odor,
            "resident_b_taste8": chemical.taste,
            "batch_worlds_restored": restored_batch.len(),
            "claims": [
                "host-measured pump and saliva gates, not learned feeding competence",
                "inherited response parameters, not scripted ecological roles",
                "host-only chemistry samples must be transduced through CNS afferents"
            ]
        }))?
    );
    Ok(())
}
