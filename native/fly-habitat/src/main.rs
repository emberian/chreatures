// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::path::PathBuf;

const POOLS: usize = 8;
const SPAWN_CLEARANCE_MM: f64 = 1.2;

#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn signed(&mut self, amplitude: f64) -> f64 {
        (2.0 * self.unit() - 1.0) * amplitude
    }
}

#[derive(Serialize, Clone)]
struct Geometry {
    id: String,
    category: String,
    shape: String,
    material: String,
    dynamic: bool,
    position_mm: Option<[f64; 3]>,
    quaternion_wxyz: Option<[f64; 4]>,
    euler_rad: Option<[f64; 3]>,
    size_mm: Vec<f64>,
    fromto_mm: Option<[f64; 6]>,
    mass_model_units: Option<f64>,
}

#[derive(Serialize, Clone)]
struct Region {
    id: String,
    geometry_id: String,
    center_m: [f64; 3],
    volume_m3: f64,
    capacity: [f64; POOLS],
    initial: [f64; POOLS],
}

#[derive(Serialize)]
struct Route {
    id: String,
    a: String,
    b: String,
    length_m: f64,
    cross_section_m2: f64,
    hydraulic_capacity_m3_s: f64,
    base_open_fraction: f64,
}

#[derive(Serialize)]
struct Packet {
    id: String,
    geometry_id: String,
    volume_m3: f64,
    capacity: [f64; POOLS],
    initial: [f64; POOLS],
}

#[derive(Serialize)]
struct ColonySite {
    id: String,
    geometry_id: String,
    anchored_region: String,
    initial: [f64; POOLS],
    atp: f64,
}

#[derive(Serialize)]
struct Spawn {
    resident: usize,
    position_mm: [f64; 3],
    yaw_rad: f64,
}

#[derive(Serialize)]
struct Parameters {
    width_mm: f64,
    depth_mm: f64,
    height_mm: f64,
    residents: usize,
    ground_tiles: [usize; 2],
    stem_networks: usize,
    movable_grains: usize,
    movable_pods: usize,
}

#[derive(Serialize)]
struct Validation {
    geometry_count: usize,
    dynamic_geometry_count: usize,
    material_region_count: usize,
    route_count: usize,
    connected_material_graph: bool,
    spawn_clearance_mm: f64,
    estimated_total_geoms_with_flies: usize,
}

#[derive(Serialize)]
struct Plan {
    format: &'static str,
    generator: &'static str,
    seed: u64,
    units: Units,
    bounds_mm: [[f64; 3]; 2],
    parameters: Parameters,
    geometries: Vec<Geometry>,
    regions: Vec<Region>,
    routes: Vec<Route>,
    packets: Vec<Packet>,
    colony_sites: Vec<ColonySite>,
    spawns: Vec<Spawn>,
    validation: Validation,
    information_boundary: &'static str,
}

#[derive(Serialize)]
struct Units {
    length: &'static str,
    ecology_coordinate: &'static str,
    mass: &'static str,
}

struct Builder {
    rng: Rng,
    width: f64,
    depth: f64,
    height: f64,
    geometries: Vec<Geometry>,
    regions: Vec<Region>,
    routes: Vec<Route>,
    packets: Vec<Packet>,
    colonies: Vec<ColonySite>,
    region_by_geometry: HashMap<String, String>,
}

impl Builder {
    fn new(seed: u64, width: f64, depth: f64, height: f64) -> Self {
        Self {
            rng: Rng::new(seed),
            width,
            depth,
            height,
            geometries: vec![],
            regions: vec![],
            routes: vec![],
            packets: vec![],
            colonies: vec![],
            region_by_geometry: HashMap::new(),
        }
    }
    fn static_geom(
        &mut self,
        id: String,
        category: &str,
        shape: &str,
        material: &str,
        position: Option<[f64; 3]>,
        size: Vec<f64>,
        fromto: Option<[f64; 6]>,
        euler: Option<[f64; 3]>,
    ) {
        self.geometries.push(Geometry {
            id,
            category: category.into(),
            shape: shape.into(),
            material: material.into(),
            dynamic: false,
            position_mm: position,
            quaternion_wxyz: None,
            euler_rad: euler,
            size_mm: size,
            fromto_mm: fromto,
            mass_model_units: None,
        });
    }
    fn dynamic_geom(
        &mut self,
        id: String,
        category: &str,
        material: &str,
        position: [f64; 3],
        size: [f64; 3],
        mass: f64,
    ) {
        self.geometries.push(Geometry {
            id,
            category: category.into(),
            shape: "ellipsoid".into(),
            material: material.into(),
            dynamic: true,
            position_mm: Some(position),
            quaternion_wxyz: Some([1.0, 0.0, 0.0, 0.0]),
            euler_rad: None,
            size_mm: size.to_vec(),
            fromto_mm: None,
            mass_model_units: Some(mass),
        });
    }
    fn region(
        &mut self,
        geometry: &str,
        center_mm: [f64; 3],
        volume: f64,
        initial: [f64; POOLS],
    ) -> String {
        let id = format!("material-{geometry}");
        self.regions.push(Region {
            id: id.clone(),
            geometry_id: geometry.into(),
            center_m: center_mm.map(|v| v * 0.001),
            volume_m3: volume,
            capacity: [20.0, 12.0, 10.0, 8.0, 12.0, 12.0, 8.0, 8.0],
            initial,
        });
        self.region_by_geometry.insert(geometry.into(), id.clone());
        id
    }
    fn route(&mut self, a: &str, b: &str, width_m: f64) {
        if a == b {
            return;
        }
        let ra = self.regions.iter().find(|r| r.id == a).unwrap();
        let rb = self.regions.iter().find(|r| r.id == b).unwrap();
        let length = ra
            .center_m
            .iter()
            .zip(rb.center_m)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
            .max(0.0002);
        self.routes.push(Route {
            id: format!("route-{:03}", self.routes.len()),
            a: a.into(),
            b: b.into(),
            length_m: length,
            cross_section_m2: width_m * width_m,
            hydraulic_capacity_m3_s: width_m * width_m * 0.02,
            base_open_fraction: 1.0,
        });
    }
}

fn generate(
    seed: u64,
    residents: usize,
    width: f64,
    depth: f64,
    height: f64,
) -> Result<Plan, String> {
    if !(40.0..=160.0).contains(&width)
        || !(30.0..=120.0).contains(&depth)
        || !(15.0..=60.0).contains(&height)
    {
        return Err("habitat dimensions outside supported fly-scale envelope".into());
    }
    if !(1..=16).contains(&residents) {
        return Err("residents must be 1..16".into());
    }
    let mut b = Builder::new(seed, width, depth, height);
    let nx = 8usize;
    let ny = 6usize;
    let dx = width / nx as f64;
    let dy = depth / ny as f64;
    let mut ground = vec![vec![String::new(); ny]; nx];
    for x in 0..nx {
        for y in 0..ny {
            let px = -width / 2.0 + dx * (x as f64 + 0.5);
            let py = -depth / 2.0 + dy * (y as f64 + 0.5);
            let id = format!("ground-{x:02}-{y:02}");
            b.static_geom(
                id.clone(),
                "ground",
                "box",
                "soil",
                Some([px, py, -0.25]),
                vec![dx / 2.0, dy / 2.0, 0.25],
                None,
                None,
            );
            let wet = 0.25 + 0.25 * b.rng.unit();
            let region = b.region(
                &id,
                [px, py, 0.0],
                dx * dy * 2.0e-9,
                [2.0, 0.12 * wet, 0.04, 0.25, 6.0, 0.2, 0.0, 0.01 * wet],
            );
            ground[x][y] = region;
        }
    }
    let wall_thickness = 0.25;
    for (id, position, size) in [
        (
            "boundary-west",
            [-width * 0.5 - wall_thickness, 0.0, height * 0.5],
            [wall_thickness, depth * 0.5 + wall_thickness, height * 0.5],
        ),
        (
            "boundary-east",
            [width * 0.5 + wall_thickness, 0.0, height * 0.5],
            [wall_thickness, depth * 0.5 + wall_thickness, height * 0.5],
        ),
        (
            "boundary-south",
            [0.0, -depth * 0.5 - wall_thickness, height * 0.5],
            [width * 0.5, wall_thickness, height * 0.5],
        ),
        (
            "boundary-north",
            [0.0, depth * 0.5 + wall_thickness, height * 0.5],
            [width * 0.5, wall_thickness, height * 0.5],
        ),
    ] {
        b.static_geom(
            id.into(),
            "world-boundary",
            "box",
            "bark",
            Some(position),
            size.to_vec(),
            None,
            None,
        );
    }
    for x in 0..nx {
        for y in 0..ny {
            if x + 1 < nx {
                b.route(&ground[x][y], &ground[x + 1][y], 0.0012);
            }
            if y + 1 < ny {
                b.route(&ground[x][y], &ground[x][y + 1], 0.0012);
            }
        }
    }

    let bases = [[-0.32, -0.28], [-0.18, 0.27], [0.16, -0.23], [0.31, 0.24]];
    let mut leaf_sites = Vec::new();
    for (tree, base) in bases.iter().enumerate() {
        let bx = base[0] * width + b.rng.signed(1.2);
        let by = base[1] * depth + b.rng.signed(1.2);
        let gx = ((bx + width / 2.0) / dx)
            .floor()
            .clamp(0.0, (nx - 1) as f64) as usize;
        let gy = ((by + depth / 2.0) / dy)
            .floor()
            .clamp(0.0, (ny - 1) as f64) as usize;
        let mut previous = ground[gx][gy].clone();
        let mut points = Vec::new();
        points.push([bx, by, 0.0]);
        for segment in 0..8 {
            let prior = *points.last().unwrap();
            let next = [
                prior[0] + b.rng.signed(0.8),
                prior[1] + b.rng.signed(0.8),
                (segment as f64 + 1.0) * height * 0.09,
            ];
            let id = format!("stem-{tree:02}-{segment:02}");
            b.static_geom(
                id.clone(),
                "stem",
                "capsule",
                "stem",
                None,
                vec![0.58],
                Some([prior[0], prior[1], prior[2], next[0], next[1], next[2]]),
                None,
            );
            let region = b.region(
                &id,
                [
                    (prior[0] + next[0]) * 0.5,
                    (prior[1] + next[1]) * 0.5,
                    (prior[2] + next[2]) * 0.5,
                ],
                1.2e-8,
                [0.8, 0.08, 0.05, 0.08, 3.0, 0.15, 0.8, 0.02],
            );
            b.route(&previous, &region, 0.00065);
            previous = region;
            points.push(next);
        }
        for (branch_index, &segment) in [2usize, 4, 6].iter().enumerate() {
            for side in [-1.0, 1.0] {
                let start = points[segment + 1];
                let angle = tree as f64 * 1.37
                    + branch_index as f64 * 0.71
                    + side * 0.8
                    + b.rng.signed(0.2);
                let length = 4.5 + b.rng.unit() * 3.0;
                let end = [
                    start[0] + length * angle.cos(),
                    start[1] + length * angle.sin(),
                    (start[2] + 1.2 + b.rng.signed(0.4)).min(height - 3.0),
                ];
                let suffix = if side < 0.0 { "a" } else { "b" };
                let id = format!("branch-{tree:02}-{branch_index:02}-{suffix}");
                b.static_geom(
                    id.clone(),
                    "branch",
                    "capsule",
                    "stem",
                    None,
                    vec![0.42],
                    Some([start[0], start[1], start[2], end[0], end[1], end[2]]),
                    None,
                );
                let region = b.region(
                    &id,
                    [
                        (start[0] + end[0]) * 0.5,
                        (start[1] + end[1]) * 0.5,
                        (start[2] + end[2]) * 0.5,
                    ],
                    7e-9,
                    [0.55, 0.06, 0.04, 0.05, 3.0, 0.12, 0.5, 0.015],
                );
                let main_id = format!("material-stem-{tree:02}-{segment:02}");
                b.route(&main_id, &region, 0.0005);
                let leaf_id = format!("leaf-{tree:02}-{branch_index:02}-{suffix}");
                let leaf_center = [
                    end[0] + 1.2 * angle.cos(),
                    end[1] + 1.2 * angle.sin(),
                    end[2] + 0.25,
                ];
                let leaf_size = [4.3 + b.rng.unit(), 2.2 + b.rng.unit() * 0.6, 0.20];
                let leaf_euler = [b.rng.signed(0.16), b.rng.signed(0.2), angle];
                b.static_geom(
                    leaf_id.clone(),
                    "leaf",
                    "ellipsoid",
                    "leaf",
                    Some(leaf_center),
                    leaf_size.to_vec(),
                    None,
                    Some(leaf_euler),
                );
                let leaf_carbon = 0.22 + b.rng.unit() * 0.18;
                let leaf_volatile = 0.05 + b.rng.unit() * 0.04;
                let leaf_region = b.region(
                    &leaf_id,
                    leaf_center,
                    2.5e-8,
                    [1.4, leaf_carbon, 0.08, 0.06, 4.0, 0.18, 1.1, leaf_volatile],
                );
                b.route(&region, &leaf_region, 0.0007);
                leaf_sites.push((leaf_id, leaf_region, leaf_center));
            }
        }
        for top in 0..2 {
            let start = points[8];
            let angle = top as f64 * std::f64::consts::PI + tree as f64 * 0.4;
            let center = [
                start[0] + 3.0 * angle.cos(),
                start[1] + 3.0 * angle.sin(),
                (start[2] + 0.4).min(height - 1.0),
            ];
            let id = format!("leaf-{tree:02}-top-{top}");
            let top_pitch = b.rng.signed(0.15);
            b.static_geom(
                id.clone(),
                "leaf",
                "ellipsoid",
                "leaf",
                Some(center),
                vec![4.7, 2.5, 0.2],
                None,
                Some([0.0, top_pitch, angle]),
            );
            let region = b.region(
                &id,
                center,
                2.8e-8,
                [1.5, 0.25, 0.09, 0.05, 4.5, 0.2, 1.2, 0.08],
            );
            b.route(&previous, &region, 0.0007);
            leaf_sites.push((id, region, center));
        }
    }

    for shelter in 0..4 {
        let sx = (-0.34 + 0.22 * shelter as f64) * width;
        let sy = if shelter % 2 == 0 {
            -0.36 * depth
        } else {
            0.36 * depth
        };
        let sign = if sy < 0.0 { 1.0 } else { -1.0 };
        for (part, off, size) in [
            ("left", [-3.2, 0.0, 2.2], [0.45, 3.4, 2.2]),
            ("right", [3.2, 0.0, 2.2], [0.45, 3.4, 2.2]),
            ("roof", [0.0, 0.0, 4.4], [3.65, 3.4, 0.4]),
            ("back", [0.0, -sign * 3.0, 2.2], [3.2, 0.4, 2.2]),
        ] {
            let id = format!("shelter-{shelter:02}-{part}");
            b.static_geom(
                id,
                "bark-shelter",
                "box",
                "bark",
                Some([sx + off[0], sy + off[1], off[2]]),
                size.to_vec(),
                None,
                None,
            );
        }
        let floor = format!("niche-{shelter:02}");
        let center = [sx, sy, 0.08];
        b.static_geom(
            floor.clone(),
            "sheltered-niche",
            "box",
            "moist",
            Some(center),
            vec![3.1, 2.8, 0.08],
            None,
            None,
        );
        let region = b.region(
            &floor,
            center,
            3e-8,
            [1.8, 0.35, 0.16, 0.12, 2.2, 0.15, 0.25, 0.12],
        );
        let gx = ((sx + width / 2.0) / dx)
            .floor()
            .clamp(0.0, (nx - 1) as f64) as usize;
        let gy = ((sy + depth / 2.0) / dy)
            .floor()
            .clamp(0.0, (ny - 1) as f64) as usize;
        b.route(&ground[gx][gy], &region, 0.001);
    }
    for ramp in 0..8 {
        let angle = ramp as f64 * std::f64::consts::TAU / 8.0;
        let radius = 0.29 * width.min(depth);
        let pos = [radius * angle.cos(), radius * angle.sin(), 1.5];
        let id = format!("ramp-{ramp:02}");
        b.static_geom(
            id.clone(),
            "ramp",
            "box",
            "bark",
            Some(pos),
            vec![4.6, 1.25, 0.16],
            None,
            Some([0.0, -0.31, angle]),
        );
        let region = b.region(
            &id,
            pos,
            1.2e-8,
            [0.45, 0.04, 0.02, 0.06, 4.0, 0.1, 0.35, 0.01],
        );
        let gx = ((pos[0] + width / 2.0) / dx)
            .floor()
            .clamp(0.0, (nx - 1) as f64) as usize;
        let gy = ((pos[1] + depth / 2.0) / dy)
            .floor()
            .clamp(0.0, (ny - 1) as f64) as usize;
        b.route(&ground[gx][gy], &region, 0.0008);
    }
    for rock in 0..24 {
        let x = b.rng.signed(width * 0.43);
        let y = b.rng.signed(depth * 0.43);
        let size = [0.7 + b.rng.unit(), 0.5 + b.rng.unit() * 0.6, 0.35];
        let yaw = b.rng.unit() * 3.14;
        let id = format!("bark-fragment-{rock:02}");
        b.static_geom(
            id,
            "bark-fragment",
            "ellipsoid",
            "bark",
            Some([x, y, 0.35]),
            size.to_vec(),
            None,
            Some([0.0, 0.0, yaw]),
        );
    }
    for wet in 0..8 {
        let x = (-0.37 + 0.105 * wet as f64) * width;
        let y = if wet % 2 == 0 {
            -0.08 * depth
        } else {
            0.10 * depth
        };
        let id = format!("moist-patch-{wet:02}");
        let center = [x, y, 0.055];
        b.static_geom(
            id.clone(),
            "moist-patch",
            "cylinder",
            "moist",
            Some(center),
            vec![1.7, 0.055],
            None,
            None,
        );
        let region = b.region(
            &id,
            center,
            1.5e-8,
            [1.7, 0.22, 0.1, 0.04, 2.5, 0.1, 0.0, 0.09],
        );
        let gx = ((x + width / 2.0) / dx).floor().clamp(0.0, (nx - 1) as f64) as usize;
        let gy = ((y + depth / 2.0) / dy).floor().clamp(0.0, (ny - 1) as f64) as usize;
        b.route(&ground[gx][gy], &region, 0.0012);
    }
    b.static_geom(
        "visual-screen".into(),
        "visual-screen",
        "box",
        "leaf",
        Some([width / 2.0 - 0.1, 0.0, height * 0.42]),
        vec![0.08, depth * 0.31, height * 0.32],
        None,
        None,
    );

    for i in 0..20 {
        let angle = i as f64 * 2.399963229728653 + b.rng.signed(0.12);
        let radius = 5.0 + (i % 5) as f64 * 2.2;
        let size = [0.34, 0.25, 0.22];
        let preferred = [radius * angle.cos(), radius * angle.sin(), 0.42];
        let pos = clear_ground_packet_position(&b, preferred, size)?;
        let id = format!("grain-{i:02}");
        b.dynamic_geom(id.clone(), "grain", "grain", pos, size, 2e-6);
        b.packets.push(Packet {
            id: format!("packet-{id}"),
            geometry_id: id,
            volume_m3: 8e-11,
            capacity: [3.0; POOLS],
            initial: [0.18, 0.45, 0.28, 0.02, 0.01, 0.0, 0.28, 0.02],
        });
    }
    for i in 0..8 {
        let (_, _, leaf) = &leaf_sites[(i * 3 + 1) % leaf_sites.len()];
        let pos = [leaf[0], leaf[1], leaf[2] + 0.58];
        let id = format!("pod-{i:02}");
        b.dynamic_geom(
            id.clone(),
            "elevated-food-pod",
            "pod",
            pos,
            [0.48, 0.32, 0.28],
            3e-6,
        );
        b.packets.push(Packet {
            id: format!("packet-{id}"),
            geometry_id: id,
            volume_m3: 1.2e-10,
            capacity: [3.0; POOLS],
            initial: [0.25, 0.75, 0.42, 0.03, 0.01, 0.0, 0.45, 0.04],
        });
    }
    for i in 0..4 {
        let (geometry, region, _) = &leaf_sites[(i * 7 + 2) % leaf_sites.len()];
        b.colonies.push(ColonySite {
            id: format!("colony-{i}"),
            geometry_id: geometry.clone(),
            anchored_region: region.clone(),
            initial: [1.5, 1.2, 0.8, 0.6, 0.8, 0.6, 1.6, 0.05],
            atp: 0.8,
        });
    }

    let spawns = select_spawns(&mut b, residents, nx, ny, dx, dy)?;
    validate(&b, &spawns)?;
    let dynamic_count = b.geometries.iter().filter(|g| g.dynamic).count();
    // The compiled fixture has 69 semantic mesh segments plus two adhesion
    // helper geoms per resident.
    let estimated = b.geometries.len() + residents * 71;
    if !(300..=600).contains(&estimated) {
        return Err(format!(
            "estimated compiled geometry budget {estimated} outside 300..600"
        ));
    }
    Ok(Plan { format:"chreatures.fly-habitat-plan.v1",generator:"chreatures-fly-habitat-0.1.0",seed,
        units:Units{length:"millimeter",ecology_coordinate:"meter",mass:"model-unit-explicit-per-geometry"},
        bounds_mm:[[-width/2.0,-depth/2.0,-0.5],[width/2.0,depth/2.0,height]],
        parameters:Parameters{width_mm:width,depth_mm:depth,height_mm:height,residents,ground_tiles:[nx,ny],stem_networks:4,movable_grains:20,movable_pods:8},
        validation:Validation{geometry_count:b.geometries.len(),dynamic_geometry_count:dynamic_count,material_region_count:b.regions.len(),route_count:b.routes.len(),connected_material_graph:true,spawn_clearance_mm:SPAWN_CLEARANCE_MM,estimated_total_geoms_with_flies:estimated},
        geometries:b.geometries,regions:b.regions,routes:b.routes,packets:b.packets,colony_sites:b.colonies,spawns,
        information_boundary:"Habitat geometry, region IDs, route graph, stores, seed and spawn coordinates are physics/ecology configuration and observer provenance; none are controller inputs." })
}

fn clear_ground_packet_position(
    builder: &Builder,
    preferred: [f64; 3],
    size: [f64; 3],
) -> Result<[f64; 3], String> {
    const GAP_MM: f64 = 0.04;
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
    for attempt in 0..512 {
        let radius = if attempt == 0 {
            0.0
        } else {
            0.55 * (attempt as f64).sqrt()
        };
        let angle = attempt as f64 * GOLDEN_ANGLE;
        let point = [
            (preferred[0] + radius * angle.cos()).clamp(
                -builder.width * 0.5 + size[0],
                builder.width * 0.5 - size[0],
            ),
            (preferred[1] + radius * angle.sin()).clamp(
                -builder.depth * 0.5 + size[1],
                builder.depth * 0.5 - size[1],
            ),
            preferred[2],
        ];
        if builder
            .geometries
            .iter()
            .all(|geometry| ground_packet_clears_geometry(point, size, geometry, GAP_MM))
        {
            return Ok(point);
        }
    }
    Err("could not place a collision-free movable grain".into())
}

fn ground_packet_clears_geometry(
    point: [f64; 3],
    packet_size: [f64; 3],
    geometry: &Geometry,
    gap: f64,
) -> bool {
    let packet_radius = packet_size.iter().map(|v| v * v).sum::<f64>().sqrt();
    if let Some(segment) = geometry.fromto_mm {
        return point_segment_distance(point, segment) > geometry.size_mm[0] + packet_radius + gap;
    }
    let Some(center) = geometry.position_mm else {
        return true;
    };
    let rotated = geometry.euler_rad.is_some()
        || geometry
            .quaternion_wxyz
            .is_some_and(|q| q != [1.0, 0.0, 0.0, 0.0]);
    if rotated {
        let geometry_radius = geometry
            .size_mm
            .iter()
            .map(|extent| extent * extent)
            .sum::<f64>()
            .sqrt();
        let distance = (0..3)
            .map(|axis| (point[axis] - center[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        return distance > geometry_radius + packet_radius + gap;
    }
    let geometry_extent = match geometry.shape.as_str() {
        "cylinder" => [
            geometry.size_mm[0],
            geometry.size_mm[0],
            geometry.size_mm[1],
        ],
        _ => [
            geometry.size_mm[0],
            *geometry.size_mm.get(1).unwrap_or(&geometry.size_mm[0]),
            *geometry.size_mm.get(2).unwrap_or(&geometry.size_mm[0]),
        ],
    };
    !(0..3).all(|axis| {
        (point[axis] - center[axis]).abs() <= geometry_extent[axis] + packet_size[axis] + gap
    })
}

fn point_segment_distance(point: [f64; 3], segment: [f64; 6]) -> f64 {
    let a = [segment[0], segment[1], segment[2]];
    let delta = [segment[3] - a[0], segment[4] - a[1], segment[5] - a[2]];
    let from_a = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let length_squared = delta.iter().map(|v| v * v).sum::<f64>();
    let t = if length_squared > 0.0 {
        from_a.iter().zip(delta).map(|(x, y)| x * y).sum::<f64>() / length_squared
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    (0..3)
        .map(|axis| (point[axis] - (a[axis] + t * delta[axis])).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn clears_geometry(point: [f64; 3], geometry: &Geometry, clearance: f64) -> bool {
    if let Some(segment) = geometry.fromto_mm {
        return point_segment_distance(point, segment) > geometry.size_mm[0] + clearance;
    }
    let Some(center) = geometry.position_mm else {
        return true;
    };
    let rotated = geometry.euler_rad.is_some()
        || geometry
            .quaternion_wxyz
            .is_some_and(|q| q != [1.0, 0.0, 0.0, 0.0]);
    if rotated {
        // An orientation-independent bounding sphere deliberately rejects
        // some clear points, but cannot admit a point inside a rotated solid.
        let radius = geometry
            .size_mm
            .iter()
            .map(|extent| extent * extent)
            .sum::<f64>()
            .sqrt();
        let distance = (0..3)
            .map(|axis| (point[axis] - center[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        return distance > radius + clearance;
    }
    // For axis-aligned primitives, the expanded AABB is exact for boxes and
    // conservative for ellipsoids and cylinders.
    !(0..3).all(|axis| {
        let half_extent = geometry
            .size_mm
            .get(axis)
            .copied()
            .unwrap_or(geometry.size_mm[0]);
        (point[axis] - center[axis]).abs() <= half_extent + clearance
    })
}

fn select_spawns(
    builder: &mut Builder,
    residents: usize,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<Vec<Spawn>, String> {
    let mut candidates = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            let point = [
                -builder.width / 2.0 + dx * (x as f64 + 0.5),
                -builder.depth / 2.0 + dy * (y as f64 + 0.5),
                1.8,
            ];
            if builder
                .geometries
                .iter()
                .all(|geometry| clears_geometry(point, geometry, SPAWN_CLEARANCE_MM))
            {
                candidates.push(point);
            }
        }
    }
    if candidates.len() < residents {
        return Err(format!(
            "only {} physically clear ground spawn cells for {residents} residents",
            candidates.len()
        ));
    }
    let first = (builder.rng.next() as usize) % candidates.len();
    let mut selected = vec![candidates.swap_remove(first)];
    while selected.len() < residents {
        let (index, _) = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let nearest = selected
                    .iter()
                    .map(|prior| {
                        (0..2)
                            .map(|axis| (prior[axis] - candidate[axis]).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                (index, nearest)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or("spawn candidate selection failed")?;
        selected.push(candidates.swap_remove(index));
    }
    Ok(selected
        .into_iter()
        .enumerate()
        .map(|(resident, position_mm)| Spawn {
            resident,
            position_mm,
            yaw_rad: (-position_mm[1]).atan2(-position_mm[0]),
        })
        .collect())
}

fn validate(b: &Builder, spawns: &[Spawn]) -> Result<(), String> {
    let ids = b
        .geometries
        .iter()
        .map(|g| g.id.as_str())
        .collect::<HashSet<_>>();
    if ids.len() != b.geometries.len() {
        return Err("duplicate geometry ID".into());
    }
    if b.regions
        .iter()
        .any(|r| !ids.contains(r.geometry_id.as_str()))
    {
        return Err("region lacks physical geometry".into());
    }
    let region_ids = b
        .regions
        .iter()
        .map(|r| r.id.as_str())
        .collect::<HashSet<_>>();
    if region_ids.len() != b.regions.len() {
        return Err("duplicate region ID".into());
    }
    if b.routes
        .iter()
        .any(|r| !region_ids.contains(r.a.as_str()) || !region_ids.contains(r.b.as_str()))
    {
        return Err("route endpoint missing".into());
    }
    let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
    for r in &b.routes {
        graph.entry(&r.a).or_default().push(&r.b);
        graph.entry(&r.b).or_default().push(&r.a)
    }
    let start = b.regions.first().ok_or("no material regions")?.id.as_str();
    let mut seen = HashSet::new();
    let mut q = VecDeque::from([start]);
    while let Some(a) = q.pop_front() {
        if seen.insert(a) {
            for &n in graph.get(a).into_iter().flatten() {
                q.push_back(n)
            }
        }
    }
    if seen.len() != b.regions.len() {
        return Err(format!(
            "material graph disconnected: {}/{} regions",
            seen.len(),
            b.regions.len()
        ));
    }
    let clearance = SPAWN_CLEARANCE_MM;
    for (index, s) in spawns.iter().enumerate() {
        let [x, y, z] = s.position_mm;
        if x.abs() > b.width / 2.0 - 2.0 || y.abs() > b.depth / 2.0 - 2.0 || z < 1.2 || z > b.height
        {
            return Err("spawn outside clearance bounds".into());
        }
        if b.geometries
            .iter()
            .any(|geometry| !clears_geometry(s.position_mm, geometry, clearance))
        {
            return Err(format!(
                "resident {index} spawn intersects expanded habitat geometry"
            ));
        }
        if spawns[..index].iter().any(|other| {
            (0..3)
                .map(|axis| (other.position_mm[axis] - s.position_mm[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
                <= 2.0 * clearance
        }) {
            return Err(format!("resident {index} spawn lacks inter-fly clearance"));
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let mut output = None::<PathBuf>;
    let mut seed = 20260908u64;
    let mut residents = 4usize;
    let mut width = 80.0;
    let mut depth = 60.0;
    let mut height = 30.0;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--output" => output = Some(PathBuf::from(value()?)),
            "--seed" => seed = value()?.parse().map_err(|_| "invalid seed")?,
            "--residents" => residents = value()?.parse().map_err(|_| "invalid residents")?,
            "--width-mm" => width = value()?.parse().map_err(|_| "invalid width")?,
            "--depth-mm" => depth = value()?.parse().map_err(|_| "invalid depth")?,
            "--height-mm" => height = value()?.parse().map_err(|_| "invalid height")?,
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    let output = output.ok_or("--output is required")?;
    let plan = generate(seed, residents, width, depth, height)?;
    let bytes = serde_json::to_vec_pretty(&plan).map_err(|e| e.to_string())?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?
    }
    fs::write(&output, &bytes).map_err(|e| e.to_string())?;
    println!("{{\"path\":\"{}\",\"seed\":{},\"geometries\":{},\"regions\":{},\"routes\":{},\"estimated_total_geoms\":{}}}",output.display(),seed,plan.geometries.len(),plan.regions.len(),plan.routes.len(),plan.validation.estimated_total_geoms_with_flies);
    Ok(())
}
