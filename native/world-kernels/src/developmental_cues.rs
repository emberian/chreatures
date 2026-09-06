//! Local physical surface cues for resource-funded developmental growth.
//!
//! The kernel evaluates all bud/geometry pairs without Python object traffic.
//! It only reports closest points on supported MuJoCo primitives; topology
//! compilation remains the authoritative collision check.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn norm(value: [f64; 3]) -> f64 {
    value
        .into_iter()
        .map(|axis| axis * axis)
        .sum::<f64>()
        .sqrt()
}

fn ellipsoid_closest(point: [f64; 3], radii: [f64; 3]) -> [f64; 3] {
    let signs = point.map(|value| if value < 0.0 { -1.0 } else { 1.0 });
    let absolute = point.map(f64::abs);
    let level = (0..3)
        .map(|axis| (absolute[axis] / radii[axis]).powi(2))
        .sum::<f64>();
    if level <= 1.0 {
        let axis = (0..3)
            .min_by(|a, b| (radii[*a] - absolute[*a]).total_cmp(&(radii[*b] - absolute[*b])))
            .unwrap_or(0);
        let mut closest = point;
        closest[axis] = signs[axis] * radii[axis];
        return closest;
    }
    let squared = radii.map(|value| value * value);
    let constraint = |lambda: f64| {
        (0..3)
            .map(|axis| {
                let ratio = radii[axis] * absolute[axis] / (lambda + squared[axis]);
                ratio * ratio
            })
            .sum::<f64>()
    };
    let mut lower = 0.0;
    let mut upper = squared.into_iter().fold(0.0_f64, f64::max);
    while constraint(upper) > 1.0 {
        upper *= 2.0;
    }
    for _ in 0..64 {
        let middle = 0.5 * (lower + upper);
        if constraint(middle) > 1.0 {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    std::array::from_fn(|axis| {
        signs[axis] * squared[axis] * absolute[axis] / (upper + squared[axis])
    })
}

fn primitive_closest(kind: i32, point: [f64; 3], size: [f64; 3]) -> Option<[f64; 3]> {
    match kind {
        // MuJoCo mjtGeom: plane=0, sphere=2, capsule=3, ellipsoid=4,
        // cylinder=5 and box=6. Meshes and height fields provide no cue.
        0 => Some([point[0], point[1], 0.0]),
        2 => {
            let length = norm(point);
            let direction = if length > 1.0e-14 {
                point.map(|value| value / length)
            } else {
                [0.0, 0.0, 1.0]
            };
            Some(direction.map(|value| value * size[0]))
        }
        3 => {
            let axis = point[2].clamp(-size[1], size[1]);
            let delta = [point[0], point[1], point[2] - axis];
            let length = norm(delta);
            let direction = if length > 1.0e-14 {
                delta.map(|value| value / length)
            } else {
                [1.0, 0.0, 0.0]
            };
            Some([
                direction[0] * size[0],
                direction[1] * size[0],
                axis + direction[2] * size[0],
            ])
        }
        4 => Some(ellipsoid_closest(point, size)),
        5 => {
            let radial = point[0].hypot(point[1]);
            let radial_direction = if radial > 1.0e-14 {
                [point[0] / radial, point[1] / radial]
            } else {
                [1.0, 0.0]
            };
            let outside_radial = radial > size[0];
            let outside_axial = point[2].abs() > size[1];
            if outside_radial || outside_axial {
                Some([
                    radial_direction[0] * radial.min(size[0]),
                    radial_direction[1] * radial.min(size[0]),
                    point[2].clamp(-size[1], size[1]),
                ])
            } else {
                let radial_gap = size[0] - radial;
                let axial_gap = size[1] - point[2].abs();
                if radial_gap <= axial_gap {
                    Some([
                        radial_direction[0] * size[0],
                        radial_direction[1] * size[0],
                        point[2],
                    ])
                } else {
                    Some([
                        point[0],
                        point[1],
                        if point[2] < 0.0 { -size[1] } else { size[1] },
                    ])
                }
            }
        }
        6 => {
            let clamped = std::array::from_fn(|axis| point[axis].clamp(-size[axis], size[axis]));
            if clamped != point {
                return Some(clamped);
            }
            let axis = (0..3)
                .min_by(|a, b| {
                    (size[*a] - point[*a].abs()).total_cmp(&(size[*b] - point[*b].abs()))
                })
                .unwrap_or(0);
            let mut closest = point;
            closest[axis] = if point[axis] < 0.0 {
                -size[axis]
            } else {
                size[axis]
            };
            Some(closest)
        }
        _ => None,
    }
}

#[pyfunction]
#[pyo3(signature = (
    bud_positions, bud_colonies, geom_types, geom_positions, geom_rotations,
    geom_sizes, geom_bounds, geom_enabled, geom_colonies, maximum_distance
))]
#[allow(clippy::too_many_arguments)]
pub fn developmental_surface_cues<'py>(
    py: Python<'py>,
    bud_positions: PyReadonlyArray2<'_, f64>,
    bud_colonies: PyReadonlyArray1<'_, i32>,
    geom_types: PyReadonlyArray1<'_, i32>,
    geom_positions: PyReadonlyArray2<'_, f64>,
    geom_rotations: PyReadonlyArray2<'_, f64>,
    geom_sizes: PyReadonlyArray2<'_, f64>,
    geom_bounds: PyReadonlyArray1<'_, f64>,
    geom_enabled: PyReadonlyArray1<'_, bool>,
    geom_colonies: PyReadonlyArray1<'_, i32>,
    maximum_distance: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let points = bud_positions.as_array();
    let bud_colonies = bud_colonies.as_slice()?;
    let types = geom_types.as_slice()?;
    let positions = geom_positions.as_array();
    let rotations = geom_rotations.as_array();
    let sizes = geom_sizes.as_array();
    let bounds = geom_bounds.as_slice()?;
    let enabled = geom_enabled.as_slice()?;
    let geom_colonies = geom_colonies.as_slice()?;
    let buds = points.nrows();
    let geoms = types.len();
    if points.ncols() != 3
        || buds > 16_384
        || bud_colonies.len() != buds
        || geoms == 0
        || geoms > 1_000_000
        || positions.shape() != [geoms, 3]
        || rotations.shape() != [geoms, 9]
        || sizes.shape() != [geoms, 3]
        || bounds.len() != geoms
        || enabled.len() != geoms
        || geom_colonies.len() != geoms
        || !maximum_distance.is_finite()
        || !(0.002..=100.0).contains(&maximum_distance)
        || points
            .iter()
            .chain(positions.iter())
            .chain(rotations.iter())
            .chain(sizes.iter())
            .chain(bounds)
            .any(|value| !value.is_finite())
        || sizes.iter().any(|value| *value < 0.0)
        || bounds.iter().any(|value| *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "invalid developmental surface cue batch",
        ));
    }
    let mut distances = vec![maximum_distance; buds];
    let mut directions = vec![0.0; buds * 3];
    let mut hits = vec![-1; buds];
    for bud in 0..buds {
        let point = [points[[bud, 0]], points[[bud, 1]], points[[bud, 2]]];
        for geom in 0..geoms {
            if !enabled[geom] || geom_colonies[geom] == bud_colonies[bud] {
                continue;
            }
            let delta = [
                point[0] - positions[[geom, 0]],
                point[1] - positions[[geom, 1]],
                point[2] - positions[[geom, 2]],
            ];
            if types[geom] != 0 && norm(delta) > bounds[geom] + distances[bud] {
                continue;
            }
            let local = [
                rotations[[geom, 0]] * delta[0]
                    + rotations[[geom, 3]] * delta[1]
                    + rotations[[geom, 6]] * delta[2],
                rotations[[geom, 1]] * delta[0]
                    + rotations[[geom, 4]] * delta[1]
                    + rotations[[geom, 7]] * delta[2],
                rotations[[geom, 2]] * delta[0]
                    + rotations[[geom, 5]] * delta[1]
                    + rotations[[geom, 8]] * delta[2],
            ];
            let size = [sizes[[geom, 0]], sizes[[geom, 1]], sizes[[geom, 2]]];
            let Some(closest_local) = primitive_closest(types[geom], local, size) else {
                continue;
            };
            let local_vector: [f64; 3] =
                std::array::from_fn(|axis| closest_local[axis] - local[axis]);
            let world_vector = [
                rotations[[geom, 0]] * local_vector[0]
                    + rotations[[geom, 1]] * local_vector[1]
                    + rotations[[geom, 2]] * local_vector[2],
                rotations[[geom, 3]] * local_vector[0]
                    + rotations[[geom, 4]] * local_vector[1]
                    + rotations[[geom, 5]] * local_vector[2],
                rotations[[geom, 6]] * local_vector[0]
                    + rotations[[geom, 7]] * local_vector[1]
                    + rotations[[geom, 8]] * local_vector[2],
            ];
            let distance = norm(world_vector);
            if distance < distances[bud] {
                distances[bud] = distance;
                hits[bud] = geom as i32;
                if distance > 1.0e-14 {
                    for axis in 0..3 {
                        directions[bud * 3 + axis] = world_vector[axis] / distance;
                    }
                }
            }
        }
    }
    let direction_array = PyArray2::from_vec2(
        py,
        &directions
            .chunks_exact(3)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    Ok((
        distances.into_pyarray(py),
        direction_array,
        hits.into_pyarray(py),
    ))
}
