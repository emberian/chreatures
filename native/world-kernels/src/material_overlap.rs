use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn ellipsoid_distance(point: [f64; 3], radii: [f64; 3]) -> f64 {
    let absolute = point.map(f64::abs);
    let inside = (0..3)
        .map(|axis| (absolute[axis] / radii[axis]).powi(2))
        .sum::<f64>();
    if inside <= 1.0 {
        return 0.0;
    }
    let squared = radii.map(|value| value * value);
    let constraint = |value: f64| {
        (0..3)
            .map(|axis| {
                let ratio = radii[axis] * absolute[axis] / (value + squared[axis]);
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
        let middle = (lower + upper) * 0.5;
        if constraint(middle) > 1.0 {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    (0..3)
        .map(|axis| {
            let closest = squared[axis] * absolute[axis] / (upper + squared[axis]);
            (absolute[axis] - closest).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

fn point_geom_distance(kind: i32, point: [f64; 3], size: [f64; 3]) -> Option<f64> {
    let norm = |value: [f64; 3]| {
        value
            .into_iter()
            .map(|item| item * item)
            .sum::<f64>()
            .sqrt()
    };
    match kind {
        // MuJoCo mjtGeom values: plane=0, sphere=2, capsule=3,
        // ellipsoid=4, cylinder=5, box=6. H-fields and meshes retain the
        // authoritative compiled-candidate path.
        0 => Some(point[2].max(0.0)),
        2 => Some((norm(point) - size[0]).max(0.0)),
        3 => {
            let radial = point[0].hypot(point[1]);
            let axial = (point[2].abs() - size[1]).max(0.0);
            Some((radial.hypot(axial) - size[0]).max(0.0))
        }
        4 => Some(ellipsoid_distance(point, size)),
        5 => {
            let radial = (point[0].hypot(point[1]) - size[0]).max(0.0);
            let axial = (point[2].abs() - size[1]).max(0.0);
            Some(radial.hypot(axial))
        }
        6 => Some(norm([
            (point[0].abs() - size[0]).max(0.0),
            (point[1].abs() - size[1]).max(0.0),
            (point[2].abs() - size[2]).max(0.0),
        ])),
        _ => None,
    }
}

#[pyfunction]
#[pyo3(signature = (
    packet_positions, packet_radii, geom_types, geom_positions, geom_rotations,
    geom_sizes, geom_bounds, geom_contype, geom_conaffinity, geom_enabled,
    tolerance=1e-6
))]
pub fn guaranteed_sphere_overlap_batch<'py>(
    py: Python<'py>,
    packet_positions: PyReadonlyArray2<'_, f64>,
    packet_radii: PyReadonlyArray1<'_, f64>,
    geom_types: PyReadonlyArray1<'_, i32>,
    geom_positions: PyReadonlyArray2<'_, f64>,
    geom_rotations: PyReadonlyArray2<'_, f64>,
    geom_sizes: PyReadonlyArray2<'_, f64>,
    geom_bounds: PyReadonlyArray1<'_, f64>,
    geom_contype: PyReadonlyArray1<'_, i32>,
    geom_conaffinity: PyReadonlyArray1<'_, i32>,
    geom_enabled: PyReadonlyArray1<'_, bool>,
    tolerance: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let positions = packet_positions.as_array();
    let radii = packet_radii.as_slice()?;
    let types = geom_types.as_slice()?;
    let geom_position = geom_positions.as_array();
    let rotations = geom_rotations.as_array();
    let sizes = geom_sizes.as_array();
    let bounds = geom_bounds.as_slice()?;
    let contype = geom_contype.as_slice()?;
    let conaffinity = geom_conaffinity.as_slice()?;
    let enabled = geom_enabled.as_slice()?;
    let packet_count = positions.nrows();
    let geom_count = types.len();
    if positions.ncols() != 3
        || packet_count == 0
        || packet_count > 256
        || radii.len() != packet_count
        || geom_count == 0
        || geom_count > 1_000_000
        || geom_position.shape() != [geom_count, 3]
        || rotations.shape() != [geom_count, 9]
        || sizes.shape() != [geom_count, 3]
        || bounds.len() != geom_count
        || contype.len() != geom_count
        || conaffinity.len() != geom_count
        || enabled.len() != geom_count
        || !tolerance.is_finite()
        || !(0.0..=0.05).contains(&tolerance)
        || positions.iter().any(|value| !value.is_finite())
        || radii
            .iter()
            .any(|value| !value.is_finite() || *value <= tolerance)
        || geom_position.iter().any(|value| !value.is_finite())
        || rotations.iter().any(|value| !value.is_finite())
        || sizes.iter().any(|value| !value.is_finite() || *value < 0.0)
        || bounds
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || contype.iter().any(|value| *value < 0)
        || conaffinity.iter().any(|value| *value < 0)
    {
        return Err(PyValueError::new_err(
            "invalid material packet overlap batch",
        ));
    }

    let mut result = vec![false; packet_count];
    for packet in 0..packet_count {
        let position = [
            positions[[packet, 0]],
            positions[[packet, 1]],
            positions[[packet, 2]],
        ];
        let radius = radii[packet];
        for geom in 0..geom_count {
            // Dormant material spheres use MuJoCo's default type/affinity 1.
            if !enabled[geom] || ((contype[geom] & 1) == 0 && (conaffinity[geom] & 1) == 0) {
                continue;
            }
            let delta = [
                position[0] - geom_position[[geom, 0]],
                position[1] - geom_position[[geom, 1]],
                position[2] - geom_position[[geom, 2]],
            ];
            if types[geom] != 0 {
                let bound = bounds[geom] + radius;
                if delta.into_iter().map(|value| value * value).sum::<f64>() >= bound * bound {
                    continue;
                }
            }
            // MuJoCo xmat is row-major local-to-world. Its transpose maps the
            // world displacement into the authored primitive frame.
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
            if let Some(distance) = point_geom_distance(types[geom], local, size) {
                if distance < radius - tolerance {
                    result[packet] = true;
                    break;
                }
            }
        }
    }
    Ok(result.into_pyarray(py))
}
