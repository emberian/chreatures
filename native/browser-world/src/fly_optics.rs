use crate::fly_types::*;

pub fn rays(
    c: &Config,
    row: usize,
    positions: &[f64],
    rotations: &[f64],
) -> Result<Vec<f64>, String> {
    let b = c.bodies.get(row).ok_or("unknown resident")?;
    if !finite(positions)
        || !finite(rotations)
        || b.eyes
            .iter()
            .any(|e| e.body * 3 + 3 > positions.len() || e.body * 9 + 9 > rotations.len())
    {
        return Err("invalid physical eye frames".into());
    }
    let mut out = Vec::with_capacity(2 * (3 + SITES * 3));
    for eye in &b.eyes {
        let m = &rotations[eye.body * 9..eye.body * 9 + 9];
        let offset = rotate(m, eye.position);
        out.extend((0..3).map(|k| positions[eye.body * 3 + k] + offset[k]));
        for direction in &c.retinal_directions {
            out.extend(rotate(m, rotate(&eye.rotation, *direction)));
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn retina(
    c: &Config,
    row: usize,
    rays: &[f64],
    hits: &[i32],
    distances: &[f64],
    positions: &[f64],
    rotations: &[f64],
    sizes: &[f64],
    colors: &[f64],
    frame: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, String> {
    let ng = positions.len() / 3;
    if row >= c.bodies.len()
        || rays.len() != 2 * (3 + SITES * 3)
        || hits.len() != 2 * SITES
        || distances.len() != 2 * SITES
        || rotations.len() != ng * 9
        || sizes.len() != ng * 3
        || colors.len() != ng * 4
        || width == 0
        || height == 0
        || width > 2048
        || height > 2048
        || frame.len() != width * height * 3
        || [rays, distances, positions, rotations, sizes, colors]
            .iter()
            .any(|x| !finite(x))
        || frame
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        || hits.iter().any(|g| *g >= ng as i32)
    {
        return Err("invalid physical retinal rays".into());
    }
    let mut rgb = vec![0.0; SITES * 3];
    for site in 0..SITES {
        if !c.supported_sites[site] {
            continue;
        }
        let eye = c.anatomical_sites[site][0] as usize - 1;
        let hit = hits[eye * SITES + site];
        let distance = distances[eye * SITES + site];
        let value = if hit < 0 || distance < 0.0 || distance > c.ray_distance_mm {
            [0.22, 0.28, 0.36]
        } else {
            let g = hit as usize;
            if g == c.screen_geom {
                let base = eye * (3 + SITES * 3);
                let p = std::array::from_fn(|k| {
                    rays[base + k] + distance * rays[base + 3 + site * 3 + k] - positions[g * 3 + k]
                });
                let lp = local(&rotations[g * 9..g * 9 + 9], p);
                let ld = local(
                    &rotations[g * 9..g * 9 + 9],
                    std::array::from_fn(|k| rays[base + 3 + site * 3 + k]),
                );
                if ld[0] <= 0.0 {
                    [0.02; 3]
                } else {
                    let u = (0.5 - lp[1] / (2.0 * sizes[g * 3 + 1])).clamp(0.0, 1.0);
                    let v = (0.5 - lp[2] / (2.0 * sizes[g * 3 + 2])).clamp(0.0, 1.0);
                    let x = u * (width - 1) as f64;
                    let y = v * (height - 1) as f64;
                    let x0 = x.floor() as usize;
                    let y0 = y.floor() as usize;
                    let x1 = (x0 + 1).min(width - 1);
                    let y1 = (y0 + 1).min(height - 1);
                    let fx = (x - x0 as f64) as f32;
                    let fy = (y - y0 as f64) as f32;
                    std::array::from_fn(|k| {
                        let top = frame[(y0 * width + x0) * 3 + k] * (1.0 - fx)
                            + frame[(y0 * width + x1) * 3 + k] * fx;
                        let bottom = frame[(y1 * width + x0) * 3 + k] * (1.0 - fx)
                            + frame[(y1 * width + x1) * 3 + k] * fx;
                        top * (1.0 - fy) + bottom * fy
                    })
                }
            } else {
                std::array::from_fn(|k| colors[g * 4 + k] as f32 * 0.85)
            }
        };
        rgb[site * 3..site * 3 + 3].copy_from_slice(&value);
    }
    Ok(rgb)
}
