//! Bilateral, body-bound optical rays for a finite world-space video screen.
//!
//! Hex membership is anatomical. Eye offsets and the affine hex-to-angle map
//! are explicitly engineered optics. Only sampled RGB belongs at the CNS
//! afferent boundary; hit/distance/UV are observer diagnostics, never policy input.
use numpy::{
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use sha2::{Digest, Sha256};

const SITES: usize = 1771;
const MAX_RESIDENTS: usize = 4096;
const FORMAT: &str = "chreatures-bilateral-optic-screen-v1";

/// Pure body-local afferent transduction. Raw18 is odor6, linear3, angular3,
/// touch2, sound3, shade1. Contacts stay separate until native opponent pooling.
fn transduce_nonvisual(
    raw: &[f32],
    normals: &[f32],
    counts: &[u8],
    physiology: &[f32],
) -> Result<Vec<f32>, String> {
    let residents = counts.len();
    if residents == 0
        || residents > MAX_RESIDENTS
        || raw.len() != residents * 18
        || normals.len() != residents * 24
        || physiology.len() != residents * 12
        || counts.iter().any(|count| *count > 8)
        || raw
            .iter()
            .chain(normals)
            .chain(physiology)
            .any(|x| !x.is_finite())
    {
        return Err("nonvisual afferent raw dimensions/values differ".into());
    }
    let mut output = vec![0.0; residents * 43];
    for row in 0..residents {
        let input = &raw[row * 18..row * 18 + 18];
        let out = &mut output[row * 43..row * 43 + 43];
        for i in 0..6 {
            out[i] = (input[i] / 4.0).clamp(0.0, 1.0);
        }
        for (source, destination, maximum) in [(6, 6, 4.0), (9, 12, 8.0)] {
            for axis in 0..3 {
                let value = (input[source + axis] / maximum).clamp(-1.0, 1.0);
                out[destination + 2 * axis] = value.max(0.0);
                out[destination + 2 * axis + 1] = (-value).max(0.0);
            }
        }
        for contact in 0..counts[row] as usize {
            let start = row * 24 + contact * 3;
            let normal = &normals[start..start + 3];
            let norm = normal.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1.0001 {
                return Err("contact normal magnitude exceeds 1".into());
            }
            for axis in 0..3 {
                out[18 + 2 * axis] = out[18 + 2 * axis].max(normal[axis].max(0.0));
                out[19 + 2 * axis] = out[19 + 2 * axis].max((-normal[axis]).max(0.0));
            }
        }
        out[24] = counts[row] as f32 / 8.0;
        out[25] = input[12].clamp(0.0, 1.0);
        out[26] = input[13].clamp(0.0, 1.0);
        for i in 0..3 {
            out[27 + i] = (input[14 + i] / 2.0).clamp(0.0, 1.0);
        }
        out[30] = input[17].clamp(0.0, 1.0);
        out[31..43].copy_from_slice(&physiology[row * 12..row * 12 + 12]);
    }
    Ok(output)
}

/// Exact retinal-v2 nonretinal31 scaling plus interoception12, without retina.
#[pyfunction]
pub fn nonvisual_afferent_batch<'py>(
    py: Python<'py>,
    raw_senses: PyReadonlyArray2<'_, f32>,
    contact_normals: PyReadonlyArray3<'_, f32>,
    contact_counts: PyReadonlyArray1<'_, u8>,
    physiology: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let residents = raw_senses.shape()[0];
    if raw_senses.shape() != [residents, 18]
        || contact_normals.shape() != [residents, 8, 3]
        || contact_counts.len() != residents
        || physiology.shape() != [residents, 12]
    {
        return Err(PyValueError::new_err(
            "nonvisual afferent input shapes differ",
        ));
    }
    let raw = raw_senses.as_slice()?;
    let normals = contact_normals.as_slice()?;
    let counts = contact_counts.as_slice()?;
    let physiology = physiology.as_slice()?;
    let output = py
        .detach(|| transduce_nonvisual(raw, normals, counts, physiology))
        .map_err(PyValueError::new_err)?;
    Ok(Array2::from_shape_vec((residents, 43), output)
        .unwrap()
        .into_pyarray(py))
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn rotate(m: &[f64], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn proper_rotation(m: &[f64]) -> bool {
    if m.len() != 9 || m.iter().any(|x| !x.is_finite()) {
        return false;
    }
    for i in 0..3 {
        for j in 0..3 {
            let value: f64 = (0..3).map(|k| m[3 * k + i] * m[3 * k + j]).sum();
            if (value - f64::from(i == j)).abs() > 1e-5 {
                return false;
            }
        }
    }
    let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    (determinant - 1.0).abs() < 1e-5
}

fn bilinear(frame: &[f32], height: usize, width: usize, uv: [f64; 2]) -> [f32; 3] {
    let x = uv[0].clamp(0.0, 1.0) * (width - 1) as f64;
    let y = uv[1].clamp(0.0, 1.0) * (height - 1) as f64;
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let mut rgb = [0.0; 3];
    for c in 0..3 {
        let top =
            frame[(y0 * width + x0) * 3 + c] * (1.0 - fx) + frame[(y0 * width + x1) * 3 + c] * fx;
        let bottom =
            frame[(y1 * width + x0) * 3 + c] * (1.0 - fx) + frame[(y1 * width + x1) * 3 + c] * fx;
        rgb[c] = top * (1.0 - fy) + bottom * fy;
    }
    rgb
}

struct Screen<'a> {
    center: [f64; 3],
    rotation: &'a [f64],
    size: [f64; 2],
    frame: &'a [f32],
    height: usize,
    width: usize,
}

fn intersect_screen(
    origin: [f64; 3],
    direction: [f64; 3],
    screen: &Screen<'_>,
    max_range: f64,
    occluder_distance: f64,
) -> Option<(f64, [f64; 2], [f32; 3])> {
    let m = screen.rotation;
    let right = [m[0], m[3], m[6]];
    let up = [m[1], m[4], m[7]];
    let normal = [m[2], m[5], m[8]];
    let denominator = dot(&direction, &normal);
    // One-sided screen; local +Z is its emitting front face.
    if denominator >= -1e-10 {
        return None;
    }
    let to_center = std::array::from_fn(|i| screen.center[i] - origin[i]);
    let distance = dot(&to_center, &normal) / denominator;
    if distance <= 1e-9 || distance > max_range || distance >= occluder_distance {
        return None;
    }
    let relative = std::array::from_fn(|i| origin[i] + distance * direction[i] - screen.center[i]);
    let x = dot(&relative, &right);
    let y = dot(&relative, &up);
    if x.abs() > 0.5 * screen.size[0] || y.abs() > 0.5 * screen.size[1] {
        return None;
    }
    let uv = [x / screen.size[0] + 0.5, 0.5 - y / screen.size[1]];
    Some((
        distance,
        uv,
        bilinear(screen.frame, screen.height, screen.width, uv),
    ))
}

/// Fixed anatomy order and optics; no neural state, privileged world labels or RNG.
#[pyclass]
pub struct OpticRetina {
    sites: Vec<i16>,
    supported: Vec<bool>,
    eye_origins: [[f64; 3]; 2],
    directions: Vec<[f64; 3]>,
    max_range: f64,
    background: [f32; 3],
    atlas_sha256: String,
    identity: String,
}

#[pymethods]
impl OpticRetina {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        site_side_hex: PyReadonlyArray2<'_, i16>,
        supported_site_mask: PyReadonlyArray1<'_, bool>,
        eye_origins: PyReadonlyArray2<'_, f64>,
        eye_calibration: PyReadonlyArray2<'_, f64>,
        max_range: f64,
        background_rgb: PyReadonlyArray1<'_, f32>,
        atlas_sha256: String,
    ) -> PyResult<Self> {
        let sites = site_side_hex.as_slice()?;
        let supported = supported_site_mask.as_slice()?;
        let origins = eye_origins.as_slice()?;
        let calibration = eye_calibration.as_slice()?;
        let background = background_rgb.as_slice()?;
        if site_side_hex.shape() != [SITES, 3]
            || supported.len() != SITES
            || eye_origins.shape() != [2, 3]
            || eye_calibration.shape() != [2, 8]
            || background.len() != 3
            || !max_range.is_finite()
            || max_range <= 0.0
            || origins.iter().chain(calibration).any(|x| !x.is_finite())
            || background
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || atlas_sha256.len() != 64
            || !atlas_sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(PyValueError::new_err(
                "invalid bilateral optic atlas/optics",
            ));
        }
        let mut directions = Vec::with_capacity(SITES);
        let mut side_counts = [0; 2];
        let mut previous = None;
        for site in sites.chunks_exact(3) {
            let key = [site[0], site[1], site[2]];
            if !matches!(site[0], 1 | 2)
                || !(1..=36).contains(&site[1])
                || !(1..=39).contains(&site[2])
                || previous.is_some_and(|p| p >= key)
            {
                return Err(PyValueError::new_err(
                    "optic sites must be unique sorted measured side/hex rows",
                ));
            }
            previous = Some(key);
            let eye = site[0] as usize - 1;
            side_counts[eye] += 1;
            // Per-eye [q0,r0,az0,el0,daz/dq,daz/dr,del/dq,del/dr], radians.
            let c = &calibration[eye * 8..eye * 8 + 8];
            let q = site[1] as f64 - c[0];
            let r = site[2] as f64 - c[1];
            let azimuth = c[2] + c[4] * q + c[5] * r;
            let elevation = c[3] + c[6] * q + c[7] * r;
            if !azimuth.is_finite()
                || !elevation.is_finite()
                || elevation.abs() > std::f64::consts::FRAC_PI_2
            {
                return Err(PyValueError::new_err(
                    "engineered optic elevation must be within +/-pi/2",
                ));
            }
            directions.push([
                elevation.cos() * azimuth.cos(),
                elevation.cos() * azimuth.sin(),
                elevation.sin(),
            ]);
        }
        if side_counts != [879, 892] {
            return Err(PyValueError::new_err(
                "optic atlas must contain 879 left and 892 right sites",
            ));
        }
        let mut hash = Sha256::new();
        hash.update(FORMAT.as_bytes());
        hash.update(atlas_sha256.as_bytes());
        for value in sites {
            hash.update(value.to_le_bytes());
        }
        for value in supported {
            hash.update([u8::from(*value)]);
        }
        for value in origins.iter().chain(calibration) {
            hash.update(value.to_le_bytes());
        }
        hash.update(max_range.to_le_bytes());
        for value in background {
            hash.update(value.to_le_bytes());
        }
        Ok(Self {
            sites: sites.to_vec(),
            supported: supported.to_vec(),
            eye_origins: [
                origins[..3].try_into().unwrap(),
                origins[3..].try_into().unwrap(),
            ],
            directions,
            max_range,
            background: background.try_into().unwrap(),
            atlas_sha256,
            identity: format!("{:x}", hash.finalize()),
        })
    }

    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("format", FORMAT)?;
        out.set_item("identity", &self.identity)?;
        out.set_item("atlas_sha256", &self.atlas_sha256)?;
        out.set_item("site_count", SITES)?;
        out.set_item("optical_scalars", SITES * 3)?;
        out.set_item(
            "supported_sites",
            self.supported.iter().filter(|x| **x).count(),
        )?;
        out.set_item(
            "site_side_hex",
            Array2::from_shape_vec((SITES, 3), self.sites.clone())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "supported_site_mask",
            Array1::from_vec(self.supported.clone()).into_pyarray(py),
        )?;
        out.set_item(
            "ray_directions_head",
            Array2::from_shape_vec(
                (SITES, 3),
                self.directions.iter().flatten().copied().collect(),
            )
            .unwrap()
            .into_pyarray(py),
        )?;
        out.set_item(
            "eye_origins",
            Array2::from_shape_vec((2, 3), self.eye_origins.iter().flatten().copied().collect())
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("maximum_range", self.max_range)?;
        out.set_item("background_rgb", self.background)?;
        out.set_item(
            "body_axis_order",
            "+X forward,+Y left,+Z up; rotation maps head-local to world",
        )?;
        out.set_item(
            "screen_axes",
            "local X right,Y up,+Z emitting face; frame row0 top",
        )?;
        out.set_item("optics_provenance", "measured side/hex membership; engineered eye origins and affine hex-to-angle calibration")?;
        Ok(out)
    }

    /// Screen texture is optical input, not an image directly copied to neural cells.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (head_positions, head_rotations, screen_position, screen_rotation, screen_size, frame_rgb, occlusion_distance=None, scene_rgb=None, occlusion_geom=None, screen_geom=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        head_positions: PyReadonlyArray2<'_, f64>,
        head_rotations: PyReadonlyArray3<'_, f64>,
        screen_position: PyReadonlyArray1<'_, f64>,
        screen_rotation: PyReadonlyArray2<'_, f64>,
        screen_size: PyReadonlyArray1<'_, f64>,
        frame_rgb: PyReadonlyArray3<'_, f32>,
        occlusion_distance: Option<PyReadonlyArray2<'_, f64>>,
        scene_rgb: Option<PyReadonlyArray3<'_, f32>>,
        occlusion_geom: Option<PyReadonlyArray2<'_, i32>>,
        screen_geom: Option<i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let residents = head_positions.shape()[0];
        let positions = head_positions.as_slice()?;
        let rotations = head_rotations.as_slice()?;
        let center = screen_position.as_slice()?;
        let orientation = screen_rotation.as_slice()?;
        let size = screen_size.as_slice()?;
        let frame = frame_rgb.as_slice()?;
        let shape = frame_rgb.shape();
        let obstruction = occlusion_distance
            .as_ref()
            .map(|x| x.as_slice())
            .transpose()?;
        let scene = scene_rgb.as_ref().map(|x| x.as_slice()).transpose()?;
        let nearest_geom = occlusion_geom.as_ref().map(|x| x.as_slice()).transpose()?;
        if residents == 0
            || residents > MAX_RESIDENTS
            || head_positions.shape() != [residents, 3]
            || head_rotations.shape() != [residents, 3, 3]
            || center.len() != 3
            || screen_rotation.shape() != [3, 3]
            || size.len() != 2
            || shape[0] == 0
            || shape[1] == 0
            || shape[2] != 3
            || positions.iter().chain(center).any(|x| !x.is_finite())
            || !proper_rotation(orientation)
            || rotations.chunks_exact(9).any(|m| !proper_rotation(m))
            || size.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || frame
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || occlusion_distance
                .as_ref()
                .is_some_and(|x| x.shape() != [residents, SITES])
            || obstruction.is_some_and(|x| x.iter().any(|v| v.is_nan() || *v < 0.0))
            || scene_rgb
                .as_ref()
                .is_some_and(|x| x.shape() != [residents, SITES, 3])
            || scene.is_some_and(|x| x.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
            || (scene.is_some() && obstruction.is_none())
            || (nearest_geom.is_some() != screen_geom.is_some())
            || (nearest_geom.is_some() && obstruction.is_none())
            || screen_geom.is_some_and(|id| id < 0)
            || occlusion_geom
                .as_ref()
                .is_some_and(|x| x.shape() != [residents, SITES])
            || nearest_geom.is_some_and(|x| x.iter().any(|id| *id < -1))
        {
            return Err(PyValueError::new_err(
                "invalid optic screen/pose/frame sample",
            ));
        }
        let screen = Screen {
            center: center.try_into().unwrap(),
            rotation: orientation,
            size: size.try_into().unwrap(),
            frame,
            height: shape[0],
            width: shape[1],
        };
        let (rgb, hit, distance, uv) = py.detach(|| {
            let rays = residents * SITES;
            let mut rgb = scene.map_or_else(|| self.background.repeat(rays), |x| x.to_vec());
            let mut hit = vec![false; rays];
            let mut distance = vec![self.max_range; rays];
            let mut uv = vec![0.0; rays * 2];
            for row in 0..residents {
                let rotation = &rotations[row * 9..row * 9 + 9];
                let eyes: [[f64; 3]; 2] = std::array::from_fn(|eye| {
                    let offset = rotate(rotation, &self.eye_origins[eye]);
                    std::array::from_fn(|i| positions[row * 3 + i] + offset[i])
                });
                for site in 0..SITES {
                    let i = row * SITES + site;
                    if screen_geom.is_some_and(|screen| nearest_geom.unwrap()[i] != screen) {
                        continue;
                    }
                    let eye = self.sites[site * 3] as usize - 1;
                    let direction = rotate(rotation, &self.directions[site]);
                    if let Some((d, pixel, color)) = intersect_screen(
                        eyes[eye],
                        direction,
                        &screen,
                        self.max_range,
                        if screen_geom.is_some() {
                            f64::INFINITY
                        } else {
                            obstruction.map_or(f64::INFINITY, |x| x[i])
                        },
                    ) {
                        // Physical mode requires the same front-face hit as MuJoCo.
                        // The screen must not be removed from the scene ray cast.
                        if screen_geom.is_some() && (d - obstruction.unwrap()[i]).abs() > 1e-5 {
                            continue;
                        }
                        hit[i] = true;
                        distance[i] = d;
                        uv[i * 2..i * 2 + 2].copy_from_slice(&pixel);
                        rgb[i * 3..i * 3 + 3].copy_from_slice(&color);
                    }
                }
            }
            (rgb, hit, distance, uv)
        });
        let out = PyDict::new(py);
        out.set_item(
            "optic_rgb",
            Array3::from_shape_vec((residents, SITES, 3), rgb)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "screen_hit",
            Array2::from_shape_vec((residents, SITES), hit)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "screen_distance",
            Array2::from_shape_vec((residents, SITES), distance)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item(
            "screen_uv",
            Array3::from_shape_vec((residents, SITES, 2), uv)
                .unwrap()
                .into_pyarray(py),
        )?;
        out.set_item("scene_occlusion_supplied", obstruction.is_some())?;
        out.set_item("screen_geometry_verified", screen_geom.is_some())?;
        out.set_item("identity", &self.identity)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_screen_pose_texture_sidedness_and_occlusion() {
        let rotation = [0., 0., -1., -1., 0., 0., 0., 1., 0.];
        assert!(proper_rotation(&rotation));
        let frame = [1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.];
        let screen = Screen {
            center: [3., 0., 0.],
            rotation: &rotation,
            size: [2., 2.],
            frame: &frame,
            height: 2,
            width: 2,
        };
        let (distance, uv, rgb) =
            intersect_screen([0.; 3], [1., 0., 0.], &screen, 10., f64::INFINITY).unwrap();
        assert_eq!((distance, uv, rgb), (3., [0.5, 0.5], [0.5; 3]));
        let (_, left_uv, _) =
            intersect_screen([0., 0.5, 0.], [1., 0., 0.], &screen, 10., f64::INFINITY).unwrap();
        let (_, right_uv, _) =
            intersect_screen([0., -0.5, 0.], [1., 0., 0.], &screen, 10., f64::INFINITY).unwrap();
        assert!(left_uv[0] < right_uv[0]);
        assert!(intersect_screen([0.; 3], [1., 0., 0.], &screen, 10., 2.).is_none());
        assert!(intersect_screen([0.; 3], [1., 0., 0.], &screen, 2., f64::INFINITY).is_none());
        assert!(
            intersect_screen([4., 0., 0.], [-1., 0., 0.], &screen, 10., f64::INFINITY).is_none()
        );
        assert!(intersect_screen([0.; 3], [0., 1., 0.], &screen, 10., f64::INFINITY).is_none());
        assert!(
            intersect_screen([0., 3., 0.], [1., 0., 0.], &screen, 10., f64::INFINITY).is_none()
        );
        let yaw_pi = [-1., 0., 0., 0., -1., 0., 0., 0., 1.];
        assert!(intersect_screen(
            [0.; 3],
            rotate(&yaw_pi, &[1., 0., 0.]),
            &screen,
            10.,
            f64::INFINITY
        )
        .is_none());
    }
}
