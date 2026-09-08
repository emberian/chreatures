//! Endogenous, wing-kinematics-driven near-field disturbance for fly bodies.
//!
//! This is a deliberately low-order engineering transduction.  It is not a
//! computational-fluid-dynamics model, does not generate lift, and does not
//! contain a song or wingbeat oscillator.  Every non-zero source sample comes
//! from a supplied wing rigid-body velocity relative to its thorax.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

pub const BODY_ACOUSTIC_BANDS: usize = 16;
pub const BODY_ACOUSTIC_CENTRES_HZ: [f64; BODY_ACOUSTIC_BANDS] = [
    40.0, 51.152, 65.414, 83.651, 106.973, 136.798, 174.938, 223.711, 286.083, 365.844, 467.843,
    598.279, 765.082, 978.390, 1251.169, 1600.0,
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidMotion {
    /// Body-frame origin in world coordinates.
    pub position_mm: [f64; 3],
    /// Row-major rotation from body-local coordinates to world coordinates.
    pub world_from_local: [f64; 9],
    /// XBODY angular velocity in world coordinates.
    pub angular_velocity_rad_s: [f64; 3],
    /// XBODY linear velocity of `position_mm` in world coordinates.
    pub linear_velocity_mm_s: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentKinematics {
    pub root: RigidMotion,
    pub wings: [RigidMotion; 2],
    /// Acoustic source points on the actual wing bodies, in wing-local mm.
    pub wing_centroid_local_mm: [[f64; 3]; 2],
    /// Geometry-derived, dimensionless source gains for left and right wings.
    /// These must not encode resident identity or a desired semantic message.
    pub wing_source_gain: [f64; 2],
    /// Actual left and right antennal anchor positions in world mm.
    pub antenna_position_mm: [[f64; 3]; 2],
    /// Row-major antenna-local to world rotations, left then right.
    pub antenna_world_from_local: [[f64; 9]; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcousticConfig {
    /// Sampling cadence of the supplied physical kinematics.
    pub sample_hz: f64,
    /// Removes static and very slow relative wing motion after sampling.
    pub highpass_hz: f64,
    /// Low-pass and highest reportable endogenous band as a fraction of sample rate.
    pub lowpass_fraction_of_sample_hz: f64,
    /// History duration used for the effective BODY-band RMS estimate.
    pub spectrum_window_s: f64,
    /// Softened dipole radius. It bounds coupling as separation approaches zero.
    pub dipole_radius_mm: f64,
    /// Global engineering scale multiplying geometry-derived wing source gains.
    pub coupling_scale: f64,
    /// Effective BODY amplitude equals measured proxy RMS divided by this value.
    pub body_band_reference_mm_s: f64,
}

impl Default for AcousticConfig {
    fn default() -> Self {
        Self {
            sample_hz: 1000.0,
            highpass_hz: 20.0,
            lowpass_fraction_of_sample_hz: 0.4,
            spectrum_window_s: 0.128,
            dipole_radius_mm: 0.55,
            coupling_scale: 0.12,
            body_band_reference_mm_s: 1.0,
        }
    }
}

impl AcousticConfig {
    pub fn sample_period_s(&self) -> f64 {
        self.sample_hz.recip()
    }

    /// The ceiling is deliberately below Nyquist. This is post-sampling
    /// conditioning, not an anti-aliasing guarantee for the incoming motion.
    pub fn endogenous_band_limit_hz(&self) -> f64 {
        self.sample_hz * self.lowpass_fraction_of_sample_hz
    }

    pub fn spectrum_window_samples(&self) -> usize {
        (self.sample_hz * self.spectrum_window_s).round() as usize
    }

    fn validate(&self) -> Result<(), AcousticError> {
        let values = [
            self.sample_hz,
            self.highpass_hz,
            self.lowpass_fraction_of_sample_hz,
            self.spectrum_window_s,
            self.dipole_radius_mm,
            self.coupling_scale,
            self.body_band_reference_mm_s,
        ];
        if values.iter().any(|x| !x.is_finite())
            || self.sample_hz <= 0.0
            || self.highpass_hz <= 0.0
            || !(0.0..0.5).contains(&self.lowpass_fraction_of_sample_hz)
            || self.highpass_hz >= self.endogenous_band_limit_hz()
            || self.spectrum_window_samples() < 16
            || self.spectrum_window_samples() > 16_384
            || self.dipole_radius_mm <= 0.0
            || self.coupling_scale < 0.0
            || self.body_band_reference_mm_s <= 0.0
        {
            return Err(AcousticError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AcousticFrame {
    /// Wing-generated air particle velocity at each antenna, antenna-local mm/s.
    pub antenna_airflow_local_mm_s: Vec<[[f64; 3]; 2]>,
    /// Effective RMS for BODY807 channels 46:62. Unsupported bands are exactly zero.
    pub acoustic_bands: Vec<[f64; BODY_ACOUSTIC_BANDS]>,
    pub supported_band_limit_hz: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcousticSnapshot {
    pub resident_count: usize,
    pub last_sample_time_s: Option<f64>,
    pub source_filters: Vec<SourceFilter>,
    pub listener_rings: Vec<[VectorRing; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlyAcoustics {
    config: AcousticConfig,
    resident_count: usize,
    last_sample_time_s: Option<f64>,
    source_filters: Vec<SourceFilter>,
    listener_rings: Vec<[VectorRing; 2]>,
    spectrum_basis: SpectrumBasis,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceFilter {
    pub slow: [f64; 3],
    pub fast: [f64; 3],
}

impl SourceFilter {
    fn zero() -> Self {
        Self {
            slow: [0.0; 3],
            fast: [0.0; 3],
        }
    }

    fn update(&mut self, input: [f64; 3], dt: f64, highpass_hz: f64, lowpass_hz: f64) -> [f64; 3] {
        let a_slow = 1.0 - (-2.0 * PI * highpass_hz * dt).exp();
        let a_fast = 1.0 - (-2.0 * PI * lowpass_hz * dt).exp();
        for axis in 0..3 {
            self.slow[axis] += a_slow * (input[axis] - self.slow[axis]);
            self.fast[axis] += a_fast * (input[axis] - self.fast[axis]);
        }
        sub(self.fast, self.slow)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorRing {
    pub values: Vec<[f64; 3]>,
    pub cursor: usize,
    pub count: usize,
}

impl VectorRing {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![[0.0; 3]; capacity],
            cursor: 0,
            count: 0,
        }
    }

    fn push(&mut self, value: [f64; 3]) {
        self.values[self.cursor] = value;
        self.cursor = (self.cursor + 1) % self.values.len();
        self.count = (self.count + 1).min(self.values.len());
    }

    fn ordered(&self, index: usize) -> [f64; 3] {
        let start = if self.count == self.values.len() {
            self.cursor
        } else {
            0
        };
        self.values[(start + index) % self.values.len()]
    }

    fn tone_rms(&self, basis: &ToneBasis) -> f64 {
        if self.count < self.values.len() {
            return 0.0;
        }
        let mut re = [0.0; 3];
        let mut im = [0.0; 3];
        for (i, kernel) in basis.kernel.iter().enumerate() {
            let value = self.ordered(i);
            for axis in 0..3 {
                re[axis] += value[axis] * kernel[0];
                im[axis] += value[axis] * kernel[1];
            }
        }
        let squared = (0..3)
            .map(|axis| re[axis] * re[axis] + im[axis] * im[axis])
            .sum::<f64>();
        // Hann-window coherent-amplitude correction followed by sinusoid RMS.
        basis.rms_scale * squared.sqrt()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ToneBasis {
    body_band: usize,
    /// `[Hann*cos(phase), -Hann*sin(phase)]` in chronological ring order.
    kernel: Vec<[f64; 2]>,
    rms_scale: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct SpectrumBasis {
    tones: Vec<ToneBasis>,
}

impl SpectrumBasis {
    fn new(config: AcousticConfig) -> Self {
        let n = config.spectrum_window_samples();
        let mut hann = Vec::with_capacity(n);
        let mut weight_sum = 0.0;
        for i in 0..n {
            let weight = 0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            hann.push(weight);
            weight_sum += weight;
        }
        let tones = BODY_ACOUSTIC_CENTRES_HZ
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, frequency_hz)| *frequency_hz <= config.endogenous_band_limit_hz())
            .map(|(body_band, frequency_hz)| {
                let kernel = hann
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, weight)| {
                        let phase = 2.0 * PI * frequency_hz * i as f64 / config.sample_hz;
                        [weight * phase.cos(), -weight * phase.sin()]
                    })
                    .collect();
                ToneBasis {
                    body_band,
                    kernel,
                    rms_scale: 2.0_f64.sqrt() / weight_sum,
                }
            })
            .collect();
        Self { tones }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcousticError {
    InvalidConfig,
    ResidentCount,
    NonFiniteKinematics,
    InvalidRotation,
    InvalidSampleTime,
    SnapshotShape,
}

impl std::fmt::Display for AcousticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidConfig => "invalid fly-acoustics configuration",
            Self::ResidentCount => "fly-acoustics resident count mismatch",
            Self::NonFiniteKinematics => "non-finite fly-acoustics kinematics",
            Self::InvalidRotation => "invalid fly-acoustics rotation matrix",
            Self::InvalidSampleTime => "fly-acoustics samples must follow the configured cadence",
            Self::SnapshotShape => "fly-acoustics snapshot shape mismatch",
        })
    }
}

impl std::error::Error for AcousticError {}

impl FlyAcoustics {
    pub fn new(config: AcousticConfig, resident_count: usize) -> Result<Self, AcousticError> {
        config.validate()?;
        if resident_count == 0 {
            return Err(AcousticError::ResidentCount);
        }
        let ring_len = config.spectrum_window_samples();
        Ok(Self {
            config,
            resident_count,
            last_sample_time_s: None,
            source_filters: vec![SourceFilter::zero(); resident_count * 2],
            listener_rings: (0..resident_count)
                .map(|_| [VectorRing::new(ring_len), VectorRing::new(ring_len)])
                .collect(),
            spectrum_basis: SpectrumBasis::new(config),
        })
    }

    pub fn config(&self) -> AcousticConfig {
        self.config
    }

    pub fn snapshot(&self) -> AcousticSnapshot {
        AcousticSnapshot {
            resident_count: self.resident_count,
            last_sample_time_s: self.last_sample_time_s,
            source_filters: self.source_filters.clone(),
            listener_rings: self.listener_rings.clone(),
        }
    }

    pub fn restore(
        config: AcousticConfig,
        snapshot: AcousticSnapshot,
    ) -> Result<Self, AcousticError> {
        config.validate()?;
        let ring_len = config.spectrum_window_samples();
        if snapshot.resident_count == 0
            || snapshot.source_filters.len() != snapshot.resident_count * 2
            || snapshot.listener_rings.len() != snapshot.resident_count
            || snapshot.listener_rings.iter().flatten().any(|ring| {
                ring.values.len() != ring_len
                    || ring.cursor >= ring_len
                    || ring.count > ring_len
                    || ring.values.iter().flatten().any(|x| !x.is_finite())
            })
            || snapshot.source_filters.iter().any(|state| {
                state
                    .slow
                    .iter()
                    .chain(state.fast.iter())
                    .any(|x| !x.is_finite())
            })
            || snapshot.last_sample_time_s.is_some_and(|x| !x.is_finite())
        {
            return Err(AcousticError::SnapshotShape);
        }
        Ok(Self {
            config,
            resident_count: snapshot.resident_count,
            last_sample_time_s: snapshot.last_sample_time_s,
            source_filters: snapshot.source_filters,
            listener_rings: snapshot.listener_rings,
            spectrum_basis: SpectrumBasis::new(config),
        })
    }

    /// Update filters and per-antenna history for one physical sample without
    /// evaluating the spectral observation. Production calls this for every
    /// physical packet sample, then calls [`Self::frame`] once per control tick.
    pub fn push_sample(
        &mut self,
        time_s: f64,
        residents: &[ResidentKinematics],
    ) -> Result<(), AcousticError> {
        if residents.len() != self.resident_count {
            return Err(AcousticError::ResidentCount);
        }
        self.validate_time(time_s)?;
        for resident in residents {
            validate_resident(resident)?;
        }

        let dt = self.config.sample_period_s();
        let lowpass_hz = self.config.endogenous_band_limit_hz();
        let mut sources = Vec::with_capacity(self.resident_count * 2);
        for (row, resident) in residents.iter().enumerate() {
            for wing_index in 0..2 {
                let wing = resident.wings[wing_index];
                let centre = transform_point(
                    wing.position_mm,
                    wing.world_from_local,
                    resident.wing_centroid_local_mm[wing_index],
                );
                let wing_velocity = point_velocity(wing, centre);
                let rigid_root_velocity = point_velocity(resident.root, centre);
                let relative = sub(wing_velocity, rigid_root_velocity);
                let filtered = self.source_filters[row * 2 + wing_index].update(
                    relative,
                    dt,
                    self.config.highpass_hz,
                    lowpass_hz,
                );
                sources.push((centre, filtered, resident.wing_source_gain[wing_index]));
            }
        }

        for (listener, resident) in residents.iter().enumerate() {
            for antenna in 0..2 {
                let at = resident.antenna_position_mm[antenna];
                let mut field_world = [0.0; 3];
                for (source_position, source_velocity, source_gain) in &sources {
                    field_world = add(
                        field_world,
                        dipole_field(
                            *source_position,
                            *source_velocity,
                            *source_gain * self.config.coupling_scale,
                            self.config.dipole_radius_mm,
                            at,
                        ),
                    );
                }
                let local = inverse_rotate(resident.antenna_world_from_local[antenna], field_world);
                self.listener_rings[listener][antenna].push(local);
            }
        }

        self.last_sample_time_s = Some(time_s);
        Ok(())
    }

    /// Extract the current sensory frame. This performs no trigonometric work;
    /// all supported Hann/DFT kernels were built once from configuration.
    pub fn frame(&self) -> AcousticFrame {
        let mut antenna_airflow_local_mm_s = vec![[[0.0; 3]; 2]; self.resident_count];
        let mut acoustic_bands = vec![[0.0; BODY_ACOUSTIC_BANDS]; self.resident_count];
        for listener in 0..self.resident_count {
            for antenna in 0..2 {
                let ring = &self.listener_rings[listener][antenna];
                if ring.count > 0 {
                    let latest = (ring.cursor + ring.values.len() - 1) % ring.values.len();
                    antenna_airflow_local_mm_s[listener][antenna] = ring.values[latest];
                }
            }
            for basis in &self.spectrum_basis.tones {
                let left = self.listener_rings[listener][0].tone_rms(basis);
                let right = self.listener_rings[listener][1].tone_rms(basis);
                acoustic_bands[listener][basis.body_band] = ((left * left + right * right) * 0.5)
                    .sqrt()
                    / self.config.body_band_reference_mm_s;
            }
        }
        AcousticFrame {
            antenna_airflow_local_mm_s,
            acoustic_bands,
            supported_band_limit_hz: self.config.endogenous_band_limit_hz(),
        }
    }

    /// Compatibility convenience for probes that need a frame after every
    /// sample. The production host should use `push_sample` then `frame`.
    pub fn ingest_sample(
        &mut self,
        time_s: f64,
        residents: &[ResidentKinematics],
    ) -> Result<AcousticFrame, AcousticError> {
        self.push_sample(time_s, residents)?;
        Ok(self.frame())
    }

    fn validate_time(&self, time_s: f64) -> Result<(), AcousticError> {
        if !time_s.is_finite() {
            return Err(AcousticError::InvalidSampleTime);
        }
        if let Some(previous) = self.last_sample_time_s {
            let error = (time_s - previous - self.config.sample_period_s()).abs();
            if error > self.config.sample_period_s() * 1e-6 + 1e-12 {
                return Err(AcousticError::InvalidSampleTime);
            }
        }
        Ok(())
    }
}

fn validate_resident(resident: &ResidentKinematics) -> Result<(), AcousticError> {
    let mut finite = resident
        .wing_centroid_local_mm
        .iter()
        .flatten()
        .chain(resident.wing_source_gain.iter())
        .chain(resident.antenna_position_mm.iter().flatten())
        .chain(resident.antenna_world_from_local.iter().flatten())
        .all(|x| x.is_finite());
    for rigid in std::iter::once(&resident.root).chain(resident.wings.iter()) {
        finite &= rigid
            .position_mm
            .iter()
            .chain(rigid.world_from_local.iter())
            .chain(rigid.angular_velocity_rad_s.iter())
            .chain(rigid.linear_velocity_mm_s.iter())
            .all(|x| x.is_finite());
        if !rotation_is_valid(rigid.world_from_local) {
            return Err(AcousticError::InvalidRotation);
        }
    }
    if resident.wing_source_gain.iter().any(|x| *x < 0.0) {
        finite = false;
    }
    if !finite {
        return Err(AcousticError::NonFiniteKinematics);
    }
    if resident
        .antenna_world_from_local
        .iter()
        .any(|rotation| !rotation_is_valid(*rotation))
    {
        return Err(AcousticError::InvalidRotation);
    }
    Ok(())
}

fn rotation_is_valid(rotation: [f64; 9]) -> bool {
    let row = |i: usize| [rotation[3 * i], rotation[3 * i + 1], rotation[3 * i + 2]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    (dot(r0, r0) - 1.0).abs() < 1e-5
        && (dot(r1, r1) - 1.0).abs() < 1e-5
        && (dot(r2, r2) - 1.0).abs() < 1e-5
        && dot(r0, r1).abs() < 1e-5
        && dot(r0, r2).abs() < 1e-5
        && dot(r1, r2).abs() < 1e-5
        && dot(r0, cross(r1, r2)) > 0.99999
}

fn dipole_field(
    source_mm: [f64; 3],
    source_velocity_mm_s: [f64; 3],
    gain: f64,
    radius_mm: f64,
    listener_mm: [f64; 3],
) -> [f64; 3] {
    let displacement = sub(listener_mm, source_mm);
    let distance = norm(displacement);
    if distance <= 1e-12 || gain == 0.0 {
        return [0.0; 3];
    }
    let direction = scale(displacement, distance.recip());
    let radial = dot(direction, source_velocity_mm_s);
    let tensor_velocity = sub(scale(direction, 3.0 * radial), source_velocity_mm_s);
    let radius3 = radius_mm.powi(3);
    let attenuation = gain * radius3 / (distance.powi(3) + radius3);
    scale(tensor_velocity, attenuation)
}

fn point_velocity(body: RigidMotion, point_world_mm: [f64; 3]) -> [f64; 3] {
    add(
        body.linear_velocity_mm_s,
        cross(
            body.angular_velocity_rad_s,
            sub(point_world_mm, body.position_mm),
        ),
    )
}

fn transform_point(origin: [f64; 3], rotation: [f64; 9], local: [f64; 3]) -> [f64; 3] {
    add(origin, rotate(rotation, local))
}

fn rotate(rotation: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| rotation[3 * i + j] * vector[j]).sum())
}

fn inverse_rotate(rotation: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| rotation[3 * j + i] * vector[j]).sum())
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}

fn scale(a: [f64; 3], scalar: f64) -> [f64; 3] {
    a.map(|x| x * scalar)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn rigid(position_mm: [f64; 3], angular: [f64; 3], linear: [f64; 3]) -> RigidMotion {
        RigidMotion {
            position_mm,
            world_from_local: I,
            angular_velocity_rad_s: angular,
            linear_velocity_mm_s: linear,
        }
    }

    fn resident(x: f64) -> ResidentKinematics {
        ResidentKinematics {
            root: rigid([x, 0.0, 0.0], [0.0; 3], [0.0; 3]),
            wings: [
                rigid([x, -0.5, 0.0], [0.0; 3], [0.0; 3]),
                rigid([x, 0.5, 0.0], [0.0; 3], [0.0; 3]),
            ],
            wing_centroid_local_mm: [[0.0; 3]; 2],
            wing_source_gain: [1.0; 2],
            antenna_position_mm: [[x + 0.5, -0.15, 0.0], [x + 0.5, 0.15, 0.0]],
            antenna_world_from_local: [I; 2],
        }
    }

    fn rotate_z_90(v: [f64; 3]) -> [f64; 3] {
        [-v[1], v[0], v[2]]
    }

    fn rotate_resident(mut r: ResidentKinematics) -> ResidentKinematics {
        let rz = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for body in std::iter::once(&mut r.root).chain(r.wings.iter_mut()) {
            body.position_mm = rotate_z_90(body.position_mm);
            body.world_from_local = rz;
            body.angular_velocity_rad_s = rotate_z_90(body.angular_velocity_rad_s);
            body.linear_velocity_mm_s = rotate_z_90(body.linear_velocity_mm_s);
        }
        for antenna in 0..2 {
            r.antenna_position_mm[antenna] = rotate_z_90(r.antenna_position_mm[antenna]);
            r.antenna_world_from_local[antenna] = rz;
        }
        r
    }

    #[test]
    fn chirp_rigid_motion_rotation_and_restore() {
        let config = AcousticConfig::default();

        // Any common rigid motion, including rotation about the thorax, cancels.
        let omega = [0.0, 0.0, 13.0];
        let translation = [4.0, -3.0, 2.0];
        let mut moving = resident(0.0);
        moving.root.angular_velocity_rad_s = omega;
        moving.root.linear_velocity_mm_s = translation;
        for wing in &mut moving.wings {
            wing.angular_velocity_rad_s = omega;
            wing.linear_velocity_mm_s = add(
                translation,
                cross(omega, sub(wing.position_mm, moving.root.position_mm)),
            );
        }
        let mut zero = FlyAcoustics::new(config, 1).unwrap();
        let frame = zero.ingest_sample(0.0, &[moving]).unwrap();
        assert_eq!(frame.antenna_airflow_local_mm_s, vec![[[0.0; 3]; 2]]);
        assert_eq!(frame.acoustic_bands, vec![[0.0; BODY_ACOUSTIC_BANDS]]);

        // A physical 175-Hz wing velocity is localized near the matching BODY bin.
        let mut engine = FlyAcoustics::new(config, 2).unwrap();
        let mut rotated = FlyAcoustics::new(config, 2).unwrap();
        let mut final_frame = None;
        for sample in 0..320 {
            let t = sample as f64 / config.sample_hz;
            let velocity = 18.0 * (2.0 * PI * BODY_ACOUSTIC_CENTRES_HZ[6] * t).sin();
            let mut source = resident(0.0);
            source.wings[0].linear_velocity_mm_s = [velocity, 0.0, 0.0];
            let listener = resident(2.0);
            engine.push_sample(t, &[source, listener]).unwrap();
            rotated
                .push_sample(t, &[rotate_resident(source), rotate_resident(listener)])
                .unwrap();
            if sample % 10 == 9 {
                let a = engine.frame();
                let b = rotated.frame();
                for antenna in 0..2 {
                    for axis in 0..3 {
                        assert!(
                            (a.antenna_airflow_local_mm_s[1][antenna][axis]
                                - b.antenna_airflow_local_mm_s[1][antenna][axis])
                                .abs()
                                < 1e-11
                        );
                    }
                }
                final_frame = Some(a);
            }
        }
        let frame = final_frame.unwrap();
        let peak = frame.acoustic_bands[1][..10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, 6);
        assert!(frame.acoustic_bands[1][6] > frame.acoustic_bands[1][5] * 1.5);
        assert!(frame.acoustic_bands[1][10..].iter().all(|x| *x == 0.0));

        // Snapshot continuation preserves filters, ring content, cursor and cadence.
        let snapshot = engine.snapshot();
        let encoded = serde_json::to_vec(&snapshot).unwrap();
        let decoded: AcousticSnapshot = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(snapshot, decoded);
        let mut restored = FlyAcoustics::restore(config, decoded).unwrap();
        let t = 320.0 / config.sample_hz;
        let velocity = 18.0 * (2.0 * PI * BODY_ACOUSTIC_CENTRES_HZ[6] * t).sin();
        let mut source = resident(0.0);
        source.wings[0].linear_velocity_mm_s = [velocity, 0.0, 0.0];
        let input = [source, resident(2.0)];
        engine.push_sample(t, &input).unwrap();
        restored.push_sample(t, &input).unwrap();
        assert_eq!(engine.frame(), restored.frame());
    }
}
