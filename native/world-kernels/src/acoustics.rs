use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const LEDGERS: usize = 7;
unsafe extern "C" {
    fn chreatures_acoustic_hinges(
        m: *const std::ffi::c_void,
        d: *mut std::ffi::c_void,
        n: i32,
        dofs: *const i32,
        damping: *const f64,
        limit: *const f64,
        dt: f64,
        work: *mut f64,
    ) -> i32;
    fn chreatures_acoustic_sample(
        m: *const std::ffi::c_void,
        d: *mut std::ffi::c_void,
        n: i32,
        bodies: *const i32,
        offsets: *const f64,
        energy: *const f64,
        gain: *const f64,
        reference: *const f64,
        range: *const f64,
        occlusion: *const f64,
        listener: *const f64,
        exclude: i32,
        out: *mut f64,
    ) -> i32;
}

#[pyclass]
pub struct AcousticEngine {
    n: usize,
    bodies: Vec<i32>,
    dofs: Vec<i32>,
    offsets: Vec<f64>,
    tones: Vec<f64>,
    capacity: Vec<f64>,
    efficiency: Vec<f64>,
    threshold: Vec<f64>,
    min_speed: Vec<f64>,
    cooldown_duration: Vec<f64>,
    decay: Vec<f64>,
    radiative: Vec<f64>,
    gain: Vec<f64>,
    reference: Vec<f64>,
    range: Vec<f64>,
    occlusion: Vec<f64>,
    damping: Vec<f64>,
    torque_limit: Vec<f64>,
    drive_contact: Vec<bool>,
    energy: Vec<f64>,
    cooldown: Vec<f64>,
    events: Vec<i64>,
    ledger: [f64; LEDGERS],
    work: Vec<f64>,
}
impl AcousticEngine {
    fn harvest(&mut self, i: usize, work: f64) {
        let available = work * self.efficiency[i];
        let stored: f64 = self.energy[3 * i..3 * i + 3].iter().sum();
        let captured = available.min((self.capacity[i] - stored).max(0.0));
        for t in 0..3 {
            self.energy[3 * i + t] += self.tones[3 * i + t] * captured;
        }
        self.ledger[2] += captured;
        self.ledger[4] += (available - captured).max(0.0);
        self.ledger[3] += work - captured;
    }
}
#[pymethods]
impl AcousticEngine {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bodies: Vec<i32>,
        dofs: Vec<i32>,
        offsets: Vec<f64>,
        tones: Vec<f64>,
        capacity: Vec<f64>,
        efficiency: Vec<f64>,
        threshold: Vec<f64>,
        min_speed: Vec<f64>,
        cooldown_duration: Vec<f64>,
        decay: Vec<f64>,
        radiative: Vec<f64>,
        gain: Vec<f64>,
        reference: Vec<f64>,
        range: Vec<f64>,
        occlusion: Vec<f64>,
        damping: Vec<f64>,
        torque_limit: Vec<f64>,
        drive_contact: Vec<bool>,
        energy: Vec<f64>,
    ) -> PyResult<Self> {
        let n = bodies.len();
        let finite_nonnegative = |values: &[f64]| values.iter().all(|v| v.is_finite() && *v >= 0.0);
        if n > 64
            || dofs.len() != n
            || offsets.len() != 3 * n
            || tones.len() != 3 * n
            || energy.len() != 3 * n
            || [
                capacity.len(),
                efficiency.len(),
                threshold.len(),
                min_speed.len(),
                cooldown_duration.len(),
                decay.len(),
                radiative.len(),
                gain.len(),
                reference.len(),
                range.len(),
                occlusion.len(),
                damping.len(),
                torque_limit.len(),
                drive_contact.len(),
            ]
            .iter()
            .any(|&x| x != n)
            || !offsets.iter().all(|v| v.is_finite())
            || !finite_nonnegative(&tones)
            || !finite_nonnegative(&capacity)
            || !finite_nonnegative(&efficiency)
            || efficiency.iter().any(|v| *v > 1.0)
            || !finite_nonnegative(&threshold)
            || !finite_nonnegative(&min_speed)
            || !finite_nonnegative(&cooldown_duration)
            || !finite_nonnegative(&decay)
            || decay.iter().any(|v| *v <= 0.0)
            || !finite_nonnegative(&radiative)
            || radiative.iter().any(|v| *v > 1.0)
            || !finite_nonnegative(&gain)
            || !finite_nonnegative(&reference)
            || reference.iter().any(|v| *v <= 0.0)
            || !finite_nonnegative(&range)
            || range.iter().any(|v| *v <= 0.0)
            || !finite_nonnegative(&occlusion)
            || occlusion.iter().any(|v| *v > 1.0)
            || !finite_nonnegative(&damping)
            || !finite_nonnegative(&torque_limit)
            || !finite_nonnegative(&energy)
            || (0..n).any(|i| {
                let sum: f64 = tones[3 * i..3 * i + 3].iter().sum();
                !sum.is_finite()
                    || (sum - 1.0).abs() > 1e-12
                    || energy[3 * i..3 * i + 3].iter().sum::<f64>() > capacity[i] + 1e-12
            })
        {
            return Err(PyValueError::new_err("invalid acoustic dimensions"));
        }
        Ok(Self {
            n,
            bodies,
            dofs,
            offsets,
            tones,
            capacity,
            efficiency,
            threshold,
            min_speed,
            cooldown_duration,
            decay,
            radiative,
            gain,
            reference,
            range,
            occlusion,
            damping,
            torque_limit,
            drive_contact,
            energy,
            cooldown: vec![0.; n],
            events: vec![0; n],
            ledger: [0.; LEDGERS],
            work: vec![0.; n],
        })
    }
    fn before_substep(&mut self, py: Python<'_>, ma: usize, da: usize, dt: f64) -> PyResult<()> {
        if ma == 0 || da == 0 || !dt.is_finite() || dt <= 0.0 {
            return Err(PyValueError::new_err("invalid acoustic substep"));
        }
        let got = py.detach(|| unsafe {
            chreatures_acoustic_hinges(
                ma as *const _,
                da as *mut _,
                self.n as i32,
                self.dofs.as_ptr(),
                self.damping.as_ptr(),
                self.torque_limit.as_ptr(),
                dt,
                self.work.as_mut_ptr(),
            )
        });
        if got != self.n as i32 {
            return Err(PyValueError::new_err("native acoustic hinge failure"));
        }
        for i in 0..self.n {
            let w = self.work[i];
            self.ledger[1] += w;
            self.harvest(i, w);
        }
        Ok(())
    }
    fn ingest_contacts(
        &mut self,
        emitters: PyReadonlyArray1<'_, i32>,
        work: PyReadonlyArray1<'_, f64>,
        speed: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let es = emitters.as_slice()?;
        let ws = work.as_slice()?;
        let ss = speed.as_slice()?;
        if es.len() != ws.len()
            || es.len() != ss.len()
            || es.iter().any(|&i| i < 0 || i as usize >= self.n)
            || ws.iter().any(|v| !v.is_finite() || *v < 0.0)
            || ss.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(PyValueError::new_err("invalid acoustic contacts"));
        }
        for k in 0..es.len() {
            let i = es[k] as usize;
            if !self.drive_contact[i] {
                continue;
            }
            let w = ws[k];
            self.ledger[0] += w;
            if self.cooldown[i] > 0. || w < self.threshold[i] || ss[k] < self.min_speed[i] {
                self.ledger[3] += w;
            } else {
                self.harvest(i, w);
                self.cooldown[i] = self.cooldown_duration[i];
                self.events[i] += 1;
            }
        }
        Ok(())
    }
    fn advance(&mut self, dt: f64) -> PyResult<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PyValueError::new_err("invalid acoustic interval"));
        }
        for i in 0..self.n {
            self.cooldown[i] = (self.cooldown[i] - dt).max(0.);
            let factor = -(-dt / self.decay[i]).exp_m1();
            for t in 0..3 {
                let loss = self.energy[3 * i + t] * factor;
                self.energy[3 * i + t] -= loss;
                self.ledger[5] += loss * self.radiative[i];
                self.ledger[6] += loss * (1. - self.radiative[i]);
            }
        }
        Ok(())
    }
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        ma: usize,
        da: usize,
        listener: PyReadonlyArray1<'_, f64>,
        exclude: i32,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let l = listener.as_slice()?;
        if ma == 0 || da == 0 || l.len() != 3 || l.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("listener must have 3 values"));
        }
        let mut out = [0.; 3];
        let got = py.detach(|| unsafe {
            chreatures_acoustic_sample(
                ma as *const _,
                da as *mut _,
                self.n as i32,
                self.bodies.as_ptr(),
                self.offsets.as_ptr(),
                self.energy.as_ptr(),
                self.gain.as_ptr(),
                self.reference.as_ptr(),
                self.range.as_ptr(),
                self.occlusion.as_ptr(),
                l.as_ptr(),
                exclude,
                out.as_mut_ptr(),
            )
        });
        if got != self.n as i32 {
            return Err(PyValueError::new_err("native acoustic sample failure"));
        }
        Ok(PyArray1::from_slice(py, &out))
    }
    fn state<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Vec<i64>,
        Vec<f64>,
    ) {
        (
            PyArray1::from_slice(py, &self.energy),
            PyArray1::from_slice(py, &self.cooldown),
            self.events.clone(),
            self.ledger.to_vec(),
        )
    }
    fn restore_state(
        &mut self,
        energy: Vec<f64>,
        cooldown: Vec<f64>,
        events: Vec<i64>,
        ledger: Vec<f64>,
    ) -> PyResult<()> {
        if energy.len() != 3 * self.n
            || cooldown.len() != self.n
            || events.len() != self.n
            || ledger.len() != LEDGERS
            || energy.iter().any(|v| !v.is_finite() || *v < 0.0)
            || cooldown.iter().any(|v| !v.is_finite() || *v < 0.0)
            || events.iter().any(|v| *v < 0)
            || ledger.iter().any(|v| !v.is_finite() || *v < 0.0)
            || (0..self.n).any(|i| {
                cooldown[i] > self.cooldown_duration[i] + 1e-12
                    || energy[3 * i..3 * i + 3].iter().sum::<f64>() > self.capacity[i] + 1e-12
            })
        {
            return Err(PyValueError::new_err("invalid acoustic state"));
        }
        self.energy = energy;
        self.cooldown = cooldown;
        self.events = events;
        self.ledger.copy_from_slice(&ledger);
        Ok(())
    }
}
