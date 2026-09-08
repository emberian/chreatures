//! Canonical native CPU implementation of the immutable MaleCNS V4 service.
use crate::{gemm_into, linear, Linear};

const FORMAT: &str = "chreatures-cns-service-v4";
const GRAPH_STORAGE: &str = "ieee-754-binary16-bits-little-endian";
const GRAPH_ROUNDING: &str = "round-to-nearest-ties-to-even";
const GRAPH_COMPUTE: &str = "decode-once-to-float32";
const N: usize = 165122;
const E: usize = 25563197;
const SITES: usize = 1771;
const RECEPTORS: usize = 4107;
const TYPES: usize = 10;
const NT: usize = 11752;
const SE: usize = 4669;
const BODY_ROWS: usize = 11798;
const BODY: usize = 807;
const CTX: usize = 12;
const CR: usize = 1314;
const MOTOR: usize = 92;
const MR: usize = 815;
const INPUT: usize = SITES * 3 + BODY;
const LATENT: usize = 512;
const RANK: usize = 64;
#[inline]
fn sig(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
#[inline]
fn sp(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
#[inline]
fn half(v: u16) -> f32 {
    let s = (v as u32 & 0x8000) << 16;
    let e = (v >> 10) & 31;
    let f = v & 1023;
    let b = match e {
        0 if f == 0 => s,
        0 => {
            let mut q = f as u32;
            let mut x = 113;
            while q & 0x400 == 0 {
                q <<= 1;
                x -= 1
            }
            s | (x << 23) | ((q & 1023) << 13)
        }
        31 => s | 0x7f800000 | ((f as u32) << 13),
        _ => s | ((e as u32 + 112) << 23) | ((f as u32) << 13),
    };
    f32::from_bits(b)
}
fn take(p: &[f32], c: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let e = c.checked_add(n).ok_or("V4 parameter overflow")?;
    let v = p.get(*c..e).ok_or("truncated V4 parameters")?.to_vec();
    *c = e;
    Ok(v)
}
fn ids(a: &[u32], n: usize, b: usize, name: &str) -> Result<Vec<usize>, String> {
    if a.len() != n || a.iter().any(|&x| x as usize >= b) {
        Err(format!("invalid V4 {name}"))
    } else {
        Ok(a.iter().map(|&x| x as usize).collect())
    }
}
fn sorted(a: &[usize]) -> bool {
    a.windows(2).all(|w| w[0] < w[1])
}
fn sha(v: &str, name: &str) -> Result<String, String> {
    if v.len() != 64
        || !v
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        Err(format!("V4 {name} must be lowercase SHA-256"))
    } else {
        Ok(v.into())
    }
}

#[derive(Clone)]
pub struct CnsState {
    pub rate: Vec<f32>,
    pub adaptation: Vec<f32>,
    pub support: Vec<f32>,
    pub release: Vec<f32>,
    pub modulation: Vec<f32>,
}
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct CnsAdapter {
    identities: [String; 8],
    crow: Vec<usize>,
    col: Vec<usize>,
    weight: Vec<f32>,
    channel: Vec<u32>,
    rr: Vec<usize>,
    rt: Vec<usize>,
    ptr: Vec<usize>,
    si: Vec<usize>,
    sw: Vec<f32>,
    br: Vec<usize>,
    bw: Vec<f32>,
    bb: Vec<f32>,
    cr: Vec<usize>,
    mr: Vec<usize>,
    nt: Vec<usize>,
    spectral: Vec<f32>,
    og: Vec<f32>,
    ob: Vec<f32>,
    mean: Vec<f32>,
    inv_scale: Vec<f32>,
    dyns: [Vec<f32>; 7],
    mg: Vec<f32>,
    ma: Vec<f32>,
    mt: [f32; 3],
    neutral: Vec<f32>,
    cw: Vec<f32>,
    cb: Vec<f32>,
    proj: Linear,
    out: Linear,
    motor_reference: Vec<f32>,
    motor_inv_scale: Vec<f32>,
    mw: Vec<f32>,
    mb: Vec<f32>,
}
impl CnsAdapter {
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed(
        format: &str,
        graph_storage: &str,
        graph_rounding: &str,
        graph_compute: &str,
        identity_values: [&str; 8],
        p: &[f32],
        neutral: &[f32],
        crow: &[u32],
        col: &[u32],
        weight_bits: &[u16],
        channel: &[u32],
        rr: &[u32],
        rt: &[u32],
        ptr: &[u32],
        si: &[u32],
        sw: &[f32],
        br: &[u32],
        bm: &[f32],
        cr: &[u32],
        mr: &[u32],
        mm: &[f32],
        nt: &[u32],
    ) -> Result<Self, String> {
        if format != FORMAT
            || graph_storage != GRAPH_STORAGE
            || graph_rounding != GRAPH_ROUNDING
            || graph_compute != GRAPH_COMPUTE
        {
            return Err("requires exact CHCNS4 format and graph quantization".into());
        }
        let names = [
            "graph_sha256",
            "atlas_sha256",
            "anatomy_sha256",
            "morphology_sha256",
            "sensory_schema_sha256",
            "actuator_schema_sha256",
            "motor_calibration_sha256",
            "graph_source_weight_sha256",
        ];
        let mut iv = Vec::new();
        for (v, n) in identity_values.into_iter().zip(names) {
            iv.push(sha(v, n)?)
        }
        let identities = iv.try_into().unwrap();
        if p.iter()
            .chain(neutral)
            .chain(sw)
            .chain(bm)
            .chain(mm)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite V4 tensor".into());
        }
        let crow = ids(crow, N + 1, E + 1, "graph.crow")?;
        let col = ids(col, E, N, "graph.col")?;
        if weight_bits.len() != E
            || channel.len() != N
            || channel.iter().any(|&x| x > 4)
            || crow[0] != 0
            || crow[N] != E
            || crow.windows(2).any(|w| w[0] > w[1])
        {
            return Err("invalid V4 graph".into());
        }
        let weight: Vec<_> = weight_bits.iter().map(|&v| half(v)).collect();
        if weight.iter().any(|x| !x.is_finite())
            || weight
                .iter()
                .zip(&col)
                .any(|(&w, &s)| channel[s] == 0 && w != 0.0)
        {
            return Err("invalid canonical V4 graph weights".into());
        }
        let rr = ids(rr, RECEPTORS, N, "receptor rows")?;
        let rt = ids(rt, RECEPTORS, TYPES, "receptor types")?;
        let ptr = ids(ptr, RECEPTORS + 1, SE + 1, "receptor ptr")?;
        let si = ids(si, SE, SITES, "site indices")?;
        let br = ids(br, BODY_ROWS, N, "body rows")?;
        let cr = ids(cr, CR, N, "context rows")?;
        let mr = ids(mr, MR, N, "motor rows")?;
        let nt = ids(nt, N, NT, "neuron types")?;
        if sw.len() != SE
            || sw.iter().any(|&x| x <= 0.0)
            || bm.len() != BODY_ROWS * BODY
            || mm.len() != MOTOR * MR
            || bm.iter().chain(mm).any(|&x| x != 0.0 && x != 1.0)
            || ptr[0] != 0
            || ptr[RECEPTORS] != SE
            || ptr.windows(2).any(|w| w[0] > w[1])
            || !sorted(&rr)
            || !sorted(&br)
            || !sorted(&cr)
            || !sorted(&mr)
        {
            return Err("invalid V4 atlas/mask".into());
        }
        for j in 0..RECEPTORS {
            if ptr[j] < ptr[j + 1]
                && (sw[ptr[j]..ptr[j + 1]].iter().sum::<f32>() - 1.0).abs() > 2e-6
            {
                return Err("V4 receptor mixture must sum to one".into());
            }
        }
        let mut injected = vec![false; N];
        for &i in rr.iter().chain(&br).chain(&cr) {
            if injected[i] {
                return Err("V4 injected rows overlap".into());
            }
            injected[i] = true
        }
        if mr.iter().any(|&i| injected[i]) {
            return Err("V4 motor rows overlap injected rows".into());
        }
        let mut c = 0;
        let mut spectral = take(p, &mut c, TYPES * 3)?;
        for r in spectral.chunks_exact_mut(3) {
            let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = r.iter().map(|x| (*x - m).exp()).sum();
            for x in r {
                *x = (*x - m).exp() / z
            }
        }
        let og = take(p, &mut c, TYPES)?.into_iter().map(sp).collect();
        let ob = take(p, &mut c, TYPES)?;
        let mean = take(p, &mut c, BODY)?;
        let scale = take(p, &mut c, BODY)?;
        if scale.iter().any(|&x| x <= 0.0) {
            return Err("V4 body scale must be positive".into());
        }
        let inv_scale = scale.into_iter().map(|x| 1.0 / x).collect();
        let raw = take(p, &mut c, BODY_ROWS * BODY)?;
        let bw = raw.into_iter().zip(bm).map(|(w, &m)| w * m).collect();
        let bb = take(p, &mut c, BODY_ROWS)?;
        let cw = take(p, &mut c, CR * CTX)?;
        let cb = take(p, &mut c, CR)?;
        let mut dyns: [Vec<f32>; 7] = std::array::from_fn(|_| vec![]);
        for (i, a) in dyns.iter_mut().enumerate() {
            *a = take(p, &mut c, NT)?
                .into_iter()
                .map(|x| match i {
                    0 => 0.05 + 0.4 * sig(x),
                    1 => 0.5 + 1.5 * sig(x),
                    2 => 0.02 + 0.23 * sig(x),
                    3 => 0.5 * sig(x),
                    4 => 0.25 + 4.75 * sig(x),
                    5 => 0.05 + 1.95 * sig(x),
                    6 => 0.01 + 0.49 * sig(x),
                    _ => unreachable!(),
                })
                .collect()
        }
        let mg = take(p, &mut c, NT * 3)?;
        let ma = take(p, &mut c, NT * 3)?;
        let x = take(p, &mut c, 3)?;
        let mt = [
            0.1 + 4.9 * sig(x[0]),
            0.1 + 4.9 * sig(x[1]),
            0.1 + 4.9 * sig(x[2]),
        ];
        let pw = take(p, &mut c, RANK * N)?;
        let out = linear(p, &mut c, LATENT, RANK)?;
        let motor_reference = take(p, &mut c, MR)?;
        if motor_reference.iter().any(|&x| !(0.0..=1.0).contains(&x)) {
            return Err("V4 motor reference outside [0,1]".into());
        }
        let ms = take(p, &mut c, MR)?;
        if ms.iter().any(|&x| x <= 0.0) {
            return Err("V4 motor scale must be positive".into());
        }
        let motor_inv_scale = ms.into_iter().map(|x| 1.0 / x).collect();
        let raw = take(p, &mut c, MOTOR * MR)?;
        let mw = raw.into_iter().zip(mm).map(|(w, &m)| w * m).collect();
        let mb = take(p, &mut c, MOTOR)?;
        if c != p.len() {
            return Err("trailing V4 parameters".into());
        }
        if neutral.len() != N
            || neutral
                .iter()
                .enumerate()
                .any(|(i, &x)| !(0.0..=1.0).contains(&x) || (!injected[i] && x != 0.0))
        {
            return Err("invalid V4 neutral drive".into());
        }
        for j in 0..RECEPTORS {
            let expected = if ptr[j] < ptr[j + 1] {
                sig(ob[rt[j]])
            } else {
                0.0
            };
            if (neutral[rr[j]] - expected).abs() > 2e-7 {
                return Err("V4 optic neutral drive differs".into());
            }
        }
        for (j, &row) in br.iter().enumerate() {
            if (neutral[row] - sig(bb[j])).abs() > 2e-7 {
                return Err("V4 body neutral drive differs".into());
            }
        }
        if cr.iter().any(|&row| neutral[row] != 0.0) {
            return Err("V4 context neutral drive must be zero".into());
        }
        let mut proj = Linear {
            out: RANK,
            input: N,
            weight: pw,
            bias: vec![0.0; RANK],
        };
        for r in proj.weight.chunks_exact_mut(N) {
            for &i in rr.iter().chain(&br).chain(&cr) {
                r[i] = 0.0
            }
        }
        Ok(Self {
            identities,
            crow,
            col,
            weight,
            channel: channel.to_vec(),
            rr,
            rt,
            ptr,
            si,
            sw: sw.to_vec(),
            br,
            bw,
            bb,
            cr,
            mr,
            nt,
            spectral,
            og,
            ob,
            mean,
            inv_scale,
            dyns,
            mg,
            ma,
            mt,
            neutral: neutral.to_vec(),
            cw,
            cb,
            proj,
            out,
            motor_reference,
            motor_inv_scale,
            mw,
            mb,
        })
    }
    fn ex(&self, k: usize) -> Vec<f32> {
        self.nt.iter().map(|&t| self.dyns[k][t]).collect()
    }
    pub fn identities(&self) -> &[String; 8] {
        &self.identities
    }
    pub fn initial_state(&self, b: usize) -> Result<CnsState, String> {
        if !(1..=32).contains(&b) {
            return Err("batch outside 1..32".into());
        }
        let r0 = self.ex(0);
        Ok(CnsState {
            rate: (0..b).flat_map(|_| r0.iter().copied()).collect(),
            adaptation: vec![0.0; b * N],
            support: vec![1.0; b * N],
            release: vec![1.0; b * N],
            modulation: vec![0.0; b * N * 3],
        })
    }
    pub fn encode_drive_flat(&self, s: &[f32], ctx: &[f32], b: usize) -> Result<Vec<f32>, String> {
        if !(1..=32).contains(&b)
            || s.len() != b * INPUT
            || ctx.len() != b * CTX
            || s.iter().chain(ctx).any(|x| !x.is_finite())
            || s.chunks_exact(INPUT)
                .any(|x| x[..SITES * 3].iter().any(|&v| !(0.0..=1.0).contains(&v)))
            || ctx.iter().any(|x| !(-1.0..=1.0).contains(x))
        {
            return Err("invalid V4 sensory/context".into());
        }
        let mut d = vec![0.0; b * N];
        let mut z = vec![0.0; BODY];
        for q in 0..b {
            for j in 0..RECEPTORS {
                let mut mix = 0.0;
                for e in self.ptr[j]..self.ptr[j + 1] {
                    for k in 0..3 {
                        mix += self.sw[e]
                            * self.spectral[self.rt[j] * 3 + k]
                            * s[q * INPUT + self.si[e] * 3 + k]
                    }
                }
                if self.ptr[j] < self.ptr[j + 1] {
                    d[q * N + self.rr[j]] =
                        sig(self.ob[self.rt[j]] + self.og[self.rt[j]] * (2.0 * mix - 1.0))
                }
            }
            for i in 0..BODY {
                z[i] = ((s[q * INPUT + SITES * 3 + i] - self.mean[i]) * self.inv_scale[i])
                    .clamp(-8.0, 8.0)
            }
            for (j, &row) in self.br.iter().enumerate() {
                let w = &self.bw[j * BODY..(j + 1) * BODY];
                let mut u = self.bb[j];
                for i in 0..BODY {
                    u += w[i] * z[i]
                }
                d[q * N + row] = sig(u)
            }
            for (j, &row) in self.cr.iter().enumerate() {
                let mut u = self.cb[j];
                for i in 0..CTX {
                    u += self.cw[j * CTX + i] * ctx[q * CTX + i]
                }
                d[q * N + row] = 0.15 * (u.tanh() - self.cb[j].tanh())
            }
        }
        Ok(d)
    }
    pub fn outputs_flat(&self, r: &[f32], b: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        if r.len() != b * N || r.iter().any(|x| !x.is_finite()) {
            return Err("invalid V4 rate shape".into());
        }
        let r0 = self.ex(0);
        let mut x = r.to_vec();
        for z in x.chunks_exact_mut(N) {
            for i in 0..N {
                z[i] -= r0[i]
            }
        }
        let mut h = vec![];
        gemm_into(&x, b, N, &self.proj, &mut h);
        let mut lat = vec![];
        gemm_into(&h, b, RANK, &self.out, &mut lat);
        lat.iter_mut().for_each(|x| *x = x.tanh());
        let mut m = vec![0.0; b * MOTOR];
        for q in 0..b {
            for o in 0..MOTOR {
                let mut u = self.mb[o];
                let w = &self.mw[o * MR..(o + 1) * MR];
                for j in 0..MR {
                    u += w[j]
                        * (r[q * N + self.mr[j]] - self.motor_reference[j])
                        * self.motor_inv_scale[j]
                }
                m[q * MOTOR + o] = if o < 84 { u.tanh() } else { sig(u) }
            }
        }
        Ok((m, lat))
    }
    pub fn step_flat(
        &self,
        drive: &[f32],
        st: &CnsState,
        dt: f32,
        b: usize,
    ) -> Result<CnsState, String> {
        let l = b * N;
        if drive.len() != l
            || st.rate.len() != l
            || st.adaptation.len() != l
            || st.support.len() != l
            || st.release.len() != l
            || st.modulation.len() != l * 3
            || dt <= 0.0
            || dt > 0.2
        {
            return Err("invalid V4 state/dt".into());
        }
        let r0 = self.ex(0);
        let g = self.ex(1);
        let tau = self.ex(2);
        let k = self.ex(3);
        let ta = self.ex(4);
        let tr = self.ex(5);
        let use_ = self.ex(6);
        let mut r = st.rate.clone();
        let mut m = st.modulation.clone();
        let mut nr = vec![0.0; l];
        let mut nm = vec![0.0; l * 3];
        let delta = dt * 0.5;
        for _ in 0..2 {
            for q in 0..b {
                for row in 0..N {
                    let ix = q * N + row;
                    let mut fast = 0.0;
                    let mut mi = [0.0; 3];
                    for e in self.crow[row]..self.crow[row + 1] {
                        let src = self.col[e];
                        let x = r[q * N + src] - r0[src];
                        match self.channel[src] {
                            1 => fast += self.weight[e] * x * st.release[q * N + src],
                            2..=4 => mi[(self.channel[src] - 2) as usize] += self.weight[e] * x,
                            _ => {}
                        }
                    }
                    let h = r0[row].min(1.0 - r0[row]);
                    let ty = self.nt[row];
                    let (mut mg, mut ma) = (0.0, 0.0);
                    for z in 0..3 {
                        let v = m[ix * 3 + z]
                            + (1.0 - (-delta / self.mt[z]).exp()) * (mi[z] - m[ix * 3 + z]);
                        nm[ix * 3 + z] = v;
                        mg += 0.5 * self.mg[ty * 3 + z].tanh() * v / h;
                        ma += 0.5 * self.ma[ty * 3 + z].tanh() * v / h
                    }
                    let u = drive[ix] - self.neutral[row] + g[row] * (0.5 * mg.tanh()).exp() * fast
                        - k[row] * (1.0 + 0.5 * ma.tanh()) * st.adaptation[ix];
                    let target = r0[row] + st.support[ix] * h * (u / h).tanh();
                    nr[ix] = r[ix] + (1.0 - (-delta / tau[row]).exp()) * (target - r[ix])
                }
            }
            std::mem::swap(&mut r, &mut nr);
            std::mem::swap(&mut m, &mut nm)
        }
        let mut a = st.adaptation.clone();
        let mut s = st.support.clone();
        let mut rel = st.release.clone();
        for q in 0..b {
            for row in 0..N {
                let i = q * N + row;
                let x = r[i] - r0[row];
                let h = r0[row].min(1.0 - r0[row]);
                a[i] += (1.0 - (-dt / ta[row]).exp()) * (x - a[i]);
                s[i] = (s[i] + dt * (0.024 * (1.0 - s[i]) - 0.003 * x.abs() / h)).clamp(0.65, 1.0);
                rel[i] = (rel[i]
                    + dt * ((1.0 - rel[i]) / tr[row] - use_[row] * x.abs() / h * rel[i]))
                    .clamp(0.2, 1.0)
            }
        }
        Ok(CnsState {
            rate: r,
            adaptation: a,
            support: s,
            release: rel,
            modulation: m,
        })
    }
    pub fn neutral_drive(&self) -> &[f32] {
        &self.neutral
    }
}
#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use super::half;

    #[test]
    fn canonical_binary16_decodes_exact_special_and_normal_values() {
        assert_eq!(half(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(half(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert!(half(0x7c00).is_infinite());
        assert!(half(0x7e00).is_nan());
    }
}
