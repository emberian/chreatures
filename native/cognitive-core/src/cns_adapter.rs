//! Canonical CPU reference for Anatomical CNS V3.
use crate::{gemm_into, linear, Linear};
const N: usize = 165122;
const SITES: usize = 1771;
const RECEPTORS: usize = 4107;
const TYPES: usize = 10;
const NT: usize = 11752;
const SE: usize = 4669;
const BODY: usize = 110;
const BR: usize = 11233;
const CTX: usize = 12;
const CR: usize = 1314;
const MOTOR: usize = 34;
const MR: usize = 815;
const INPUT: usize = SITES * 3 + BODY;
const LATENT: usize = 512;
const RANK: usize = 64;
fn sig(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
fn sp(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn take(p: &[f32], c: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let e = c.checked_add(n).ok_or("overflow")?;
    let v = p.get(*c..e).ok_or("truncated V3 parameters")?.to_vec();
    *c = e;
    Ok(v)
}
fn ids(a: &[u32], n: usize, b: usize) -> Result<Vec<usize>, String> {
    if a.len() != n || a.iter().any(|&x| x as usize >= b) {
        Err("V3 index shape/range differs".into())
    } else {
        Ok(a.iter().map(|&x| x as usize).collect())
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
    rr: Vec<usize>,
    rt: Vec<usize>,
    ptr: Vec<usize>,
    si: Vec<usize>,
    sw: Vec<f32>,
    br: Vec<usize>,
    bm: Vec<f32>,
    cr: Vec<usize>,
    mr: Vec<usize>,
    mm: Vec<f32>,
    nt: Vec<usize>,
    spectral: Vec<f32>,
    og: Vec<f32>,
    ob: Vec<f32>,
    mean: Vec<f32>,
    scale: Vec<f32>,
    bw: Vec<f32>,
    bb: Vec<f32>,
    dyns: [Vec<f32>; 7],
    mg: Vec<f32>,
    ma: Vec<f32>,
    mt: [f32; 3],
    neutral: Vec<f32>,
    cw: Vec<f32>,
    cb: Vec<f32>,
    proj: Linear,
    out: Linear,
    mw: Vec<f32>,
    mb: Vec<f32>,
}
impl CnsAdapter {
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed(
        p: &[f32],
        neutral: &[f32],
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
        if p.iter()
            .chain(neutral)
            .chain(sw)
            .chain(bm)
            .chain(mm)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite V3 tensor".into());
        }
        let rr = ids(rr, RECEPTORS, N)?;
        let rt = ids(rt, RECEPTORS, TYPES)?;
        let ptr = ids(ptr, RECEPTORS + 1, SE + 1)?;
        let si = ids(si, SE, SITES)?;
        let br = ids(br, BR, N)?;
        let cr = ids(cr, CR, N)?;
        let mr = ids(mr, MR, N)?;
        let nt = ids(nt, N, NT)?;
        if sw.len() != SE
            || bm.len() != BR * BODY
            || mm.len() != MOTOR * MR
            || bm.iter().chain(mm).any(|&x| x != 0.0 && x != 1.0)
            || ptr[0] != 0
            || ptr[RECEPTORS] != SE
        {
            return Err("invalid V3 topology/mask".into());
        }
        let mut inj = vec![false; N];
        for &i in rr.iter().chain(&br).chain(&cr) {
            if inj[i] {
                return Err("injected rows overlap".into());
            }
            inj[i] = true
        }
        let mut c = 0;
        let mut spectral = take(p, &mut c, TYPES * 3)?;
        for r in spectral.chunks_exact_mut(3) {
            let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = r.iter().map(|x| (*x - m).exp()).sum();
            for x in r {
                *x = (*x - m).exp() / s
            }
        }
        let og = take(p, &mut c, TYPES)?.into_iter().map(sp).collect();
        let ob = take(p, &mut c, TYPES)?;
        let mean = take(p, &mut c, BODY)?;
        let scale = take(p, &mut c, BODY)?;
        if scale.iter().any(|&x| x <= 0.0) {
            return Err("body.scale must be positive".into());
        }
        let bw = take(p, &mut c, BR * BODY)?;
        let bb = take(p, &mut c, BR)?;
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
                    _ => 0.0,
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
        let mw = take(p, &mut c, MOTOR * MR)?.into_iter().map(sp).collect();
        let mb = take(p, &mut c, MOTOR)?;
        if c != p.len() {
            return Err("trailing V3 parameters".into());
        }
        if neutral.len() != N
            || neutral
                .iter()
                .enumerate()
                .any(|(i, &x)| !(0.0..=1.0).contains(&x) || (!inj[i] && x != 0.0))
        {
            return Err("invalid neutral drive".into());
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
            rr,
            rt,
            ptr,
            si,
            sw: sw.to_vec(),
            br,
            bm: bm.to_vec(),
            cr,
            mr,
            mm: mm.to_vec(),
            nt,
            spectral,
            og,
            ob,
            mean,
            scale,
            bw,
            bb,
            dyns,
            mg,
            ma,
            mt,
            neutral: neutral.to_vec(),
            cw,
            cb,
            proj,
            out,
            mw,
            mb,
        })
    }
    fn ex(&self, k: usize) -> Vec<f32> {
        self.nt.iter().map(|&t| self.dyns[k][t]).collect()
    }
    pub fn initial_state(&self, b: usize) -> Result<CnsState, String> {
        if !(1..=32).contains(&b) {
            return Err("batch outside 1.0.32".into());
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
            || ctx.iter().any(|x| !(-1.0..=1.0).contains(x))
        {
            return Err("invalid V3 sensory/context".into());
        }
        let mut d = vec![0.0; b * N];
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
            for (j, &row) in self.br.iter().enumerate() {
                let mut u = self.bb[j];
                for i in 0..BODY {
                    u += self.bw[j * BODY + i]
                        * self.bm[j * BODY + i]
                        * ((s[q * INPUT + SITES * 3 + i] - self.mean[i]) / self.scale[i])
                            .clamp(-8.0, 8.0)
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
        if r.len() != b * N {
            return Err("invalid rate shape".into());
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
                for j in 0..MR {
                    u += self.mw[o * MR + j] * self.mm[o * MR + j] * r[q * N + self.mr[j]]
                }
                let v = sig(u);
                m[q * MOTOR + o] = if o == 24 || o == 25 { 2.0 * v - 1.0 } else { v }
            }
        }
        Ok((m, lat))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn step_flat(
        &self,
        crow: &[u32],
        col: &[u32],
        w: &[f32],
        ch: &[u32],
        drive: &[f32],
        st: &CnsState,
        dt: f32,
        b: usize,
    ) -> Result<CnsState, String> {
        let l = b * N;
        if crow.len() != N + 1
            || col.len() != w.len()
            || ch.len() != N
            || drive.len() != l
            || st.rate.len() != l
            || st.adaptation.len() != l
            || st.support.len() != l
            || st.release.len() != l
            || st.modulation.len() != l * 3
            || dt <= 0.0
            || dt > 0.2
        {
            return Err("invalid V3 graph/state".into());
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
        for _ in 0..2 {
            for q in 0..b {
                for row in 0..N {
                    let ix = q * N + row;
                    let mut fast = 0.0;
                    let mut mi = [0.0; 3];
                    for e in crow[row] as usize..crow[row + 1] as usize {
                        let src = col[e] as usize;
                        let x = r[q * N + src] - r0[src];
                        match ch[src] {
                            1 => fast += w[e] * x * st.release[q * N + src],
                            2..=4 => mi[(ch[src] - 2) as usize] += w[e] * x,
                            _ => {}
                        }
                    }
                    let h = r0[row].min(1.0 - r0[row]);
                    let ty = self.nt[row];
                    let (mut mg, mut ma): (f32, f32) = (0.0, 0.0);
                    for z in 0..3 {
                        let v = m[ix * 3 + z]
                            + (1.0 - (-dt / (2.0 * self.mt[z])).exp()) * (mi[z] - m[ix * 3 + z]);
                        nm[ix * 3 + z] = v;
                        mg += 0.5 * self.mg[ty * 3 + z].tanh() * v / h;
                        ma += 0.5 * self.ma[ty * 3 + z].tanh() * v / h
                    }
                    let u = drive[ix] - self.neutral[ix % N]
                        + g[row] * (0.5 * mg.tanh()).exp() * fast
                        - k[row] * (1.0 + 0.5 * ma.tanh()) * st.adaptation[ix];
                    let target = r0[row] + st.support[ix] * h * (u / h).tanh();
                    nr[ix] = r[ix] + (1.0 - (-dt / (2.0 * tau[row])).exp()) * (target - r[ix])
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
