//! Native, body-local transduction of the physical fly. No controller lives here.
use crate::fly_types::*;

pub struct Sensed {
    pub afferents: Vec<f32>,
    /// Host-only material binding index and sampled contact fraction per fly.
    pub mouth_contacts: Vec<Vec<(usize, f64)>>,
    pub mechanical_work_rate: Vec<f64>,
}

impl PhysicalPacket<'_> {
    pub fn samples(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    pub fn nbody(&self) -> usize {
        self.positions.len() / (self.samples().max(1) * 3)
    }
    pub fn validate(&self, c: &Config) -> Result<(), String> {
        let ns = self.samples();
        let nb = self.nbody();
        if ns == 0
            || ns > 100
            || nb == 0
            || nb > 16384
            || self.positions.len() != ns * nb * 3
            || self.rotations.len() != ns * nb * 9
            || self.velocities.len() != ns * nb * 6
            || self.contacts.len() % 20 != 0
            || self.offsets.first() != Some(&0)
            || self.offsets.last().copied().unwrap_or(0) as usize != self.contacts.len() / 20
            || self.offsets.windows(2).any(|v| v[0] > v[1])
            || self.irradiance.len() != c.bodies.len()
            || self.loads.len() != self.qvel.len()
            || [
                self.qpos,
                self.qvel,
                self.loads,
                self.positions,
                self.rotations,
                self.velocities,
                self.contacts,
                self.irradiance,
            ]
            .iter()
            .any(|x| !finite(x))
        {
            return Err("invalid sampled fly physics packet".into());
        }
        for b in &c.bodies {
            if b.qpos.iter().any(|i| *i >= self.qpos.len())
                || b.dofs.iter().any(|i| *i >= self.qvel.len())
                || b.segments.iter().any(|i| *i >= nb)
                || [b.root, b.head, b.mouth.body, b.halteres[0], b.halteres[1]]
                    .iter()
                    .any(|i| *i >= nb)
                || b.eyes.iter().any(|s| s.body >= nb)
                || b.olfactory_sites.iter().any(|s| s.body >= nb)
            {
                return Err("fly physics body addresses differ".into());
            }
        }
        if c.geoms.iter().any(|g| g.body >= nb) {
            return Err("geometry body address differs".into());
        }
        for r in self.contacts.chunks_exact(20) {
            if r[..2]
                .iter()
                .any(|v| *v < 0.0 || v.fract() != 0.0 || *v as usize >= c.geoms.len())
            {
                return Err("contact geometry address differs".into());
            }
        }
        Ok(())
    }
    pub fn pos(&self, s: usize, b: usize) -> [f64; 3] {
        let k = (s * self.nbody() + b) * 3;
        self.positions[k..k + 3].try_into().unwrap()
    }
    pub fn rot(&self, s: usize, b: usize) -> &[f64] {
        let k = (s * self.nbody() + b) * 9;
        &self.rotations[k..k + 9]
    }
    pub fn vel(&self, s: usize, b: usize) -> &[f64] {
        let k = (s * self.nbody() + b) * 6;
        &self.velocities[k..k + 6]
    }
    pub fn site(&self, s: usize, b: usize, offset: [f64; 3]) -> [f64; 3] {
        let p = self.pos(s, b);
        let d = rotate(self.rot(s, b), offset);
        std::array::from_fn(|i| p[i] + d[i])
    }
    fn point_velocity(&self, s: usize, b: usize, p: [f64; 3]) -> [f64; 3] {
        let v = self.vel(s, b);
        let origin = self.pos(s, b);
        let av = cross(
            v[..3].try_into().unwrap(),
            std::array::from_fn(|i| p[i] - origin[i]),
        );
        std::array::from_fn(|i| v[i + 3] + av[i])
    }
}

/// Finite-volume concentrations interpolated from nearby regional cells.
/// The host supplies SI positions; the output contains no cell IDs or coordinates.
pub fn local_chemistry(
    c: &Config,
    eco: &chreatures_ecology_core::EcologyWorld,
    p_mm: [f64; 3],
) -> [f64; 8] {
    let p = p_mm.map(|v| v * 0.001);
    let mut near: Vec<(f64, usize)> = c
        .ecology
        .regions
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let d2 = (0..3).map(|k| (p[k] - r.center_m[k]).powi(2)).sum::<f64>();
            (d2, i)
        })
        .collect();
    near.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = [0.0; 8];
    let mut sum = 0.0;
    for (d2, i) in near.into_iter().take(4) {
        let r = &c.ecology.regions[i];
        let regularizer = r.volume_m3.powf(2.0 / 3.0) * 0.01;
        let weight = 1.0 / (d2 + regularizer.max(1e-18));
        sum += weight;
        for (k, v) in out.iter_mut().enumerate() {
            *v += weight * eco.state().regions[i].quantity[k] / r.volume_m3;
        }
    }
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

pub fn store_concentration(
    eco: &chreatures_ecology_core::EcologyWorld,
    store: &chreatures_ecology_core::StoreId,
) -> [f64; 8] {
    use chreatures_ecology_core::StoreId;
    let value = match store {
        StoreId::Region(id) => eco
            .config()
            .regions
            .iter()
            .position(|r| &r.id == id)
            .map(|i| {
                (
                    &eco.state().regions[i].quantity,
                    eco.config().regions[i].volume_m3,
                )
            }),
        StoreId::Organism(id) => eco
            .state()
            .organisms
            .iter()
            .find(|r| &r.id == id)
            .map(|r| (&r.material.quantity, r.internal_volume_m3)),
        StoreId::Packet(id) => eco
            .state()
            .packets
            .iter()
            .find(|r| &r.id == id && r.active)
            .map(|r| (&r.material.quantity, r.volume_m3)),
        StoreId::Structure(id) => eco
            .state()
            .structures
            .iter()
            .find(|r| &r.id == id && r.active)
            .map(|r| (&r.material.quantity, r.nominal_volume_m3)),
    };
    value
        .map(|(q, v)| std::array::from_fn(|i| q[i] / v))
        .unwrap_or([0.0; 8])
}

pub fn transduce(
    c: &Config,
    state: &Saved,
    eco: &chreatures_ecology_core::EcologyWorld,
    p: &PhysicalPacket<'_>,
) -> Result<Sensed, String> {
    p.validate(c)?;
    let ns = p.samples();
    let last = ns - 1;
    let weight = 1.0 / ns as f64;
    let mut out = vec![0.0; c.bodies.len() * CHANNELS];
    let mut mouth_contacts = vec![Vec::new(); c.bodies.len()];
    let mut mechanical_work_rate = vec![0.0; c.bodies.len()];
    for (row, b) in c.bodies.iter().enumerate() {
        let r = &state.residents[row];
        let o = &mut out[row * CHANNELS..(row + 1) * CHANNELS];
        let root_rot = p.rot(last, b.root);
        for (i, site) in b.olfactory_sites.iter().enumerate() {
            let at = p.site(last, site.body, site.position);
            let chemical = local_chemistry(c, eco, at);
            for k in 0..8 {
                o[i * 8 + k] = (chemical[k] * c.volatile_fraction[k]) as f32;
            }
            if i < 2 {
                let v = p.point_velocity(last, site.body, at);
                let relative = local(
                    p.rot(last, site.body),
                    std::array::from_fn(|k| c.airflow_mm_s[k] - v[k]),
                );
                for k in 0..3 {
                    o[40 + i * 3 + k] = relative[k] as f32;
                }
            }
        }
        let thorax_velocity = p.vel(last, b.root);
        let lv = local(root_rot, thorax_velocity[3..6].try_into().unwrap());
        let av = local(root_rot, thorax_velocity[..3].try_into().unwrap());
        for k in 0..3 {
            o[62 + k] = lv[k] as f32;
            o[65 + k] = av[k] as f32;
        }
        o[68] = p.irradiance[row].max(0.0) as f32;
        let internal = eco
            .fly_interoception12(&b.ecology_id, &c.interoception)
            .map_err(|e| e.to_string())?;
        for k in 0..12 {
            o[69 + k] = internal[k] as f32;
        }
        for j in 0..JOINTS {
            o[81 + j] = p.qpos[b.qpos[j]] as f32;
            o[207 + j] = p.qvel[b.dofs[j]] as f32;
            o[333 + j] = p.loads[b.dofs[j]] as f32;
            mechanical_work_rate[row] += (p.loads[b.dofs[j]] * p.qvel[b.dofs[j]]).abs();
        }
        let mut mouth_hits = vec![0.0; c.material_bindings.len()];
        let mut mouth_normal = [0.0; 3];
        let mut mouth_fraction = 0.0;
        let mut foot_counts = [0.0; 6];
        for s in 0..ns {
            let mut touched = vec![false; c.material_bindings.len()];
            let mut mouth_sample = false;
            let mouth = p.site(s, b.mouth.body, b.mouth.position);
            for ci in p.offsets[s] as usize..p.offsets[s + 1] as usize {
                let rec = &p.contacts[ci * 20..ci * 20 + 20];
                let g = [rec[0] as usize, rec[1] as usize];
                let bodies = [c.geoms[g[0]].body, c.geoms[g[1]].body];
                let position: [f64; 3] = rec[2..5].try_into().unwrap();
                // MuJoCo contact frame rows are basis vectors; force is on geom2.
                let world_force = local(&rec[5..14], rec[14..17].try_into().unwrap());
                for side in 0..2 {
                    let Some(segment) = b.segments.iter().position(|id| *id == bodies[side]) else {
                        continue;
                    };
                    let sign = if side == 0 { -1.0 } else { 1.0 };
                    let force = world_force.map(|x| x * sign);
                    let segment_force = local(p.rot(s, bodies[side]), force);
                    for k in 0..3 {
                        o[495 + segment * 3 + k] += (weight * segment_force[k]) as f32;
                    }
                    for (foot, members) in b.feet.iter().enumerate() {
                        if !members.contains(&bodies[side]) {
                            continue;
                        }
                        let f = local(p.rot(s, b.root), force);
                        let va = p.point_velocity(s, bodies[side], position);
                        let vb = p.point_velocity(s, bodies[1 - side], position);
                        let relative: [f64; 3] = std::array::from_fn(|k| va[k] - vb[k]);
                        let normal: [f64; 3] = rec[5..8].try_into().unwrap();
                        let vn = (0..3).map(|k| relative[k] * normal[k]).sum::<f64>();
                        let slip = local(
                            p.rot(s, b.root),
                            std::array::from_fn(|k| relative[k] - vn * normal[k]),
                        );
                        for k in 0..3 {
                            o[459 + foot * 6 + k] += (weight * f[k]) as f32;
                            o[462 + foot * 6 + k] += (weight * slip[k]) as f32;
                        }
                        foot_counts[foot] += weight;
                    }
                    if bodies[side] == b.mouth.body
                        && norm(std::array::from_fn(|k| position[k] - mouth[k]))
                            <= b.mouth.contact_radius_mm
                    {
                        mouth_sample = true;
                        let normal = local(
                            p.rot(s, b.mouth.body),
                            std::array::from_fn(|k| rec[5 + k] * sign),
                        );
                        for k in 0..3 {
                            mouth_normal[k] += weight * normal[k];
                        }
                        for (mi, material) in c.material_bindings.iter().enumerate() {
                            if material.exposed && material.geoms.contains(&g[1 - side]) {
                                touched[mi] = true;
                            }
                        }
                    }
                }
            }
            if mouth_sample {
                mouth_fraction += weight;
            }
            for (i, t) in touched.iter().enumerate() {
                if *t {
                    mouth_hits[i] += weight;
                }
            }
        }
        for foot in 0..6 {
            // Force sums across contact points; slip is their conditional average.
            if foot_counts[foot] > 0.0 {
                for k in 0..3 {
                    o[462 + foot * 6 + k] /= foot_counts[foot] as f32;
                }
            }
        }
        let mn = norm(mouth_normal);
        if mn > 0.0 {
            for k in 0..3 {
                o[708 + k] = (mouth_normal[k] / mn) as f32;
            }
        }
        o[711] = mouth_fraction as f32;
        let total_mouth = mouth_hits.iter().sum::<f64>();
        for (i, fraction) in mouth_hits.iter().enumerate().filter(|(_, v)| **v > 0.0) {
            mouth_contacts[row].push((i, *fraction));
            let chemistry = store_concentration(eco, &c.material_bindings[i].store);
            for k in 0..8 {
                o[32 + k] += (chemistry[k] * fraction / total_mouth.max(1e-12)) as f32;
            }
        }
        for (i, h) in b.halteres.iter().enumerate() {
            let v = p.vel(last, *h);
            let relative = std::array::from_fn(|k| v[3 + k] - thorax_velocity[3 + k]);
            let coriolis =
                cross(thorax_velocity[..3].try_into().unwrap(), relative).map(|x| 2.0 * x);
            let inertial = local(p.rot(last, *h), coriolis);
            for k in 0..3 {
                o[702 + i * 3 + k] = inertial[k] as f32;
            }
        }
        for k in 0..MOTOR_CHANNELS {
            o[712 + k] = r.fatigue[k] as f32;
        }
        o[804] = r.pump_phase.sin() as f32;
        o[805] = r.pump_phase.cos() as f32;
        o[806] = r.pump_flow as f32;
        let at = p.pos(last, b.head);
        for emission in &state.emissions {
            let distance = norm(std::array::from_fn(|k| at[k] - emission.origin_mm[k]));
            let arrived = state.time - emission.born - distance / 343000.0;
            if !(0.0..emission.duration).contains(&arrived) {
                continue;
            }
            let envelope = (arrived / 0.002)
                .clamp(0.0, 1.0)
                .min(((emission.duration - arrived) / 0.005).clamp(0.0, 1.0));
            for k in 0..16 {
                let centre = 40.0_f64.ln() + (1600.0_f64 / 40.0).ln() * k as f64 / 15.0;
                let tuning = (-0.5
                    * ((emission.frequency_hz.ln() - centre) / (0.35 * 2.0_f64.ln())).powi(2))
                .exp();
                o[46 + k] += (emission.envelope * envelope * tuning
                    / (1.0 + (distance / 10.0).powi(2))) as f32;
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite physical afferent".into());
    }
    Ok(Sensed {
        afferents: out,
        mouth_contacts,
        mechanical_work_rate,
    })
}
