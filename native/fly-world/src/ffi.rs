// SPDX-License-Identifier: AGPL-3.0-or-later
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawDimensions {
    header_version: c_int,
    runtime_version: c_int,
    nq: c_int,
    nv: c_int,
    nu: c_int,
    na: c_int,
    nbody: c_int,
    njnt: c_int,
    ngeom: c_int,
    nsite: c_int,
    nsensor: c_int,
    nsensordata: c_int,
    nkey: c_int,
    nmocap: c_int,
    nuserdata: c_int,
    neq: c_int,
    npair: c_int,
    nmat: c_int,
    nmesh: c_int,
    nmeshvert: c_int,
    nmeshnormal: c_int,
    nmeshface: c_int,
    integration_state_size: c_int,
    ncon: c_int,
    integrator: c_int,
    timestep: f64,
    time: f64,
}
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Dimensions {
    pub version: i32,
    pub nq: usize,
    pub nv: usize,
    pub nu: usize,
    pub na: usize,
    pub nbody: usize,
    pub njnt: usize,
    pub ngeom: usize,
    pub nsite: usize,
    pub nsensor: usize,
    pub nsensordata: usize,
    pub nkey: usize,
    pub nmocap: usize,
    pub nuserdata: usize,
    pub neq: usize,
    pub npair: usize,
    pub nmat: usize,
    pub nmesh: usize,
    pub integrator: i32,
    pub timestep: f64,
    pub time: f64,
}
#[allow(dead_code)]
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum NumField {
    Qpos = 1,
    Qvel = 2,
    Act = 3,
    QaccWarmstart = 4,
    Ctrl = 5,
    QfrcActuator = 6,
    QfrcConstraint = 7,
    QfrcApplied = 8,
    XfrcApplied = 9,
    MocapPos = 10,
    MocapQuat = 11,
    UserData = 12,
    BodyXpos = 13,
    BodyXquat = 14,
    BodyXmat = 15,
    GeomXpos = 16,
    GeomXmat = 17,
    SensorData = 18,
    Time = 19,
    ActuatorForce = 20,
    BodyXipos = 21,
    GeomSize = 100,
    GeomPos = 101,
    GeomQuat = 102,
    GeomRgba = 103,
    GeomFriction = 104,
    ActuatorForceRange = 105,
    ActuatorGainPrm = 106,
    MatRgba = 107,
    MeshVert = 108,
    MeshNormal = 109,
}
#[allow(dead_code)]
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum IntField {
    GeomBodyId = 1,
    GeomType = 2,
    GeomMatId = 3,
    GeomContype = 4,
    GeomConaffinity = 5,
    GeomGroup = 6,
    MeshVertAdr = 7,
    MeshVertNum = 8,
    MeshNormalAdr = 9,
    MeshNormalNum = 10,
    MeshFaceAdr = 11,
    MeshFaceNum = 12,
    MeshFace = 13,
    GeomDataId = 14,
}
unsafe extern "C" {
    fn fly_world_version_compatible() -> c_int;
    fn fly_world_load_xml(path: *const c_char, error: *mut c_char, capacity: usize) -> *mut c_void;
    fn fly_world_make_data(model: *const c_void) -> *mut c_void;
    fn fly_world_delete_data(data: *mut c_void);
    fn fly_world_delete_model(model: *mut c_void);
    fn fly_world_material_albedo_len(model: *const c_void) -> i64;
    fn fly_world_material_albedo(model: *const c_void, output: *mut f64, count: usize) -> c_int;
    fn fly_world_dimensions(
        model: *const c_void,
        data: *const c_void,
        out: *mut RawDimensions,
    ) -> c_int;
    fn fly_world_reset_keyframe(model: *const c_void, data: *mut c_void, key: c_int) -> c_int;
    fn fly_world_forward(model: *const c_void, data: *mut c_void) -> c_int;
    fn fly_world_step1(model: *const c_void, data: *mut c_void) -> c_int;
    fn fly_world_step2(model: *const c_void, data: *mut c_void) -> c_int;
    fn fly_world_set_const(model: *mut c_void, data: *mut c_void) -> c_int;
    fn fly_world_state_integration() -> c_int;
    fn fly_world_state_size(model: *const c_void, spec: c_int) -> c_int;
    fn fly_world_get_state(
        model: *const c_void,
        data: *const c_void,
        spec: c_int,
        out: *mut f64,
        len: usize,
    ) -> c_int;
    fn fly_world_set_state(
        model: *const c_void,
        data: *mut c_void,
        spec: c_int,
        input: *const f64,
        len: usize,
    ) -> c_int;
    fn fly_world_num_len(model: *const c_void, field: c_int) -> i64;
    fn fly_world_num_read(
        model: *const c_void,
        data: *const c_void,
        field: c_int,
        offset: usize,
        out: *mut f64,
        count: usize,
    ) -> c_int;
    fn fly_world_num_write(
        model: *mut c_void,
        data: *mut c_void,
        field: c_int,
        offset: usize,
        input: *const f64,
        count: usize,
    ) -> c_int;
    fn fly_world_int_len(model: *const c_void, field: c_int) -> i64;
    fn fly_world_int_read(
        model: *const c_void,
        field: c_int,
        offset: usize,
        out: *mut i32,
        count: usize,
    ) -> c_int;
    fn fly_world_int_write(
        model: *mut c_void,
        field: c_int,
        offset: usize,
        input: *const i32,
        count: usize,
    ) -> c_int;
    fn fly_world_xbody_velocity_all(
        model: *const c_void,
        data: *const c_void,
        out: *mut f64,
        len: usize,
    ) -> c_int;
    fn fly_world_xbody_velocity_selected(
        model: *const c_void,
        data: *const c_void,
        bodies: *const i32,
        count: usize,
        out: *mut f64,
    ) -> c_int;
    fn fly_world_contact_count(data: *const c_void) -> c_int;
    fn fly_world_contact_records(
        model: *const c_void,
        data: *const c_void,
        records: *mut f64,
        distances: *mut f64,
        capacity: usize,
    ) -> c_int;
    fn fly_world_multi_ray_masked(
        model: *mut c_void,
        data: *mut c_void,
        origin: *const f64,
        directions: *const f64,
        ray_count: c_int,
        hidden: *const i32,
        hidden_count: usize,
        cutoff: f64,
        geoms: *mut c_int,
        distances: *mut f64,
        normals: *mut f64,
    ) -> c_int;
    fn fly_world_rays(
        model: *mut c_void,
        data: *const c_void,
        origins: *const f64,
        directions: *const f64,
        ray_count: c_int,
        cutoff: f64,
        geoms: *mut c_int,
        distances: *mut f64,
    ) -> c_int;
    fn fly_world_endpoint_inside(
        model: *mut c_void,
        data: *mut c_void,
        proxy_geom: c_int,
        source_geom_count: c_int,
        points: *const f64,
        point_count: usize,
        inside: *mut u8,
    ) -> c_int;
    fn fly_world_geom_distance(
        model: *const c_void,
        data: *mut c_void,
        g1: c_int,
        g2: c_int,
        maximum: f64,
        from_to: *mut f64,
        distance: *mut f64,
    ) -> c_int;
    fn fly_world_body_name_to_id(model: *const c_void, name: *const c_char) -> c_int;
    fn fly_world_geom_name_to_id(model: *const c_void, name: *const c_char) -> c_int;
}

pub struct Physics {
    model: NonNull<c_void>,
    data: NonNull<c_void>,
}
impl Drop for Physics {
    fn drop(&mut self) {
        unsafe {
            fly_world_delete_data(self.data.as_ptr());
            fly_world_delete_model(self.model.as_ptr())
        }
    }
}
impl Physics {
    pub fn load(path: &Path) -> Result<Self, String> {
        if unsafe { fly_world_version_compatible() } != 1 {
            return Err("MuJoCo header/runtime must both be 3.12".into());
        }
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "MuJoCo path contains NUL")?;
        let mut error = vec![0i8; 2048];
        let model = NonNull::new(unsafe {
            fly_world_load_xml(path.as_ptr(), error.as_mut_ptr(), error.len())
        })
        .ok_or_else(|| {
            unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })?;
        let data = match NonNull::new(unsafe { fly_world_make_data(model.as_ptr()) }) {
            Some(data) => data,
            None => {
                unsafe { fly_world_delete_model(model.as_ptr()) };
                return Err("MuJoCo data allocation failed".into());
            }
        };
        Ok(Self { model, data })
    }
    pub fn dimensions(&self) -> Dimensions {
        let mut r = RawDimensions::default();
        assert_eq!(
            unsafe { fly_world_dimensions(self.model.as_ptr(), self.data.as_ptr(), &mut r) },
            1
        );
        Dimensions {
            version: r.runtime_version,
            nq: r.nq as usize,
            nv: r.nv as usize,
            nu: r.nu as usize,
            na: r.na as usize,
            nbody: r.nbody as usize,
            njnt: r.njnt as usize,
            ngeom: r.ngeom as usize,
            nsite: r.nsite as usize,
            nsensor: r.nsensor as usize,
            nsensordata: r.nsensordata as usize,
            nkey: r.nkey as usize,
            nmocap: r.nmocap as usize,
            nuserdata: r.nuserdata as usize,
            neq: r.neq as usize,
            npair: r.npair as usize,
            nmat: r.nmat as usize,
            nmesh: r.nmesh as usize,
            integrator: r.integrator,
            timestep: r.timestep,
            time: r.time,
        }
    }
    pub fn reset_keyframe(&mut self, key: i32) -> Result<(), String> {
        if unsafe { fly_world_reset_keyframe(self.model.as_ptr(), self.data.as_ptr(), key) } == 1 {
            Ok(())
        } else {
            Err("MuJoCo keyframe reset failed".into())
        }
    }
    pub fn forward(&mut self) -> Result<(), String> {
        if unsafe { fly_world_forward(self.model.as_ptr(), self.data.as_ptr()) } == 1 {
            Ok(())
        } else {
            Err("MuJoCo forward failed".into())
        }
    }
    pub fn step1(&mut self) -> Result<(), String> {
        if unsafe { fly_world_step1(self.model.as_ptr(), self.data.as_ptr()) } == 1 {
            Ok(())
        } else {
            Err("MuJoCo step1 failed".into())
        }
    }
    pub fn step2(&mut self) -> Result<(), String> {
        if unsafe { fly_world_step2(self.model.as_ptr(), self.data.as_ptr()) } == 1 {
            Ok(())
        } else {
            Err("MuJoCo step2 failed".into())
        }
    }
    pub fn set_const(&mut self) -> Result<(), String> {
        if unsafe { fly_world_set_const(self.model.as_ptr(), self.data.as_ptr()) } == 1 {
            Ok(())
        } else {
            Err("MuJoCo constant refresh failed".into())
        }
    }
    pub fn time(&self) -> f64 {
        self.dimensions().time
    }
    pub fn material_texture_albedo(&self) -> Result<Vec<f64>, String> {
        let length = unsafe { fly_world_material_albedo_len(self.model.as_ptr()) };
        if length < 0 {
            return Err("MuJoCo material texture albedo shape differs".into());
        }
        let mut output = vec![0.0; length as usize];
        if unsafe {
            fly_world_material_albedo(self.model.as_ptr(), output.as_mut_ptr(), output.len())
        } == 1
        {
            Ok(output)
        } else {
            Err("MuJoCo material texture albedo read failed".into())
        }
    }
    pub fn num(&self, field: NumField) -> Result<Vec<f64>, String> {
        let n = unsafe { fly_world_num_len(self.model.as_ptr(), field as i32) };
        if n < 0 {
            return Err("unknown numeric MuJoCo field".into());
        }
        let mut out = vec![0.0; n as usize];
        if unsafe {
            fly_world_num_read(
                self.model.as_ptr(),
                self.data.as_ptr(),
                field as i32,
                0,
                out.as_mut_ptr(),
                out.len(),
            )
        } == 1
        {
            Ok(out)
        } else {
            Err("MuJoCo numeric read failed".into())
        }
    }
    pub fn read_num_into(&self, field: NumField, output: &mut [f64]) -> Result<(), String> {
        let n = unsafe { fly_world_num_len(self.model.as_ptr(), field as i32) };
        if n < 0 || n as usize != output.len() {
            return Err("MuJoCo numeric output shape differs".into());
        }
        if unsafe {
            fly_world_num_read(
                self.model.as_ptr(),
                self.data.as_ptr(),
                field as i32,
                0,
                output.as_mut_ptr(),
                output.len(),
            )
        } == 1
        {
            Ok(())
        } else {
            Err("MuJoCo numeric read failed".into())
        }
    }
    pub fn write_num(&mut self, field: NumField, input: &[f64]) -> Result<(), String> {
        self.write_num_at(field, 0, input)
    }
    pub fn write_num_at(
        &mut self,
        field: NumField,
        offset: usize,
        input: &[f64],
    ) -> Result<(), String> {
        if unsafe {
            fly_world_num_write(
                self.model.as_ptr(),
                self.data.as_ptr(),
                field as i32,
                offset,
                input.as_ptr(),
                input.len(),
            )
        } == 1
        {
            Ok(())
        } else {
            Err("MuJoCo numeric write failed".into())
        }
    }
    pub fn int(&self, field: IntField) -> Result<Vec<i32>, String> {
        let n = unsafe { fly_world_int_len(self.model.as_ptr(), field as i32) };
        if n < 0 {
            return Err("unknown integer MuJoCo field".into());
        }
        let mut out = vec![0; n as usize];
        if unsafe {
            fly_world_int_read(
                self.model.as_ptr(),
                field as i32,
                0,
                out.as_mut_ptr(),
                out.len(),
            )
        } == 1
        {
            Ok(out)
        } else {
            Err("MuJoCo integer read failed".into())
        }
    }
    pub fn write_int(&mut self, field: IntField, input: &[i32]) -> Result<(), String> {
        if unsafe {
            fly_world_int_write(
                self.model.as_ptr(),
                field as i32,
                0,
                input.as_ptr(),
                input.len(),
            )
        } == 1
        {
            Ok(())
        } else {
            Err("MuJoCo integer write failed".into())
        }
    }
    pub fn xbody_velocities(&self) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0; self.dimensions().nbody * 6];
        if unsafe {
            fly_world_xbody_velocity_all(
                self.model.as_ptr(),
                self.data.as_ptr(),
                out.as_mut_ptr(),
                out.len(),
            )
        } == 1
        {
            Ok(out)
        } else {
            Err("MuJoCo body velocity read failed".into())
        }
    }
    pub fn xbody_velocities_selected(
        &self,
        bodies: &[i32],
        output: &mut [f64],
    ) -> Result<(), String> {
        if output.len() != bodies.len() * 6 {
            return Err("selected body velocity output shape differs".into());
        }
        if unsafe {
            fly_world_xbody_velocity_selected(
                self.model.as_ptr(),
                self.data.as_ptr(),
                bodies.as_ptr(),
                bodies.len(),
                output.as_mut_ptr(),
            )
        } == 1
        {
            Ok(())
        } else {
            Err("selected MuJoCo body velocity read failed".into())
        }
    }
    pub fn contacts(&self) -> Result<Vec<f64>, String> {
        let n = unsafe { fly_world_contact_count(self.data.as_ptr()) };
        if n < 0 {
            return Err("MuJoCo contact count failed".into());
        }
        let mut out = vec![0.0; n as usize * 20];
        if unsafe {
            fly_world_contact_records(
                self.model.as_ptr(),
                self.data.as_ptr(),
                out.as_mut_ptr(),
                std::ptr::null_mut(),
                n as usize,
            )
        } == n
        {
            Ok(out)
        } else {
            Err("MuJoCo contact extraction failed".into())
        }
    }
    pub fn contacts_with_distances(&self) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = unsafe { fly_world_contact_count(self.data.as_ptr()) };
        if n < 0 {
            return Err("MuJoCo contact count failed".into());
        }
        let mut out = vec![0.0; n as usize * 20];
        let mut distances = vec![0.0; n as usize];
        if unsafe {
            fly_world_contact_records(
                self.model.as_ptr(),
                self.data.as_ptr(),
                out.as_mut_ptr(),
                distances.as_mut_ptr(),
                n as usize,
            )
        } == n
        {
            Ok((out, distances))
        } else {
            Err("MuJoCo contact extraction failed".into())
        }
    }
    pub fn multi_ray(
        &mut self,
        origin: [f64; 3],
        directions: &[f64],
        hidden: &[i32],
        cutoff: f64,
    ) -> Result<(Vec<i32>, Vec<f64>), String> {
        if directions.len() % 3 != 0 {
            return Err("ray directions stride differs".into());
        }
        let n = directions.len() / 3;
        let mut geoms = vec![-1; n];
        let mut distances = vec![cutoff; n];
        if unsafe {
            fly_world_multi_ray_masked(
                self.model.as_ptr(),
                self.data.as_ptr(),
                origin.as_ptr(),
                directions.as_ptr(),
                n as i32,
                hidden.as_ptr(),
                hidden.len(),
                cutoff,
                geoms.as_mut_ptr(),
                distances.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        } == 1
        {
            Ok((geoms, distances))
        } else {
            Err("MuJoCo masked multi-ray failed".into())
        }
    }
    pub fn rays(
        &mut self,
        origins: &[f64],
        directions: &[f64],
        cutoff: f64,
    ) -> Result<(Vec<i32>, Vec<f64>), String> {
        if origins.len() != directions.len() || origins.len() % 3 != 0 {
            return Err("native ray batch shape differs".into());
        }
        let n = origins.len() / 3;
        let mut geoms = vec![-1; n];
        let mut distances = vec![cutoff; n];
        if unsafe {
            fly_world_rays(
                self.model.as_ptr(),
                self.data.as_ptr(),
                origins.as_ptr(),
                directions.as_ptr(),
                n as i32,
                cutoff,
                geoms.as_mut_ptr(),
                distances.as_mut_ptr(),
            )
        } == 1
        {
            Ok((geoms, distances))
        } else {
            Err("native arbitrary ray batch failed".into())
        }
    }
    pub fn endpoints_inside(
        &mut self,
        proxy: usize,
        source_geoms: usize,
        points: &[f64],
    ) -> Result<Vec<bool>, String> {
        if points.len() % 3 != 0 {
            return Err("route endpoint stride differs".into());
        }
        let mut raw = vec![0u8; points.len() / 3];
        if unsafe {
            fly_world_endpoint_inside(
                self.model.as_ptr(),
                self.data.as_ptr(),
                proxy as i32,
                source_geoms as i32,
                points.as_ptr(),
                raw.len(),
                raw.as_mut_ptr(),
            )
        } != 1
        {
            return Err("route endpoint containment failed".into());
        }
        Ok(raw.into_iter().map(|v| v != 0).collect())
    }
    pub fn geom_distance(
        &mut self,
        g1: usize,
        g2: usize,
        maximum: f64,
    ) -> Result<(f64, [f64; 6]), String> {
        let mut d = 0.0;
        let mut points = [0.0; 6];
        if unsafe {
            fly_world_geom_distance(
                self.model.as_ptr(),
                self.data.as_ptr(),
                g1 as i32,
                g2 as i32,
                maximum,
                points.as_mut_ptr(),
                &mut d,
            )
        } == 1
        {
            Ok((d, points))
        } else {
            Err("MuJoCo geometry distance failed".into())
        }
    }
    pub fn body_name_to_id(&self, name: &str) -> Result<usize, String> {
        let n = CString::new(name).map_err(|_| "body name contains NUL")?;
        let id = unsafe { fly_world_body_name_to_id(self.model.as_ptr(), n.as_ptr()) };
        usize::try_from(id).map_err(|_| format!("MuJoCo body name missing: {name}"))
    }
    pub fn geom_name_to_id(&self, name: &str) -> Result<usize, String> {
        let n = CString::new(name).map_err(|_| "geom name contains NUL")?;
        let id = unsafe { fly_world_geom_name_to_id(self.model.as_ptr(), n.as_ptr()) };
        usize::try_from(id).map_err(|_| format!("MuJoCo geom name missing: {name}"))
    }
    pub fn state(&self) -> Result<Vec<f64>, String> {
        let spec = unsafe { fly_world_state_integration() };
        let n = unsafe { fly_world_state_size(self.model.as_ptr(), spec) };
        if n < 0 {
            return Err("MuJoCo state size failed".into());
        }
        let mut out = vec![0.0; n as usize];
        if unsafe {
            fly_world_get_state(
                self.model.as_ptr(),
                self.data.as_ptr(),
                spec,
                out.as_mut_ptr(),
                out.len(),
            )
        } == 1
        {
            Ok(out)
        } else {
            Err("MuJoCo state read failed".into())
        }
    }
    pub fn set_state(&mut self, state: &[f64]) -> Result<(), String> {
        let spec = unsafe { fly_world_state_integration() };
        if unsafe {
            fly_world_set_state(
                self.model.as_ptr(),
                self.data.as_ptr(),
                spec,
                state.as_ptr(),
                state.len(),
            )
        } == 1
        {
            Ok(())
        } else {
            Err("MuJoCo state write failed".into())
        }
    }
    pub fn clear_applied_forces(&mut self) -> Result<(), String> {
        self.write_num(NumField::QfrcApplied, &vec![0.0; self.dimensions().nv])?;
        self.write_num(
            NumField::XfrcApplied,
            &vec![0.0; self.dimensions().nbody * 6],
        )
    }
    pub fn copy_prefix_from(&mut self, source: &Physics) -> Result<(), String> {
        for field in [
            NumField::Qpos,
            NumField::Qvel,
            NumField::Act,
            NumField::QaccWarmstart,
            NumField::Ctrl,
            NumField::QfrcApplied,
            NumField::XfrcApplied,
            NumField::MocapPos,
            NumField::MocapQuat,
            NumField::UserData,
            NumField::Time,
            NumField::GeomSize,
            NumField::GeomPos,
            NumField::GeomQuat,
            NumField::GeomRgba,
            NumField::GeomFriction,
            NumField::ActuatorForceRange,
            NumField::ActuatorGainPrm,
        ] {
            let values = source.num(field)?;
            if !values.is_empty() {
                self.write_num_at(field, 0, &values)?
            }
        }
        for field in [IntField::GeomContype, IntField::GeomConaffinity] {
            let values = source.int(field)?;
            if !values.is_empty() {
                let mut all = self.int(field)?;
                all[..values.len()].copy_from_slice(&values);
                self.write_int(field, &all)?
            }
        }
        let state = self.state()?;
        let prior = source.state()?;
        if state.len() == prior.len() {
            self.set_state(&prior)?;
        } else if state.len() < prior.len() {
            return Err("appended model integration state shrank".into());
        }
        self.forward()
    }
}
