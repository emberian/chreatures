use std::ffi::{c_char, c_int, c_void, CString};
use std::fs;
use std::path::PathBuf;

#[repr(C)]
#[derive(Default)]
struct Summary {
    header_version: c_int,
    runtime_version: c_int,
    nq: c_int,
    nv: c_int,
    nu: c_int,
    nbody: c_int,
    njnt: c_int,
    ngeom: c_int,
    nsite: c_int,
    nsensor: c_int,
    nsensordata: c_int,
    nkey: c_int,
    ncon: c_int,
    timestep: f64,
    time: f64,
    max_abs_qpos: f64,
}

unsafe extern "C" {
    fn fly_body_load(path: *const c_char, error: *mut c_char, error_size: c_int) -> *mut c_void;
    fn fly_body_make_data(model: *const c_void) -> *mut c_void;
    fn fly_body_delete_model(model: *mut c_void);
    fn fly_body_delete_data(data: *mut c_void);
    fn fly_body_startup(
        model: *mut c_void,
        data: *mut c_void,
        steps: c_int,
        summary: *mut Summary,
    ) -> c_int;
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let model_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/neuromechfly-2.1.0-ca65a510-ypr/model/model.xml"));
    let steps = args
        .next()
        .and_then(|v| v.to_string_lossy().parse::<i32>().ok())
        .unwrap_or(100);
    let receipt_path = args.next().map(PathBuf::from);
    let canonical = model_path
        .canonicalize()
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", model_path.display()));
    let cpath = CString::new(canonical.to_string_lossy().as_bytes()).expect("model path has NUL");
    let mut error = vec![0_i8; 4096];
    let model = unsafe { fly_body_load(cpath.as_ptr(), error.as_mut_ptr(), error.len() as i32) };
    if model.is_null() {
        let message = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        panic!("MuJoCo could not load {}: {message}", canonical.display());
    }
    let data = unsafe { fly_body_make_data(model) };
    if data.is_null() {
        unsafe { fly_body_delete_model(model) };
        panic!("MuJoCo could not allocate MjData");
    }
    let mut summary = Summary::default();
    let ok = unsafe { fly_body_startup(model, data, steps, &mut summary) };
    unsafe {
        fly_body_delete_data(data);
        fly_body_delete_model(model);
    }
    assert_eq!(ok, 1, "non-finite state during neutral body startup");
    assert_eq!(summary.njnt, 127, "free joint + 126 anatomical hinges");
    assert_eq!(summary.nu, 90, "84 position + 6 adhesion actuators");
    assert_eq!(summary.nsensor, 6, "one ground-contact sensor per leg");
    assert_eq!(summary.nkey, 1, "neutral keyframe required");

    let json = format!(
        concat!(
            "{{\n",
            "  \"engine\": \"mujoco-native\",\n",
            "  \"model\": \"{}\",\n",
            "  \"header_version\": {},\n",
            "  \"runtime_version\": {},\n",
            "  \"steps\": {},\n",
            "  \"time_s\": {:.17},\n",
            "  \"timestep_s\": {:.17},\n",
            "  \"nq\": {}, \"nv\": {}, \"nu\": {},\n",
            "  \"nbody\": {}, \"njnt\": {}, \"ngeom\": {}, \"nsite\": {},\n",
            "  \"nsensor\": {}, \"nsensordata\": {}, \"nkey\": {},\n",
            "  \"contacts_after_startup\": {},\n",
            "  \"max_abs_qpos\": {:.17},\n",
            "  \"finite\": true\n",
            "}}\n"
        ),
        escape_json(&canonical.to_string_lossy()),
        summary.header_version,
        summary.runtime_version,
        steps,
        summary.time,
        summary.timestep,
        summary.nq,
        summary.nv,
        summary.nu,
        summary.nbody,
        summary.njnt,
        summary.ngeom,
        summary.nsite,
        summary.nsensor,
        summary.nsensordata,
        summary.nkey,
        summary.ncon,
        summary.max_abs_qpos,
    );
    if let Some(path) = receipt_path {
        fs::write(&path, &json)
            .unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
    }
    print!("{json}");
}
