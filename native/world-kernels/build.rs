use std::env;
use std::path::PathBuf;

fn main() {
    let include = env::var_os("MUJOCO_INCLUDE_DIR")
        .map(PathBuf::from)
        .expect("MUJOCO_INCLUDE_DIR must point to the directory containing mujoco/mujoco.h");
    let library = env::var_os("MUJOCO_LIB_DIR")
        .map(PathBuf::from)
        .expect("MUJOCO_LIB_DIR must point to the directory containing libmujoco");
    cc::Build::new()
        .files([
            "src/contact_shim.c",
            "src/actuation_shim.c",
            "src/environment_shim.c",
            "src/sensorium_shim.c",
        ])
        .include(include)
        .flag_if_supported("-O3")
        .flag_if_supported("-ffp-contract=off")
        .compile("chreatures_contact_shim");
    let dynamic = std::fs::read_dir(&library)
        .expect("MUJOCO_LIB_DIR cannot be read")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("libmujoco")
                        && (name.contains(".so") || name.contains(".dylib"))
                })
        })
        .expect("MUJOCO_LIB_DIR contains no libmujoco shared library");
    println!("cargo:rustc-link-arg={}", dynamic.display());
    let target_os = env::var("CARGO_CFG_TARGET_OS");
    if target_os.as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    } else if target_os.as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library.display());
    }
    println!("cargo:rerun-if-changed=src/contact_shim.c");
    println!("cargo:rerun-if-changed=src/actuation_shim.c");
    println!("cargo:rerun-if-changed=src/environment_shim.c");
    println!("cargo:rerun-if-changed=src/sensorium_shim.c");
    println!("cargo:rerun-if-env-changed=MUJOCO_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MUJOCO_LIB_DIR");
}
