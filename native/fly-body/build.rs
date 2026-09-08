use std::env;
use std::path::PathBuf;

fn main() {
    let include = env::var_os("MUJOCO_INCLUDE_DIR")
        .map(PathBuf::from)
        .expect("MUJOCO_INCLUDE_DIR must contain mujoco/mujoco.h");
    let library = env::var_os("MUJOCO_LIB_DIR")
        .map(PathBuf::from)
        .expect("MUJOCO_LIB_DIR must contain the MuJoCo shared library");

    cc::Build::new()
        .file("src/mujoco_shim.c")
        .include(include)
        .flag_if_supported("-O2")
        .compile("chreatures_fly_body_mujoco");

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
        .expect("MUJOCO_LIB_DIR contains no MuJoCo shared library");
    println!("cargo:rustc-link-arg={}", dynamic.display());
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // The macOS Python wheel's dylib has a framework-style install name even
        // though the wheel ships a flat file. Recreate that private layout in the
        // Cargo output rather than mutating the project environment.
        let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR missing"));
        let framework = output.join("mujoco.framework/Versions/A");
        std::fs::create_dir_all(&framework).expect("cannot create private MuJoCo framework path");
        let filename = dynamic.file_name().expect("MuJoCo dylib has no filename");
        std::fs::copy(&dynamic, framework.join(filename))
            .expect("cannot copy MuJoCo dylib into private framework path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", output.display());
    } else if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library.display());
    }
    println!("cargo:rerun-if-changed=src/mujoco_shim.c");
    println!("cargo:rerun-if-env-changed=MUJOCO_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MUJOCO_LIB_DIR");
}
