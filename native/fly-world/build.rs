// SPDX-License-Identifier: AGPL-3.0-or-later
use std::env;
use std::path::{Path, PathBuf};

fn find_mujoco_library(directory: &Path) -> PathBuf {
    std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("MUJOCO_LIB_DIR cannot be read: {error}"))
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
        .unwrap_or_else(|| {
            panic!(
                "MUJOCO_LIB_DIR contains no MuJoCo shared library: {}",
                directory.display()
            )
        })
}

fn main() {
    let include = env::var_os("MUJOCO_INCLUDE_DIR")
        .map(PathBuf::from)
        .expect("MUJOCO_INCLUDE_DIR must point to the directory containing mujoco/mujoco.h");
    let library_dir = env::var_os("MUJOCO_LIB_DIR")
        .map(PathBuf::from)
        .expect("MUJOCO_LIB_DIR must point to the directory containing libmujoco");
    let library = find_mujoco_library(&library_dir);

    cc::Build::new()
        .file("src/mujoco_shim.c")
        .include(include)
        .flag_if_supported("-std=c11")
        .flag_if_supported("-O3")
        .flag_if_supported("-ffp-contract=off")
        .warnings(true)
        .compile("chreatures_fly_world_mujoco");

    // The wheel does not ship an unversioned libmujoco symlink, so link the
    // resolved shared library directly. Its macOS install name expects the
    // framework-style path recreated below.
    println!("cargo:rustc-link-arg={}", library.display());
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR missing"));
            let framework = output.join("mujoco.framework/Versions/A");
            std::fs::create_dir_all(&framework)
                .expect("cannot create private MuJoCo framework path");
            let filename = library.file_name().expect("MuJoCo dylib has no filename");
            std::fs::copy(&library, framework.join(filename))
                .expect("cannot copy MuJoCo dylib into private framework path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", output.display());
        }
        Ok("linux") => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_dir.display());
        }
        _ => {}
    }

    println!("cargo:rerun-if-changed=src/mujoco_shim.c");
    println!("cargo:rerun-if-env-changed=MUJOCO_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MUJOCO_LIB_DIR");
}
