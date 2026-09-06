#!/usr/bin/env python3
"""Build the native world kernels for the Python interpreter running this file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="directory in which to install the importable extension (default: repository root)",
    )
    args = parser.parse_args()

    import mujoco

    crate = Path(__file__).resolve().parent
    package = Path(mujoco.__file__).resolve().parent
    header = package / "include" / "mujoco" / "mujoco.h"
    libraries = sorted(package.glob("libmujoco*.dylib")) + sorted(package.glob("libmujoco*.so*"))
    if not header.is_file() or not libraries:
        raise SystemExit(f"MuJoCo wheel lacks native headers or library under {package}")

    env = os.environ.copy()
    env.update(
        PYO3_PYTHON=str(Path(sys.executable).resolve()),
        MUJOCO_INCLUDE_DIR=str(header.parent.parent),
        MUJOCO_LIB_DIR=str(package),
    )
    subprocess.run(["cargo", "build", "--release"], cwd=crate, env=env, check=True)

    source_name = "lib_world_kernels.dylib" if sys.platform == "darwin" else "lib_world_kernels.so"
    source = crate / "target" / "release" / source_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / ("_world_kernels" + sysconfig.get_config_var("EXT_SUFFIX"))
    shutil.copy2(source, destination)

    if sys.platform == "darwin":
        linked = subprocess.check_output(["otool", "-L", destination], text=True)
        dependency = next(
            (line.strip().split(" ", 1)[0] for line in linked.splitlines() if "libmujoco" in line),
            None,
        )
        if dependency is None:
            raise SystemExit("built extension has no MuJoCo dependency")
        subprocess.run(
            ["install_name_tool", "-change", dependency, str(libraries[0]), destination],
            check=True,
        )
    print(destination)


if __name__ == "__main__":
    main()
