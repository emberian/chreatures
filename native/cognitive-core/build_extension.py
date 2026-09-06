#!/usr/bin/env python3
"""Build the cognitive-core extension for this Python interpreter."""

import argparse, os, shutil, subprocess, sys, sysconfig
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[2]
    )
    a = p.parse_args()
    crate = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(Path(sys.executable).resolve())
    subprocess.run(["cargo", "build", "--release"], cwd=crate, env=env, check=True)
    source = (
        crate
        / "target"
        / "release"
        / (
            "lib_cognitive_core.dylib"
            if sys.platform == "darwin"
            else "lib_cognitive_core.so"
        )
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    destination = a.output_dir / (
        "_cognitive_core" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    shutil.copy2(source, destination)
    print(destination)


if __name__ == "__main__":
    main()
