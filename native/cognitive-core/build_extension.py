#!/usr/bin/env python3
"""Build the cognitive-core extension for this Python interpreter."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
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
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=crate, env=env, text=True,
    ))
    source = (
        Path(metadata["target_directory"])
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
    descriptor, staging_name = tempfile.mkstemp(prefix=".cognitive-core-", dir=a.output_dir)
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        shutil.copy2(source, staging)
        # Preserve code pages mapped by existing lives on the old inode.
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
