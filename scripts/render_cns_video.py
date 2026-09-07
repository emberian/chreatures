#!/usr/bin/env python3
"""Compose verified stimulus, MaleCNS rate-state, and body frames into an MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FORMAT = "chreatures-bad-apple-response-bundle-v1"
RECEIPT_FORMAT = "chreatures-bad-apple-composition-receipt-v1"
NEURONS = 165_122
STREAMS = {
    "time_seconds": (np.dtype("float64"), 1),
    "stimulus_frames": (np.dtype("uint8"), None),
    "neural_rates": (np.dtype("float32"), 2),
    "soma_positions": (np.dtype("float32"), 2),
    "soma_valid": (np.dtype("bool"), 1),
    "body_frames": (np.dtype("uint8"), 4),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def public_neural_display(projection: Mapping[str, Any]) -> str:
    if projection["display_mode"] == "signed delta rate":
        return "maximum positive and negative rate delta per soma display pixel from first recorded frame; fixed symmetric global scale; not spikes or calcium"
    return "maximum clipped absolute rate per soma display pixel; fixed global scale; not spikes or calcium"


def relative_file(root: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"{label} file must be a relative path")
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{label} file escapes the bundle")
    if path.suffix != ".npy" or not path.is_file():
        raise ValueError(f"{label} must name an existing .npy file")
    return path


def load_bundle(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Path]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"bundle manifest must use {FORMAT}")
    if manifest.get("status") != "recorded synchronized source, rate-model, and physical frames":
        raise ValueError("bundle status does not attest recorded synchronized inputs")
    frame_count = manifest.get("frame_count")
    fps = manifest.get("playback_fps")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError("frame_count must be an integer of at least two")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 1 <= fps <= 120:
        raise ValueError("playback_fps must lie in [1,120]")
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping) or set(streams) != set(STREAMS):
        raise ValueError(f"streams must be exactly {sorted(STREAMS)}")
    arrays: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for name, (dtype, dimensions) in STREAMS.items():
        spec = streams[name]
        if not isinstance(spec, Mapping) or set(spec) != {"file", "sha256", "shape", "dtype"}:
            raise ValueError(f"{name} stream specification differs")
        path = relative_file(root, spec["file"], name)
        if sha256(path) != strict_sha(spec["sha256"], f"{name} stream"):
            raise ValueError(f"{name} stream SHA-256 differs")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != dtype or list(array.shape) != spec["shape"]:
            raise ValueError(f"{name} stream dtype or shape differs")
        if dimensions is not None and array.ndim != dimensions:
            raise ValueError(f"{name} stream rank differs")
        arrays[name] = array
        paths[name] = path
    time_values = arrays["time_seconds"]
    stimulus = arrays["stimulus_frames"]
    rates = arrays["neural_rates"]
    soma = arrays["soma_positions"]
    valid = arrays["soma_valid"]
    bodies = arrays["body_frames"]
    if time_values.shape != (frame_count,) or not np.isfinite(time_values).all():
        raise ValueError("time_seconds must provide one finite value per frame")
    if np.any(np.diff(time_values) <= 0):
        raise ValueError("time_seconds must increase strictly")
    if stimulus.ndim not in (3, 4) or stimulus.shape[0] != frame_count:
        raise ValueError("stimulus_frames must be [T,H,W] or [T,H,W,C]")
    if stimulus.ndim == 4 and stimulus.shape[3] not in (1, 3, 4):
        raise ValueError("stimulus_frames channel count must be one, three, or four")
    if rates.shape != (frame_count, NEURONS):
        raise ValueError(f"neural_rates must be [T,{NEURONS}]")
    if soma.shape != (NEURONS, 3) or valid.shape != (NEURONS,):
        raise ValueError("soma atlas must align exactly with all 165,122 graph rows")
    if bodies.shape[0] != frame_count or bodies.shape[3] not in (3, 4):
        raise ValueError("body_frames must be [T,H,W,3|4]")
    if not valid.any() or not np.isfinite(soma[valid]).all():
        raise ValueError("soma atlas has no finite valid positions")
    projection = manifest.get("neural_projection")
    if not isinstance(projection, Mapping) or set(projection) != {
        "axes", "bounds_source_units", "display_mode", "delta_reference",
        "rate_floor", "rate_ceiling", "rate_unit",
    }:
        raise ValueError("neural_projection contract differs")
    axes = projection["axes"]
    bounds = projection["bounds_source_units"]
    if (
        not isinstance(axes, list) or len(axes) != 2
        or any(isinstance(axis, bool) or not isinstance(axis, int) or axis not in range(3) for axis in axes)
        or axes[0] == axes[1]
        or not isinstance(bounds, list) or len(bounds) != 2
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in bounds)
    ):
        raise ValueError("neural projection axes or bounds differ")
    for lower, upper in bounds:
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (lower, upper)) or lower >= upper:
            raise ValueError("neural projection bounds must be finite increasing pairs")
    floor = projection["rate_floor"]
    ceiling = projection["rate_ceiling"]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (floor, ceiling)) or floor < 0 or floor >= ceiling:
        raise ValueError("rate display bounds must be finite, nonnegative, and increasing")
    if not isinstance(projection["rate_unit"], str) or not projection["rate_unit"].strip():
        raise ValueError("rate display unit is absent")
    display_mode = projection["display_mode"]
    delta_reference = projection["delta_reference"]
    if display_mode == "absolute rate":
        if delta_reference != "none":
            raise ValueError("absolute-rate display must use delta_reference=none")
    elif display_mode == "signed delta rate":
        if delta_reference != "first recorded frame" or floor != 0:
            raise ValueError("signed-delta display requires first-frame reference and rate_floor=0")
    else:
        raise ValueError("neural display mode differs")
    sources = manifest.get("sources")
    required_sources = {
        "stimulus_sha256", "stimulus_receipt_sha256", "malecns_graph_sha256",
        "rate_capture_sha256", "body_recording_sha256", "world_source_revision",
    }
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise ValueError(f"sources must be exactly {sorted(required_sources)}")
    for key in required_sources - {"world_source_revision"}:
        strict_sha(sources[key], key)
    if not isinstance(sources["world_source_revision"], str) or not sources["world_source_revision"]:
        raise ValueError("world source revision is absent")
    neural_model = manifest.get("neural_model")
    if not isinstance(neural_model, Mapping) or set(neural_model) != {
        "status", "artifact_sha256", "training_receipt_sha256",
    }:
        raise ValueError("neural_model contract differs")
    status = neural_model["status"]
    if status not in {
        "initialized, untrained rate model",
        "trained rate model with cited training receipt",
    }:
        raise ValueError("neural model status differs")
    strict_sha(neural_model["artifact_sha256"], "neural model artifact")
    training = neural_model["training_receipt_sha256"]
    if status.startswith("initialized"):
        if training is not None:
            raise ValueError("untrained neural model cannot cite a training receipt")
    else:
        strict_sha(training, "neural model training receipt")
    controller = manifest.get("controller")
    if not isinstance(controller, Mapping) or set(controller) != {
        "status", "artifact_sha256", "training_receipt_sha256",
    }:
        raise ValueError("controller contract differs")
    controller_status = controller["status"]
    if controller_status not in {
        "initialized controller; motor and memory core untrained",
        "briefly trained controller with cited receipt; no competence claim",
        "trained controller with cited receipt; no competence claim",
    }:
        raise ValueError("controller status differs")
    strict_sha(controller["artifact_sha256"], "controller artifact")
    controller_training = controller["training_receipt_sha256"]
    if controller_status.startswith("initialized"):
        if controller_training is not None:
            raise ValueError("initialized controller cannot cite a training receipt")
    else:
        strict_sha(controller_training, "controller training receipt")
    return manifest, arrays, paths


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_frame(value: np.ndarray, size: tuple[int, int]) -> Image.Image:
    if value.ndim == 2:
        image = Image.fromarray(value, "L").convert("RGB")
    elif value.shape[2] == 1:
        image = Image.fromarray(value[..., 0], "L").convert("RGB")
    else:
        image = Image.fromarray(value[..., :3], "RGB")
    return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=(9, 16, 14))


class NeuralRaster:
    def __init__(
        self, positions: np.ndarray, valid: np.ndarray, projection: Mapping[str, Any],
        size: tuple[int, int], baseline_rates: np.ndarray,
    ):
        self.width, self.height = size
        axes = projection["axes"]
        bounds = projection["bounds_source_units"]
        rows = np.flatnonzero(valid)
        x = np.asarray(positions[rows, axes[0]], dtype=np.float64)
        y = np.asarray(positions[rows, axes[1]], dtype=np.float64)
        inside = (x >= bounds[0][0]) & (x <= bounds[0][1]) & (y >= bounds[1][0]) & (y <= bounds[1][1])
        self.rows = rows[inside]
        if not len(self.rows):
            raise ValueError("projection bounds contain no valid soma positions")
        x = x[inside]; y = y[inside]
        px = np.rint((x - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * (self.width - 1)).astype(np.int64)
        py = np.rint((1 - (y - bounds[1][0]) / (bounds[1][1] - bounds[1][0])) * (self.height - 1)).astype(np.int64)
        self.pixel = py * self.width + px
        density = np.zeros(self.width * self.height, dtype=np.float32)
        np.add.at(density, self.pixel, 1)
        density = np.log1p(density)
        self.density = density / max(float(density.max()), 1.0)
        self.floor = float(projection["rate_floor"])
        self.ceiling = float(projection["rate_ceiling"])
        self.display_mode = str(projection["display_mode"])
        self.baseline = np.asarray(baseline_rates[self.rows], dtype=np.float32).copy()

    def render(self, rates: np.ndarray) -> Image.Image:
        values = np.asarray(rates[self.rows], dtype=np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("neural rate sample contains invalid values")
        base = self.density.reshape(self.height, self.width)
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        if self.display_mode == "absolute rate":
            level = np.clip((values - self.floor) / (self.ceiling - self.floor), 0, 1)
            activity = np.zeros(self.width * self.height, dtype=np.float32)
            np.maximum.at(activity, self.pixel, level)
            hot = activity.reshape(self.height, self.width)
            rgb[..., 0] = np.rint(12 + 75 * base + 168 * hot).clip(0, 255)
            rgb[..., 1] = np.rint(23 + 105 * base + 94 * hot).clip(0, 255)
            rgb[..., 2] = np.rint(20 + 84 * base + 38 * hot).clip(0, 255)
        else:
            delta = (values - self.baseline) / self.ceiling
            positive = np.zeros(self.width * self.height, dtype=np.float32)
            negative = np.zeros(self.width * self.height, dtype=np.float32)
            np.maximum.at(positive, self.pixel, np.clip(delta, 0, 1))
            np.maximum.at(negative, self.pixel, np.clip(-delta, 0, 1))
            pos = positive.reshape(self.height, self.width)
            neg = negative.reshape(self.height, self.width)
            rgb[..., 0] = np.rint(10 + 38 * base + 18 * pos + 195 * neg).clip(0, 255)
            rgb[..., 1] = np.rint(16 + 42 * base + 168 * pos + 92 * neg).clip(0, 255)
            rgb[..., 2] = np.rint(17 + 45 * base + 205 * pos + 22 * neg).clip(0, 255)
        return Image.fromarray(rgb, "RGB")


def draw_panel(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int], label: str, detail: str) -> None:
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill=(12, 26, 22), outline=(91, 114, 95), width=2)
    inner = (left + 10, top + 58, right - 10, bottom - 10)
    canvas.paste(ImageOps.fit(image, (inner[2] - inner[0], inner[3] - inner[1]), method=Image.Resampling.LANCZOS), inner[:2])
    draw.text((left + 14, top + 10), label, font=font(24), fill=(240, 233, 211))
    draw.text((right - 14, top + 38), detail, font=font(12), fill=(152, 174, 157), anchor="ra")


def compose_frame(
    index: int, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    neural: NeuralRaster, size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    canvas = Image.new("RGB", size, (7, 17, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 25), "BAD APPLE / ONE RECORDED SENSORIMOTOR PASS", font=font(18), fill=(202, 173, 106))
    draw.text((36, 58), "source frame → full MaleCNS rate state → recorded physical response", font=font(30), fill=(244, 239, 222))
    draw.text((width - 36, 34), f"frame {index + 1:,} / {manifest['frame_count']:,}", font=font(18), fill=(181, 194, 181), anchor="ra")
    draw.text((width - 36, 65), f"t = {float(arrays['time_seconds'][index]):.3f} s", font=font(18), fill=(181, 194, 181), anchor="ra")
    margin, gap, top, bottom = 36, 18, 112, height - 116
    available = width - 2 * margin - 2 * gap
    side = int(available * 0.27)
    center = available - 2 * side
    boxes = (
        (margin, top, margin + side, bottom),
        (margin + side + gap, top, margin + side + gap + center, bottom),
        (margin + side + gap + center + gap, top, width - margin, bottom),
    )
    stimulus = fit_frame(arrays["stimulus_frames"][index], (side - 20, bottom - top - 68))
    brain = neural.render(arrays["neural_rates"][index])
    body = fit_frame(arrays["body_frames"][index], (side - 20, bottom - top - 68))
    projection = manifest["neural_projection"]
    draw_panel(canvas, stimulus, boxes[0], "01 / SCREEN STIMULUS", "recorded source frame")
    if projection["display_mode"] == "signed delta rate":
        neural_detail = f"Δ rate from first sample · fixed ±{projection['rate_ceiling']:g} {projection['rate_unit']}"
    else:
        neural_detail = f"absolute rate · fixed {projection['rate_floor']:g}–{projection['rate_ceiling']:g} {projection['rate_unit']}"
    draw_panel(canvas, brain, boxes[1], "02 / FULL MALECNS", neural_detail)
    draw_panel(canvas, body, boxes[2], "03 / BODY", "recorded physical frame")
    sources = manifest["sources"]
    model_status = str(manifest["neural_model"]["status"]).upper()
    controller_status = str(manifest["controller"]["status"]).upper()
    draw.text((36, height - 92), f"NEURAL: {model_status} — RATE STATE, NOT SPIKES OR CALCIUM", font=font(15), fill=(219, 135, 88))
    draw.text((36, height - 64), f"CONTROLLER: {controller_status}", font=font(14), fill=(224, 192, 124))
    draw.text((36, height - 36), f"MaleCNS {sources['malecns_graph_sha256'][:12]} · rates {sources['rate_capture_sha256'][:12]} · body {sources['body_recording_sha256'][:12]} · world {sources['world_source_revision'][:12]}", font=font(13), fill=(145, 165, 150))
    return np.asarray(canvas, dtype=np.uint8)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Directory containing manifest.json and hashed .npy streams")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--receipt", type=Path, help="Composition receipt path; defaults beside output")
    parser.add_argument("--public-manifest", type=Path, help="Write the completed public playback manifest after composition")
    parser.add_argument("--video-url", help="Same-site MP4 URL recorded in --public-manifest")
    parser.add_argument("--receipt-url", help="Same-site receipt URL recorded in --public-manifest")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.width < 960 or args.height < 540 or args.width % 2 or args.height % 2:
        raise SystemExit("video dimensions must be even and at least 960×540")
    if not 0 <= args.crf <= 51:
        raise SystemExit("CRF must lie in [0,51]")
    if bool(args.public_manifest) != bool(args.video_url and args.receipt_url):
        raise SystemExit("--public-manifest requires both --video-url and --receipt-url")
    root = args.bundle.resolve()
    manifest, arrays, paths = load_bundle(root)
    neural_width = int((args.width - 72 - 36) * 0.46) - 20
    neural_height = args.height - 112 - 116 - 68
    raster = NeuralRaster(
        arrays["soma_positions"], arrays["soma_valid"],
        manifest["neural_projection"], (neural_width, neural_height),
        arrays["neural_rates"][0],
    )
    # Validate every dense rate row before declaring the bundle renderable.
    for start in range(0, manifest["frame_count"], 64):
        block = np.asarray(arrays["neural_rates"][start:start + 64])
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError("neural_rates contains nonfinite or negative rate-model state")
    if args.validate_only:
        print(json.dumps({
            "format": manifest["format"], "frames": manifest["frame_count"],
            "neurons": NEURONS, "valid_soma": int(arrays["soma_valid"].sum()),
            "bundle_content_sha256": digest(manifest), "validated_only": True,
        }, sort_keys=True))
        return 0
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise RuntimeError(f"FFmpeg executable not found: {args.ffmpeg}")
    receipt_path = args.receipt or args.output.with_suffix(".receipt.json")
    final_paths = [args.output, receipt_path]
    if args.public_manifest is not None:
        final_paths.append(args.public_manifest)
    for path in final_paths:
        if path.exists():
            raise FileExistsError(f"composition refuses existing output: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if args.public_manifest is not None:
        args.public_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
        "-r", str(manifest["playback_fps"]), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(manifest["frame_count"]):
            process.stdin.write(compose_frame(
                index, manifest, arrays, raster, (args.width, args.height),
            ).tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill(); process.wait()
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, args.output)
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": "composed from hash-verified recorded streams; no generated neural or body frames",
        "bundle_format": manifest["format"],
        "bundle_content_sha256": digest(manifest),
        "stream_files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "frames": manifest["frame_count"],
        "playback_fps": manifest["playback_fps"],
        "source_time_seconds": [float(arrays["time_seconds"][0]), float(arrays["time_seconds"][-1])],
        "neural_values_per_frame": NEURONS,
        "valid_soma_positions": int(arrays["soma_valid"].sum()),
        "neural_raster_semantics": (
            "maximum positive and negative rate delta among actual soma rows landing in each display pixel, "
            "relative to the first recorded frame on a fixed symmetric scale"
            if manifest["neural_projection"]["display_mode"] == "signed delta rate"
            else "maximum clipped absolute rate-model value among actual soma rows landing in each display pixel on a fixed global scale"
        ),
        "neural_projection": dict(manifest["neural_projection"]),
        "video": {"file": args.output.name, "sha256": sha256(args.output), "bytes": args.output.stat().st_size, "width": args.width, "height": args.height},
        "sources": dict(manifest["sources"]),
        "neural_model": dict(manifest["neural_model"]),
        "controller": dict(manifest["controller"]),
    }
    receipt["content_sha256"] = digest(receipt)
    receipt_path.write_bytes(json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
    public_path = None
    if args.public_manifest is not None:
        public = {
            "format": "chreatures-bad-apple-public-playback-v1",
            "status": "recorded synchronized stimulus, MaleCNS rate-model state, and physical response",
            "frames": manifest["frame_count"],
            "playback_fps": manifest["playback_fps"],
            "source_time_seconds": receipt["source_time_seconds"],
            "neurons": NEURONS,
            "valid_soma_positions": receipt["valid_soma_positions"],
            "model_status": manifest["neural_model"]["status"],
            "controller_status": manifest["controller"]["status"],
            "neural_display": public_neural_display(manifest["neural_projection"]),
            "sources": dict(manifest["sources"]),
            "video": {"url": args.video_url, **receipt["video"]},
            "receipt": {"url": args.receipt_url, "sha256": sha256(receipt_path)},
        }
        args.public_manifest.write_bytes(json.dumps(public, indent=2, sort_keys=True).encode() + b"\n")
        public_path = str(args.public_manifest)
    print(json.dumps({"output": str(args.output), "receipt": str(receipt_path), "public_manifest": public_path, **receipt["video"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
