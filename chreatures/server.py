"""Local interactive home. Start with `uv run chreatures`."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .physics import MODEL_DT

ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("chreatures")


def create_app(
    checkpoint: Path | None = None,
    seed=7,
    autostep=True,
    brain_url="http://127.0.0.1:18765",
    body_mode="articulated",
    ecology="diffusion",
    resources=None,
    biosphere=None,
    acoustics=None,
    motor_genome=None,
    personal_memory=False,
    habitat_spec=None,
    perception_url=None,
    physics_backend=None,
    personal_plasticity=False,
    predictive_model=None,
):
    checkpoint = checkpoint or ROOT / "runs/hollow-garden.json"
    authority = {"habitat": None, "error": None, "alive": True}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        try:
            from .runtime3d import Habitat3D

            restoring = checkpoint.exists()
            authority["habitat"] = (
                Habitat3D.load(checkpoint, brain_url)
                if restoring
                else Habitat3D(
                    seed,
                    brain_url,
                    body_mode=body_mode,
                    ecology=ecology,
                    resources=resources,
                    biosphere=biosphere,
                    acoustics=acoustics,
                    motor_genome=motor_genome,
                    personal_memory=personal_memory,
                    perception_url=perception_url,
                    physics_backend=physics_backend,
                    personal_plasticity=personal_plasticity,
                    predictive_model=predictive_model,
                    spec=json.loads(Path(habitat_spec).read_text())
                    if habitat_spec is not None
                    else None,
                )
            )
            if not restoring:
                # Persist the first coherent state before the first neural tick.
                # A failed startup must never advance an unsaved new world.
                authority["habitat"].save(checkpoint)
        except Exception as exc:
            authority["error"] = str(exc)
            if authority["habitat"] is not None:
                authority["habitat"].paused = True
            log.exception("Could not start habitat")

        async def advance():
            last_save = time.monotonic()
            while authority["alive"]:
                started = time.monotonic()
                async with lock:
                    habitat = authority["habitat"]
                    if habitat and not habitat.paused and autostep:
                        try:
                            habitat.step(habitat.speed)
                        except Exception as exc:
                            habitat.paused = True
                            authority["error"] = str(exc)
                            log.exception("Paused after simulation error")
                    if (
                        habitat and habitat.pending_step is None
                        and not habitat.neural.uncertain
                        and time.monotonic() - last_save > 30
                    ):
                        try:
                            habitat.save(checkpoint)
                        except Exception as exc:
                            habitat.paused = True
                            authority["error"] = str(exc)
                            log.exception("Could not checkpoint habitat")
                        last_save = time.monotonic()
                await asyncio.sleep(max(0.001, MODEL_DT - (time.monotonic() - started)))

        task = asyncio.create_task(advance())
        yield
        authority["alive"] = False
        await task
        if authority["habitat"]:
            try:
                authority["habitat"].save(checkpoint)
            except Exception:
                log.exception(
                    "Preserving the previous checkpoint after an incomplete tick"
                )
            vision = getattr(authority["habitat"], "vision", None)
            if vision is not None:
                vision.close()

    app = FastAPI(title="Chreatures", lifespan=lifespan)
    from .observatory import router as observatory_router

    app.include_router(observatory_router)
    app.state.authority = authority
    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

    def get_habitat():
        if authority["habitat"] is None:
            raise HTTPException(503, authority["error"] or "Habitat is starting")
        return authority["habitat"]

    def check_origin(headers):
        origin = headers.get("origin")
        host = headers.get("host")
        if origin and origin not in (f"http://{host}", f"https://{host}"):
            raise HTTPException(403, "Commands must come from this habitat")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/garden.html")

    @app.get("/observatory")
    def observatory():
        return FileResponse(ROOT / "web/observatory.html")

    @app.get("/api/state")
    async def state():
        async with lock:
            value = get_habitat().view()
            value["error"] = authority["error"]
            return value

    @app.get("/api/connectome")
    async def connectome():
        return get_habitat().neural.metadata["brain"]

    def visitor_habitat():
        habitat = get_habitat()
        return habitat

    async def visitor_payload(request):
        check_origin(request.headers)
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(415, "Use application/json")
        if len(await request.body()) > 32768:
            raise HTTPException(413, "Performance is too large")
        try:
            return await request.json()
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(400, "Invalid JSON") from error

    @app.get("/api/visitor")
    async def visitor_state():
        async with lock:
            habitat = visitor_habitat()
            return habitat.visitor.view(habitat.tick, habitat.paused)

    @app.post("/api/visitor/motifs")
    async def visitor_motif(request: Request):
        value = await visitor_payload(request)
        async with lock:
            habitat = visitor_habitat()
            try:
                result = habitat.visitor.add_motif(habitat.world, value)
                habitat.note(
                    "visitor-motif",
                    "A visitor saved a sensory performance.",
                    motif=result,
                )
                return result
            except (ValueError, KeyError, TypeError) as error:
                raise HTTPException(400, str(error)) from error

    @app.post("/api/visitor/schedules")
    async def visitor_schedule(request: Request):
        value = await visitor_payload(request)
        async with lock:
            habitat = visitor_habitat()
            try:
                result = habitat.visitor.schedule(habitat.world, habitat.tick, value)
                habitat.note(
                    "visitor-performance",
                    "A visitor scheduled a sensory performance.",
                    performance=result,
                )
                return result
            except (ValueError, KeyError, TypeError) as error:
                raise HTTPException(400, str(error)) from error

    @app.delete("/api/visitor/schedules/{identifier}")
    async def visitor_cancel(identifier: str, request: Request):
        check_origin(request.headers)
        async with lock:
            habitat = visitor_habitat()
            try:
                return habitat.visitor.cancel(habitat.world, identifier)
            except StopIteration as error:
                raise HTTPException(404, "Unknown performance") from error

    @app.get("/api/checkpoint")
    async def export_checkpoint():
        async with lock:
            get_habitat().save(checkpoint)
        return FileResponse(
            checkpoint,
            filename="chreatures-checkpoint.json",
            media_type="application/json",
        )

    @app.get("/api/vision/{resident}/frame")
    async def visual_frame(resident: str):
        async with lock:
            vision = getattr(get_habitat(), "vision", None)
            if vision is None or resident not in vision.frames:
                raise HTTPException(404, "No delivered native view for this resident")
            return Response(
                base64.b64decode(vision.frames[resident]),
                media_type="image/png",
                headers={
                    "ETag": '"' + vision.latest[resident]["frame_sha256"] + '"',
                    "Cache-Control": "no-store",
                },
            )

    @app.post("/api/command")
    async def command(request: Request):
        check_origin(request.headers)
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(415, "Use application/json")
        if len(await request.body()) > 16_384:
            raise HTTPException(413, "Command is too large")
        try:
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError("Command must be an object")
            async with lock:
                habitat = get_habitat()
                if value.get("op") == "save":
                    return {"ok": True, "sha256": habitat.save(checkpoint)}
                return {"ok": True, "result": habitat.command(value)}
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.websocket("/ws")
    async def stream(websocket: WebSocket):
        try:
            check_origin(websocket.headers)
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                async with lock:
                    if authority["habitat"]:
                        value = get_habitat().view()
                        value["error"] = authority["error"]
                    else:
                        value = {"error": authority["error"] or "Starting"}
                await websocket.send_json(value)
                await asyncio.sleep(0.10)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Open an embodied connectome nursery")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--brain-url", default="http://127.0.0.1:18765")
    parser.add_argument(
        "--body",
        choices=("crawler", "articulated"),
        default="articulated",
        help="Body for new worlds; saved worlds retain their original body",
    )
    parser.add_argument(
        "--ecology",
        choices=("analytic", "diffusion"),
        default="diffusion",
        help="Chemical ecology for new worlds; saved worlds retain their original model",
    )
    parser.add_argument("--checkpoint", type=Path)
    environment_group = parser.add_mutually_exclusive_group()
    environment_group.add_argument(
        "--resources",
        type=Path,
        help="Finite resource growth configuration for new 3D worlds; saved worlds preserve their ecology",
    )
    environment_group.add_argument(
        "--biosphere",
        type=Path,
        help="Native metabolic and developmental configuration for new 3D research worlds",
    )
    parser.add_argument(
        "--acoustics",
        type=Path,
        help="Physical acoustic transducers for new 3D worlds; saved worlds preserve their mechanisms",
    )
    parser.add_argument(
        "--motor-genome",
        type=Path,
        help="Inherited NumPy motor artifact for new worlds; existing lives keep their controllers",
    )
    parser.add_argument(
        "--personal-memory",
        action="store_true",
        help="Learn private action consequences around the inherited motor for new worlds",
    )
    parser.add_argument(
        "--personal-plasticity",
        action="store_true",
        help="Learn a private motor adapter from actual bodily consequences in new worlds",
    )
    parser.add_argument(
        "--predictive-model",
        type=Path,
        help=(
            "Native physical-unit predictive-state artifact for opt-in private "
            "foresight in new personal-memory worlds"
        ),
    )
    parser.add_argument(
        "--habitat",
        type=Path,
        help="Physical habitat specification for new 3D worlds; saved worlds contain their own specification",
    )
    parser.add_argument(
        "--perception-url",
        help="Optional native visual feature service for new personal-memory residents",
    )
    parser.add_argument(
        "--physics-backend",
        choices=("reference", "vectorized"),
        help="Execution path for new worlds; articulated worlds default to vectorized",
    )
    args = parser.parse_args()
    uvicorn.run(
        create_app(
            args.checkpoint,
            args.seed,
            brain_url=args.brain_url,
            body_mode=args.body,
            ecology=args.ecology,
            resources=args.resources,
            biosphere=args.biosphere,
            acoustics=args.acoustics,
            motor_genome=args.motor_genome,
            personal_memory=args.personal_memory,
            habitat_spec=args.habitat,
            perception_url=args.perception_url,
            physics_backend=args.physics_backend,
            personal_plasticity=args.personal_plasticity,
            predictive_model=args.predictive_model,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
