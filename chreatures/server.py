"""Local interactive home. Start with `uv run chreatures`."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .runtime import Habitat
from .brain import ROOT
from .world import MODEL_DT

log = logging.getLogger("chreatures")


def create_app(checkpoint: Path | None = None, seed=7, autostep=True, dimension=2, brain_url="http://127.0.0.1:18765",
               body_mode="articulated", ecology="diffusion", resources=None, acoustics=None, motor_genome=None):
    checkpoint = checkpoint or ROOT / ("runs/hollow-garden.json" if dimension == 3 else "runs/residents.json")
    authority = {"habitat": None, "error": None, "alive": True}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        try:
            if dimension == 3:
                from .runtime3d import Habitat3D
                authority["habitat"] = Habitat3D.load(checkpoint, brain_url) if checkpoint.exists() else Habitat3D(
                    seed, brain_url, body_mode=body_mode, ecology=ecology, resources=resources,
                    acoustics=acoustics, motor_genome=motor_genome)
            else:
                authority["habitat"] = Habitat.load(checkpoint) if checkpoint.exists() else Habitat(seed)
        except Exception as exc:
            authority["error"] = str(exc)
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
                    if habitat and time.monotonic() - last_save > 30:
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
                log.exception("Preserving the previous checkpoint after an incomplete tick")

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
        return FileResponse(ROOT / ("web/garden.html" if dimension == 3 else "web/index.html"))

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
        if dimension == 3:
            return get_habitat().neural.metadata["brain"]
        return next(iter(get_habitat().brains.values())).graph.summary()

    @app.get("/api/checkpoint")
    async def export_checkpoint():
        async with lock:
            get_habitat().save(checkpoint)
        return FileResponse(checkpoint, filename="chreatures-checkpoint.json", media_type="application/json")

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


def main(default_dimension=2):
    import uvicorn
    parser = argparse.ArgumentParser(description="Open an embodied connectome nursery")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=default_dimension)
    parser.add_argument("--brain-url", default="http://127.0.0.1:18765")
    parser.add_argument("--body", choices=("crawler", "articulated"), default="articulated",
                        help="Body for new worlds; saved worlds retain their original body")
    parser.add_argument("--ecology", choices=("analytic", "diffusion"), default="diffusion",
                        help="Chemical ecology for new worlds; saved worlds retain their original model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resources", type=Path,
                        help="Finite resource growth configuration for new 3D worlds; saved worlds preserve their ecology")
    parser.add_argument("--acoustics", type=Path,
                        help="Physical acoustic transducers for new 3D worlds; saved worlds preserve their mechanisms")
    parser.add_argument("--motor-genome", type=Path,
                        help="Inherited NumPy motor artifact for new worlds; existing lives keep their controllers")
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint, args.seed, dimension=args.dimension, brain_url=args.brain_url,
                          body_mode=args.body, ecology=args.ecology, resources=args.resources,
                          acoustics=args.acoustics, motor_genome=args.motor_genome), host=args.host, port=args.port)


def main3d():
    main(default_dimension=3)


if __name__ == "__main__":
    main()
