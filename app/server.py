from __future__ import annotations

import asyncio
import json
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.orchestrator import FederatedOrchestrator, SessionConfig

STATIC_DIR = Path(__file__).resolve().parent / "static"

orchestrator = FederatedOrchestrator()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    orchestrator.stop_session()


app = FastAPI(
    title="FedIris gRPC — Aprendizagem Federada",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class StartSessionRequest(BaseModel):
    num_clients: int = Field(default=5, ge=2, le=10)
    total_rounds: int = Field(default=10, ge=1, le=25)
    local_epochs: int = Field(default=1, ge=1, le=10)
    poll_interval: float = Field(default=0.15, ge=0.05, le=2.0)
    visual_delay: float = Field(default=0.45, ge=0.0, le=3.0)
    step_mode: bool = False
    port: int = Field(default=50051, ge=1024, le=65535)


class BaselineRequest(BaseModel):
    num_clients: int = Field(default=5, ge=2, le=10)
    total_rounds: int = Field(default=10, ge=1, le=25)
    local_epochs: int = Field(default=1, ge=1, le=10)


class InjectRpcRequest(BaseModel):
    kind: Literal["stale_round", "duplicate", "after_finish"]
    cid: int = Field(default=0, ge=0, le=99)


class PredictRequest(BaseModel):
    petal_length: float = Field(ge=0.0, le=10.0)
    petal_width: float = Field(ge=0.0, le=5.0)
    sepal_length: float = Field(ge=0.0, le=10.0)
    sepal_width: float = Field(ge=0.0, le=6.0)


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/state")
def get_state():
    return orchestrator.get_state_snapshot()


@app.get("/api/events")
async def stream_events():
    q = orchestrator.subscribe()

    async def event_generator():
        try:
            initial = {"type": "state", "data": orchestrator.get_state_snapshot()}
            yield f"data: {json.dumps(initial)}\n\n"
            while True:
                try:
                    event = await asyncio.to_thread(q.get, True, 1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            orchestrator.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest):
    cfg = SessionConfig(
        num_clients=req.num_clients,
        total_rounds=req.total_rounds,
        local_epochs=req.local_epochs,
        poll_interval=req.poll_interval,
        visual_delay=req.visual_delay,
        step_mode=req.step_mode,
        port=req.port,
    )
    try:
        return await asyncio.to_thread(orchestrator.start_session, cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/session/next-round")
async def next_round():
    return await asyncio.to_thread(orchestrator.advance_step)


@app.post("/api/session/stop")
async def stop_session():
    return await asyncio.to_thread(orchestrator.stop_session)


@app.post("/api/session/inject-rpc")
async def inject_rpc(req: InjectRpcRequest):
    try:
        return await asyncio.to_thread(orchestrator.inject_validation_rpc, req.kind, req.cid)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/manual-baseline")
async def manual_baseline(req: BaselineRequest):
    try:
        return await asyncio.to_thread(
            orchestrator.run_manual_baseline,
            req.num_clients,
            req.total_rounds,
            req.local_epochs,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        return await asyncio.to_thread(
            orchestrator.predict_iris,
            req.petal_length,
            req.petal_width,
            req.sepal_length,
            req.sepal_width,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/tests/run")
async def run_tests():
    try:
        return await asyncio.to_thread(orchestrator.run_unit_tests)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
