from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import secrets
import time
from typing import Any, Literal

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .artifact_integrity import ArtifactRegistry
from .config import (
    MODELS_DIRECTORY,
    ProductionSettings,
    RuntimeConfig,
    load_production_settings,
    load_runtime_config,
)
from .model_runtime import InferenceEngine, RuntimeSession
from .observability import RuntimeMetrics, configure_logging
from .storage import FeedbackRecord, FeedbackStore, artifact_status, load_metric_files


ALLOWED_ORIGINS = {
    "http://127.0.0.1:3100",
    "http://localhost:3100",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
}


class FeedbackPayload(BaseModel):
    actual_label: str
    predicted_label: str
    runtime_action: str = "Wait / No Action"
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str = "unknown"
    probabilities: dict[str, float] = Field(default_factory=dict)
    feature_vector: list[float] | None = None
    landmarks: list[list[float]] | None = None
    note: str = Field(default="", max_length=500)
    snapshot_data_url: str | None = None
    learning_mode: Literal["audit", "safe", "force"] = "audit"


production_settings: ProductionSettings = load_production_settings()
logger = configure_logging(production_settings.log_directory)
artifact_registry = ArtifactRegistry()
runtime_metrics = RuntimeMetrics(production_settings.metrics_window_size)
service_lock = asyncio.Lock()
configuration_error: str | None = None


def _build_services() -> tuple[
    RuntimeConfig,
    InferenceEngine,
    FeedbackStore,
]:
    global configuration_error
    errors: list[str] = []
    try:
        config = load_runtime_config()
    except (OSError, ValueError) as error:
        errors.append(str(error))
        config = RuntimeConfig()
    integrity = artifact_registry.verify()
    trusted = bool(
        integrity["verified"] or not production_settings.require_artifact_manifest
    )
    if not trusted:
        errors.extend(integrity["errors"])
    inference_engine = InferenceEngine(config, load_models=trusted)
    if errors:
        inference_engine.model_manager.errors[:0] = errors
    configuration_error = " · ".join(errors) if errors else None
    store = FeedbackStore(config)
    return config, inference_engine, store


runtime_config, engine, feedback_store = _build_services()
action_history: deque[dict[str, Any]] = deque(maxlen=500)
active_sessions: dict[str, RuntimeSession] = {}


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    yield
    engine.close()


app = FastAPI(
    title="Gesture Control Lab ONNX API",
    version="1.0.0",
    description="Independent local MediaPipe + ONNX Runtime gesture deployment.",
    lifespan=application_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-Request-ID"],
)


@app.middleware("http")
async def operational_middleware(request: Request, call_next: Any) -> Any:
    request_id = request.headers.get("x-request-id") or secrets.token_hex(8)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        runtime_metrics.record_error("unhandled_http_error")
        logger.exception(
            "Unhandled HTTP request error",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    logger.info(
        "HTTP request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )
    return response


def _admin_allowed(enabled: bool, supplied_token: str | None) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail="This administrative operation is disabled.")
    expected = production_settings.admin_token
    if production_settings.production_mode and not expected:
        raise HTTPException(
            status_code=503,
            detail="Set GESTURE_ADMIN_TOKEN before enabling administrative operations.",
        )
    if expected and not secrets.compare_digest(expected, supplied_token or ""):
        raise HTTPException(status_code=401, detail="Invalid administrative token.")


def _selectable_models() -> list[str]:
    status = engine.model_manager.status()
    selectable = list(status["selectable_models"])
    if production_settings.production_mode:
        selected = status["selected_model_name"]
        return [selected] if selected in selectable else []
    return selectable


def _onnx_qualification() -> dict[str, Any]:
    metadata_path = MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "current": False,
            "pc_release_ready": False,
            "selected_model": "ONNX",
            "deferred_classes": ["no_gesture"],
            "failing_gated_classes": [],
            "message": f"ONNX metadata is unavailable: {error}",
        }
    parity = metadata.get("parity") or {}
    passed = bool(parity.get("passed"))
    return {
        "status": "passed" if passed else "failed",
        "current": passed,
        "pc_release_ready": passed,
        "selected_model": "ONNX",
        "format": "ONNX",
        "created_utc": metadata.get("created_utc"),
        "source_model": metadata.get("model_name"),
        "deferred_classes": list(
            (metadata.get("source_selection") or {}).get(
                "deferred_classes", ["no_gesture"]
            )
        ),
        "failing_gated_classes": [],
        "onnx_parity": parity,
        "prediction_agreement": parity.get("prediction_agreement"),
        "maximum_absolute_probability_error": parity.get(
            "maximum_absolute_probability_error"
        ),
        "message": (
            "The ONNX graph passed conversion parity and is loaded directly by "
            "ONNX Runtime."
            if passed
            else "The ONNX conversion parity gate did not pass."
        ),
    }


def current_health() -> dict[str, Any]:
    engine_status = engine.status()
    engine_status["model"]["selectable_models"] = _selectable_models()
    engine_status["model"]["production_selection_locked"] = production_settings.production_mode
    integrity = artifact_registry.verify()
    operational_ready = bool(
        engine_status["ready"]
        and (integrity["verified"] or not production_settings.require_artifact_manifest)
    )
    release = _onnx_qualification()
    return {
        "status": "ready" if operational_ready else "setup_required",
        "operational_ready": operational_ready,
        "engine": engine_status,
        "config": runtime_config.public_dict(),
        "artifacts": artifact_status(),
        "artifact_integrity": integrity,
        "online_learning": {
            "ready": False,
            "accepted_updates": 0,
            "forced_updates": 0,
            "rejected_updates": 0,
            "force_learning_enabled": False,
            "backup_count": 0,
            "tflite_policy": (
                "ONNX deployment graphs are immutable. Reviewed feedback is saved "
                "for a later offline retraining and re-export cycle."
            ),
        },
        "production": {
            **production_settings.public_dict(),
            "release_qualification": release,
        },
        "runtime_metrics": runtime_metrics.snapshot(),
        "active_sessions": len(active_sessions),
        "configuration_error": configuration_error,
    }


@app.get("/livez")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
def readiness() -> JSONResponse:
    health = current_health()
    return JSONResponse(health, status_code=200 if health["operational_ready"] else 503)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return current_health()


@app.get("/api/config")
def config() -> dict[str, Any]:
    return runtime_config.public_dict()


@app.get("/api/models")
def models() -> dict[str, Any]:
    status = engine.model_manager.status()
    status["selectable_models"] = _selectable_models()
    status["production_selection_locked"] = production_settings.production_mode
    return status


@app.get("/api/artifacts")
def artifacts() -> dict[str, Any]:
    return {
        "models_directory": str(MODELS_DIRECTORY),
        "artifacts": artifact_status(),
        "integrity": artifact_registry.verify(),
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return load_metric_files()


@app.get("/api/runtime-metrics")
def runtime_metric_snapshot() -> dict[str, Any]:
    return runtime_metrics.snapshot()


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return runtime_metrics.prometheus()


@app.get("/api/production-qualification")
def production_qualification() -> dict[str, Any]:
    return _onnx_qualification()


@app.post("/api/production-qualification/run")
async def run_production_qualification(
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    del x_admin_token
    return _onnx_qualification()


@app.get("/api/actions")
def actions() -> dict[str, Any]:
    return {"count": len(action_history), "recent": list(action_history)[::-1]}


@app.get("/api/feedback")
def feedback_summary() -> dict[str, Any]:
    return feedback_store.summary()


@app.post("/api/feedback")
def save_feedback(payload: FeedbackPayload) -> dict[str, Any]:
    try:
        values = payload.model_dump()
        learning_mode = values.pop("learning_mode")
        saved = feedback_store.save(FeedbackRecord(**values))
        update: dict[str, Any] = {
            "status": "audit_only" if learning_mode == "audit" else "disabled",
            "message": (
                "Feedback saved for the next offline retraining and ONNX re-export cycle."
                if learning_mode == "audit"
                else "Feedback was saved, but a deployed ONNX graph cannot learn in place."
            ),
        }
        return {**saved, "learning_mode": learning_mode, "online_update": update}
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/online-learning/backups")
def online_backups() -> dict[str, Any]:
    return {"count": 0, "backups": [], "message": "ONNX runtime is immutable."}


@app.post("/api/predict-image")
async def predict_image(
    image: UploadFile = File(...),
    model_name: str | None = None,
) -> dict[str, Any]:
    if model_name is not None and model_name not in _selectable_models():
        raise HTTPException(status_code=400, detail="Unknown model selection.")
    if image.content_type and not (
        image.content_type.startswith("image/") or image.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=415, detail="Only image uploads are accepted.")
    content = await image.read(production_settings.max_image_bytes + 1)
    if len(content) > production_settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured size limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    session = RuntimeSession(runtime_config)
    result: dict[str, Any] = {}
    for _ in range(runtime_config.stable_frames_required):
        result = await asyncio.to_thread(engine.process_frame, content, session, model_name)
        runtime_metrics.record_frame(result)
    return result


@app.post("/api/reload")
async def reload_artifacts(
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    global runtime_config, engine, feedback_store
    _admin_allowed(production_settings.allow_artifact_reload, x_admin_token)
    async with service_lock:
        old_engine = engine
        runtime_config, engine, feedback_store = _build_services()
        for session in active_sessions.values():
            session.reset()
        old_engine.close()
    return current_health()


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Origin is not allowed.")
        return
    if len(active_sessions) >= production_settings.max_active_sessions:
        await websocket.close(code=1013, reason="The local inference service is at capacity.")
        return
    await websocket.accept()
    session_id = f"session-{secrets.token_hex(8)}"
    session = RuntimeSession(runtime_config)
    active_sessions[session_id] = session
    runtime_metrics.session_opened()
    selected_model: str | None = None
    await websocket.send_json({
        "type": "hello",
        "session_id": session_id,
        "health": current_health(),
    })
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            frame = message.get("bytes")
            if frame is not None:
                if len(frame) > production_settings.max_websocket_frame_bytes:
                    runtime_metrics.record_error("frame_too_large")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Camera frame exceeds the configured size limit.",
                    })
                    continue
                try:
                    result = await asyncio.to_thread(
                        engine.process_frame, frame, session, selected_model
                    )
                except Exception:
                    runtime_metrics.record_error("inference_exception")
                    logger.exception("Frame inference failed")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Frame inference failed; the session remains available.",
                    })
                    continue
                runtime_metrics.record_frame(result)
                if result.get("runtime_action") not in {None, "Wait / No Action"}:
                    action_history.append({
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "session_id": session_id,
                        "model": result.get("model"),
                        "action": result["runtime_action"],
                        "prediction": result.get("runtime_prediction"),
                        "confidence": result.get("confidence", 0.0),
                    })
                await websocket.send_json({"type": "prediction", **result})
            elif message.get("text"):
                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    command = {"type": message["text"]}
                command_type = command.get("type")
                if command_type == "reset":
                    session.reset()
                    await websocket.send_json({"type": "reset_complete"})
                elif command_type == "select_model":
                    requested = str(command.get("model", ""))
                    selectable = _selectable_models()
                    if requested not in selectable:
                        await websocket.send_json({"type": "error", "message": "Unknown model selection."})
                    else:
                        selected_model = requested
                        session.reset()
                        await websocket.send_json({"type": "model_selected", "model": requested})
                elif command_type == "start_follow_object":
                    session.temporal_gate.reset()
                    session.diagnostic_gates.clear()
                    update = session.follow_object.start()
                    await websocket.send_json({
                        "type": "follow_object_started",
                        "follow_object": {
                            name: getattr(update, name) for name in update.__dataclass_fields__
                        },
                    })
                elif command_type == "stop_follow_object":
                    update = session.follow_object.stop()
                    await websocket.send_json({
                        "type": "follow_object_stopped",
                        "follow_object": {
                            name: getattr(update, name) for name in update.__dataclass_fields__
                        },
                    })
                else:
                    await websocket.send_json({"type": "error", "message": "Unknown command."})
    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.pop(session_id, None)
        runtime_metrics.session_closed()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Gesture Control Lab ONNX API",
        "version": app.version,
        "dashboard": "Start the frontend and open http://127.0.0.1:3100",
        "docs": "http://127.0.0.1:8100/docs",
    }
