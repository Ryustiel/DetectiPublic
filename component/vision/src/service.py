from __future__ import annotations

import asyncio
import io
import os
import re
import secrets
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, field_validator

from processor.depth import (
    DEPTH_BACKEND_NONE,
    get_available_depth_backends,
    get_loaded_depth_backends,
    is_depth_backend_loaded,
    normalize_depth_backend,
    predict_depth_map,
    select_depth_backend,
    unload_depth_backend,
)
from processor.grounding_dino import (
    DETECTION_BACKEND_NONE,
    get_available_detection_backends,
    get_loaded_detection_backends,
    is_detection_backend_loaded,
    normalize_detection_backend,
    predict_objects,
    sanitize_keywords,
    select_detection_backend,
    unload_detection_backend,
)


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
DEBUG_HTML_FILE = TEMPLATE_DIR / "debug.html"
CAMERA_HTML_FILE = TEMPLATE_DIR / "camera.html"


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _host_from_base_url(value: str) -> str:
    raw = value.strip()
    if "://" not in raw:
        raw = "https://" + raw
    return (urllib.parse.urlparse(raw).netloc or raw).lower()


def _comparable_host(value: str) -> str:
    host = value.strip().lower()
    for suffix in (":443", ":80"):
        if host.endswith(suffix):
            return host[: -len(suffix)]
    return host


VISION_PUBLIC_BASE_URL = os.getenv("VISION_PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
CAMERA_PUBLIC_BASE_URL = os.getenv("CAMERA_PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
VISION_PUBLIC_HOST = _comparable_host(_host_from_base_url(VISION_PUBLIC_BASE_URL))
CAMERA_PUBLIC_HOST = _comparable_host(_host_from_base_url(CAMERA_PUBLIC_BASE_URL))

TEMP_IMAGE_DIR = Path(os.getenv("VISION_TEMP_IMAGE_DIR", "/app/assets/temp")).resolve()
TEMP_URL_MAX = _env_int("VISION_TEMP_URL_MAX", 20, minimum=1)
TEMP_URL_TTL_SECONDS = _env_int("VISION_TEMP_URL_TTL_SECONDS", 24 * 60 * 60, minimum=60)
GENERATED_IMAGE_MAX = _env_int("VISION_GENERATED_IMAGE_MAX", 100, minimum=1)
GENERATED_IMAGE_TTL_SECONDS = _env_int(
    "VISION_GENERATED_IMAGE_TTL_SECONDS",
    24 * 60 * 60,
    minimum=60,
)
STREAM_FRAME_MAX = _env_int("VISION_STREAM_FRAME_MAX", 100, minimum=1)
STREAM_FRAME_TTL_SECONDS = _env_int("VISION_STREAM_FRAME_TTL_SECONDS", 24 * 60 * 60, minimum=60)
CAMERA_FRAME_MAX = _env_int("VISION_CAMERA_FRAME_MAX", STREAM_FRAME_MAX, minimum=1)
CAMERA_FRAME_TTL_SECONDS = _env_int(
    "VISION_CAMERA_FRAME_TTL_SECONDS",
    STREAM_FRAME_TTL_SECONDS,
    minimum=60,
)
CAMERA_CODE_INACTIVITY_SECONDS = _env_int(
    "VISION_CAMERA_CODE_INACTIVITY_SECONDS",
    10 * 60,
    minimum=30,
)
MAX_IMAGE_EDGE = _env_int("VISION_MAX_IMAGE_EDGE", 1280, minimum=128)
JPEG_QUALITY = _env_int("VISION_JPEG_QUALITY", 82, minimum=35)
MAX_DOWNLOAD_BYTES = _env_int("VISION_MAX_DOWNLOAD_BYTES", 15_000_000, minimum=100_000)
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = _env_float(
    "VISION_IMAGE_DOWNLOAD_TIMEOUT_SECONDS",
    10.0,
    minimum=1.0,
)
CALLBACK_TIMEOUT_SECONDS = _env_float("VISION_CALLBACK_TIMEOUT_SECONDS", 5.0, minimum=1.0)
DEFAULT_CAMERA_FPS = _env_float("VISION_DEFAULT_CAMERA_FPS", 2.0, minimum=0.05)

STREAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")

TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vision API")


class StopConditionPayload(BaseModel):
    type: Literal["first_detection", "consecutive_detection"]
    label: str
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    frames: int = Field(default=1, ge=1)
    cut: Literal["last_detected", "middle_detected"] = "last_detected"

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        cleaned = " ".join(str(value).strip().lower().split())
        if not cleaned:
            raise ValueError("label is required")
        return cleaned


class StreamCreatePayload(BaseModel):
    id: str
    detection_model: str
    labels: list[str]
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    max_objects: int = Field(default=10, ge=1, le=100)
    process_every_ms: int = Field(default=0, ge=0)
    stop_conditions: list[StopConditionPayload] = Field(default_factory=list)
    callback_url: str | None = None
    stream: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not STREAM_ID_RE.match(value):
            raise ValueError("id must be alphanumeric and may contain _ . : -")
        return value

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        cleaned = sanitize_keywords(value)
        if not cleaned:
            raise ValueError("at least one label is required")
        return cleaned


class StreamUpdatePayload(BaseModel):
    detection_model: str | None = None
    labels: list[str] | None = None
    sensitivity: float | None = Field(default=None, ge=0.0, le=1.0)
    max_objects: int | None = Field(default=None, ge=1, le=100)
    process_every_ms: int | None = Field(default=None, ge=0)
    stop_conditions: list[StopConditionPayload] | None = None
    callback_url: str | None = None
    stream: str | None = None

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = sanitize_keywords(value)
        if not cleaned:
            raise ValueError("at least one label is required")
        return cleaned


class CameraCodePayload(BaseModel):
    code: str | None = None
    capture_fps: float = Field(default=DEFAULT_CAMERA_FPS, gt=0.0, le=30.0)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not CODE_RE.match(cleaned):
            raise ValueError("code must be 3-64 chars and may contain letters, numbers, _ or -")
        return cleaned


@dataclass
class FrameRecord:
    sequence: int
    created_at: float
    path: Path
    width: int
    height: int
    detections: list[dict[str, Any]]
    detection_model: str
    labels: list[str]


@dataclass
class CameraFrameRecord:
    sequence: int
    created_at: float
    path: Path
    width: int
    height: int


RetainedFrame = FrameRecord | CameraFrameRecord


@dataclass
class StreamState:
    id: str
    detection_model: str
    labels: list[str]
    sensitivity: float
    max_objects: int
    process_every_ms: int
    stop_conditions: list[StopConditionPayload]
    callback_url: str | None = None
    source_code: str | None = None
    status: Literal["active", "stopped"] = "active"
    stop_reason: str | None = None
    stop_frame_sequence: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    frames: list[FrameRecord] = field(default_factory=list)
    next_sequence: int = 0
    last_processed_at: float | None = None
    stop_streaks: dict[int, list[int]] = field(default_factory=dict)


@dataclass
class CameraCodeState:
    code: str
    capture_fps: float
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    frames: list[CameraFrameRecord] = field(default_factory=list)
    next_sequence: int = 0


@dataclass
class TempUrlState:
    token: str
    path: Path
    created_at: float
    expires_at: float
    kind: Literal["frame", "generated"]


@dataclass
class GeneratedImageState:
    image_id: str
    path: Path
    created_at: float
    expires_at: float


STREAMS: dict[str, StreamState] = {}
CAMERA_CODES: dict[str, CameraCodeState] = {}
TEMP_URLS: dict[str, TempUrlState] = {}
GENERATED_IMAGES: dict[str, GeneratedImageState] = {}
STATE_LOCK = asyncio.Lock()
MODEL_LOCK = asyncio.Lock()
SUBSCRIBER_LOCK = asyncio.Lock()
STREAM_EVENT_SUBSCRIBERS: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
ALL_STREAMS_SUBSCRIPTION = "*"


def _raise(status_code: int, code: str, message: str, **extra: Any) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"error": code, "message": message, **extra},
    )


def _header_host(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(",", 1)[0].strip().lower()


def _forwarded_host(value: str | None) -> str | None:
    if not value:
        return None
    first = value.split(",", 1)[0]
    for part in first.split(";"):
        key, _, raw = part.strip().partition("=")
        if key.lower() == "host" and raw:
            return raw.strip("\"").lower()
    return None


def _url_header_host(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.netloc:
        return parsed.netloc.lower()
    return None


def original_host(request: Request | WebSocket) -> str:
    return (
        _header_host(request.headers.get("cf-forwarded-host"))
        or _header_host(request.headers.get("x-forwarded-host"))
        or _header_host(request.headers.get("x-original-host"))
        or _url_header_host(request.headers.get("x-original-url"))
        or _url_header_host(request.headers.get("x-forwarded-url"))
        or _forwarded_host(request.headers.get("forwarded"))
        or _header_host(request.headers.get("host"))
        or ""
    )


def is_camera_host(request: Request | WebSocket) -> bool:
    return _comparable_host(original_host(request)) == CAMERA_PUBLIC_HOST


def is_local_host(request: Request | WebSocket) -> bool:
    host = original_host(request).split(":", 1)[0]
    return host in {"localhost", "127.0.0.1", "::1"}


def request_base_url(request: Request) -> str:
    host = original_host(request)
    if not host:
        return VISION_PUBLIC_BASE_URL
    proto = (
        request.headers.get("x-forwarded-proto")
        or urllib.parse.urlparse(VISION_PUBLIC_BASE_URL).scheme
        or request.url.scheme
    )
    return f"{proto}://{host}"


def camera_url(code: str) -> str:
    return f"{CAMERA_PUBLIC_BASE_URL}/{code}"


@app.middleware("http")
async def block_api_from_camera_host(request: Request, call_next):
    if is_camera_host(request):
        path = request.url.path
        if path.startswith(("/api", "/docs", "/redoc")) or path in {"/openapi.json", "/health"}:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "camera_host_has_no_api",
                    "message": "This host only serves active camera capture codes.",
                },
            )
    return await call_next(request)


def vram_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "free_bytes": None, "total_bytes": None}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def detection_model_ids() -> set[str]:
    return {
        item["id"]
        for item in get_available_detection_backends()
        if item["id"] != DETECTION_BACKEND_NONE and item.get("enabled", True)
    }


def depth_model_ids() -> set[str]:
    return {
        item["id"]
        for item in get_available_depth_backends()
        if item["id"] != DEPTH_BACKEND_NONE and not item.get("disabled", False)
    }


def normalize_model_id(model_id: str) -> tuple[Literal["detection", "depth"], str]:
    try:
        normalized = normalize_detection_backend(model_id)
        if normalized != DETECTION_BACKEND_NONE and normalized in detection_model_ids():
            return "detection", normalized
    except ValueError:
        pass

    try:
        normalized = normalize_depth_backend(model_id)
        if normalized != DEPTH_BACKEND_NONE and normalized in depth_model_ids():
            return "depth", normalized
    except ValueError:
        pass

    _raise(404, "unknown_model", f"Unknown model '{model_id}'.")


def model_is_loaded(kind: str, model_id: str) -> bool:
    if kind == "detection":
        return is_detection_backend_loaded(model_id)
    if kind == "depth":
        return is_depth_backend_loaded(model_id)
    return False


def model_catalog() -> list[dict[str, Any]]:
    loaded_detection = set(get_loaded_detection_backends())
    loaded_depth = set(get_loaded_depth_backends())
    models: list[dict[str, Any]] = []

    for item in get_available_detection_backends():
        if item["id"] == DETECTION_BACKEND_NONE or not item.get("enabled", True):
            continue
        models.append(
            {
                **item,
                "kind": "detection",
                "loaded": item["id"] in loaded_detection,
                "output_modes": ["text", "image"],
                "usable_for_streams": True,
            }
        )

    for item in get_available_depth_backends():
        if item["id"] == DEPTH_BACKEND_NONE or item.get("disabled", False):
            continue
        models.append(
            {
                **item,
                "kind": "depth",
                "loaded": item["id"] in loaded_depth,
                "output_modes": ["image"],
                "usable_for_streams": False,
            }
        )

    return models


def resize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) <= MAX_IMAGE_EDGE:
        return image
    copy = image.copy()
    copy.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return copy


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return resize_image(image)
    except Exception as exc:
        _raise(400, "invalid_image", f"Could not decode image: {exc}")


def save_image(image: Image.Image, prefix: str) -> Path:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMP_IMAGE_DIR / f"{prefix}-{uuid.uuid4().hex}.jpg"
    image.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return path


def delete_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

    for token, state in list(TEMP_URLS.items()):
        if state.path == path:
            TEMP_URLS.pop(token, None)


def prune_temp_urls(now: float | None = None) -> None:
    now = now or time.time()
    for token, state in list(TEMP_URLS.items()):
        if state.expires_at <= now or not state.path.exists():
            TEMP_URLS.pop(token, None)

    while len(TEMP_URLS) > TEMP_URL_MAX:
        oldest_token = min(TEMP_URLS.values(), key=lambda item: item.created_at).token
        TEMP_URLS.pop(oldest_token, None)


def prune_generated_images(now: float | None = None) -> None:
    now = now or time.time()
    for image_id, state in list(GENERATED_IMAGES.items()):
        if state.expires_at <= now or not state.path.exists():
            delete_path(state.path)
            GENERATED_IMAGES.pop(image_id, None)

    while len(GENERATED_IMAGES) > GENERATED_IMAGE_MAX:
        oldest = min(GENERATED_IMAGES.values(), key=lambda item: item.created_at)
        delete_path(oldest.path)
        GENERATED_IMAGES.pop(oldest.image_id, None)


def prune_stream_frames(stream: StreamState, now: float | None = None) -> None:
    now = now or time.time()
    candidates: list[FrameRecord] = []
    for frame in stream.frames:
        if (now - frame.created_at) > STREAM_FRAME_TTL_SECONDS:
            delete_path(frame.path)
        else:
            candidates.append(frame)

    overflow_count = max(0, len(candidates) - STREAM_FRAME_MAX)
    for frame in candidates[:overflow_count]:
        delete_path(frame.path)
    stream.frames = candidates[overflow_count:]


def prune_camera_frames(camera: CameraCodeState, now: float | None = None) -> None:
    now = now or time.time()
    candidates: list[CameraFrameRecord] = []
    for frame in camera.frames:
        if (now - frame.created_at) > CAMERA_FRAME_TTL_SECONDS:
            delete_path(frame.path)
        else:
            candidates.append(frame)

    overflow_count = max(0, len(candidates) - CAMERA_FRAME_MAX)
    for frame in candidates[:overflow_count]:
        delete_path(frame.path)
    camera.frames = candidates[overflow_count:]


def create_temp_url(path: Path, kind: Literal["frame", "generated"], request: Request) -> dict[str, Any]:
    now = time.time()
    token = secrets.token_urlsafe(18)
    state = TempUrlState(
        token=token,
        path=path,
        created_at=now,
        expires_at=now + TEMP_URL_TTL_SECONDS,
        kind=kind,
    )
    TEMP_URLS[token] = state
    prune_temp_urls(now)
    return {
        "url": f"{request_base_url(request)}/tmp/{token}",
        "expires_at": state.expires_at,
        "kind": kind,
    }


def register_generated_image(image: Image.Image, request: Request) -> dict[str, Any]:
    now = time.time()
    image_id = uuid.uuid4().hex
    path = save_image(image, "generated")
    GENERATED_IMAGES[image_id] = GeneratedImageState(
        image_id=image_id,
        path=path,
        created_at=now,
        expires_at=now + GENERATED_IMAGE_TTL_SECONDS,
    )
    prune_generated_images(now)
    return create_temp_url(path, "generated", request)


def detection_matches(detection: dict[str, Any], label: str, min_confidence: float) -> bool:
    candidates = {
        str(detection.get("keyword") or "").strip().lower(),
        str(detection.get("label") or "").strip().lower(),
        str(detection.get("raw_label") or "").strip().lower(),
    }
    score = float(detection.get("score") or 0.0)
    normalized_label = " ".join(label.strip().lower().split())
    return score >= min_confidence and any(
        normalized_label == candidate
        or normalized_label in candidate
        or candidate in normalized_label
        for candidate in candidates
        if candidate
    )


def matching_detections(
    detections: list[dict[str, Any]],
    label: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    return [
        detection
        for detection in detections
        if detection_matches(detection, label, min_confidence)
    ]


def detection_text_payload(
    detections: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    items = []
    lines = []
    for detection in detections:
        x = float(detection.get("x") or 0.0)
        y = float(detection.get("y") or 0.0)
        w = max(0.0, float(detection.get("w") or 0.0))
        h = max(0.0, float(detection.get("h") or 0.0))
        center_x_pct = (((x + w / 2.0) - (width / 2.0)) / max(width, 1)) * 100.0
        center_y_pct = (((y + h / 2.0) - (height / 2.0)) / max(height, 1)) * 100.0
        width_pct = (w / max(width, 1)) * 100.0
        height_pct = (h / max(height, 1)) * 100.0
        label = str(detection.get("keyword") or detection.get("label") or "object")
        score = float(detection.get("score") or 0.0)
        item = {
            "label": label,
            "score": score,
            "center_offset": {
                "x_percent": round(center_x_pct, 2),
                "y_percent": round(center_y_pct, 2),
            },
            "box_size": {
                "width_percent": round(width_pct, 2),
                "height_percent": round(height_pct, 2),
            },
        }
        items.append(item)
        lines.append(
            f"{label}: confidence {score:.2f}, center "
            f"{center_x_pct:+.1f}%x {center_y_pct:+.1f}%y, "
            f"width {width_pct:.1f}%, height {height_pct:.1f}%"
        )
    return {"text": "\n".join(lines), "detections": items}


def stream_frame_reference(stream_id: str, frame: FrameRecord) -> dict[str, Any]:
    quoted_stream_id = urllib.parse.quote(stream_id, safe="")
    return {
        "id": f"{stream_id}:{frame.sequence}",
        "sequence": frame.sequence,
        "created_at": frame.created_at,
        "width": frame.width,
        "height": frame.height,
        "cache_expires_at": frame.created_at + STREAM_FRAME_TTL_SECONDS,
        "query_endpoint": f"/api/streams/{quoted_stream_id}/frames/{frame.sequence}/query",
        "raw_endpoint": f"/api/streams/{quoted_stream_id}/frames/{frame.sequence}/raw",
    }


def camera_frame_reference(code: str, frame: CameraFrameRecord) -> dict[str, Any]:
    quoted_code = urllib.parse.quote(code, safe="")
    return {
        "id": f"{code}:{frame.sequence}",
        "sequence": frame.sequence,
        "created_at": frame.created_at,
        "width": frame.width,
        "height": frame.height,
        "cache_expires_at": frame.created_at + CAMERA_FRAME_TTL_SECONDS,
        "query_endpoint": f"/api/camera-codes/{quoted_code}/frames/{frame.sequence}/query",
        "raw_endpoint": f"/api/camera-codes/{quoted_code}/frames/{frame.sequence}/raw",
    }


def stream_frame_query_payload(
    stream_id: str,
    frame: FrameRecord,
    warning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "stream": stream_id,
        "frame": {
            **stream_frame_reference(stream_id, frame),
            "detection_model": frame.detection_model,
            "labels": frame.labels,
        },
        "detections": detection_text_payload(frame.detections, frame.width, frame.height),
    }
    if warning:
        payload["warning"] = warning
    return payload


def camera_frame_query_payload(
    code: str,
    frame: CameraFrameRecord,
    warning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "frame": camera_frame_reference(code, frame),
    }
    if warning:
        payload["warning"] = warning
    return payload


def stream_detection_event_payload(
    stream: StreamState,
    frame: FrameRecord,
    event: str = "detection",
    reason: str | None = None,
    condition: StopConditionPayload | None = None,
    matched_detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "stream": serialize_stream(stream),
        "frame": stream_frame_reference(stream.id, frame),
        "detections": detection_text_payload(frame.detections, frame.width, frame.height),
    }
    if matched_detections is not None:
        payload["matched_detections"] = detection_text_payload(
            matched_detections,
            frame.width,
            frame.height,
        )
    if condition is not None:
        payload["condition"] = condition.model_dump()
    if reason:
        payload["reason"] = reason
    return payload


def stream_stopped_event_payload(
    stream: StreamState,
    reason: str,
    frame: FrameRecord | None = None,
    condition: StopConditionPayload | None = None,
    matched_detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if frame is None:
        return {
            "event": "stream_stopped",
            "reason": reason,
            "stream": serialize_stream(stream),
        }
    return stream_detection_event_payload(
        stream=stream,
        frame=frame,
        event="stream_stopped",
        reason=reason,
        condition=condition,
        matched_detections=matched_detections,
    )


async def publish_stream_event(stream_id: str, payload: dict[str, Any]) -> None:
    async with SUBSCRIBER_LOCK:
        queues = list(STREAM_EVENT_SUBSCRIBERS.get(stream_id, set()))
        queues.extend(STREAM_EVENT_SUBSCRIBERS.get(ALL_STREAMS_SUBSCRIPTION, set()))

    for queue in queues:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    palette = [
        "#0ea5e9",
        "#f97316",
        "#22c55e",
        "#a855f7",
        "#ef4444",
        "#14b8a6",
    ]
    for index, detection in enumerate(detections):
        color = palette[index % len(palette)]
        x = float(detection.get("x") or 0.0)
        y = float(detection.get("y") or 0.0)
        w = max(1.0, float(detection.get("w") or 1.0))
        h = max(1.0, float(detection.get("h") or 1.0))
        label = str(detection.get("keyword") or detection.get("label") or "object")
        score = float(detection.get("score") or 0.0)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        text = f"{label} {score:.2f}"
        text_bbox = draw.textbbox((x, y), text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle((x, max(0, y - text_h - 8), x + text_w + 10, y), fill=color)
        draw.text((x + 5, max(0, y - text_h - 5)), text, fill="white")
    return output


def depth_map_to_image(depth_map: np.ndarray) -> Image.Image:
    valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid.size == 0:
        _raise(500, "empty_depth_map", "Depth model returned no finite positive values.")
    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth_map - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0

    stops = np.array(
        [
            [20, 34, 82],
            [14, 165, 233],
            [34, 197, 94],
            [250, 204, 21],
            [239, 68, 68],
        ],
        dtype=np.float32,
    )
    scaled = normalized * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    alpha = (scaled - lower)[..., None]
    rgb = (stops[lower] * (1.0 - alpha)) + (stops[upper] * alpha)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


async def run_detection(
    image: Image.Image,
    model_id: str,
    labels: list[str],
    sensitivity: float,
    max_objects: int,
) -> list[dict[str, Any]]:
    async with MODEL_LOCK:
        return await asyncio.to_thread(
            predict_objects,
            image,
            labels,
            sensitivity,
            max_objects,
            model_id,
        )


async def run_depth(image: Image.Image, model_id: str) -> np.ndarray:
    async with MODEL_LOCK:
        return await asyncio.to_thread(predict_depth_map, image, model_id)


async def fetch_image_url(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        _raise(400, "invalid_url", "Only http and https image URLs are supported.")

    async with httpx.AsyncClient(timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    if len(content) > MAX_DOWNLOAD_BYTES:
        _raise(
            413,
            "image_too_large",
            f"Downloaded image exceeds {MAX_DOWNLOAD_BYTES} bytes.",
        )
    return content


def serialize_stream(stream: StreamState, request: Request | None = None) -> dict[str, Any]:
    payload = {
        "id": stream.id,
        "status": stream.status,
        "detection_model": stream.detection_model,
        "labels": stream.labels,
        "sensitivity": stream.sensitivity,
        "max_objects": stream.max_objects,
        "process_every_ms": stream.process_every_ms,
        "stop_conditions": [condition.model_dump() for condition in stream.stop_conditions],
        "callback_url": stream.callback_url,
        "stream": stream.source_code,
        "frame_count": len(stream.frames),
        "stop_reason": stream.stop_reason,
        "stop_frame_sequence": stream.stop_frame_sequence,
        "created_at": stream.created_at,
        "updated_at": stream.updated_at,
    }
    if stream.frames:
        latest = stream.frames[-1]
        payload["latest_frame"] = {
            **stream_frame_reference(stream.id, latest),
            "detection_count": len(latest.detections),
        }
    return payload


def serialize_code(code: CameraCodeState) -> dict[str, Any]:
    payload = {
        "code": code.code,
        "capture_fps": code.capture_fps,
        "capture_interval_ms": int(1000 / max(code.capture_fps, 0.001)),
        "url": camera_url(code.code),
        "frame_count": len(code.frames),
        "created_at": code.created_at,
        "last_seen_at": code.last_seen_at,
        "expires_at": code.last_seen_at + CAMERA_CODE_INACTIVITY_SECONDS,
    }
    if code.frames:
        payload["latest_frame"] = camera_frame_reference(code.code, code.frames[-1])
    return payload


def stop_stream_locked(
    stream: StreamState,
    reason: str,
    stop_frame_sequence: int | None = None,
    frame: FrameRecord | None = None,
    condition: StopConditionPayload | None = None,
    matched_detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if stream.status == "stopped":
        return None
    stream.status = "stopped"
    stream.stop_reason = reason
    stream.stop_frame_sequence = stop_frame_sequence
    stream.updated_at = time.time()
    payload = stream_stopped_event_payload(
        stream=stream,
        reason=reason,
        frame=frame,
        condition=condition,
        matched_detections=matched_detections,
    )
    return {
        "callback_url": stream.callback_url,
        "payload": payload,
        "stream_id": stream.id,
    }


async def send_callback(callback_url: str, payload: dict[str, Any]) -> None:
    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:
            await client.post(callback_url, json=payload)
    except Exception as exc:
        print(f"Callback failed for {callback_url}: {exc}")


def evaluate_stop_conditions_locked(
    stream: StreamState,
    frame: FrameRecord,
) -> dict[str, Any] | None:
    for index, condition in enumerate(stream.stop_conditions):
        matched = matching_detections(
            frame.detections,
            condition.label,
            condition.min_confidence,
        )

        if condition.type == "first_detection":
            if matched:
                return stop_stream_locked(
                    stream,
                    f"first_detection:{condition.label}",
                    frame.sequence,
                    frame=frame,
                    condition=condition,
                    matched_detections=matched,
                )
            continue

        streak = stream.stop_streaks.setdefault(index, [])
        if matched:
            streak.append(frame.sequence)
        else:
            streak.clear()

        if len(streak) >= condition.frames:
            if condition.cut == "middle_detected":
                stop_sequence = streak[(len(streak) - 1) // 2]
            else:
                stop_sequence = streak[-1]
            return stop_stream_locked(
                stream,
                f"consecutive_detection:{condition.label}:{condition.frames}",
                stop_sequence,
                frame=frame,
                condition=condition,
                matched_detections=matched,
            )
    return None


def resolve_frame_from_history(
    frames: Sequence[RetainedFrame],
    index: int | None,
    age_ms: int | None,
) -> tuple[RetainedFrame, dict[str, Any] | None]:
    if not frames:
        _raise(404, "no_frames", "This source has no retained frames.")

    if index is not None:
        if index < 0 or index >= len(frames):
            _raise(
                404,
                "frame_index_not_found",
                f"Frame index {index} is outside the retained history.",
                retained_frames=len(frames),
            )
        return frames[-1 - index], None

    if age_ms is None:
        return frames[-1], None

    latest = frames[-1]
    oldest = frames[0]
    oldest_delta_ms = int((latest.created_at - oldest.created_at) * 1000)
    last_interval_ms = 0
    if len(frames) >= 2:
        last_interval_ms = int(
            (frames[-1].created_at - frames[-2].created_at) * 1000
        )

    if age_ms > oldest_delta_ms + last_interval_ms:
        _raise(
            416,
            "frame_interval_too_old",
            "Requested relative time is older than retained frames plus the latest frame interval.",
            requested_age_ms=age_ms,
            oldest_retained_age_ms=oldest_delta_ms,
            latest_frame_interval_ms=last_interval_ms,
        )

    warning = None
    if age_ms > oldest_delta_ms:
        warning = {
            "code": "clamped_to_oldest_frame",
            "message": "Requested age is older than the oldest frame; returning the oldest retained frame.",
            "oldest_retained_age_ms": oldest_delta_ms,
        }

    target_time = latest.created_at - (age_ms / 1000.0)
    selected = min(frames, key=lambda frame: abs(frame.created_at - target_time))
    return selected, warning


def resolve_frame_from_stream(
    stream: StreamState,
    index: int | None,
    age_ms: int | None,
) -> tuple[FrameRecord, dict[str, Any] | None]:
    frame, warning = resolve_frame_from_history(stream.frames, index, age_ms)
    return frame, warning


def resolve_camera_frame(
    camera: CameraCodeState,
    index: int | None,
    age_ms: int | None,
) -> tuple[CameraFrameRecord, dict[str, Any] | None]:
    frame, warning = resolve_frame_from_history(camera.frames, index, age_ms)
    return frame, warning


def resolve_frame_by_sequence(
    frames: Sequence[RetainedFrame],
    frame_sequence: int,
) -> RetainedFrame:
    if not frames:
        _raise(404, "no_frames", "This source has no retained frames.")
    for frame in frames:
        if frame.sequence == frame_sequence:
            return frame
    _raise(
        404,
        "frame_id_not_found",
        f"Frame id {frame_sequence} is not in the retained history.",
        retained_frame_ids=[frame.sequence for frame in frames],
    )


async def ingest_frame_for_stream(
    stream_id: str,
    image: Image.Image,
) -> dict[str, Any]:
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        if stream.status != "active":
            _raise(409, "stream_stopped", f"Stream '{stream_id}' is stopped.")
        if not is_detection_backend_loaded(stream.detection_model):
            stop_result = stop_stream_locked(stream, "model_unloaded")
            if stop_result:
                asyncio.create_task(
                    publish_stream_event(stop_result["stream_id"], stop_result["payload"])
                )
                if stop_result["callback_url"]:
                    asyncio.create_task(
                        send_callback(stop_result["callback_url"], stop_result["payload"])
                    )
            _raise(
                409,
                "model_not_loaded",
                f"Detection model '{stream.detection_model}' is not loaded.",
            )

        now = time.time()
        if (
            stream.process_every_ms
            and stream.last_processed_at is not None
            and (now - stream.last_processed_at) * 1000 < stream.process_every_ms
        ):
            return {"status": "skipped", "reason": "process_every_ms", "stream": stream_id}

        snapshot = {
            "model": stream.detection_model,
            "labels": list(stream.labels),
            "sensitivity": stream.sensitivity,
            "max_objects": stream.max_objects,
        }

    detections = await run_detection(
        image=image,
        model_id=snapshot["model"],
        labels=snapshot["labels"],
        sensitivity=snapshot["sensitivity"],
        max_objects=snapshot["max_objects"],
    )
    frame_path = save_image(image, f"stream-{stream_id}")

    stop_result = None
    event_payload = None
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            delete_path(frame_path)
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        if stream.status != "active":
            delete_path(frame_path)
            _raise(409, "stream_stopped", f"Stream '{stream_id}' is stopped.")

        frame = FrameRecord(
            sequence=stream.next_sequence,
            created_at=time.time(),
            path=frame_path,
            width=image.width,
            height=image.height,
            detections=detections,
            detection_model=snapshot["model"],
            labels=snapshot["labels"],
        )
        stream.next_sequence += 1
        stream.frames.append(frame)
        stream.last_processed_at = frame.created_at
        stream.updated_at = frame.created_at
        prune_stream_frames(stream, frame.created_at)
        stop_result = evaluate_stop_conditions_locked(stream, frame)
        if stop_result:
            event_payload = stop_result["payload"]
        elif frame.detections:
            event_payload = stream_detection_event_payload(stream, frame)
        payload = {
            "status": "ok",
            "stream": serialize_stream(stream),
            "frame": {
                **stream_frame_reference(stream.id, frame),
                "detection_count": len(frame.detections),
            },
            "detections": detection_text_payload(
                frame.detections,
                frame.width,
                frame.height,
            ),
        }

    if event_payload:
        await publish_stream_event(stream_id, event_payload)
    if stop_result and stop_result["callback_url"]:
        asyncio.create_task(send_callback(stop_result["callback_url"], stop_result["payload"]))
    return payload


async def image_from_request_body(request: Request) -> Image.Image:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image") or form.get("file")
        if upload is None or not hasattr(upload, "read"):
            _raise(400, "missing_image", "multipart/form-data must include image or file.")
        image_bytes = await upload.read()
    else:
        image_bytes = await request.body()
    if not image_bytes:
        _raise(400, "missing_image", "Image body is required.")
    return decode_image_bytes(image_bytes)


async def parse_apply_request(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        labels = form.getlist("labels") or form.getlist("label")
        if not labels and form.get("labels"):
            labels = [str(form.get("labels"))]
        upload = form.get("image") or form.get("file")
        image = None
        if upload is not None and hasattr(upload, "read"):
            image = decode_image_bytes(await upload.read())
        return {
            "model": form.get("model"),
            "output": form.get("output", "text"),
            "labels": labels,
            "sensitivity": form.get("sensitivity", 0.5),
            "max_objects": form.get("max_objects", 10),
            "source_url": form.get("source_url") or form.get("url"),
            "stream_id": form.get("stream_id"),
            "camera_code": form.get("camera_code") or form.get("camera"),
            "index": form.get("index"),
            "age_ms": form.get("age_ms"),
            "frame_sequence": form.get("frame_sequence") or form.get("frame_id"),
            "image": image,
        }

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return payload


def parse_frame_sequence_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    raw = str(value).strip()
    if ":" in raw:
        raw = raw.rsplit(":", 1)[1]
    return int(raw)


async def image_from_apply_payload(payload: dict[str, Any]) -> Image.Image:
    if isinstance(payload.get("image"), Image.Image):
        return payload["image"]

    if payload.get("source_url"):
        return decode_image_bytes(await fetch_image_url(str(payload["source_url"])))

    if payload.get("stream_id"):
        stream_id = str(payload["stream_id"])
        index = payload.get("index")
        age_ms = payload.get("age_ms")
        frame_sequence = parse_frame_sequence_value(
            payload.get("frame_sequence") or payload.get("frame_id")
        )
        index_value = int(index) if index not in (None, "") else None
        age_value = int(age_ms) if age_ms not in (None, "") else None
        async with STATE_LOCK:
            stream = STREAMS.get(stream_id)
            if stream is None:
                _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
            if frame_sequence is not None:
                frame = resolve_frame_by_sequence(stream.frames, frame_sequence)
            else:
                frame, _ = resolve_frame_from_stream(stream, index_value, age_value)
            path = frame.path
        with Image.open(path) as image:
            return image.convert("RGB")

    camera_code = payload.get("camera_code") or payload.get("camera")
    if camera_code:
        code = str(camera_code)
        index = payload.get("index")
        age_ms = payload.get("age_ms")
        frame_sequence = parse_frame_sequence_value(
            payload.get("frame_sequence") or payload.get("frame_id")
        )
        index_value = int(index) if index not in (None, "") else None
        age_value = int(age_ms) if age_ms not in (None, "") else None
        async with STATE_LOCK:
            camera = CAMERA_CODES.get(code)
            if camera is None:
                _raise(404, "unknown_camera_code", f"Camera code '{code}' is not active.")
            if frame_sequence is not None:
                frame = resolve_frame_by_sequence(camera.frames, frame_sequence)
            else:
                frame, _ = resolve_camera_frame(camera, index_value, age_value)
            path = frame.path
        with Image.open(path) as image:
            return image.convert("RGB")

    _raise(400, "missing_source", "Provide an image, source_url, stream_id, or camera_code.")


def parse_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return sanitize_keywords([value])
    if isinstance(value, list):
        return sanitize_keywords([str(item) for item in value])
    return sanitize_keywords([str(value)])


async def cleanup_inactive_codes() -> None:
    now = time.time()
    async with STATE_LOCK:
        for code, state in list(CAMERA_CODES.items()):
            if now - state.last_seen_at > CAMERA_CODE_INACTIVITY_SECONDS:
                for frame in state.frames:
                    delete_path(frame.path)
                CAMERA_CODES.pop(code, None)


@app.get("/")
async def root(request: Request):
    if is_camera_host(request):
        return Response(status_code=404, content="Unknown camera code")
    return FileResponse(DEBUG_HTML_FILE, media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "vision"}


@app.get("/api/models")
async def list_models():
    return {"models": model_catalog()}


@app.get("/api/models/status")
async def models_status():
    return {
        "models": model_catalog(),
        "loaded_models": get_loaded_detection_backends() + get_loaded_depth_backends(),
        "vram": vram_status(),
    }


@app.post("/api/models/{model_id}/load")
async def load_model_endpoint(model_id: str):
    kind, normalized = normalize_model_id(model_id)
    async with MODEL_LOCK:
        if kind == "detection":
            await asyncio.to_thread(select_detection_backend, normalized)
        else:
            await asyncio.to_thread(select_depth_backend, normalized)
    return {"status": "ok", "model": normalized, "kind": kind, "vram": vram_status()}


@app.post("/api/models/{model_id}/unload")
async def unload_model_endpoint(model_id: str):
    kind, normalized = normalize_model_id(model_id)
    async with MODEL_LOCK:
        if kind == "detection":
            await asyncio.to_thread(unload_detection_backend, normalized)
        else:
            await asyncio.to_thread(unload_depth_backend, normalized)

    stop_results = []
    if kind == "detection":
        async with STATE_LOCK:
            for stream in STREAMS.values():
                if stream.detection_model == normalized:
                    stop_result = stop_stream_locked(stream, "model_unloaded")
                    if stop_result:
                        stop_results.append(stop_result)

    for stop_result in stop_results:
        asyncio.create_task(publish_stream_event(stop_result["stream_id"], stop_result["payload"]))
        if stop_result["callback_url"]:
            asyncio.create_task(
                send_callback(stop_result["callback_url"], stop_result["payload"])
            )

    return {"status": "ok", "model": normalized, "kind": kind, "vram": vram_status()}


@app.get("/api/streams")
async def list_streams(request: Request):
    async with STATE_LOCK:
        return {"streams": [serialize_stream(stream, request) for stream in STREAMS.values()]}


async def stream_events_websocket(websocket: WebSocket, stream_id: str | None = None) -> None:
    if is_camera_host(websocket):
        await websocket.close(code=1008)
        return

    stream_exists = True
    if stream_id is not None:
        async with STATE_LOCK:
            stream_exists = stream_id in STREAMS

    await websocket.accept()
    if not stream_exists:
        await websocket.send_json(
            {
                "event": "error",
                "error": "unknown_stream",
                "message": f"Unknown stream '{stream_id}'.",
            }
        )
        await websocket.close(code=1008)
        return

    key = stream_id or ALL_STREAMS_SUBSCRIPTION
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
    async with SUBSCRIBER_LOCK:
        STREAM_EVENT_SUBSCRIBERS.setdefault(key, set()).add(queue)

    try:
        await websocket.send_json(
            {
                "event": "subscribed",
                "stream": stream_id,
                "scope": "stream" if stream_id else "all_streams",
            }
        )
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                payload = {
                    "event": "heartbeat",
                    "stream": stream_id,
                    "created_at": time.time(),
                }
            await websocket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        async with SUBSCRIBER_LOCK:
            subscribers = STREAM_EVENT_SUBSCRIBERS.get(key)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    STREAM_EVENT_SUBSCRIBERS.pop(key, None)


@app.websocket("/api/streams/events")
async def all_stream_events_socket(websocket: WebSocket):
    await stream_events_websocket(websocket)


@app.websocket("/api/streams/{stream_id}/events")
async def stream_events_socket(stream_id: str, websocket: WebSocket):
    await stream_events_websocket(websocket, stream_id)


@app.post("/api/streams")
async def create_stream(payload: StreamCreatePayload, request: Request):
    kind, model_id = normalize_model_id(payload.detection_model)
    if kind != "detection":
        _raise(400, "invalid_stream_model", "Streams can only use detection models.")
    if not is_detection_backend_loaded(model_id):
        _raise(409, "model_not_loaded", f"Detection model '{model_id}' is not loaded.")

    await cleanup_inactive_codes()
    async with STATE_LOCK:
        if payload.id in STREAMS:
            _raise(409, "stream_exists", f"Stream '{payload.id}' already exists.")
        if payload.stream and payload.stream not in CAMERA_CODES:
            _raise(400, "invalid_camera_code", f"Camera code '{payload.stream}' is not active.")

        stream = StreamState(
            id=payload.id,
            detection_model=model_id,
            labels=payload.labels,
            sensitivity=payload.sensitivity,
            max_objects=payload.max_objects,
            process_every_ms=payload.process_every_ms,
            stop_conditions=payload.stop_conditions,
            callback_url=payload.callback_url,
            source_code=payload.stream,
        )
        STREAMS[stream.id] = stream
        return {"status": "ok", "stream": serialize_stream(stream, request)}


@app.patch("/api/streams/{stream_id}")
async def update_stream(stream_id: str, payload: StreamUpdatePayload, request: Request):
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")

        if payload.detection_model is not None:
            kind, model_id = normalize_model_id(payload.detection_model)
            if kind != "detection":
                _raise(400, "invalid_stream_model", "Streams can only use detection models.")
            if stream.status == "active" and not is_detection_backend_loaded(model_id):
                _raise(409, "model_not_loaded", f"Detection model '{model_id}' is not loaded.")
            stream.detection_model = model_id

        if payload.labels is not None:
            stream.labels = payload.labels
        if payload.sensitivity is not None:
            stream.sensitivity = payload.sensitivity
        if payload.max_objects is not None:
            stream.max_objects = payload.max_objects
        if payload.process_every_ms is not None:
            stream.process_every_ms = payload.process_every_ms
        if payload.stop_conditions is not None:
            stream.stop_conditions = payload.stop_conditions
            stream.stop_streaks.clear()
        if payload.callback_url is not None:
            stream.callback_url = payload.callback_url
        if payload.stream is not None:
            if payload.stream and payload.stream not in CAMERA_CODES:
                _raise(400, "invalid_camera_code", f"Camera code '{payload.stream}' is not active.")
            stream.source_code = payload.stream or None

        stream.updated_at = time.time()
        return {"status": "ok", "stream": serialize_stream(stream, request)}


@app.delete("/api/streams/{stream_id}")
async def delete_stream(stream_id: str):
    async with STATE_LOCK:
        stream = STREAMS.pop(stream_id, None)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        for frame in stream.frames:
            delete_path(frame.path)
    return {"status": "ok", "stream": stream_id}


@app.post("/api/streams/{stream_id}/frames")
async def ingest_stream_frame(stream_id: str, request: Request):
    image = await image_from_request_body(request)
    return await ingest_frame_for_stream(stream_id, image)


@app.post("/api/streams/{stream_id}/url")
async def ingest_stream_url(stream_id: str, payload: dict[str, str]):
    url = payload.get("url")
    if not url:
        _raise(400, "missing_url", "url is required.")
    image = decode_image_bytes(await fetch_image_url(url))
    return await ingest_frame_for_stream(stream_id, image)


@app.get("/api/streams/{stream_id}/frames/query")
async def query_stream_frame(
    stream_id: str,
    index: int | None = Query(default=None, ge=0),
    age_ms: int | None = Query(default=None, ge=0),
):
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        frame, warning = resolve_frame_from_stream(stream, index, age_ms)
        return stream_frame_query_payload(stream_id, frame, warning)


@app.get("/api/streams/{stream_id}/frames/raw")
async def query_stream_frame_raw(
    stream_id: str,
    request: Request,
    index: int | None = Query(default=None, ge=0),
    age_ms: int | None = Query(default=None, ge=0),
):
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        frame, warning = resolve_frame_from_stream(stream, index, age_ms)
        temp = create_temp_url(frame.path, "frame", request)
        payload = {
            "stream": stream_id,
            "frame": stream_frame_reference(stream_id, frame),
            "image": temp,
        }
        if warning:
            payload["warning"] = warning
        return payload


@app.get("/api/streams/{stream_id}/frames/{frame_sequence}/query")
async def query_stream_frame_by_sequence(stream_id: str, frame_sequence: int):
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        frame = resolve_frame_by_sequence(stream.frames, frame_sequence)
        return stream_frame_query_payload(stream_id, frame)


@app.get("/api/streams/{stream_id}/frames/{frame_sequence}/raw")
async def query_stream_frame_raw_by_sequence(
    stream_id: str,
    frame_sequence: int,
    request: Request,
):
    async with STATE_LOCK:
        stream = STREAMS.get(stream_id)
        if stream is None:
            _raise(404, "unknown_stream", f"Unknown stream '{stream_id}'.")
        frame = resolve_frame_by_sequence(stream.frames, frame_sequence)
        temp = create_temp_url(frame.path, "frame", request)
        return {
            "stream": stream_id,
            "frame": stream_frame_reference(stream_id, frame),
            "image": temp,
        }


@app.post("/api/apply")
async def apply_model(request: Request):
    payload = await parse_apply_request(request)
    model_id_raw = payload.get("model")
    output = str(payload.get("output") or "text").strip().lower()
    if not model_id_raw:
        _raise(400, "missing_model", "model is required.")
    if output not in {"text", "image"}:
        _raise(400, "invalid_output", "output must be text or image.")

    kind, model_id = normalize_model_id(str(model_id_raw))
    if not model_is_loaded(kind, model_id):
        _raise(409, "model_not_loaded", f"Model '{model_id}' is not loaded.")

    image = await image_from_apply_payload(payload)

    if kind == "detection":
        labels = parse_labels(payload.get("labels") or payload.get("label") or payload.get("keywords"))
        if not labels:
            _raise(400, "missing_labels", "Detection models require labels.")
        sensitivity = float(payload.get("sensitivity") or 0.5)
        max_objects = int(payload.get("max_objects") or 10)
        detections = await run_detection(image, model_id, labels, sensitivity, max_objects)
        if output == "text":
            return {
                "status": "ok",
                "model": model_id,
                "output": "text",
                "result": detection_text_payload(detections, image.width, image.height),
            }
        annotated = draw_detections(image, detections)
        return {
            "status": "ok",
            "model": model_id,
            "output": "image",
            "image": register_generated_image(annotated, request),
        }

    if output != "image":
        _raise(400, "unsupported_output", f"Model '{model_id}' does not support text output.")

    depth_map = await run_depth(image, model_id)
    depth_image = depth_map_to_image(depth_map)
    return {
        "status": "ok",
        "model": model_id,
        "output": "image",
        "image": register_generated_image(depth_image, request),
    }


@app.get("/api/camera-codes")
async def list_camera_codes():
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        return {"codes": [serialize_code(code) for code in CAMERA_CODES.values()]}


@app.post("/api/camera-codes")
async def create_camera_code(payload: CameraCodePayload):
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        if payload.code and payload.code in CAMERA_CODES:
            _raise(409, "camera_code_exists", f"Camera code '{payload.code}' is already active.")
        code = payload.code or secrets.token_hex(5)
        while not payload.code and code in CAMERA_CODES:
            code = secrets.token_hex(5)
        CAMERA_CODES[code] = CameraCodeState(code=code, capture_fps=payload.capture_fps)
        return {"status": "ok", "camera": serialize_code(CAMERA_CODES[code])}


@app.get("/api/camera-codes/{code}/frames/query")
async def query_camera_code_frame(
    code: str,
    index: int | None = Query(default=None, ge=0),
    age_ms: int | None = Query(default=None, ge=0),
):
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            _raise(404, "unknown_camera_code", f"Camera code '{code}' is not active.")
        frame, warning = resolve_camera_frame(camera, index, age_ms)
        return camera_frame_query_payload(code, frame, warning)


@app.get("/api/camera-codes/{code}/frames/raw")
async def query_camera_code_frame_raw(
    code: str,
    request: Request,
    index: int | None = Query(default=None, ge=0),
    age_ms: int | None = Query(default=None, ge=0),
):
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            _raise(404, "unknown_camera_code", f"Camera code '{code}' is not active.")
        frame, warning = resolve_camera_frame(camera, index, age_ms)
        temp = create_temp_url(frame.path, "frame", request)
        payload = {
            "code": code,
            "frame": camera_frame_reference(code, frame),
            "image": temp,
        }
        if warning:
            payload["warning"] = warning
        return payload


@app.get("/api/camera-codes/{code}/frames/{frame_sequence}/query")
async def query_camera_code_frame_by_sequence(code: str, frame_sequence: int):
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            _raise(404, "unknown_camera_code", f"Camera code '{code}' is not active.")
        frame = resolve_frame_by_sequence(camera.frames, frame_sequence)
        return camera_frame_query_payload(code, frame)


@app.get("/api/camera-codes/{code}/frames/{frame_sequence}/raw")
async def query_camera_code_frame_raw_by_sequence(
    code: str,
    frame_sequence: int,
    request: Request,
):
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            _raise(404, "unknown_camera_code", f"Camera code '{code}' is not active.")
        frame = resolve_frame_by_sequence(camera.frames, frame_sequence)
        temp = create_temp_url(frame.path, "frame", request)
        return {
            "code": code,
            "frame": camera_frame_reference(code, frame),
            "image": temp,
        }


@app.delete("/api/camera-codes/{code}")
async def delete_camera_code(code: str):
    async with STATE_LOCK:
        camera = CAMERA_CODES.pop(code, None)
        if camera is None:
            _raise(404, "unknown_camera_code", f"Unknown camera code '{code}'.")
        for frame in camera.frames:
            delete_path(frame.path)
    return {"status": "ok", "code": code}


@app.post("/camera/{code}/frame")
async def ingest_camera_code_frame(code: str, request: Request):
    image = await image_from_request_body(request)

    await cleanup_inactive_codes()
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            _raise(404, "unknown_camera_code", f"Camera code '{code}' is not active.")
        camera.last_seen_at = time.time()
        target_stream_ids = [
            stream.id
            for stream in STREAMS.values()
            if stream.source_code == code and stream.status == "active"
        ]

    camera_frame_path = save_image(image, f"camera-{code}")
    camera_frame = None
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            delete_path(camera_frame_path)
        else:
            camera_frame = CameraFrameRecord(
                sequence=camera.next_sequence,
                created_at=time.time(),
                path=camera_frame_path,
                width=image.width,
                height=image.height,
            )
            camera.next_sequence += 1
            camera.frames.append(camera_frame)
            camera.last_seen_at = camera_frame.created_at
            prune_camera_frames(camera, camera_frame.created_at)

    results = []
    for stream_id in target_stream_ids:
        try:
            results.append(await ingest_frame_for_stream(stream_id, image.copy()))
        except HTTPException as exc:
            results.append({"stream": stream_id, "status": "error", "detail": exc.detail})

    return {
        "status": "ok",
        "code": code,
        "frame": camera_frame_reference(code, camera_frame) if camera_frame else None,
        "stream_count": len(target_stream_ids),
        "results": results,
    }


@app.get("/tmp/{token}")
async def get_temp_image(token: str):
    async with STATE_LOCK:
        prune_temp_urls()
        state = TEMP_URLS.get(token)
        if state is None or not state.path.exists():
            return Response(status_code=404, content="Temporary image not found")
        path = state.path
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/{code}")
async def camera_page(code: str, request: Request):
    if not (is_camera_host(request) or is_local_host(request)):
        return Response(status_code=404, content="Not found")
    await cleanup_inactive_codes()
    async with STATE_LOCK:
        camera = CAMERA_CODES.get(code)
        if camera is None:
            return Response(status_code=404, content="Unknown or expired camera code")
        camera.last_seen_at = time.time()
        interval_ms = int(1000 / max(camera.capture_fps, 0.001))

    template = CAMERA_HTML_FILE.read_text(encoding="utf-8")
    html = (
        template.replace("__CODE__", code)
        .replace("__CAPTURE_INTERVAL_MS__", str(interval_ms))
        .replace("__MAX_IMAGE_EDGE__", str(MAX_IMAGE_EDGE))
    )
    return HTMLResponse(html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
