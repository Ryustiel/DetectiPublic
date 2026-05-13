from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Any, Sequence

os.environ.setdefault("HF_HOME", "/app/assets/models/")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

from processor.angle import (
    CameraCalibration,
    bearing_angle_for_detection,
    detection_center,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1

DEPTH_BACKEND_ZOE = "zoedepth"
DEPTH_BACKEND_DEPTH_ANYTHING_V2 = "depth_anything_v2"
DEPTH_BACKEND_NONE = "none"


@dataclass(frozen=True)
class DepthBackendConfig:
    id: str
    model_id: str
    display_name: str
    is_metric: bool
    description: str


AVAILABLE_DEPTH_BACKENDS: dict[str, DepthBackendConfig] = {
    DEPTH_BACKEND_NONE: DepthBackendConfig(
        id=DEPTH_BACKEND_NONE,
        model_id="",
        display_name="Depth Disabled",
        is_metric=False,
        description="Disable depth estimation to free VRAM. Angle estimation remains available.",
    ),
    DEPTH_BACKEND_ZOE: DepthBackendConfig(
        id=DEPTH_BACKEND_ZOE,
        model_id=os.getenv("ZOEDEPTH_MODEL_ID", "Intel/zoedepth-nyu-kitti"),
        display_name="ZoeDepth",
        is_metric=True,
        description="Metric depth. Detection results include distance in meters.",
    ),
    DEPTH_BACKEND_DEPTH_ANYTHING_V2: DepthBackendConfig(
        id=DEPTH_BACKEND_DEPTH_ANYTHING_V2,
        model_id=os.getenv(
            "DEPTH_ANYTHING_MODEL_ID",
            "depth-anything/Depth-Anything-V2-Base-hf",
        ),
        display_name="Depth Anything V2",
        is_metric=False,
        description="Relative depth only. Detection results include a relative depth value, not meters.",
    ),
}

_DEPTH_BACKEND_ALIASES = {
    "none": DEPTH_BACKEND_NONE,
    "off": DEPTH_BACKEND_NONE,
    "disabled": DEPTH_BACKEND_NONE,
    "zoe": DEPTH_BACKEND_ZOE,
    "zoedepth": DEPTH_BACKEND_ZOE,
    "depthanything": DEPTH_BACKEND_DEPTH_ANYTHING_V2,
    "depth_anything": DEPTH_BACKEND_DEPTH_ANYTHING_V2,
    "depth-anything": DEPTH_BACKEND_DEPTH_ANYTHING_V2,
    "depth_anything_v2": DEPTH_BACKEND_DEPTH_ANYTHING_V2,
    "depth-anything-v2": DEPTH_BACKEND_DEPTH_ANYTHING_V2,
}


def normalize_depth_backend(value: str | None) -> str:
    raw = str(value or DEPTH_BACKEND_NONE).strip().lower().replace("-", "_").replace(" ", "_")
    raw = _DEPTH_BACKEND_ALIASES.get(raw, raw)
    if raw not in AVAILABLE_DEPTH_BACKENDS:
        supported = ", ".join(AVAILABLE_DEPTH_BACKENDS.keys())
        raise ValueError(f"Unsupported depth backend '{value}'. Supported values: {supported}")
    return raw


try:
    DEFAULT_DEPTH_BACKEND = normalize_depth_backend(
        os.getenv("DEFAULT_DEPTH_BACKEND", DEPTH_BACKEND_NONE)
    )
except ValueError:
    DEFAULT_DEPTH_BACKEND = DEPTH_BACKEND_NONE

def get_available_depth_backends() -> list[dict[str, Any]]:
    return [
        {
            "id": config.id,
            "label": (
                config.display_name
                if config.id == DEPTH_BACKEND_NONE
                else config.display_name + (" (metric)" if config.is_metric else " (relative)")
            ),
            "metric": config.is_metric,
            "disabled": config.id == DEPTH_BACKEND_NONE,
            "description": config.description,
            "model_id": config.model_id,
        }
        for config in AVAILABLE_DEPTH_BACKENDS.values()
    ]


class LoadedDepthBackend:
    def __init__(self, config: DepthBackendConfig):
        self.config = config
        self.pipeline = None
        self.device: str | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def _load_pipeline(self, device: int):
        return pipeline(
            task="depth-estimation",
            model=self.config.model_id,
            device=device,
        )

    def load(self):
        if self.pipeline is not None:
            return

        print(
            f"Loading {self.config.display_name} "
            f"({'metric' if self.config.is_metric else 'relative'}) "
            f"on {DEVICE} using {self.config.model_id}..."
        )

        try:
            self.pipeline = self._load_pipeline(PIPELINE_DEVICE)
            self.device = DEVICE
        except Exception as exc:
            self.pipeline = None
            self.device = None

            if torch.cuda.is_available():
                print(f"Failed to load {self.config.display_name} on CUDA: {exc}")
                print(f"Retrying {self.config.display_name} on CPU...")
                try:
                    self.pipeline = self._load_pipeline(-1)
                    self.device = "cpu"
                    print(f"{self.config.display_name} CPU fallback loaded.")
                except Exception as cpu_exc:
                    self.pipeline = None
                    self.device = None
                    raise RuntimeError(
                        f"Failed to load {self.config.display_name}: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise RuntimeError(
                    f"Failed to load {self.config.display_name}: {exc}"
                ) from exc

    def unload(self):
        if self.pipeline is None:
            return

        print(f"Unloading {self.config.display_name}...")
        self.pipeline = None
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_depth_map(self, image: Image.Image) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.config.display_name} model failed to load.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        with torch.inference_mode():
            outputs = self.pipeline(image)

        if not isinstance(outputs, dict):
            raise RuntimeError(
                f"Unexpected {self.config.display_name} output type: {type(outputs)!r}"
            )

        predicted_depth = outputs.get("predicted_depth")
        if predicted_depth is None:
            raise RuntimeError(
                f"{self.config.display_name} pipeline did not return 'predicted_depth'. "
                f"Available keys: {list(outputs.keys())}"
            )

        if isinstance(predicted_depth, torch.Tensor):
            depth_map = predicted_depth.detach().float().cpu().numpy()
        else:
            depth_map = np.asarray(predicted_depth, dtype=np.float32)

        depth_map = np.squeeze(depth_map).astype(np.float32, copy=False)
        if depth_map.ndim != 2:
            raise RuntimeError(
                f"Unexpected {self.config.display_name} depth shape: {depth_map.shape}"
            )

        expected_width, expected_height = image.size
        if depth_map.shape != (expected_height, expected_width):
            depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)
            depth_map = (
                F.interpolate(
                    depth_tensor,
                    size=(expected_height, expected_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        depth_map[~np.isfinite(depth_map)] = np.nan
        depth_map[depth_map <= 0] = np.nan

        if not np.isfinite(depth_map).any():
            raise RuntimeError(
                f"{self.config.display_name} returned no valid positive depth values."
            )

        return depth_map


class DepthProcessor:
    def __init__(self):
        self._lock = threading.RLock()
        self._backends: dict[str, LoadedDepthBackend] = {
            backend_id: LoadedDepthBackend(config)
            for backend_id, config in AVAILABLE_DEPTH_BACKENDS.items()
        }

        print("Depth backend manager ready.")
        print(f"Default depth backend: {DEFAULT_DEPTH_BACKEND}")
        print(
            f"Available depth backends: {', '.join(AVAILABLE_DEPTH_BACKENDS.keys())}"
        )
        print("Depth backends are loaded only through explicit API calls.")

    def ensure_backend_loaded(self, backend_name: str) -> LoadedDepthBackend:
        normalized_backend = normalize_depth_backend(backend_name)

        with self._lock:
            backend = self._backends[normalized_backend]
            if normalized_backend != DEPTH_BACKEND_NONE and not backend.loaded:
                backend.load()

            return backend

    def unload_backend(self, backend_name: str) -> None:
        normalized_backend = normalize_depth_backend(backend_name)
        with self._lock:
            self._backends[normalized_backend].unload()

    def is_backend_loaded(self, backend_name: str) -> bool:
        normalized_backend = normalize_depth_backend(backend_name)
        with self._lock:
            return self._backends[normalized_backend].loaded

    def loaded_backend_ids(self) -> list[str]:
        with self._lock:
            return [
                backend_id
                for backend_id, backend in self._backends.items()
                if backend.loaded
            ]

    def _estimate_depth_value(
        self,
        depth_map: np.ndarray | None,
        detection: dict,
    ) -> float | None:
        if depth_map is None:
            return None

        image_height, image_width = depth_map.shape[:2]
        x = float(detection.get("x", 0.0))
        y = float(detection.get("y", 0.0))
        w = max(0.0, float(detection.get("w", 0.0)))
        h = max(0.0, float(detection.get("h", 0.0)))

        x1 = int(np.clip(np.floor(x), 0, image_width - 1))
        y1 = int(np.clip(np.floor(y), 0, image_height - 1))
        x2 = int(np.clip(np.ceil(x + w), x1 + 1, image_width))
        y2 = int(np.clip(np.ceil(y + h), y1 + 1, image_height))

        inner_margin_x = max(1, int((x2 - x1) * 0.25))
        inner_margin_y = max(1, int((y2 - y1) * 0.25))
        inner_x1 = min(x2 - 1, x1 + inner_margin_x)
        inner_y1 = min(y2 - 1, y1 + inner_margin_y)
        inner_x2 = max(inner_x1 + 1, x2 - inner_margin_x)
        inner_y2 = max(inner_y1 + 1, y2 - inner_margin_y)

        patch = depth_map[inner_y1:inner_y2, inner_x1:inner_x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]

        if valid.size < 9:
            patch = depth_map[y1:y2, x1:x2]
            valid = patch[np.isfinite(patch) & (patch > 0)]

        if valid.size == 0:
            center_x, center_y = detection_center(detection)
            px = int(np.clip(round(center_x), 0, image_width - 1))
            py = int(np.clip(round(center_y), 0, image_height - 1))
            center_value = depth_map[py, px]
            if not np.isfinite(center_value) or center_value <= 0:
                return None
            return float(center_value)

        return float(np.nanmedian(valid))

    def predict(
        self,
        image: Image.Image,
        detections: Sequence[dict],
        camera_fov_deg: float | None = None,
        distortion: dict | None = None,
        depth_backend: str | None = None,
    ) -> list[dict]:
        backend_name = normalize_depth_backend(depth_backend or DEFAULT_DEPTH_BACKEND)
        backend_config = AVAILABLE_DEPTH_BACKENDS[backend_name]

        calibration = CameraCalibration.from_values(
            fov_deg=camera_fov_deg,
            distortion=distortion,
        )

        depth_map = None
        if backend_name == DEPTH_BACKEND_NONE:
            self.ensure_backend_loaded(backend_name)
        else:
            try:
                with self._lock:
                    backend = self._backends[backend_name]
                    if not backend.loaded:
                        raise RuntimeError(
                            f"Depth backend '{backend_name}' is not loaded."
                        )
                depth_map = backend.predict_depth_map(image)
            except Exception as exc:
                print(
                    f"Depth inference failed for backend '{backend_name}', "
                    f"falling back to angle-only enrichment: {exc}"
                )

        image_width, image_height = image.size
        results: list[dict] = []

        for detection in detections:
            center_x, center_y = detection_center(detection)
            depth_value = self._estimate_depth_value(depth_map, detection)

            distance_m = None
            if backend_config.is_metric:
                distance_m = calibration.range_from_z_depth_m(
                    depth_value,
                    center_x,
                    center_y,
                    image_width,
                    image_height,
                )

            results.append(
                {
                    "distance_m": distance_m,
                    "depth_value": depth_value,
                    "depth_units": "m" if backend_config.is_metric else "relative",
                    "depth_is_metric": backend_config.is_metric,
                    "depth_backend": backend_name,
                    "angle_deg": bearing_angle_for_detection(
                        calibration,
                        detection,
                        image_width,
                        image_height,
                    ),
                }
            )

        return results


_depth_processor: DepthProcessor | None = None


def load_depth_processor() -> DepthProcessor:
    global _depth_processor
    if _depth_processor is None:
        _depth_processor = DepthProcessor()
    return _depth_processor


def select_depth_backend(depth_backend: str | None = None) -> str:
    selected_backend = normalize_depth_backend(depth_backend or DEFAULT_DEPTH_BACKEND)
    load_depth_processor().ensure_backend_loaded(selected_backend)
    return selected_backend


def unload_depth_backend(depth_backend: str | None = None) -> str:
    selected_backend = normalize_depth_backend(depth_backend or DEFAULT_DEPTH_BACKEND)
    load_depth_processor().unload_backend(selected_backend)
    return selected_backend


def get_loaded_depth_backends() -> list[str]:
    return load_depth_processor().loaded_backend_ids()


def is_depth_backend_loaded(depth_backend: str | None = None) -> bool:
    selected_backend = normalize_depth_backend(depth_backend or DEFAULT_DEPTH_BACKEND)
    return load_depth_processor().is_backend_loaded(selected_backend)


def predict_depth_map(
    image: Image.Image,
    depth_backend: str | None = None,
) -> np.ndarray:
    selected_backend = normalize_depth_backend(depth_backend or DEFAULT_DEPTH_BACKEND)
    if selected_backend == DEPTH_BACKEND_NONE:
        raise RuntimeError("Depth backend 'none' cannot produce a depth map.")

    processor = load_depth_processor()
    with processor._lock:
        backend = processor._backends[selected_backend]
        if not backend.loaded:
            raise RuntimeError(f"Depth backend '{selected_backend}' is not loaded.")

    return backend.predict_depth_map(image)


def enrich_detections(
    image: Image.Image,
    detections: Sequence[dict],
    camera_fov_deg: float | None = None,
    distortion: dict | None = None,
    depth_backend: str | None = None,
) -> list[dict]:
    if _depth_processor is None:
        load_depth_processor()

    selected_backend = normalize_depth_backend(depth_backend or DEFAULT_DEPTH_BACKEND)
    if not detections:
        if selected_backend == DEPTH_BACKEND_NONE:
            _depth_processor.ensure_backend_loaded(selected_backend)
        return []

    depth_outputs = _depth_processor.predict(
        image=image,
        detections=detections,
        camera_fov_deg=camera_fov_deg,
        distortion=distortion,
        depth_backend=selected_backend,
    )

    enriched = []
    for index, detection in enumerate(detections):
        extra = depth_outputs[index] if index < len(depth_outputs) else {}
        enriched.append(
            {
                **detection,
                "distance_m": extra.get("distance_m"),
                "depth_value": extra.get("depth_value"),
                "depth_units": extra.get("depth_units"),
                "depth_is_metric": extra.get("depth_is_metric"),
                "depth_backend": extra.get("depth_backend"),
                "angle_deg": extra.get("angle_deg"),
            }
        )

    return enriched
