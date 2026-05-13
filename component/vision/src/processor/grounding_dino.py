from __future__ import annotations

import gc
import json
import os
import re
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

os.environ.setdefault("HF_HOME", "/app/assets/models")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    import tensorrt as trt
except ImportError:
    trt = None

CACHE_DIR = os.getenv("HF_HOME", "/app/assets/models")
COMPILED_TRT_DIR = Path(
    os.getenv("GROUNDING_DINO_TRT_DIR", "/app/assets/models/compiled/gdinoonnx")
)
MODEL_ID = os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_KEYWORDS = 10
DEFAULT_WARMUP_PROMPT = "human."

DETECTION_BACKEND_NONE = "none"
DETECTION_BACKEND_KIND_DISABLED = "disabled"
DETECTION_BACKEND_HF = "grounding_dino"
DETECTION_BACKEND_KIND_HF = "huggingface"
DETECTION_BACKEND_KIND_TRT = "tensorrt"

_TRT_TO_TORCH_DTYPE = (
    {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if trt is not None
    else {}
)

_IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass(frozen=True)
class DetectionBackendConfig:
    id: str
    kind: str
    label: str
    description: str
    enabled: bool
    optimized: bool
    model_id: str | None = None
    engine_path: Path | None = None
    metadata_path: Path | None = None
    image_height: int | None = None
    image_width: int | None = None
    seq_len: int | None = None
    precision: str | None = None
    warmup_prompt: str | None = None


def normalize_keyword(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().strip(".")).strip()


def sanitize_keywords(values: Sequence[str] | None) -> list[str]:
    seen = set()
    cleaned: list[str] = []

    for value in values or []:
        if value is None:
            continue

        for part in str(value).replace(";", ",").split(","):
            keyword = normalize_keyword(part)
            if not keyword or keyword in seen:
                continue

            seen.add(keyword)
            cleaned.append(keyword)

            if len(cleaned) >= MAX_KEYWORDS:
                return cleaned

    return cleaned


def build_prompt(keywords: Sequence[str]) -> str:
    cleaned = sanitize_keywords(keywords)
    if not cleaned:
        return ""
    return " . ".join(cleaned) + " ."


def sensitivity_to_thresholds(sensitivity: float) -> tuple[float, float]:
    value = min(1.0, max(0.0, float(sensitivity)))
    box_threshold = 0.60 - (0.50 * value)
    text_threshold = max(0.05, box_threshold - 0.05)
    return box_threshold, text_threshold


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def match_keyword_index(label: str, keywords: Sequence[str]) -> int | None:
    normalized_label = normalize_keyword(label)
    if not normalized_label:
        return None

    for index, keyword in enumerate(keywords):
        if normalized_label == keyword:
            return index

    for index, keyword in enumerate(keywords):
        if keyword in normalized_label or normalized_label in keyword:
            return index

    label_tokens = set(normalized_label.split())
    best_index = None
    best_overlap = 0

    for index, keyword in enumerate(keywords):
        overlap = len(label_tokens.intersection(set(keyword.split())))
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    return best_index if best_overlap > 0 else None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _build_hf_backend_config() -> DetectionBackendConfig:
    return DetectionBackendConfig(
        id=DETECTION_BACKEND_HF,
        kind=DETECTION_BACKEND_KIND_HF,
        label="GroundingDINO (PyTorch)",
        description="Original Hugging Face GroundingDINO model.",
        enabled=True,
        optimized=False,
        model_id=MODEL_ID,
        warmup_prompt=DEFAULT_WARMUP_PROMPT,
    )


def _build_none_backend_config() -> DetectionBackendConfig:
    return DetectionBackendConfig(
        id=DETECTION_BACKEND_NONE,
        kind=DETECTION_BACKEND_KIND_DISABLED,
        label="Detector Disabled",
        description="Disable object detection until you explicitly enable a backend.",
        enabled=False,
        optimized=False,
    )


def _build_trt_backend_config(metadata_path: Path) -> DetectionBackendConfig | None:
    if trt is None:
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Skipping invalid GroundingDINO metadata {metadata_path}: {exc}")
        return None

    engine_path_raw = str(payload.get("engine_path") or "").strip()
    if not engine_path_raw:
        return None

    engine_path = Path(engine_path_raw)
    if not engine_path.exists():
        print(f"Skipping missing TensorRT engine referenced by {metadata_path}: {engine_path}")
        return None

    image_height = int(payload.get("image_height") or 0)
    image_width = int(payload.get("image_width") or 0)
    seq_len = int(payload.get("seq_len") or 0)
    precision = str(payload.get("precision") or "unknown").lower()
    engine_stem = _slugify(engine_path.stem)

    label = f"GroundingDINO TensorRT {precision.upper()} {image_height}x{image_width} {seq_len} tok"
    description = (
        "TensorRT-optimized GroundingDINO engine discovered from compiled artifacts. "
        f"Static image shape {image_height}x{image_width}. Prompt budget {seq_len} tokens."
    )

    return DetectionBackendConfig(
        id=f"grounding_dino_trt_{engine_stem}",
        kind=DETECTION_BACKEND_KIND_TRT,
        label=label,
        description=description,
        enabled=True,
        optimized=True,
        engine_path=engine_path,
        metadata_path=metadata_path,
        image_height=image_height or None,
        image_width=image_width or None,
        seq_len=seq_len or None,
        precision=precision,
        warmup_prompt=str(payload.get("caption") or DEFAULT_WARMUP_PROMPT),
    )


def _discover_detection_backends() -> dict[str, DetectionBackendConfig]:
    backends: dict[str, DetectionBackendConfig] = {
        DETECTION_BACKEND_NONE: _build_none_backend_config(),
        DETECTION_BACKEND_HF: _build_hf_backend_config(),
    }

    if not COMPILED_TRT_DIR.exists():
        return backends

    for metadata_path in sorted(COMPILED_TRT_DIR.glob("*.metadata.json")):
        config = _build_trt_backend_config(metadata_path)
        if config is not None:
            backends[config.id] = config

    return backends


AVAILABLE_DETECTION_BACKENDS = _discover_detection_backends()

_DETECTION_BACKEND_ALIASES = {
    "none": DETECTION_BACKEND_NONE,
    "off": DETECTION_BACKEND_NONE,
    "disabled": DETECTION_BACKEND_NONE,
    "hf": DETECTION_BACKEND_HF,
    "pytorch": DETECTION_BACKEND_HF,
    "groundingdino": DETECTION_BACKEND_HF,
    "grounding_dino": DETECTION_BACKEND_HF,
    "grounding_dino_hf": DETECTION_BACKEND_HF,
}


def normalize_detection_backend(value: str | None) -> str:
    raw = str(value or DETECTION_BACKEND_HF).strip().lower().replace("-", "_").replace(" ", "_")
    raw = _DETECTION_BACKEND_ALIASES.get(raw, raw)
    if raw not in AVAILABLE_DETECTION_BACKENDS:
        supported = ", ".join(AVAILABLE_DETECTION_BACKENDS.keys())
        raise ValueError(
            f"Unsupported detection backend '{value}'. Supported values: {supported}"
        )
    return raw


try:
    DEFAULT_DETECTION_BACKEND = normalize_detection_backend(
        os.getenv("DEFAULT_DETECTION_BACKEND", DETECTION_BACKEND_NONE)
    )
except ValueError:
    DEFAULT_DETECTION_BACKEND = DETECTION_BACKEND_NONE

def get_available_detection_backends() -> list[dict[str, Any]]:
    return [
        {
            "id": config.id,
            "label": config.label,
            "kind": config.kind,
            "optimized": config.optimized,
            "enabled": config.enabled,
            "description": config.description,
            "model_id": config.model_id,
            "precision": config.precision,
            "seq_len": config.seq_len,
            "image_height": config.image_height,
            "image_width": config.image_width,
            "engine_path": str(config.engine_path) if config.engine_path else None,
        }
        for config in AVAILABLE_DETECTION_BACKENDS.values()
    ]


def _post_process_detection(
    processor,
    outputs,
    input_ids,
    target_size: tuple[int, int],
    box_threshold: float,
    text_threshold: float,
):
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[target_size],
        )[0]
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[target_size],
        )[0]


def _results_to_detections(
    results,
    cleaned_keywords: Sequence[str],
    max_objects: int,
    detection_backend: str,
):
    boxes_tensor = results.get("boxes")
    scores_tensor = results.get("scores")
    labels = results.get("text_labels", results.get("labels", []))

    if boxes_tensor is None or scores_tensor is None:
        return []

    boxes = _to_numpy(boxes_tensor)
    scores = _to_numpy(scores_tensor)

    detections = []
    for index, box in enumerate(boxes):
        raw_label = labels[index] if index < len(labels) else ""
        if isinstance(raw_label, (list, tuple)):
            raw_label = " ".join(str(part) for part in raw_label)
        else:
            raw_label = str(raw_label)

        keyword_index = match_keyword_index(raw_label, cleaned_keywords)
        if keyword_index is None:
            continue

        score = float(scores[index]) if index < len(scores) else 0.0
        x1, y1, x2, y2 = [float(v) for v in box]
        display_label = raw_label or cleaned_keywords[keyword_index]

        detections.append(
            {
                "label": display_label,
                "raw_label": display_label,
                "keyword": cleaned_keywords[keyword_index],
                "keyword_index": keyword_index,
                "score": score,
                "x": x1,
                "y": y1,
                "w": max(0.0, x2 - x1),
                "h": max(0.0, y2 - y1),
                "detection_backend": detection_backend,
            }
        )

    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections[:max_objects]


def _ensure_token_type_ids(tokenized) -> None:
    if "token_type_ids" not in tokenized:
        tokenized["token_type_ids"] = torch.zeros_like(tokenized["input_ids"])


def _generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    idxs = torch.nonzero(special_tokens_mask)
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]
    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list


def _preprocess_trt_image(image: Image.Image, height: int, width: int) -> np.ndarray:
    resized = image.convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - _IMAGE_MEAN) / _IMAGE_STD
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32, copy=False)


class HuggingFaceDetectorRuntime:
    def __init__(self, config: DetectionBackendConfig):
        self.config = config
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        print(f"Loading {self.config.label} on {DEVICE}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR
        ).to(DEVICE)
        self.model.eval()

    def unload(self):
        self.processor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self,
        image: Image.Image,
        cleaned_keywords: Sequence[str],
        sensitivity: float,
        max_objects: int,
    ):
        if self.processor is None or self.model is None:
            raise RuntimeError(f"{self.config.label} is not loaded.")

        prompt = build_prompt(cleaned_keywords)
        box_threshold, text_threshold = sensitivity_to_thresholds(sensitivity)
        width, height = image.size

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = _post_process_detection(
            processor=self.processor,
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_size=(height, width),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return _results_to_detections(
            results,
            cleaned_keywords=cleaned_keywords,
            max_objects=max_objects,
            detection_backend=self.config.id,
        )


class TensorRTDetectorRuntime:
    def __init__(self, config: DetectionBackendConfig):
        self.config = config
        self.processor = None
        self.tokenizer = None
        self.logger = None
        self.runtime = None
        self.engine = None
        self.context = None

    @cached_property
    def input_names(self) -> list[str]:
        if self.engine is None:
            return []
        return [
            name
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(name := self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        ]

    @cached_property
    def output_names(self) -> list[str]:
        if self.engine is None:
            return []
        return [
            name
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(name := self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]

    def load(self):
        if self.context is not None:
            return
        if trt is None:
            raise RuntimeError("TensorRT Python bindings are not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT GroundingDINO requires CUDA.")
        if self.config.engine_path is None:
            raise RuntimeError(f"TensorRT backend '{self.config.id}' is missing an engine path.")

        print(f"Loading {self.config.label} on CUDA from {self.config.engine_path}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("GroundingDINO processor did not expose a tokenizer.")

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        self.runtime = trt.Runtime(self.logger)
        with self.config.engine_path.open("rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.config.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.config.engine_path}")

    def unload(self):
        self.context = None
        self.engine = None
        self.runtime = None
        self.logger = None
        self.processor = None
        self.tokenizer = None
        self.__dict__.pop("input_names", None)
        self.__dict__.pop("output_names", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_text_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        if self.tokenizer is None or self.config.seq_len is None:
            raise RuntimeError(f"TensorRT backend '{self.config.id}' is not ready.")

        raw_tokenized = self.tokenizer([prompt], padding="longest", return_tensors="pt")
        actual_seq_len = int(raw_tokenized["input_ids"].shape[1])
        if actual_seq_len > self.config.seq_len:
            raise RuntimeError(
                f"TensorRT backend '{self.config.label}' supports prompts up to "
                f"{self.config.seq_len} tokens, but the current prompt tokenized to "
                f"{actual_seq_len}. Choose another TensorRT backend or switch back to PyTorch."
            )

        tokenized = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.config.seq_len,
            truncation=False,
            return_tensors="pt",
        )
        _ensure_token_type_ids(tokenized)

        special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        text_token_mask, position_ids, _ = _generate_masks_with_special_tokens_and_transfer_map(
            tokenized,
            special_tokens,
        )

        return {
            "input_ids": tokenized["input_ids"].to(dtype=torch.long),
            "attention_mask": tokenized["attention_mask"].to(dtype=torch.bool),
            "position_ids": position_ids.to(dtype=torch.long),
            "token_type_ids": tokenized["token_type_ids"].to(dtype=torch.long),
            "text_token_mask": text_token_mask.to(dtype=torch.bool),
        }

    def _tensor_from_value(self, name: str, value) -> torch.Tensor:
        if self.engine is None:
            raise RuntimeError(f"TensorRT backend '{self.config.id}' is not loaded.")

        trt_dtype = self.engine.get_tensor_dtype(name)
        torch_dtype = _TRT_TO_TORCH_DTYPE.get(trt_dtype)
        if torch_dtype is None:
            raise RuntimeError(f"Unsupported TensorRT dtype for '{name}': {trt_dtype}")

        if isinstance(value, torch.Tensor):
            tensor = value.to(device="cuda", dtype=torch_dtype)
        else:
            tensor = torch.as_tensor(value, device="cuda", dtype=torch_dtype)
        return tensor.contiguous()

    def _execute(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.context is None or self.engine is None:
            raise RuntimeError(f"TensorRT backend '{self.config.id}' is not loaded.")

        bound_tensors: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            tensor = inputs[name]
            self.context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
            bound_tensors[name] = tensor

        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = _TRT_TO_TORCH_DTYPE.get(trt_dtype)
            if torch_dtype is None:
                raise RuntimeError(f"Unsupported TensorRT dtype for '{name}': {trt_dtype}")
            bound_tensors[name] = torch.empty(shape, device="cuda", dtype=torch_dtype)

        for name, tensor in bound_tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream()
        executed = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not executed:
            raise RuntimeError(f"TensorRT execution failed for backend '{self.config.id}'.")
        stream.synchronize()

        return {name: bound_tensors[name].detach().cpu() for name in self.output_names}

    def predict(
        self,
        image: Image.Image,
        cleaned_keywords: Sequence[str],
        sensitivity: float,
        max_objects: int,
    ):
        if self.processor is None or self.context is None:
            raise RuntimeError(f"{self.config.label} is not loaded.")
        if self.config.image_height is None or self.config.image_width is None:
            raise RuntimeError(f"TensorRT backend '{self.config.id}' is missing image metadata.")

        prompt = build_prompt(cleaned_keywords)
        box_threshold, text_threshold = sensitivity_to_thresholds(sensitivity)
        width, height = image.size

        text_inputs = self._prepare_text_inputs(prompt)
        image_input = _preprocess_trt_image(
            image,
            height=self.config.image_height,
            width=self.config.image_width,
        )

        outputs = self._execute(
            {
                "img": self._tensor_from_value("img", image_input),
                "input_ids": self._tensor_from_value("input_ids", text_inputs["input_ids"]),
                "attention_mask": self._tensor_from_value(
                    "attention_mask", text_inputs["attention_mask"]
                ),
                "position_ids": self._tensor_from_value("position_ids", text_inputs["position_ids"]),
                "token_type_ids": self._tensor_from_value(
                    "token_type_ids", text_inputs["token_type_ids"]
                ),
                "text_token_mask": self._tensor_from_value(
                    "text_token_mask",
                    text_inputs["text_token_mask"],
                ),
            }
        )

        model_outputs = SimpleNamespace(
            logits=outputs["logits"],
            pred_boxes=outputs["boxes"],
        )
        results = _post_process_detection(
            processor=self.processor,
            outputs=model_outputs,
            input_ids=text_inputs["input_ids"],
            target_size=(height, width),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return _results_to_detections(
            results,
            cleaned_keywords=cleaned_keywords,
            max_objects=max_objects,
            detection_backend=self.config.id,
        )


class LoadedDetectionBackend:
    def __init__(self, config: DetectionBackendConfig):
        self.config = config
        self.runtime = None

    @property
    def loaded(self) -> bool:
        return self.runtime is not None

    def load(self):
        if not self.config.enabled:
            return
        if self.runtime is not None:
            return

        try:
            if self.config.kind == DETECTION_BACKEND_KIND_TRT:
                self.runtime = TensorRTDetectorRuntime(self.config)
            else:
                self.runtime = HuggingFaceDetectorRuntime(self.config)
            self.runtime.load()
        except Exception:
            self.runtime = None
            raise

    def unload(self):
        if self.runtime is None:
            return

        print(f"Unloading {self.config.label}...")
        self.runtime.unload()
        self.runtime = None

    def predict(
        self,
        image: Image.Image,
        cleaned_keywords: Sequence[str],
        sensitivity: float,
        max_objects: int,
    ):
        if self.runtime is None:
            raise RuntimeError(f"Detection backend '{self.config.id}' is not loaded.")
        return self.runtime.predict(
            image=image,
            cleaned_keywords=cleaned_keywords,
            sensitivity=sensitivity,
            max_objects=max_objects,
        )


class DetectorManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._backends: dict[str, LoadedDetectionBackend] = {
            backend_id: LoadedDetectionBackend(config)
            for backend_id, config in AVAILABLE_DETECTION_BACKENDS.items()
        }
        print("Detection backend manager ready.")
        print(f"Default detection backend: {DEFAULT_DETECTION_BACKEND}")
        print(
            "Available detection backends: "
            + ", ".join(AVAILABLE_DETECTION_BACKENDS.keys())
        )
        print("Detection backends are loaded only through explicit API calls.")

    def _annotate_detections(self, detections: list[dict], requested_backend: str) -> list[dict]:
        for detection in detections:
            detection["requested_detection_backend"] = requested_backend
            detection["effective_detection_backend"] = requested_backend
            detection["detection_backend_fallback"] = False
        return detections

    def ensure_backend_loaded(self, backend_name: str) -> LoadedDetectionBackend:
        normalized_backend = normalize_detection_backend(backend_name)

        with self._lock:
            backend = self._backends[normalized_backend]
            if not backend.config.enabled:
                return backend
            if not backend.loaded:
                backend.load()
            return backend

    def unload_backend(self, backend_name: str) -> None:
        normalized_backend = normalize_detection_backend(backend_name)
        with self._lock:
            self._backends[normalized_backend].unload()

    def is_backend_loaded(self, backend_name: str) -> bool:
        normalized_backend = normalize_detection_backend(backend_name)
        with self._lock:
            return self._backends[normalized_backend].loaded

    def loaded_backend_ids(self) -> list[str]:
        with self._lock:
            return [
                backend_id
                for backend_id, backend in self._backends.items()
                if backend.loaded
            ]

    def predict(
        self,
        image: Image.Image,
        keywords: Sequence[str] | None = None,
        sensitivity: float = 0.5,
        max_objects: int = 10,
        detection_backend: str | None = None,
    ):
        requested_backend = normalize_detection_backend(detection_backend or DEFAULT_DETECTION_BACKEND)
        if requested_backend == DETECTION_BACKEND_NONE:
            self.ensure_backend_loaded(requested_backend)
            return []

        cleaned_keywords = sanitize_keywords(keywords)
        if not cleaned_keywords:
            return []

        with self._lock:
            backend = self._backends[requested_backend]
            if not backend.loaded:
                raise RuntimeError(
                    f"Detection backend '{requested_backend}' is not loaded."
                )

        detections = backend.predict(
            image=image,
            cleaned_keywords=cleaned_keywords,
            sensitivity=sensitivity,
            max_objects=max_objects,
        )
        return self._annotate_detections(detections, requested_backend=requested_backend)


_detector_manager: DetectorManager | None = None


def load_model() -> DetectorManager:
    global _detector_manager
    if _detector_manager is None:
        _detector_manager = DetectorManager()
    return _detector_manager


def select_detection_backend(detection_backend: str | None = None) -> str:
    selected_backend = normalize_detection_backend(detection_backend or DEFAULT_DETECTION_BACKEND)
    load_model().ensure_backend_loaded(selected_backend)
    return selected_backend


def unload_detection_backend(detection_backend: str | None = None) -> str:
    selected_backend = normalize_detection_backend(detection_backend or DEFAULT_DETECTION_BACKEND)
    load_model().unload_backend(selected_backend)
    return selected_backend


def get_loaded_detection_backends() -> list[str]:
    return load_model().loaded_backend_ids()


def is_detection_backend_loaded(detection_backend: str | None = None) -> bool:
    selected_backend = normalize_detection_backend(detection_backend or DEFAULT_DETECTION_BACKEND)
    return load_model().is_backend_loaded(selected_backend)


def predict_objects(
    image: Image.Image,
    keywords: Sequence[str] | None = None,
    sensitivity: float = 0.5,
    max_objects: int = 10,
    detection_backend: str | None = None,
):
    if _detector_manager is None:
        load_model()
    return _detector_manager.predict(
        image=image,
        keywords=keywords,
        sensitivity=sensitivity,
        max_objects=max_objects,
        detection_backend=detection_backend,
    )
