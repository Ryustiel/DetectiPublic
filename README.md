# Vision API

Vision API is a containerized FastAPI service for object detection, depth maps,
retained camera frames, and event callbacks. It is designed to run locally,
expose a small debug UI, and optionally discover TensorRT-compiled GroundingDINO
engines built by the companion model-build container.

---

## TLDR

- **Vision API Runtime:** FastAPI service with model load/unload endpoints,
  image inference, stream ingestion, retained frames, and a browser debug UI.
- **Camera Capture Flow:** Create short camera codes, open a browser capture
  page, and retain frames that can be queried or attached to detection streams.
- **Detection Streams:** Register streams with labels, stop conditions,
  callbacks, and WebSocket events for frame-by-frame detection results.
- **Depth and Geometry:** Depth backends can enrich results with relative or
  metric depth, distance estimates, and bearing angles.
- **Containerized Runtime:** Docker Compose runs the API service on port `8000`
  with runtime assets mounted under `component/vision/assets/`.
- **Optional TensorRT Builder:** The `gdinoonnx` profile builds
  GroundingDINO ONNX/TensorRT artifacts for the vision service to discover.

---

## Tech Stack and Techniques

- **Backend API**
  - FastAPI and Uvicorn for the HTTP, WebSocket, and browser-facing routes.
  - Pydantic for request validation and structured API payloads.
  - `asyncio` for stream state, callbacks, and retained-frame coordination.
- **Vision Models**
  - PyTorch and Transformers for Hugging Face model execution.
  - GroundingDINO for open-vocabulary object detection.
  - ZoeDepth and Depth Anything V2 style backends for depth estimation.
  - Optional TensorRT support for compiled GroundingDINO engines.
- **Image and Runtime Handling**
  - Pillow and OpenCV headless for image decoding, resizing, annotation, and
    temporary JPEG output.
  - Runtime assets, model caches, generated images, and compiled engines are
    stored outside git under `component/vision/assets/`.
- **Deployment**
  - Docker and Docker Compose for the API, optional model-build container, and
    optional Cloudflare tunnel.
  - `uv` for Python dependency installation inside the service images.

---

## Highlights

- **One API for images and streams** Vision API can process one-off image
  requests through `/api/apply` or keep stream state and retained frames under
  `/api/streams`.
- **Camera-code capture pages** The API can create a temporary capture code and
  serve a browser page at `/{code}`. Frames posted by that page are retained and
  can also feed active streams.
- **Explicit model lifecycle** Models are listed, loaded, unloaded, and queried
  through API endpoints, which makes VRAM use easier to control on local GPU
  machines.
- **Runtime model discovery** Compiled GroundingDINO TensorRT artifacts are
  discovered from `component/vision/assets/models/compiled/gdinoonnx/`, so the
  API can use newly built engines after a restart.
- **Separate builder profile** The ONNX/TensorRT build environment is isolated
  behind the `model-build` Compose profile, keeping normal API startup focused
  on the runtime service.

---

## Installation

### Prerequisites

- Docker and Docker Compose
- An NVIDIA GPU runtime if you want GPU-backed model execution
- Optional Hugging Face token if you need authenticated model downloads
- Optional Cloudflare tunnel token if you want to expose the service through
  the `tunnel` profile

### Setup and Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ryustiel/DetectiPublic
   cd DetectiPublic
   ```

2. **Configure environment files**

   Copy the examples in `environ/` to matching `.env` files when you need to
   override defaults:

   - `environ/vision.env`: API runtime settings, public URLs, and optional Hugging Face token.
   - `environ/gdinoonnx.env`: optional GroundingDINO ONNX/TensorRT build settings.
   - `environ/tunnel.env`: Cloudflare tunnel token.

3. **Build and run the API container**

   ```bash
   docker compose up --build vision
   ```

   The debug UI and API are available at `http://localhost:8000`.

---

## API Surface

Useful routes:

- `GET /`
- `GET /health`
- `GET /api/models`
- `GET /api/models/status`
- `POST /api/models/{model_id}/load`
- `POST /api/models/{model_id}/unload`
- `POST /api/apply`
- `GET /api/streams`
- `POST /api/streams`
- `PATCH /api/streams/{stream_id}`
- `DELETE /api/streams/{stream_id}`
- `POST /api/streams/{stream_id}/frames`
- `POST /api/streams/{stream_id}/url`
- `WS /api/streams/events`
- `WS /api/streams/{stream_id}/events`
- `GET /api/camera-codes`
- `POST /api/camera-codes`
- `DELETE /api/camera-codes/{code}`
- `GET /api/camera-codes/{code}/frames/query`
- `GET /api/camera-codes/{code}/frames/raw`

---

## GroundingDINO TensorRT Builder

The TensorRT builder is optional and is not needed to boot the API container.
Build or rebuild the GroundingDINO ONNX/TensorRT artifacts with:

```bash
docker compose --profile model-build run --rm --build gdinoonnx bash /app/src/build_gdino.sh
```

Artifacts are written to
`component/vision/assets/models/compiled/gdinoonnx/`. Restart a TensorRT-enabled
`vision` image after a new build so the API can rediscover the compiled backend.

The default API image skips the TensorRT Python bindings so normal container
startup does not depend on the TensorRT wheel stack. To include those bindings
in the `vision` image, build with:

```bash
VISION_EXTRAS=tensorrt docker compose build vision
```

---

## Project Structure

```text
.
|-- docker-compose.yaml          # Orchestrates the API, builder, and tunnel services.
|-- environ/                     # Optional environment files and examples.
|   |-- vision.env.example       # Runtime settings for the vision API.
|   |-- gdinoonnx.env.example    # TensorRT builder configuration.
|   `-- tunnel.env.example       # Cloudflare tunnel configuration.
|-- component/
|   |-- vision/                  # FastAPI runtime service.
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   `-- src/
|   |       |-- service.py       # API entry point.
|   |       |-- processor/       # Detection, depth, and angle logic.
|   |       `-- templates/       # Debug UI and camera capture page.
|   `-- gdinoonnx/               # Optional GroundingDINO ONNX/TensorRT builder.
|       |-- Dockerfile
|       |-- pyproject.toml
|       `-- src/                 # Bootstrap, checkpoint, patch, and build scripts.
`-- pyproject.toml               # Local development workspace metadata.
```

Model weights, compiled engines, generated images, temporary files, and cloned
upstream wrappers are intentionally excluded from git. They live under
`component/vision/assets/` at runtime.

---

## License

MIT License. See `LICENSE`.
