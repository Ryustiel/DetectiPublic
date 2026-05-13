# GroundingDINO ONNX/TensorRT Builder

Build container for the external GroundingDINO ONNX/TensorRT wrapper.

The upstream repository is cloned at runtime into
`component/vision/assets/models/wrappers/gdinoonnx/`. Only the local bootstrap,
patch, checkpoint resolver, and build scripts are tracked in this repository.

Build the ONNX and TensorRT artifacts from the repository root:

```powershell
docker compose --profile model-build run --rm --build gdinoonnx bash /app/src/build_gdino.sh
```
