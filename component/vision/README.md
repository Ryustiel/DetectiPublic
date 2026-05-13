# Vision API

FastAPI service for model control, detection streams, retained camera frames,
and image outputs.

The service stores runtime data in `/app/assets`, mounted from
`component/vision/assets/`. That directory is ignored by git.

Start it from the repository root:

```powershell
docker compose up --build vision
```
