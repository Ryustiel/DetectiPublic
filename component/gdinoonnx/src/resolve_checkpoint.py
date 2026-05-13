import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_HF_REPO_ID = "ShilongLiu/GroundingDINO"
DEFAULT_HF_FILENAME = "groundingdino_swint_ogc.pth"
DEFAULT_CHECKPOINT_URLS = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
)
DEFAULT_LOCAL_SOURCES = (
    "/app/assets/models/checkpoints/groundingdino_swint_ogc.pth",
    "/app/assets/models/groundingdino_swint_ogc.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the GroundingDINO checkpoint when Git LFS is unavailable."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    return parser.parse_args()


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    return first_line == "version https://git-lfs.github.com/spec/v1"


def parse_env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def candidate_sources(checkpoint_path: Path) -> list[Path]:
    configured = parse_env_list("GDINO_CHECKPOINT_SOURCES")
    if source := os.environ.get("GDINO_CHECKPOINT_SOURCE"):
        configured.insert(0, source)

    ordered_sources: list[Path] = []
    for raw_source in configured + list(DEFAULT_LOCAL_SOURCES):
        source_path = Path(raw_source)
        if source_path.resolve() == checkpoint_path.resolve():
            continue
        ordered_sources.append(source_path)
    return ordered_sources


def candidate_urls() -> list[str]:
    configured = parse_env_list("GDINO_CHECKPOINT_URLS")
    if url := os.environ.get("GDINO_CHECKPOINT_URL"):
        configured.insert(0, url)
    return configured + list(DEFAULT_CHECKPOINT_URLS)


def hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None


def copy_from_local_source(target_path: Path) -> bool:
    for source_path in candidate_sources(target_path):
        if not source_path.exists():
            continue
        if is_lfs_pointer(source_path):
            print(f"Skipping local checkpoint source because it is still a Git LFS pointer: {source_path}")
            continue
        print(f"Copying GroundingDINO checkpoint from local source: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return True
    return False


def download_from_huggingface(target_path: Path) -> bool:
    repo_id = os.environ.get("GDINO_CHECKPOINT_HF_REPO_ID", DEFAULT_HF_REPO_ID).strip() or DEFAULT_HF_REPO_ID
    filename = os.environ.get("GDINO_CHECKPOINT_HF_FILENAME", DEFAULT_HF_FILENAME).strip() or DEFAULT_HF_FILENAME
    token = hf_token()

    print(f"Downloading GroundingDINO checkpoint from Hugging Face: {repo_id}/{filename}")
    if token:
        print("Using Hugging Face token from environment.")
    else:
        print("No Hugging Face token found in environment. Trying anonymous download.")

    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=token,
                cache_dir=os.environ.get("HF_HOME", "/app/assets/models"),
            )
        )
    except Exception as exc:
        print(f"Hugging Face download failed: {exc}")
        return False

    if is_lfs_pointer(downloaded_path):
        print(f"Hugging Face returned a Git LFS pointer instead of weights: {downloaded_path}")
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, target_path)
    return True


def download_checkpoint(target_path: Path) -> bool:
    for url in candidate_urls():
        print(f"Downloading GroundingDINO checkpoint from: {url}")
        with tempfile.NamedTemporaryFile(delete=False, dir=str(target_path.parent), suffix=".tmp") as handle:
            temp_path = Path(handle.name)
        try:
            with urllib.request.urlopen(url) as response, temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            if is_lfs_pointer(temp_path):
                print(f"Downloaded file from {url} is still a Git LFS pointer. Trying the next source.")
                temp_path.unlink(missing_ok=True)
                continue
            temp_path.replace(target_path)
            return True
        except Exception as exc:
            print(f"Checkpoint download failed from {url}: {exc}")
            temp_path.unlink(missing_ok=True)
    return False


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and not is_lfs_pointer(checkpoint_path):
        print(f"GroundingDINO checkpoint is already available: {checkpoint_path}")
        return 0

    if copy_from_local_source(checkpoint_path):
        print(f"GroundingDINO checkpoint is ready at: {checkpoint_path}")
        return 0

    if download_from_huggingface(checkpoint_path):
        print(f"GroundingDINO checkpoint is ready at: {checkpoint_path}")
        return 0

    if download_checkpoint(checkpoint_path):
        print(f"GroundingDINO checkpoint is ready at: {checkpoint_path}")
        return 0

    message = [
        f"Unable to materialize the GroundingDINO checkpoint at {checkpoint_path}.",
        "Git LFS is unavailable for the upstream wrapper repository, and all fallback sources failed.",
        "Place a real groundingdino_swint_ogc.pth file in /app/assets/models/checkpoints/ or set HF_TOKEN / GDINO_CHECKPOINT_HF_REPO_ID / GDINO_CHECKPOINT_HF_FILENAME / GDINO_CHECKPOINT_SOURCE / GDINO_CHECKPOINT_URL.",
    ]
    print("\n".join(message), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
