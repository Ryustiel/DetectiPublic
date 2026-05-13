import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import onnx
import tensorrt as trt
import torch
from runtime_patch import apply_groundingdino_runtime_patch

DEFAULT_REPO_DIR = Path("/app/assets/models/wrappers/gdinoonnx")
DEFAULT_OUTPUT_DIR = Path("/app/assets/models/compiled/gdinoonnx")
DEFAULT_RESOLVE_CHECKPOINT_SCRIPT = Path("/app/src/resolve_checkpoint.py")
DEFAULT_PROMPT = "car ."
DEFAULT_HEIGHT = 800
DEFAULT_WIDTH = 1200
DEFAULT_MAX_TEXT_LEN = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GroundingDINO to ONNX and TensorRT."
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--max-text-len", type=int, default=DEFAULT_MAX_TEXT_LEN)
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-onnx", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def ensure_repo(repo_dir: Path) -> None:
    required_paths = [
        repo_dir / "setup.py",
        repo_dir / "groundingdino",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Wrapper repository is incomplete. Run boot.sh first. Missing: "
            + ", ".join(missing)
        )


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    location = f" (cwd={cwd})" if cwd else ""
    print(f"$ {shlex.join(command)}{location}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    return first_line == "version https://git-lfs.github.com/spec/v1"


def ensure_checkpoint(repo_dir: Path) -> Path:
    checkpoint_path = repo_dir / "weights" / "groundingdino_swint_ogc.pth"
    if checkpoint_path.exists() and not is_lfs_pointer(checkpoint_path):
        return checkpoint_path

    if not DEFAULT_RESOLVE_CHECKPOINT_SCRIPT.exists():
        raise FileNotFoundError(f"Checkpoint resolver script not found: {DEFAULT_RESOLVE_CHECKPOINT_SCRIPT}")

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/app/assets/models")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_command(
        [
            sys.executable,
            str(DEFAULT_RESOLVE_CHECKPOINT_SCRIPT),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
        env=env,
    )
    return checkpoint_path


def add_repo_to_sys_path(repo_dir: Path) -> None:
    repo_path = str(repo_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def normalize_prompt(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split()).lower()
    if not cleaned:
        raise ValueError("Prompt must not be empty.")
    if not cleaned.endswith("."):
        cleaned += " ."
    return cleaned


def default_paths(repo_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = args.checkpoint or repo_dir / "weights" / "groundingdino_swint_ogc.pth"
    config = args.config or repo_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    return checkpoint, config


def load_model(repo_dir: Path, config_path: Path, checkpoint_path: Path):
    add_repo_to_sys_path(repo_dir)
    apply_groundingdino_runtime_patch(repo_dir)

    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    model_args = SLConfig.fromfile(str(config_path))
    model_args.device = "cpu"
    model_args.use_checkpoint = False
    model_args.use_transformer_ckpt = False

    model = build_model(model_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.eval()
    return model


def prepare_text_inputs(model, prompt: str, max_text_len: int):
    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    caption = normalize_prompt(prompt)
    raw_tokenized = model.tokenizer([caption], padding="longest", return_tensors="pt")
    actual_seq_len = int(raw_tokenized["input_ids"].shape[1])
    if actual_seq_len > max_text_len:
        raise ValueError(
            f"Prompt tokenized to {actual_seq_len} tokens, which exceeds the fixed "
            f"TensorRT text window of {max_text_len}. Shorten the prompt or rebuild "
            "with a larger --max-text-len."
        )

    tokenized = model.tokenizer(
        [caption],
        padding="max_length",
        max_length=max_text_len,
        truncation=False,
        return_tensors="pt",
    )
    special_tokens = model.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text_token_mask, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized,
        special_tokens,
        model.tokenizer,
    )
    return {
        "caption": caption,
        "input_ids": tokenized["input_ids"].to(dtype=torch.long),
        "attention_mask": tokenized["attention_mask"].to(dtype=torch.bool),
        "position_ids": position_ids.to(dtype=torch.long),
        "token_type_ids": tokenized["token_type_ids"].to(dtype=torch.long),
        "text_token_mask": text_token_mask.to(dtype=torch.bool),
        "seq_len": max_text_len,
    }


def export_onnx_model(
    model,
    text_inputs: dict,
    onnx_output_path: Path,
    height: int,
    width: int,
    opset: int,
    dynamic_onnx: bool,
) -> None:
    image = torch.randn(1, 3, height, width, dtype=torch.float32)
    input_names = [
        "img",
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "text_token_mask",
    ]
    output_names = ["logits", "boxes"]
    dynamic_axes = None
    if dynamic_onnx:
        dynamic_axes = {
            "img": {0: "batch_size", 2: "height", 3: "width"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "token_type_ids": {0: "batch_size", 1: "seq_len"},
            "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
            "logits": {0: "batch_size"},
            "boxes": {0: "batch_size"},
        }

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            args=(
                image,
                text_inputs["input_ids"],
                text_inputs["attention_mask"],
                text_inputs["position_ids"],
                text_inputs["token_type_ids"],
                text_inputs["text_token_mask"],
            ),
            f=str(onnx_output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)


def build_engine(
    onnx_output_path: Path,
    engine_output_path: Path,
    height: int,
    width: int,
    seq_len: int,
    precision: str,
    workspace_gb: int,
    dynamic_onnx: bool,
) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_output_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise RuntimeError("TensorRT failed to parse ONNX:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("FP16 was requested but this platform does not support fast FP16.")
        config.set_flag(trt.BuilderFlag.FP16)

    if dynamic_onnx:
        profile = builder.create_optimization_profile()
        fixed_image_shape = (1, 3, height, width)
        fixed_text_shape = (1, seq_len)
        fixed_mask_shape = (1, seq_len, seq_len)
        profile.set_shape("img", fixed_image_shape, fixed_image_shape, fixed_image_shape)
        profile.set_shape("input_ids", fixed_text_shape, fixed_text_shape, fixed_text_shape)
        profile.set_shape("attention_mask", fixed_text_shape, fixed_text_shape, fixed_text_shape)
        profile.set_shape("position_ids", fixed_text_shape, fixed_text_shape, fixed_text_shape)
        profile.set_shape("token_type_ids", fixed_text_shape, fixed_text_shape, fixed_text_shape)
        profile.set_shape("text_token_mask", fixed_mask_shape, fixed_mask_shape, fixed_mask_shape)
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build returned no engine.")

    engine_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_output_path, "wb") as handle:
        handle.write(serialized_engine)


def write_metadata(
    metadata_path: Path,
    caption: str,
    seq_len: int,
    height: int,
    width: int,
    precision: str,
    onnx_output_path: Path,
    engine_output_path: Path | None,
    checkpoint_path: Path,
    config_path: Path,
) -> None:
    metadata = {
        "caption": caption,
        "seq_len": seq_len,
        "image_height": height,
        "image_width": width,
        "precision": precision,
        "onnx_path": str(onnx_output_path),
        "engine_path": str(engine_output_path) if engine_output_path else None,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/app/assets/models")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.height <= 0 or args.width <= 0:
        raise ValueError("Image height and width must be positive integers.")

    repo_dir = args.repo_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_repo(repo_dir)
    ensure_checkpoint(repo_dir)

    checkpoint_path, config_path = default_paths(repo_dir, args)
    model = load_model(repo_dir, config_path.resolve(), checkpoint_path.resolve())
    text_inputs = prepare_text_inputs(model, args.prompt, args.max_text_len)

    base_name = f"groundingdino_swint_ogc_{args.height}x{args.width}_{text_inputs['seq_len']}tok"
    onnx_output_path = output_dir / f"{base_name}.onnx"
    engine_output_path = output_dir / f"{base_name}_{args.precision}.engine"
    metadata_path = output_dir / f"{base_name}.metadata.json"

    if not args.skip_onnx:
        print(f"Exporting ONNX to {onnx_output_path}")
        export_onnx_model(
            model=model,
            text_inputs=text_inputs,
            onnx_output_path=onnx_output_path,
            height=args.height,
            width=args.width,
            opset=args.opset,
            dynamic_onnx=args.dynamic_onnx,
        )

    if not args.skip_engine:
        if not onnx_output_path.exists():
            raise FileNotFoundError(
                f"ONNX file not found at {onnx_output_path}. Build ONNX first or drop --skip-onnx."
            )
        print(f"Building TensorRT engine to {engine_output_path}")
        build_engine(
            onnx_output_path=onnx_output_path,
            engine_output_path=engine_output_path,
            height=args.height,
            width=args.width,
            seq_len=text_inputs["seq_len"],
            precision=args.precision,
            workspace_gb=args.workspace_gb,
            dynamic_onnx=args.dynamic_onnx,
        )

    write_metadata(
        metadata_path=metadata_path,
        caption=text_inputs["caption"],
        seq_len=text_inputs["seq_len"],
        height=args.height,
        width=args.width,
        precision=args.precision,
        onnx_output_path=onnx_output_path,
        engine_output_path=None if args.skip_engine else engine_output_path,
        checkpoint_path=checkpoint_path.resolve(),
        config_path=config_path.resolve(),
    )

    print(f"Prompt fixed into build: {text_inputs['caption']}")
    print(f"Artifacts written to {output_dir}")
    if args.skip_engine:
        print(f"ONNX: {onnx_output_path}")
    else:
        print(f"ONNX: {onnx_output_path}")
        print(f"TensorRT: {engine_output_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
