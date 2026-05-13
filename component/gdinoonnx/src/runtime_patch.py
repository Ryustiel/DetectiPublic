from __future__ import annotations

import importlib
import inspect
import sys
from functools import wraps
from pathlib import Path


def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
    if head_mask is None:
        return [None] * num_hidden_layers

    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    elif head_mask.dim() != 5:
        raise ValueError(f"head_mask must have dim 1, 2, or 5, got {head_mask.dim()}")

    dtype = getattr(self, "dtype", None)
    if dtype is None:
        embeddings = getattr(self, "embeddings", None)
        word_embeddings = getattr(embeddings, "word_embeddings", None)
        weight = getattr(word_embeddings, "weight", None)
        dtype = getattr(weight, "dtype", None)
    if dtype is not None:
        head_mask = head_mask.to(dtype=dtype)

    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask


def add_repo_to_sys_path(repo_dir: Path) -> None:
    repo_path = str(repo_dir.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _patch_transformers_head_mask() -> None:
    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except ImportError:
        return

    module_utils_mixin = getattr(modeling_utils, "ModuleUtilsMixin", None)
    if module_utils_mixin is not None and not hasattr(module_utils_mixin, "get_head_mask"):
        module_utils_mixin.get_head_mask = _compat_get_head_mask

    try:
        modeling_bert = importlib.import_module("transformers.models.bert.modeling_bert")
    except ImportError:
        return

    bert_model = getattr(modeling_bert, "BertModel", None)
    if bert_model is not None and not hasattr(bert_model, "get_head_mask"):
        bert_model.get_head_mask = _compat_get_head_mask


def _patch_bertwarper(repo_dir: Path) -> None:
    import torch

    add_repo_to_sys_path(repo_dir)
    module = importlib.import_module("groundingdino.models.GroundingDINO.bertwarper")
    bert_model_warper = getattr(module, "BertModelWarper", None)
    if bert_model_warper is None:
        return

    if not hasattr(bert_model_warper, "_compat_get_head_mask"):
        bert_model_warper._compat_get_head_mask = _compat_get_head_mask

    original_init = bert_model_warper.__init__
    if getattr(original_init, "_detecti_bertwarper_init_patch", False):
        return

    @wraps(original_init)
    def patched_init(self, bert_model, *args, **kwargs):
        original_init(self, bert_model, *args, **kwargs)

        attention_mask_fn = self.get_extended_attention_mask
        parameters = list(inspect.signature(attention_mask_fn).parameters.values())
        if len(parameters) < 3 or parameters[2].name != "dtype":
            return

        if getattr(attention_mask_fn, "_detecti_extended_attention_mask_patch", False):
            return

        @wraps(attention_mask_fn)
        def compat_get_extended_attention_mask(attention_mask, input_shape, device=None, dtype=None):
            target_dtype = dtype
            if target_dtype is None:
                if isinstance(device, torch.dtype):
                    target_dtype = device
                else:
                    weight = getattr(self.embeddings.word_embeddings, "weight", None)
                    target_dtype = getattr(weight, "dtype", None)

            if target_dtype is None:
                return attention_mask_fn(attention_mask, input_shape)
            return attention_mask_fn(attention_mask, input_shape, dtype=target_dtype)

        compat_get_extended_attention_mask._detecti_extended_attention_mask_patch = True
        self.get_extended_attention_mask = compat_get_extended_attention_mask

    patched_init._detecti_bertwarper_init_patch = True
    bert_model_warper.__init__ = patched_init


def _force_legacy_torch_onnx_export() -> None:
    try:
        import torch
    except ImportError:
        return

    export_fn = torch.onnx.export
    if getattr(export_fn, "_detecti_legacy_export", False):
        return

    @wraps(export_fn)
    def export_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return export_fn(*args, **kwargs)

    export_with_legacy_default._detecti_legacy_export = True
    torch.onnx.export = export_with_legacy_default


def apply_groundingdino_runtime_patch(repo_dir: Path) -> None:
    _force_legacy_torch_onnx_export()
    _patch_transformers_head_mask()
    _patch_bertwarper(repo_dir)
