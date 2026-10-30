"""
Apply MemoryBank compression / sparse retrieval from training YAML (same logic as Engine.setup).
Use runtime_stage='eval' so _compression_enabled_now / _sparse_enabled_now take the non-train branch (return True when enabled).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from adaptation.plugins.manager import PluginManager


def _load_merged_yaml(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True) or {}


def _apply_mem_compression_arg_aliases(tc: Any) -> None:
    """Mirror run_adaptation.py main() lines ~1064-1077."""
    if bool(getattr(tc, "use_memory_compression", False)):
        setattr(tc, "mem_compression_enabled", True)
    if str(getattr(tc, "memory_compress_mode", "none")).lower() != "none":
        setattr(tc, "mem_compression_mode", str(getattr(tc, "memory_compress_mode")))
    if str(getattr(tc, "memory_compress_location", "before_attention")).lower() != "before_attention":
        setattr(tc, "mem_compression_location", str(getattr(tc, "memory_compress_location")))
    if int(getattr(tc, "memory_downsample_ratio", 2)) != 2:
        dr = int(getattr(tc, "memory_downsample_ratio", 2))
        setattr(tc, "mem_compression_kernel_size", dr)
        setattr(tc, "mem_compression_stride", dr)
    mode = str(getattr(tc, "mem_compression_mode", "none")).lower()
    if int(getattr(tc, "memory_spatial_window_size", 2)) != 2 and mode in {"spatial", "hybrid"}:
        ws = int(getattr(tc, "memory_spatial_window_size", 2))
        setattr(tc, "mem_compression_kernel_size", ws)
        setattr(tc, "mem_compression_stride", ws)
    if int(getattr(tc, "memory_global_summary_tokens", 0)) > 0:
        setattr(tc, "mem_compression_keep_global_tokens", int(getattr(tc, "memory_global_summary_tokens", 0)))


def _train_config_from_yaml(merged: Dict[str, Any]) -> Any:
    from run_adaptation import TrainConfig

    tc = TrainConfig()
    for k, v in merged.items():
        if k == "plugins":
            continue
        if hasattr(tc, k):
            setattr(tc, k, v)
    plugins = merged.get("plugins", None)
    setattr(tc, "plugins_cfg", plugins if isinstance(plugins, dict) else {})
    _apply_mem_compression_arg_aliases(tc)
    return tc


def _configure_compression_from_build_model_args(model: torch.nn.Module, args: Any) -> None:
    """Same as utils.build.build_model tail when mem_compression_enabled."""
    mem_enabled = bool(getattr(args, "mem_compression_enabled", False)) or bool(
        getattr(args, "use_memory_compression", False)
    )
    mb = model.memory_bank
    if not hasattr(mb, "configure_compression"):
        return
    mb.configure_compression(
        enabled=mem_enabled,
        select_mode=str(getattr(args, "memory_select_mode", "none")),
        recent_keep=int(getattr(args, "memory_recent_keep", 0)),
        score_keep=int(getattr(args, "memory_score_keep", 0)),
        compress_mode=str(getattr(args, "mem_compression_mode", getattr(args, "memory_compress_mode", "none"))),
        compress_location=str(
            getattr(args, "mem_compression_location", getattr(args, "memory_compress_location", "before_attention"))
        ),
        downsample_ratio=int(getattr(args, "memory_downsample_ratio", 2)),
        spatial_window_size=int(getattr(args, "memory_spatial_window_size", 2)),
        global_summary_tokens=int(getattr(args, "memory_global_summary_tokens", 0)),
        kernel_size=int(getattr(args, "mem_compression_kernel_size", 2)),
        stride=int(getattr(args, "mem_compression_stride", 2)),
        keep_global_tokens=int(getattr(args, "mem_compression_keep_global_tokens", 0)),
        apply_to_pos_enc=bool(getattr(args, "mem_compression_apply_to_pos_enc", True)),
        min_tokens=int(getattr(args, "mem_compression_min_tokens", 64)),
        warmup_iters=int(getattr(args, "mem_compression_warmup_iters", 0)),
        enable_in_train=bool(getattr(args, "mem_compression_enable_in_train", True)),
        enable_in_pseudo_label=bool(getattr(args, "mem_compression_enable_in_pseudo_label", True)),
        train_kernel_size=int(getattr(args, "mem_compression_train_kernel_size", 2)),
        pl_kernel_size=int(getattr(args, "mem_compression_pl_kernel_size", 2)),
        verbose=bool(getattr(args, "mem_compression_verbose", False)),
        log_interval=int(getattr(args, "mem_compression_log_interval", 200)),
    )


def _configure_efficient_tam_like_engine(model: torch.nn.Module, pm: PluginManager, args: Any) -> None:
    if not pm.enabled("efficient_tam_memory"):
        return
    mcfg = pm.cfg.get("efficient_tam_memory", {}) if isinstance(pm.cfg, dict) else {}
    mode = str(mcfg.get("mode", "pool")).lower()
    pool_stride = int(mcfg.get("pool_stride", 2))
    apply_in_pl = bool(mcfg.get("apply_in_pl", True))
    apply_in_train = bool(mcfg.get("apply_in_train", True))
    mb = model.memory_bank
    if not hasattr(mb, "configure_compression"):
        return
    mb.configure_compression(
        enabled=True,
        compress_mode=mode,
        compress_location=str(getattr(args, "mem_compression_location", "before_attention")),
        kernel_size=pool_stride,
        stride=pool_stride,
        min_tokens=int(getattr(args, "mem_compression_min_tokens", 64)),
        warmup_iters=int(getattr(args, "mem_compression_warmup_iters", 0)),
        enable_in_train=apply_in_train,
        enable_in_pseudo_label=apply_in_pl,
        train_kernel_size=pool_stride,
        pl_kernel_size=pool_stride,
        verbose=bool(getattr(args, "mem_compression_verbose", False)),
        log_interval=int(getattr(args, "mem_compression_log_interval", 200)),
    )


def _configure_sparse_like_engine(model: torch.nn.Module, pm: PluginManager) -> None:
    if not pm.enabled("efficient_sam2_sparse"):
        return
    scfg = pm.cfg.get("efficient_sam2_sparse", {}) if isinstance(pm.cfg, dict) else {}
    mode = str(scfg.get("mode", "attn_topk"))
    keep_ratio = float(scfg.get("keep_ratio", 0.5))
    min_keep = int(scfg.get("min_keep_tokens", 64))
    cache_sal = bool(scfg.get("cache_saliency", True))
    fallback_full = bool(scfg.get("fallback_full_when_uncertain", True))
    apply_in_pl = bool(scfg.get("apply_in_pl", True))
    apply_in_train = bool(scfg.get("apply_in_train", True))
    fallback = str(scfg.get("saliency_source", "feature_norm"))
    mb = model.memory_bank
    if not hasattr(mb, "configure_sparse_retrieval"):
        return
    mb.configure_sparse_retrieval(
        enabled=True,
        mode=mode,
        keep_ratio=keep_ratio,
        min_tokens=min_keep,
        cache_enabled=cache_sal,
        fallback=fallback,
        fallback_full_when_uncertain=fallback_full,
        enable_in_train=apply_in_train,
        enable_in_pseudo_label=apply_in_pl,
    )


def disable_memory_accel(model: torch.nn.Module) -> Dict[str, Any]:
    mb = getattr(model, "memory_bank", None)
    if mb is None:
        return {"error": "no_memory_bank"}
    mb.configure_compression(enabled=False, compress_mode="none")
    mb.configure_sparse_retrieval(enabled=False)
    mb.set_runtime_context(stage="eval", iter_idx=0)
    return memory_bank_status(mb)


def enable_memory_accel_from_yaml(model: torch.nn.Module, yaml_path: Path, num_classes: int = 13) -> Dict[str, Any]:
    mb = model.memory_bank
    mb.configure_compression(enabled=False, compress_mode="none")
    mb.configure_sparse_retrieval(enabled=False)

    merged = _load_merged_yaml(yaml_path)
    tc = _train_config_from_yaml(merged)
    device = next(model.parameters()).device
    pm = PluginManager.from_args(tc, num_classes=num_classes, device=device)

    # Compression: plugin efficient_tam_memory overrides generic mem_compression from YAML
    if pm.enabled("efficient_tam_memory"):
        _configure_efficient_tam_like_engine(model, pm, tc)
    elif bool(getattr(tc, "mem_compression_enabled", False)) or bool(getattr(tc, "use_memory_compression", False)):
        _configure_compression_from_build_model_args(model, tc)

    if pm.enabled("efficient_sam2_sparse"):
        _configure_sparse_like_engine(model, pm)

    model.memory_bank.set_runtime_context(stage="eval", iter_idx=0)
    return memory_bank_status(model.memory_bank)


def memory_bank_status(mb: Any) -> Dict[str, Any]:
    return {
        "memory_compress_enabled": bool(getattr(mb, "memory_compress_enabled", False)),
        "memory_compress_mode": str(getattr(mb, "memory_compress_mode", "")),
        "sparse_enabled": bool(getattr(mb, "sparse_enabled", False)),
        "sparse_mode": str(getattr(mb, "sparse_mode", "")),
        "runtime_stage": str(getattr(mb, "runtime_stage", "")),
        "memory_enable_in_train": bool(getattr(mb, "memory_enable_in_train", False)),
        "memory_enable_in_pseudo_label": bool(getattr(mb, "memory_enable_in_pseudo_label", False)),
        "sparse_enable_in_train": bool(getattr(mb, "sparse_enable_in_train", False)),
        "sparse_enable_in_pseudo_label": bool(getattr(mb, "sparse_enable_in_pseudo_label", False)),
    }


def log_memory_status(prefix: str, status: Dict[str, Any]) -> None:
    print(
        f"{prefix} "
        f"memory_compress_enabled={status.get('memory_compress_enabled')} "
        f"sparse_enabled={status.get('sparse_enabled')} "
        f"runtime_stage={status.get('runtime_stage')} "
        f"memory_enable_in_train={status.get('memory_enable_in_train')} "
        f"sparse_enable_in_train={status.get('sparse_enable_in_train')}",
        flush=True,
    )


def write_accel_meta(path: Path, memory_accel: str, memory_config: str, status: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "memory_accel": memory_accel,
        "memory_config": memory_config,
        **status,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
