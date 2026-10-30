import json
import os
import re
import time
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.build import *
from adaptation.pl_generator import PseudoLabelGenerator
from adaptation.trainer import Trainer
from adaptation.strategies import WithMemStrategy, NoMemStrategy
from adaptation.patch_scheduler import AdaptivePatchScheduler
from adaptation.pl_update_controller import PseudoLabelUpdateController
from adaptation.plugins.manager import PluginManager
from utils.callbacks import CheckpointCB, LRCB, LoggerCB
from utils.io import ensure_dir, clear_dir, save_checkpoint, load_checkpoint_file
from utils.lr_scheduler import build_poly_warmup_scheduler


class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_builder = None
        self.trainer = None

        self.class_names = None
        self.num_classes = None
        self.is_warmup = None

        self.target_pl_loader = None
        self.source_loader = None
        self.val_loader = None
        self.target_loader = None

        self.model_t = None
        self.model_s = None
        self.save_dir = None

        self.global_step = 0
        self.best_mIoU = 0.0
        self.best_ckpt_path: Optional[str] = None
        self.best_metrics = {
            "mIoU": 0.0,
            "mAcc": None,
            "PixAcc": None,
            "FWIoU": None,
            "mDice": None,
        }
        self.start_epoch = 0
        self.resume_path: Optional[str] = None
        self.start_iter = 0
        self.tb_writer: Optional[SummaryWriter] = None
        self.patch_scheduler: Optional[AdaptivePatchScheduler] = None
        self.pl_update_controller: Optional[PseudoLabelUpdateController] = None
        self.last_pl_stats: dict = {}
        self.last_train_stats: dict = {}
        self.pl_total_epochs: int = 0
        self.pl_update_epochs: int = 0
        self.resume_save_disabled: bool = False
        self.plugins: Optional[PluginManager] = None

    @staticmethod
    def _count_params(params):
        return int(sum(p.numel() for p in params))

    def _configure_trainable_params(self):
        stage = str(getattr(self.args, "text_train_stage", "stage2")).lower()
        # freeze all first
        for p in self.model_t.parameters():
            p.requires_grad_(False)
        groups = {
            "text_adapter": [],
            "text_fusion_alpha": [],
            "image_to_text_cross_attn": [],
            "text_fusion_beta": [],
            "text_to_image_grounding": [],
            "text_grounding_gamma": [],
            "text_grounding_logit_scale": [],
            "text_scale": [],
            "semantic_decoder": [],
            "depth_adapter": [],
            "lora": [],
            "memory": [],
            "density_estimator": [],
        }

        # stage-independent text knobs
        text_adapter_lr_mult = float(getattr(self.args, "text_adapter_lr_mult", 1.0))
        text_fusion_alpha_lr_mult = float(getattr(self.args, "text_fusion_alpha_lr_mult", 1.0))
        text_scale_lr_mult = float(getattr(self.args, "text_scale_lr_mult", 1.0))
        txt_head = getattr(self.model_t, "txt_head", None)
        if txt_head is not None:
            for p in txt_head.projector.parameters():
                p.requires_grad_(True)
                groups["text_adapter"].append(p)
            if hasattr(txt_head, "text_scale") and isinstance(txt_head.text_scale, torch.nn.Parameter):
                txt_head.text_scale.requires_grad_(True)
                groups["text_scale"].append(txt_head.text_scale)

        txt_fusion = getattr(self.model_t, "text_fusion", None)
        if txt_fusion is not None and hasattr(txt_fusion, "alpha"):
            txt_fusion.alpha.requires_grad_(bool(getattr(self.args, "text_fusion_alpha_learnable", True)))
            if txt_fusion.alpha.requires_grad:
                groups["text_fusion_alpha"].append(txt_fusion.alpha)
        txt_cross = getattr(self.model_t, "text_cross_fusion", None)
        if txt_cross is not None:
            for p in txt_cross.parameters():
                p.requires_grad_(True)
            if hasattr(txt_cross, "beta") and isinstance(txt_cross.beta, torch.nn.Parameter):
                groups["text_fusion_beta"].append(txt_cross.beta)
            for name, p in txt_cross.named_parameters():
                if name != "beta":
                    groups["image_to_text_cross_attn"].append(p)
        txt_ground = getattr(self.model_t, "text_grounding_fusion", None)
        if txt_ground is not None:
            for p in txt_ground.parameters():
                p.requires_grad_(True)
            if hasattr(txt_ground, "gamma") and isinstance(txt_ground.gamma, torch.nn.Parameter):
                groups["text_grounding_gamma"].append(txt_ground.gamma)
            if hasattr(txt_ground, "logit_scale") and isinstance(txt_ground.logit_scale, torch.nn.Parameter):
                groups["text_grounding_logit_scale"].append(txt_ground.logit_scale)
            for name, p in txt_ground.named_parameters():
                if name not in {"gamma", "logit_scale"}:
                    groups["text_to_image_grounding"].append(p)

        if stage in {"stage2", "stage3", "all"}:
            for p in self.model_t.decoder.parameters():
                p.requires_grad_(True)
                groups["semantic_decoder"].append(p)
        depth_adapter = getattr(self.model_t, "depth_adapter", None)
        if depth_adapter is not None:
            for p in depth_adapter.parameters():
                p.requires_grad_(True)
                groups["depth_adapter"].append(p)
        if stage in {"stage3", "all"}:
            for n, p in self.model_t.encoder.named_parameters():
                if "w_a" in n or "w_b" in n:
                    p.requires_grad_(True)
                    groups["lora"].append(p)

        # SG-DRF density estimator
        if bool(getattr(self.args, "enable_sgdrf", False)):
            density_est = getattr(self.model_t, "density_estimator", None)
            if density_est is not None:
                for p in density_est.parameters():
                    p.requires_grad_(True)
                    groups["density_estimator"].append(p)

        # fall back: if text disabled keep old behavior (train all)
        if not any(len(v) > 0 for v in groups.values()):
            for p in self.model_t.parameters():
                p.requires_grad_(True)
            params = [p for p in self.model_t.parameters() if p.requires_grad]
            print(f"[trainable] text-disabled fallback; total_trainable={self._count_params(params)}", flush=True)
            return [{"params": params}]

        group_list = []
        for name, ps in groups.items():
            if len(ps) == 0:
                continue
            item = {"params": ps}
            if name == "text_adapter":
                item["lr"] = float(self.args.lr) * text_adapter_lr_mult
            elif name == "text_fusion_alpha":
                item["lr"] = float(self.args.lr) * text_fusion_alpha_lr_mult
            elif name == "text_fusion_beta":
                item["lr"] = float(self.args.lr) * text_fusion_alpha_lr_mult
            elif name == "text_scale":
                item["lr"] = float(self.args.lr) * text_scale_lr_mult
            elif name == "text_grounding_gamma":
                item["lr"] = float(self.args.lr) * text_fusion_alpha_lr_mult
            elif name == "text_grounding_logit_scale":
                item["lr"] = float(self.args.lr) * text_scale_lr_mult
            elif name == "density_estimator":
                item["lr"] = float(getattr(self.args, "sgdrf_density_lr", 5e-5))
            group_list.append(item)
        print("[trainable] parameter groups:", flush=True)
        total = 0
        for name, ps in groups.items():
            n = self._count_params(ps)
            total += n
            lr_info = ""
            if name == "text_adapter":
                lr_info = f" (lr={float(self.args.lr) * text_adapter_lr_mult:.6e})"
            elif name == "text_fusion_alpha":
                lr_info = f" (lr={float(self.args.lr) * text_fusion_alpha_lr_mult:.6e})"
            elif name == "text_fusion_beta":
                lr_info = f" (lr={float(self.args.lr) * text_fusion_alpha_lr_mult:.6e})"
            elif name == "text_scale":
                lr_info = f" (lr={float(self.args.lr) * text_scale_lr_mult:.6e})"
            elif name == "text_grounding_gamma":
                lr_info = f" (lr={float(self.args.lr) * text_fusion_alpha_lr_mult:.6e})"
            elif name == "text_grounding_logit_scale":
                lr_info = f" (lr={float(self.args.lr) * text_scale_lr_mult:.6e})"
            elif name == "density_estimator":
                lr_info = f" (lr={float(getattr(self.args, 'sgdrf_density_lr', 5e-5)):.6e})"
            print(f"  - {name}: {n}{lr_info}", flush=True)
        print(f"  - total_trainable: {total}", flush=True)
        print("[text-trainable]", flush=True)
        print(f"text_projector: {self._count_params(groups['text_adapter'])}", flush=True)
        print(f"image_to_text_cross_attn: {self._count_params(groups['image_to_text_cross_attn'])}", flush=True)
        print(f"text_fusion_beta: {self._count_params(groups['text_fusion_beta'])}", flush=True)
        print(f"text_to_image_grounding: {self._count_params(groups['text_to_image_grounding'])}", flush=True)
        print(f"text_grounding_gamma: {self._count_params(groups['text_grounding_gamma'])}", flush=True)
        print(f"text_grounding_logit_scale: {self._count_params(groups['text_grounding_logit_scale'])}", flush=True)
        print(f"semantic_decoder: {self._count_params(groups['semantic_decoder'])}", flush=True)
        print(f"lora: {self._count_params(groups['lora'])}", flush=True)
        print(f"total_trainable: {total}", flush=True)
        print(f"[trainable] depth_adapter_trainable_params={self._count_params(groups['depth_adapter'])}", flush=True)
        print(f"[trainable] total_trainable_params={total}", flush=True)
        return group_list

    def _metrics_dir(self):
        d = os.path.join(self._save_dir(), "metrics")
        ensure_dir(d)
        return d

    def _load_pl_subset_filter(self):
        subset_file = str(getattr(self.args, "pl_subset_file", "") or "").strip()
        if not subset_file:
            return None
        if not os.path.isfile(subset_file):
            raise FileNotFoundError(f"pl_subset_file not found: {subset_file}")
        with open(subset_file, "r", encoding="utf-8") as f:
            names = {line.strip() for line in f if line.strip()}
        return names if names else None

    def setup(self):
        self.data_builder = DatasetBuilder(self.args)
        self.target_pl_loader, self.class_names, self.num_classes, _ = self.data_builder.build_tgt_for_pl()

        if bool(getattr(self.args, "use_tensorboard", True)):
            tb_root = getattr(self.args, "tb_log_dir", "runs")
            tb_dir = os.path.join(tb_root, f"{self.args.backbone}_{self.args.dataset}_{getattr(self.args, 'class_set', 13)}cls")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        self.model_t = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device)
        self.model_s = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device)

        load_checkpoint(self.model_t, self.args.tgt_ckpt)
        load_checkpoint(self.model_s, self.args.src_ckpt)

        # Plugin manager (config-only at this stage); attach to engine for stateful plugins.
        self.plugins = PluginManager.from_args(args=self.args, num_classes=self.num_classes, device=self.device)
        print(self.plugins.format_enabled_plugins(prefix="[plugins] "), flush=True)

        # EfficientTAM-style memory compression plugin (wrap existing memory_bank compressor).
        if self.plugins.enabled("efficient_tam_memory"):
            mcfg = self.plugins.cfg.get("efficient_tam_memory", {}) if isinstance(getattr(self.plugins, "cfg", None), dict) else {}
            mode = str(mcfg.get("mode", "pool")).lower()
            pool_stride = int(mcfg.get("pool_stride", 2))
            apply_in_pl = bool(mcfg.get("apply_in_pl", True))
            apply_in_train = bool(mcfg.get("apply_in_train", True))
            # Use existing MemoryBank.configure_compression interface.
            for _m in (self.model_t, self.model_s):
                if hasattr(_m, "memory_bank") and hasattr(_m.memory_bank, "configure_compression"):
                    _m.memory_bank.configure_compression(
                        enabled=True,
                        compress_mode=mode,
                        compress_location=str(getattr(self.args, "mem_compression_location", "before_attention")),
                        kernel_size=pool_stride,
                        stride=pool_stride,
                        min_tokens=int(getattr(self.args, "mem_compression_min_tokens", 64)),
                        warmup_iters=int(getattr(self.args, "mem_compression_warmup_iters", 0)),
                        enable_in_train=apply_in_train,
                        enable_in_pseudo_label=apply_in_pl,
                        train_kernel_size=pool_stride,
                        pl_kernel_size=pool_stride,
                        verbose=bool(getattr(self.args, "mem_compression_verbose", False)),
                        log_interval=int(getattr(self.args, "mem_compression_log_interval", 200)),
                    )

        # Efficient-SAM2-style sparse memory retrieval plugin (token selection).
        if self.plugins.enabled("efficient_sam2_sparse"):
            scfg = self.plugins.cfg.get("efficient_sam2_sparse", {}) if isinstance(getattr(self.plugins, "cfg", None), dict) else {}
            mode = str(scfg.get("mode", "attn_topk"))
            keep_ratio = float(scfg.get("keep_ratio", 0.5))
            min_keep = int(scfg.get("min_keep_tokens", 64))
            cache_sal = bool(scfg.get("cache_saliency", True))
            fallback_full = bool(scfg.get("fallback_full_when_uncertain", True))
            apply_in_pl = bool(scfg.get("apply_in_pl", True))
            apply_in_train = bool(scfg.get("apply_in_train", True))
            fallback = str(scfg.get("saliency_source", "feature_norm"))
            boundary_reserve_ratio = float(scfg.get("boundary_reserve_ratio", 0.125))  # NEW: Route 1
            rare_class_ids = list(scfg.get("rare_class_ids", []))  # NEW: Route 3
            rare_reserve_ratio = float(scfg.get("rare_reserve_ratio", 0.15))  # NEW: Route 3
            rare_selection_strategy = str(scfg.get("rare_selection_strategy", "confidence"))  # NEW: Route 3
            boundary_score_source = str(scfg.get("boundary_score_source", "feature_norm"))  # NEW: uncertainty-guided
            print(
                f"[plugins][efficient_sam2_sparse] apply_in_train={apply_in_train}, "
                f"apply_in_pl={apply_in_pl}, mode={mode}, keep_ratio={keep_ratio}, "
                f"boundary_reserve_ratio={boundary_reserve_ratio}, "
                f"rare_class_ids={rare_class_ids}, rare_reserve_ratio={rare_reserve_ratio}, "
                f"boundary_score_source={boundary_score_source}",
                flush=True,
            )
            for _m in (self.model_t, self.model_s):
                if hasattr(_m, "memory_bank") and hasattr(_m.memory_bank, "configure_sparse_retrieval"):
                    _m.memory_bank.configure_sparse_retrieval(
                        enabled=True,
                        mode=mode,
                        keep_ratio=keep_ratio,
                        min_tokens=min_keep,
                        cache_enabled=cache_sal,
                        fallback=fallback,
                        fallback_full_when_uncertain=fallback_full,
                        enable_in_train=apply_in_train,
                        enable_in_pseudo_label=apply_in_pl,
                        boundary_reserve_ratio=boundary_reserve_ratio,  # NEW: Route 1
                        rare_class_ids=rare_class_ids,  # NEW: Route 3
                        rare_reserve_ratio=rare_reserve_ratio,  # NEW: Route 3
                        rare_selection_strategy=rare_selection_strategy,  # NEW: Route 3
                        boundary_score_source=boundary_score_source,  # NEW: uncertainty-guided
                    )

        # Pano-SMB changes only per-FoV K; legacy SMR scoring and total budget are preserved.
        pcfg = self.plugins.cfg.get("pano_smb", {}) if isinstance(getattr(self.plugins, "cfg", None), dict) else {}
        pano_enabled = bool(pcfg.get("enabled", False))
        if pano_enabled and not self.plugins.enabled("efficient_sam2_sparse"):
            raise ValueError("pano_smb requires efficient_sam2_sparse.enabled=true")
        for _m in (self.model_t, self.model_s):
            if hasattr(_m, "memory_bank") and hasattr(_m.memory_bank, "configure_pano_smb"):
                _m.memory_bank.configure_pano_smb(
                    enabled=pano_enabled,
                    min_tokens=int(pcfg.get("min_tokens", 256)),
                    verbose=bool(pcfg.get("verbose", False)),
                    log_interval=int(pcfg.get("log_interval", 200)),
                )

        criteria = self.data_builder.build_criteria(self.device, self.args.dataset)
        optimizer = torch.optim.AdamW(self._configure_trainable_params(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = build_poly_warmup_scheduler(optimizer, power=1.0, warmup_iters=1800)
        mem_strategy = WithMemStrategy() if self.args.dataset != 'Stanford2D3D' else NoMemStrategy()

        callbacks = [
            CheckpointCB(
                save_dir=self._save_dir(),
                backbone=f"{self.args.backbone}_{getattr(self.args,'class_set',13)}cls",
                use_wandb=bool(getattr(self.args, "use_wandb", False)),
                tb_writer=self.tb_writer,
            ),
            LRCB(scheduler),
            LoggerCB(
                use_wandb=bool(getattr(self.args, "use_wandb", False)),
                tb_writer=self.tb_writer,
            )
        ]

        if self.args.dataset == 'Stanford2D3D':
            temp_path_file = os.path.join(self.args.pd_root, 'Stanford2D3D', self.args.backbone, 'unused_path.txt')
            if os.path.exists(temp_path_file):
                os.remove(temp_path_file)
        else:
            temp_path_file = os.path.join(self.args.pd_root, 'DensePASS', self.args.backbone, self.args.dataset, 'unused_path.txt')
            if os.path.exists(temp_path_file):
                os.remove(temp_path_file)

        self.trainer = Trainer(
            args=self.args,
            model_tgt=self.model_t,
            model_src=self.model_s,
            optimizer=optimizer,
            criteria=criteria,
            adaptation_kwargs=dict(num_classes=self.num_classes, feature_dim=256),
            mem_strategy=mem_strategy,
            callbacks=callbacks,
            device=self.device,
        )
        self.args.on_train_iter_end = self._on_train_iter_end
        self.patch_scheduler = AdaptivePatchScheduler(
            enabled=bool(getattr(self.args, "use_patch_scheduler", False)),
            mode=str(getattr(self.args, "patch_scheduler_mode", "none")),
            stride_scale=float(getattr(self.args, "patch_stride_scale", 1.0)),
            topk=int(getattr(self.args, "patch_topk", 0)),
            score_source=str(getattr(self.args, "patch_score_source", "hybrid")),
            baseline_stride=int(getattr(self.args, "stride", 384)),
            high_complexity_stride=int(getattr(self.args, "adaptive_stride_high_complexity", getattr(self.args, "stride", 384))),
            low_complexity_stride=int(getattr(self.args, "adaptive_stride_low_complexity", int(getattr(self.args, "stride", 384) * 2))),
            complexity_threshold=float(getattr(self.args, "adaptive_threshold", 0.0)),
            apply_stage=str(getattr(self.args, "adaptive_apply_stage", "pseudo_label")),
        )
        self.pl_update_controller = PseudoLabelUpdateController(
            enabled=bool(getattr(self.args, "use_pl_update_controller", False)),
            mode=str(getattr(self.args, "pl_update_mode", "every_epoch")),
            interval=int(getattr(self.args, "pl_update_interval", 1)),
            score_threshold=float(getattr(self.args, "pl_score_threshold", 0.5)),
        )

        self._try_resume()
        self._init_best_baseline_from_tgt_ckpt()

    @staticmethod
    def _parse_iou_from_ckpt_path(ckpt_path: str) -> Optional[float]:
        """Parse val mIoU from filenames like best_*_iou60.36.pth (reported as 0–100)."""
        if not ckpt_path:
            return None
        name = os.path.basename(ckpt_path)
        m = re.search(r"iou[_]?(\d+\.\d+)", name, re.IGNORECASE)
        if not m:
            return None
        return float(m.group(1))

    def _init_best_baseline_from_tgt_ckpt(self):
        """
        Use the loaded tgt_ckpt as initial best when its filename encodes mIoU (e.g. ...iou60.36.pth).
        New checkpoints are saved only when val mIoU strictly exceeds this baseline.
        """
        tgt = getattr(self.args, "tgt_ckpt", None)
        if not tgt:
            return
        abspath = os.path.abspath(tgt)
        if not os.path.isfile(abspath):
            return
        parsed = self._parse_iou_from_ckpt_path(abspath)
        if parsed is None:
            return
        if parsed > self.best_mIoU:
            self.best_mIoU = parsed
            self.best_ckpt_path = abspath
            self.best_metrics["mIoU"] = parsed
            print(
                f"[best] baseline mIoU={self.best_mIoU:.2f} from tgt_ckpt",
                flush=True,
            )

    @staticmethod
    def _metric_or_none(metrics: dict, key: str):
        v = metrics.get(key)
        if v is None:
            return None
        try:
            v = float(v)
        except Exception:
            return None
        if v != v:  # NaN
            return None
        return v

    def _should_update_best(self, cur: dict):
        """
        综合判定是否更新 best：
        1) mIoU 必须严格上涨；
        2) 其余关键指标（mAcc/PixAcc/FWIoU/mDice）不能下降；
        3) 且至少一个关键指标上涨（当 baseline 具备该指标时）。
        """
        eps = 1e-6
        cur_miou = self._metric_or_none(cur, "mIoU")
        if cur_miou is None:
            return False, "missing mIoU"
        if cur_miou <= float(self.best_mIoU) + eps:
            return False, f"mIoU not improved ({cur_miou:.2f} <= {self.best_mIoU:.2f})"

        aux_keys = ("mAcc", "PixAcc", "FWIoU", "mDice")
        saw_baseline = False
        improved_aux = False
        for k in aux_keys:
            cur_v = self._metric_or_none(cur, k)
            best_v = self._metric_or_none(self.best_metrics, k)
            if cur_v is None:
                continue
            if best_v is None:
                # 首次没有 baseline 时，不阻止更新；后续会记录 baseline。
                continue
            saw_baseline = True
            if cur_v + eps < best_v:
                return False, f"{k} dropped ({cur_v:.2f} < {best_v:.2f})"
            if cur_v > best_v + eps:
                improved_aux = True

        if saw_baseline and not improved_aux:
            return False, "mIoU improved but no auxiliary metric improved"
        return True, "all criteria satisfied"

    def need_pseudo_labels(self, epoch: int) -> bool:
        return (epoch % self.args.pseudo_every) == 0

    def generate_pseudo_labels(self, epoch: int):
        self.pl_update_epochs += 1
        self.save_dir = self._pl_epoch_dir(epoch)
        parent_save_dir = os.path.dirname(self.save_dir)
        if getattr(self.args, "regenerate_pl_each_epoch", False) and os.path.exists(self.save_dir):
            clear_dir(self.save_dir)
        ensure_dir(self.save_dir)
        max_num = self.args.warmup_pl if epoch == 0 else self.args.iters_pl
        self.is_warmup = True if epoch == 0 else False

        plg = PseudoLabelGenerator(
            model=self.model_t,
            device=self.device,
            class_names=self.class_names,
            num_classes=self.num_classes,
        )
        fixed_subset = self._load_pl_subset_filter()
        # Pass psr_pld config if present (set via YAML pseudo_label_reliability.pl_merge)
        _psr_pld_cfg = None
        _psr_raw = getattr(self.args, 'pseudo_label_reliability', None)
        if isinstance(_psr_raw, dict) and bool(_psr_raw.get('enabled', False)):
            _psr_pld_cfg = _psr_raw
        used_paths, pl_stats = plg.run(
            self.target_pl_loader,
            self.save_dir,
            max_num,
            self.args.dataset,
            self.args.uc_threshold,
            amp=bool(getattr(self.args, "amp", False)),
            patch_scheduler=self.patch_scheduler,
            fixed_subset=fixed_subset,
            timing_warmup_batches=int(getattr(self.args, "pl_timing_warmup_batches", 0)),
            psr_pld_cfg=_psr_pld_cfg,
        )
        self.last_pl_stats = dict(pl_stats or {})
        self.last_pl_stats["pseudo_label_update_ratio"] = float(self.pl_update_epochs / max(1, self.pl_total_epochs))
        self.last_pl_stats["pseudo_label_updated_this_epoch"] = 1.0
        self.last_pl_stats["epoch"] = int(epoch + 1)
        self.last_pl_stats["dataset"] = str(self.args.dataset)
        self.last_pl_stats["split"] = "target_train_for_pl"
        self.last_pl_stats["seed"] = int(getattr(self.args, "seed", 0))
        self.last_pl_stats["memory_bank_size"] = int(getattr(getattr(self.model_t, "memory_bank", None), "num_maskmem", getattr(self.args, "num_maskmem", 0)))
        self.last_pl_stats["compressor_mode"] = str(getattr(getattr(self.model_t, "memory_bank", None), "memory_compress_mode", "none"))
        self.last_pl_stats["compress_location"] = str(getattr(getattr(self.model_t, "memory_bank", None), "memory_compress_location", "before_attention"))
        try:
            rec_path = os.path.join(self._metrics_dir(), "pl_timing.jsonl")
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.last_pl_stats, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[warn] failed to write pl_timing.jsonl: {e}", flush=True)

        self.data_builder.write_splits_after_pl(parent_save_dir, used_paths)
        self._validate_pseudo_labels(self.save_dir, used_paths)

    def train(self, epoch: int):
        # SG-DRF training stage management
        if bool(getattr(self.args, "enable_sgdrf", False)):
            cold_epochs = int(getattr(self.args, "sgdrf_cold_start_epochs", 2))
            desired_stage = "cold_start" if epoch < cold_epochs else "joint"
            if hasattr(self.model_t, "sgdrf_training_stage") and self.model_t.sgdrf_training_stage != desired_stage:
                self.model_t.set_sgdrf_training_stage(desired_stage)
                print(f"[SG-DRF] Switched to {desired_stage} stage at epoch {epoch}", flush=True)
        t0 = time.perf_counter()
        if hasattr(self.model_t, "memory_bank"):
            self.model_t.memory_bank.attention_time_accum = 0.0
            self.model_t.memory_bank.attention_time_count = 0
        fixed_dir = getattr(self.args, "reuse_pl_dir", "")
        if fixed_dir:
            self._ensure_pseudo_labels_ready_dir(fixed_dir)
        else:
            if self.save_dir is None:
                self.save_dir = self._pl_epoch_dir(epoch)
            self._ensure_pseudo_labels_ready(epoch)
        self.target_loader, self.val_loader = self.data_builder.build_tgt_from_pd(self.save_dir)
        if self.source_loader is None:
            self.source_loader, _ = self.data_builder.build_src()
        start_iter = self.start_iter if epoch == self.start_epoch else 0
        self.global_step = self.trainer.train_one_epoch(
            epoch,
            self.global_step,
            self.source_loader,
            self.target_loader,
            self.is_warmup,
            start_iter=start_iter,
        )
        self.start_iter = 0
        elapsed = time.perf_counter() - t0
        train_stats = dict(getattr(self.trainer, "last_epoch_stats", {}) or {})
        train_stats.setdefault("epoch_time", float(elapsed))
        train_stats.setdefault("step_time", float(train_stats.get("step_time", 0.0)))
        train_stats["effective_memory_length"] = float(
            getattr(getattr(self.model_t, "memory_bank", None), "last_effective_memory_length", 0.0)
        )
        if torch.cuda.is_available():
            try:
                train_stats["gpu_mem_peak"] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                train_stats["peak_gpu_mem_train"] = float(train_stats["gpu_mem_peak"])
            except Exception:
                train_stats["gpu_mem_peak"] = float("nan")
                train_stats["peak_gpu_mem_train"] = float("nan")
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self.last_train_stats = train_stats
        self.last_train_stats["epoch"] = int(epoch + 1)
        self.last_train_stats["dataset"] = str(self.args.dataset)
        self.last_train_stats["seed"] = int(getattr(self.args, "seed", 0))
        self.last_train_stats["compressor_mode"] = str(getattr(getattr(self.model_t, "memory_bank", None), "memory_compress_mode", "none"))
        self.last_train_stats["compress_location"] = str(getattr(getattr(self.model_t, "memory_bank", None), "memory_compress_location", "before_attention"))
        self.last_train_stats["memory_bank_size"] = int(getattr(getattr(self.model_t, "memory_bank", None), "num_maskmem", getattr(self.args, "num_maskmem", 0)))
        # ── Task 2-2A: D/E token statistics (method A, ~7 lines) ─────────────────
        # Read already-computed attributes from memory_bank; null when module disabled.
        # compression_* = D module (SpatialMemoryCompressor), last-iter per-chunk stats.
        # sparse_*      = E module (SparseMemoryRetriever), last-iter all-chunk sum.
        # memory_tokens_{before,after}_final = existing concat-all-chunks totals (unchanged scope).
        _mb = getattr(self.model_t, "memory_bank", None)
        _cst = dict(getattr(_mb, "last_compression_stats", None) or {})
        _d_active = bool(_cst) and not _cst.get("skipped", True)
        self.last_train_stats["compression_tokens_before"] = float(_cst["original_num_tokens"])   if _d_active else None
        self.last_train_stats["compression_tokens_after"]  = float(_cst["compressed_num_tokens"]) if _d_active else None
        self.last_train_stats["compression_ratio"]         = float(_cst["compression_ratio"])     if _d_active else None
        _e_active = (
            _mb is not None
            and bool(getattr(_mb, "sparse_enabled", False))
            and str(getattr(_mb, "sparse_mode", "none")).lower() not in ("none", "")
        )
        self.last_train_stats["sparse_tokens_before"] = float(getattr(_mb, "last_sparse_tokens_before", 0.0)) if _e_active else None
        self.last_train_stats["sparse_tokens_after"]  = float(getattr(_mb, "last_sparse_tokens_after",  0.0)) if _e_active else None
        self.last_train_stats["sparse_keep_ratio"]    = float(getattr(_mb, "last_sparse_keep_ratio",    1.0)) if _e_active else None
        self.last_train_stats["memory_tokens_scope"]        = "per_attention_call_all_chunks_concat"
        self.last_train_stats["memory_tokens_before_final"] = float(self.last_train_stats.get("memory_tokens_before", 0.0))
        self.last_train_stats["memory_tokens_after_final"]  = float(self.last_train_stats.get("memory_tokens_after",  0.0))
        if _mb is not None and bool(getattr(_mb, "pano_smb_enabled", False)):
            try:
                _pano_stats = _mb.get_pano_smb_aggregate_stats()
                _pano_path = os.path.join(self._metrics_dir(), "pano_smb_quota_summary.json")
                with open(_pano_path, "w", encoding="utf-8") as _f:
                    json.dump(_pano_stats, _f, ensure_ascii=False, indent=2)
                self.last_train_stats["pano_smb_quota_summary"] = _pano_path
            except Exception as e:
                print(f"[warn] failed to write pano_smb_quota_summary.json: {e}", flush=True)
        # ─────────────────────────────────────────────────────────────────────────
        # Use patch count in target loader as effective patches in current training implementation.
        self.last_train_stats["avg_effective_patches_per_iter"] = float(self.last_train_stats.get("avg_effective_patches_per_iter", 0.0))
        # Required text-fusion dynamics for ablation diagnostics.
        dyn_keys = [
            "fusion_alpha",
            "text_scale",
            "text_logits_mean",
            "text_logits_std",
            "visual_logits_mean",
            "visual_logits_std",
            "final_logits_mean",
            "final_logits_std",
        ]
        for k in dyn_keys:
            self.last_train_stats.setdefault(f"{k}_start", 0.0)
            self.last_train_stats.setdefault(f"{k}_end", 0.0)
            self.last_train_stats.setdefault(f"{k}_mean", 0.0)
            self.last_train_stats.setdefault(f"{k}_std", 0.0)
        print(
            "[text-dyn] "
            f"epoch={epoch+1} "
            f"fusion_alpha(start={self.last_train_stats['fusion_alpha_start']:.6f},"
            f"end={self.last_train_stats['fusion_alpha_end']:.6f},"
            f"mean={self.last_train_stats['fusion_alpha_mean']:.6f},"
            f"std={self.last_train_stats['fusion_alpha_std']:.6f}) "
            f"text_scale(start={self.last_train_stats['text_scale_start']:.6f},"
            f"end={self.last_train_stats['text_scale_end']:.6f},"
            f"mean={self.last_train_stats['text_scale_mean']:.6f},"
            f"std={self.last_train_stats['text_scale_std']:.6f})",
            flush=True,
        )
        print(
            "[text-crossattn] "
            f"type={getattr(self.args, 'text_fusion_type', 'none')} "
            f"proto_mode={getattr(self.args, 'text_proto_mode', 'real')} "
            f"beta={self.last_train_stats.get('cross_beta_mean', 0.0):.6f} "
            f"attn_entropy={self.last_train_stats.get('cross_attn_entropy_mean', 0.0):.6f} "
            f"feature_delta_ratio={self.last_train_stats.get('feature_delta_ratio_mean', 0.0):.6f} "
            f"prediction_change_ratio={self.last_train_stats.get('prediction_change_ratio_mean', 0.0):.6f}",
            flush=True,
        )
        print(
            "[text-grounding] "
            f"type={getattr(self.args, 'text_fusion_type', 'none')} "
            f"proto_mode={getattr(self.args, 'text_proto_mode', 'real')} "
            f"gamma={self.last_train_stats.get('grounding_gamma_mean', 0.0):.6f} "
            f"logit_scale={self.last_train_stats.get('grounding_logit_scale_mean', 0.0):.6f} "
            f"attn_entropy={self.last_train_stats.get('text_to_image_attn_entropy_mean', 0.0):.6f} "
            f"text_only_change={self.last_train_stats.get('text_only_prediction_change_ratio_mean', 0.0):.6f} "
            f"loss_ground={self.last_train_stats.get('loss_text_grounding_ce_mean', 0.0):.6f}",
            flush=True,
        )
        try:
            rec_path = os.path.join(self._metrics_dir(), "train_timing.jsonl")
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.last_train_stats, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[warn] failed to write train_timing.jsonl: {e}", flush=True)
        # ---- Geo-TeCMA geometry diagnostic jsonl (diagnosis-only; no selection changes) ----
        try:
            _geo_recs = list(getattr(self.trainer, "geo_memory_diag_records", []) or [])
            if _geo_recs:
                _geo_cfg = {}
                if getattr(self.plugins, "cfg", None) and isinstance(self.plugins.cfg, dict):
                    _geo_cfg = self.plugins.cfg.get("geo_memory", {}) or {}
                _geo_path = str(_geo_cfg.get("log_jsonl", "experiments/geo_memory/results/geo_memory_diag.jsonl"))
                _geo_dir = os.path.dirname(_geo_path)
                if _geo_dir:
                    os.makedirs(_geo_dir, exist_ok=True)
                with open(_geo_path, "a", encoding="utf-8") as _f:
                    for _rec in _geo_recs:
                        _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                print(
                    f"[geo_memory_diag] wrote {len(_geo_recs)} frame records -> {_geo_path}",
                    flush=True,
                )
        except Exception as e:
            print(f"[warn] failed to write geo_memory_diag jsonl: {e}", flush=True)
        # ---- end Geo-TeCMA geometry diagnostic jsonl ----
        # ---- Geo-TeCMA token geometry diagnosis-only writers ----
        try:
            _geo_cfg = {}
            if getattr(self.plugins, "cfg", None) and isinstance(self.plugins.cfg, dict):
                _geo_cfg = self.plugins.cfg.get("geo_memory", {}) or {}
            _writer_cfg = _geo_cfg.get("token_writer", {}) if isinstance(_geo_cfg, dict) else {}
            if bool(_writer_cfg.get("enabled", False)):
                _summary_recs = list(getattr(self.trainer, "geo_memory_token_summary_records", []) or [])
                _token_recs = list(getattr(self.trainer, "geo_memory_token_sample_records", []) or [])
                _summary_path = str(_writer_cfg.get("summary_jsonl", "experiments/geo_memory/results/token_geometry_summary.jsonl"))
                _token_path = str(_writer_cfg.get("token_jsonl", "experiments/geo_memory/results/token_geometry_diag.jsonl"))
                if _summary_recs:
                    _summary_dir = os.path.dirname(_summary_path)
                    if _summary_dir:
                        os.makedirs(_summary_dir, exist_ok=True)
                    with open(_summary_path, "a", encoding="utf-8") as _f:
                        for _rec in _summary_recs:
                            _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                    print(f"[geo_token_summary] wrote {len(_summary_recs)} records -> {_summary_path}", flush=True)
                if _token_recs:
                    _token_dir = os.path.dirname(_token_path)
                    if _token_dir:
                        os.makedirs(_token_dir, exist_ok=True)
                    with open(_token_path, "a", encoding="utf-8") as _f:
                        for _rec in _token_recs:
                            _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                    print(f"[geo_token_diag] wrote {len(_token_recs)} sample token records -> {_token_path}", flush=True)
        except Exception as e:
            print(f"[warn] failed to write geo token diagnosis jsonl: {e}", flush=True)
        # ---- end Geo-TeCMA token geometry diagnosis-only writers ----
        # ---- PLQ diagnostic jsonl (R3 only, zero overhead when mode != plq_basic) ----
        try:
            _plq_recs = list(getattr(self.trainer, "plq_diag_records", []) or [])
            if _plq_recs:
                _plq_diag_dir = os.path.join("experiments", "pl_refiner", "results")
                os.makedirs(_plq_diag_dir, exist_ok=True)
                _plq_path = os.path.join(_plq_diag_dir, "r3_plq_5step_diag.jsonl")
                with open(_plq_path, "a", encoding="utf-8") as _f:
                    for _rec in _plq_recs:
                        _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                print(
                    f"[plq_diag] wrote {len(_plq_recs)} step records -> {_plq_path}",
                    flush=True,
                )
        except Exception as e:
            print(f"[warn] failed to write r3_plq_5step_diag.jsonl: {e}", flush=True)
        # ---- end PLQ diagnostic jsonl ----
        # ---- R4 prototype similarity diagnostic jsonl (diagnostic-only) ----
        try:
            _proto_recs = list(getattr(self.trainer, "proto_sim_diag_records", []) or [])
            if _proto_recs:
                _proto_diag_dir = os.path.join("experiments", "pl_refiner", "results")
                os.makedirs(_proto_diag_dir, exist_ok=True)
                _proto_cfg = getattr(self.args, "pseudo_label_refiner", {}) or {}
                _proto_diag_cfg = _proto_cfg.get("prototype_diagnostics", {}) if isinstance(_proto_cfg, dict) else {}
                _proto_filename = str(_proto_diag_cfg.get("output_jsonl", "r4_proto_similarity_5step_diag.jsonl"))
                if str(_proto_cfg.get("mode", "")).lower() == "proto_gate" and "output_jsonl" not in _proto_diag_cfg:
                    _proto_filename = "r4_proto_gate_5step_diag.jsonl"
                _proto_path = os.path.join(_proto_diag_dir, _proto_filename)
                with open(_proto_path, "a", encoding="utf-8") as _f:
                    for _rec in _proto_recs:
                        _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                print(
                    f"[proto_sim_diag] wrote {len(_proto_recs)} step records -> {_proto_path}",
                    flush=True,
                )
        except Exception as e:
            print(f"[warn] failed to write r4_proto_similarity_5step_diag.jsonl: {e}", flush=True)
        # ---- end R4 prototype similarity diagnostic jsonl ----
        try:
            dyn_path = os.path.join(self._metrics_dir(), "text_fusion_dynamics.jsonl")
            dyn_rec = {
                "epoch": int(epoch + 1),
                "fusion_alpha_start": float(self.last_train_stats["fusion_alpha_start"]),
                "fusion_alpha_end": float(self.last_train_stats["fusion_alpha_end"]),
                "fusion_alpha_mean": float(self.last_train_stats["fusion_alpha_mean"]),
                "fusion_alpha_std": float(self.last_train_stats["fusion_alpha_std"]),
                "text_scale_start": float(self.last_train_stats["text_scale_start"]),
                "text_scale_end": float(self.last_train_stats["text_scale_end"]),
                "text_scale_mean": float(self.last_train_stats["text_scale_mean"]),
                "text_scale_std": float(self.last_train_stats["text_scale_std"]),
                "text_logits_mean_mean": float(self.last_train_stats["text_logits_mean_mean"]),
                "text_logits_mean_std": float(self.last_train_stats["text_logits_mean_std"]),
                "text_logits_std_mean": float(self.last_train_stats["text_logits_std_mean"]),
                "text_logits_std_std": float(self.last_train_stats["text_logits_std_std"]),
                "visual_logits_mean_mean": float(self.last_train_stats["visual_logits_mean_mean"]),
                "visual_logits_mean_std": float(self.last_train_stats["visual_logits_mean_std"]),
                "visual_logits_std_mean": float(self.last_train_stats["visual_logits_std_mean"]),
                "visual_logits_std_std": float(self.last_train_stats["visual_logits_std_std"]),
                "final_logits_mean_mean": float(self.last_train_stats["final_logits_mean_mean"]),
                "final_logits_mean_std": float(self.last_train_stats["final_logits_mean_std"]),
                "final_logits_std_mean": float(self.last_train_stats["final_logits_std_mean"]),
                "final_logits_std_std": float(self.last_train_stats["final_logits_std_std"]),
                "cross_beta_mean": float(self.last_train_stats.get("cross_beta_mean", 0.0)),
                "cross_attn_mean_mean": float(self.last_train_stats.get("cross_attn_mean_mean", 0.0)),
                "cross_attn_std_mean": float(self.last_train_stats.get("cross_attn_std_mean", 0.0)),
                "cross_attn_entropy_mean": float(self.last_train_stats.get("cross_attn_entropy_mean", 0.0)),
                "cross_attn_max_mean_mean": float(self.last_train_stats.get("cross_attn_max_mean_mean", 0.0)),
                "feature_delta_ratio_mean": float(self.last_train_stats.get("feature_delta_ratio_mean", 0.0)),
                "prediction_change_ratio_mean": float(self.last_train_stats.get("prediction_change_ratio_mean", 0.0)),
                "grounding_gamma_mean": float(self.last_train_stats.get("grounding_gamma_mean", 0.0)),
                "grounding_logit_scale_mean": float(self.last_train_stats.get("grounding_logit_scale_mean", 0.0)),
                "text_to_image_attn_entropy_mean": float(self.last_train_stats.get("text_to_image_attn_entropy_mean", 0.0)),
                "text_grounding_logits_mean_mean": float(self.last_train_stats.get("text_grounding_logits_mean_mean", 0.0)),
                "text_grounding_logits_std_mean": float(self.last_train_stats.get("text_grounding_logits_std_mean", 0.0)),
                "text_only_prediction_change_ratio_mean": float(self.last_train_stats.get("text_only_prediction_change_ratio_mean", 0.0)),
                "loss_text_grounding_ce_mean": float(self.last_train_stats.get("loss_text_grounding_ce_mean", 0.0)),
            }
            with open(dyn_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dyn_rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[warn] failed to write text_fusion_dynamics.jsonl: {e}", flush=True)
        try:
            iter_dyn_path = os.path.join(self._metrics_dir(), "text_dynamics.jsonl")
            with open(iter_dyn_path, "a", encoding="utf-8") as f:
                for rec in list(getattr(self.trainer, "last_iter_metric_records", []) or []):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[warn] failed to write text_dynamics.jsonl: {e}", flush=True)
        self._save_resume_state(epoch + 1, 0)

    def validate(self, epoch: int):
        metrics = self.trainer.validate(epoch, self.val_loader, self.class_names)
        cur_mIoU = float(metrics.get("mIoU", float("nan")))
        should_update, reason = self._should_update_best(metrics)
        if should_update:
            self.best_mIoU = cur_mIoU
            self.best_metrics = {
                "mIoU": self._metric_or_none(metrics, "mIoU"),
                "mAcc": self._metric_or_none(metrics, "mAcc"),
                "PixAcc": self._metric_or_none(metrics, "PixAcc"),
                "FWIoU": self._metric_or_none(metrics, "FWIoU"),
                "mDice": self._metric_or_none(metrics, "mDice"),
            }
            self.best_ckpt_path = os.path.join(self._save_dir(), f"best_iou{cur_mIoU:.2f}.pth")
            torch.save(self.model_t.state_dict(), self.best_ckpt_path)
        self._save_resume_state(epoch + 1, 0)
        print(
            f"Epoch {epoch+1}: val mIoU={cur_mIoU:.2f}, "
            f"mAcc={metrics.get('mAcc', float('nan')):.2f}, "
            f"PixAcc={metrics.get('PixAcc', float('nan')):.2f}, "
            f"FWIoU={metrics.get('FWIoU', float('nan')):.2f}, "
            f"mDice={metrics.get('mDice', float('nan')):.2f}, "
            f"best={self.best_mIoU:.2f}; update_best={should_update} ({reason})"
        )
        # Persist a machine-readable metrics log for automation scripts.
        try:
            metrics_path = os.path.join(self._save_dir(), "metrics.jsonl")
            rec = {
                "epoch": int(epoch + 1),
                "val_mIoU": float(cur_mIoU),
                "val_mAcc": self._metric_or_none(metrics, "mAcc"),
                "val_PixAcc": self._metric_or_none(metrics, "PixAcc"),
                "val_FWIoU": self._metric_or_none(metrics, "FWIoU"),
                "val_mDice": self._metric_or_none(metrics, "mDice"),
                "best_mIoU": float(self.best_mIoU),
                "best_metrics": self.best_metrics,
                "best_ckpt_path": self.best_ckpt_path,
                "update_best": bool(should_update),
                "update_reason": reason,
                "reuse_pl_dir": getattr(self.args, "reuse_pl_dir", ""),
                "avg_patch_count_before": self.last_pl_stats.get("avg_patch_count_before", "NA"),
                "avg_patch_count_after": self.last_pl_stats.get("avg_patch_count_after", "NA"),
                "patch_reduction_ratio": self.last_pl_stats.get("patch_reduction_ratio", "NA"),
                "pseudo_label_generation_time": self.last_pl_stats.get("pseudo_label_generation_time", "NA"),
                "pseudo_label_update_ratio": self.last_pl_stats.get(
                    "pseudo_label_update_ratio",
                    float(self.pl_update_epochs / max(1, self.pl_total_epochs)) if self.pl_total_epochs > 0 else "NA",
                ),
                "step_time": self.last_train_stats.get("step_time", "NA"),
                "epoch_time": self.last_train_stats.get("epoch_time", "NA"),
                "gpu_mem_peak": self.last_train_stats.get("gpu_mem_peak", "NA"),
                "effective_memory_length": self.last_train_stats.get("effective_memory_length", "NA"),
                "pl_patch_prepare_time": self.last_pl_stats.get("pl_patch_prepare_time", "NA"),
                "pl_forward_time": self.last_pl_stats.get("pl_forward_time", "NA"),
                "pl_backward_time": self.last_pl_stats.get("pl_backward_time", "NA"),
                "pl_merge_time": self.last_pl_stats.get("pl_merge_time", "NA"),
                "pl_confidence_filter_time": self.last_pl_stats.get("pl_confidence_filter_time", "NA"),
                "pl_total_time": self.last_pl_stats.get("pl_total_time", "NA"),
                "pl_time_per_image": self.last_pl_stats.get("pl_time_per_image", "NA"),
                "pl_images_per_sec": self.last_pl_stats.get("pl_images_per_sec", "NA"),
                "raw_patch_count_total": self.last_pl_stats.get("raw_patch_count_total", "NA"),
                "effective_patch_count_total": self.last_pl_stats.get("effective_patch_count_total", "NA"),
                "high_conf_ratio": self.last_pl_stats.get("high_conf_ratio", "NA"),
                "uncertain_ratio": self.last_pl_stats.get("uncertain_ratio", "NA"),
                "valid_pseudo_ratio": self.last_pl_stats.get("valid_pseudo_ratio", "NA"),
                "peak_gpu_mem_pl": self.last_pl_stats.get("peak_gpu_mem_pl", "NA"),
                "memory_tokens_before": self.last_train_stats.get("memory_tokens_before", "NA"),
                "memory_tokens_after": self.last_train_stats.get("memory_tokens_after", "NA"),
                "memory_token_ratio": self.last_train_stats.get("memory_token_ratio", "NA"),
                "compressor_mode": self.last_train_stats.get("compressor_mode", "NA"),
                "compress_location": self.last_train_stats.get("compress_location", "NA"),
                "data_time": self.last_train_stats.get("data_time", "NA"),
                "iter_compute_time": self.last_train_stats.get("iter_compute_time", "NA"),
                "iter_total_time": self.last_train_stats.get("iter_total_time", "NA"),
                "train_images_per_sec": self.last_train_stats.get("train_images_per_sec", "NA"),
                "train_patches_per_sec": self.last_train_stats.get("train_patches_per_sec", "NA"),
                "epoch_train_time": self.last_train_stats.get("epoch_train_time", "NA"),
                "epoch_total_time": self.last_train_stats.get("epoch_total_time", "NA"),
                "fusion_alpha_start": self.last_train_stats.get("fusion_alpha_start", "NA"),
                "fusion_alpha_end": self.last_train_stats.get("fusion_alpha_end", "NA"),
                "fusion_alpha_mean": self.last_train_stats.get("fusion_alpha_mean", "NA"),
                "fusion_alpha_std": self.last_train_stats.get("fusion_alpha_std", "NA"),
                "text_scale_start": self.last_train_stats.get("text_scale_start", "NA"),
                "text_scale_end": self.last_train_stats.get("text_scale_end", "NA"),
                "text_scale_mean": self.last_train_stats.get("text_scale_mean", "NA"),
                "text_scale_std": self.last_train_stats.get("text_scale_std", "NA"),
                "text_logits_mean_mean": self.last_train_stats.get("text_logits_mean_mean", "NA"),
                "text_logits_mean_std": self.last_train_stats.get("text_logits_mean_std", "NA"),
                "text_logits_std_mean": self.last_train_stats.get("text_logits_std_mean", "NA"),
                "text_logits_std_std": self.last_train_stats.get("text_logits_std_std", "NA"),
                "visual_logits_mean_mean": self.last_train_stats.get("visual_logits_mean_mean", "NA"),
                "visual_logits_mean_std": self.last_train_stats.get("visual_logits_mean_std", "NA"),
                "visual_logits_std_mean": self.last_train_stats.get("visual_logits_std_mean", "NA"),
                "visual_logits_std_std": self.last_train_stats.get("visual_logits_std_std", "NA"),
                "final_logits_mean_mean": self.last_train_stats.get("final_logits_mean_mean", "NA"),
                "final_logits_mean_std": self.last_train_stats.get("final_logits_mean_std", "NA"),
                "final_logits_std_mean": self.last_train_stats.get("final_logits_std_mean", "NA"),
                "final_logits_std_std": self.last_train_stats.get("final_logits_std_std", "NA"),
                "cross_beta_mean": self.last_train_stats.get("cross_beta_mean", "NA"),
                "cross_beta_grad_norm_mean": self.last_train_stats.get("cross_beta_grad_norm_mean", "NA"),
                "cross_attn_mean_mean": self.last_train_stats.get("cross_attn_mean_mean", "NA"),
                "cross_attn_std_mean": self.last_train_stats.get("cross_attn_std_mean", "NA"),
                "cross_attn_entropy_mean": self.last_train_stats.get("cross_attn_entropy_mean", "NA"),
                "cross_attn_max_mean_mean": self.last_train_stats.get("cross_attn_max_mean_mean", "NA"),
                "visual_feature_mean_mean": self.last_train_stats.get("visual_feature_mean_mean", "NA"),
                "visual_feature_std_mean": self.last_train_stats.get("visual_feature_std_mean", "NA"),
                "fused_feature_mean_mean": self.last_train_stats.get("fused_feature_mean_mean", "NA"),
                "fused_feature_std_mean": self.last_train_stats.get("fused_feature_std_mean", "NA"),
                "feature_delta_mean_mean": self.last_train_stats.get("feature_delta_mean_mean", "NA"),
                "feature_delta_norm_mean": self.last_train_stats.get("feature_delta_norm_mean", "NA"),
                "feature_delta_ratio_mean": self.last_train_stats.get("feature_delta_ratio_mean", "NA"),
                "logits_delta_mean_mean": self.last_train_stats.get("logits_delta_mean_mean", "NA"),
                "logits_delta_norm_mean": self.last_train_stats.get("logits_delta_norm_mean", "NA"),
                "prediction_change_ratio_mean": self.last_train_stats.get("prediction_change_ratio_mean", "NA"),
                "prediction_change_ratio_std": self.last_train_stats.get("prediction_change_ratio_std", "NA"),
                "prediction_change_ratio_cls_1_mean": self.last_train_stats.get("prediction_change_ratio_cls_1_mean", "NA"),
                "prediction_change_ratio_cls_3_mean": self.last_train_stats.get("prediction_change_ratio_cls_3_mean", "NA"),
                "prediction_change_ratio_cls_4_mean": self.last_train_stats.get("prediction_change_ratio_cls_4_mean", "NA"),
                "prediction_change_ratio_cls_5_mean": self.last_train_stats.get("prediction_change_ratio_cls_5_mean", "NA"),
                "prediction_change_ratio_cls_6_mean": self.last_train_stats.get("prediction_change_ratio_cls_6_mean", "NA"),
                "prediction_change_ratio_cls_7_mean": self.last_train_stats.get("prediction_change_ratio_cls_7_mean", "NA"),
                "prediction_change_ratio_cls_9_mean": self.last_train_stats.get("prediction_change_ratio_cls_9_mean", "NA"),
                "cross_attn_grad_norm_mean": self.last_train_stats.get("cross_attn_grad_norm_mean", "NA"),
                "text_projector_grad_norm_mean": self.last_train_stats.get("text_adapter_grad_norm_mean", "NA"),
                "grounding_gamma_mean": self.last_train_stats.get("grounding_gamma_mean", "NA"),
                "grounding_logit_scale_mean": self.last_train_stats.get("grounding_logit_scale_mean", "NA"),
                "grounding_gamma_grad_norm_mean": self.last_train_stats.get("grounding_gamma_grad_norm_mean", "NA"),
                "grounding_logit_scale_grad_norm_mean": self.last_train_stats.get("grounding_logit_scale_grad_norm_mean", "NA"),
                "query_self_attn_grad_norm_mean": self.last_train_stats.get("query_self_attn_grad_norm_mean", "NA"),
                "query_cross_attn_grad_norm_mean": self.last_train_stats.get("query_cross_attn_grad_norm_mean", "NA"),
                "query_ffn_grad_norm_mean": self.last_train_stats.get("query_ffn_grad_norm_mean", "NA"),
                "text_to_image_attn_mean_mean": self.last_train_stats.get("text_to_image_attn_mean_mean", "NA"),
                "text_to_image_attn_std_mean": self.last_train_stats.get("text_to_image_attn_std_mean", "NA"),
                "text_to_image_attn_entropy_mean": self.last_train_stats.get("text_to_image_attn_entropy_mean", "NA"),
                "text_to_image_attn_max_mean": self.last_train_stats.get("text_to_image_attn_max_mean", "NA"),
                "text_grounding_logits_mean_mean": self.last_train_stats.get("text_grounding_logits_mean_mean", "NA"),
                "text_grounding_logits_std_mean": self.last_train_stats.get("text_grounding_logits_std_mean", "NA"),
                "text_logits_delta_norm_mean": self.last_train_stats.get("text_logits_delta_norm_mean", "NA"),
                "final_visual_logits_delta_norm_mean": self.last_train_stats.get("final_visual_logits_delta_norm_mean", "NA"),
                "text_only_prediction_change_ratio_mean": self.last_train_stats.get("text_only_prediction_change_ratio_mean", "NA"),
                "loss_text_grounding_ce_mean": self.last_train_stats.get("loss_text_grounding_ce_mean", "NA"),
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[warn] failed to write metrics.jsonl: {e}", flush=True)

    def step_epoch_end(self, epoch: int):
        if (epoch + 1) % 5 == 0:
            self.trainer.reset_optimizer(lr=self.args.lr)

    def _save_dir(self):
        class_set = getattr(self.args, "class_set", 13)
        d = os.path.join(self.args.pd_root, 'checkpoints', f"{self.args.dataset}_{class_set}cls", self.args.backbone)
        ensure_dir(d)
        return d

    def _pl_epoch_dir(self, epoch: int):
        if self.args.dataset == 'Stanford2D3D':
            return os.path.join(self.args.pd_root, 'Stanford2D3D', self.args.backbone, f'epoch{epoch + 1}')
        class_set = getattr(self.args, "class_set", 13)
        return os.path.join(self.args.pd_root, 'DensePASS', self.args.backbone, f"{self.args.dataset}_{class_set}cls", f'epoch{epoch + 1}')

    def _resume_state_path(self):
        return os.path.join(self._save_dir(), f"resume_{self.args.backbone}.pth")

    def _save_resume_state(self, next_epoch: int, next_iter: int):
        state = {
            "model_t": self.model_t.state_dict(),
            "model_s": self.model_s.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_mIoU": self.best_mIoU,
            "best_metrics": self.best_metrics,
            "best_ckpt_path": self.best_ckpt_path,
            "next_epoch": next_epoch,
            "next_iter": next_iter,
            "save_dir": self.save_dir,
        }
        save_checkpoint(state, self._resume_state_path())

    def _try_resume(self):
        resume_path = self.args.resume_ckpt or self._resume_state_path()
        if not resume_path or not os.path.isfile(resume_path):
            return
        try:
            state = load_checkpoint_file(resume_path, map_location="cpu")
        except (RuntimeError, OSError) as exc:
            # A previously interrupted write can leave a corrupted resume file.
            # Ignore it and continue from scratch for this run.
            print(f"[warn] skip corrupted resume state: {resume_path} ({exc})", flush=True)
            return
        self.model_t.load_state_dict(state["model_t"], strict=False)
        self.model_s.load_state_dict(state["model_s"], strict=False)
        self.trainer.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self.best_mIoU = state.get("best_mIoU", 0.0)
        self.best_metrics = state.get("best_metrics", self.best_metrics)
        self.best_ckpt_path = state.get("best_ckpt_path")
        self.start_epoch = state.get("next_epoch", 0)
        self.start_iter = state.get("next_iter", 0)
        self.save_dir = state.get("save_dir")
        self.resume_path = resume_path
        print(
            f"Resumed adaptation state from: {resume_path}; next epoch={self.start_epoch + 1}, next iter={self.start_iter}",
            flush=True,
        )

    def _expected_pl_labels(self, save_dir, used_paths):
        return [
            os.path.join(save_dir, "gtFine", name.lstrip("/").replace(".jpg", "_labelTrainIds.png"))
            for name in used_paths
        ]

    def _validate_pseudo_labels(self, save_dir: str, used_paths):
        if not getattr(self.args, "validate_pseudo_labels", True):
            return
        expected = self._expected_pl_labels(save_dir, used_paths)
        missing = [p for p in expected if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"Pseudo-label integrity check failed under {save_dir}. Missing {len(missing)} files. Examples: {missing[:3]}"
            )

    def _load_used_paths(self, epoch: int):
        used_path_file = os.path.join(os.path.dirname(self._pl_epoch_dir(epoch)), 'used_path.txt')
        if not os.path.isfile(used_path_file):
            return []
        with open(used_path_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _load_used_paths_from_save_dir(self, save_dir: str):
        used_path_file = os.path.join(os.path.dirname(save_dir), 'used_path.txt')
        if not os.path.isfile(used_path_file):
            return []
        with open(used_path_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _ensure_pseudo_labels_ready_dir(self, save_dir: str):
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            raise FileNotFoundError(f"Pseudo-label dir not found: {self.save_dir}")
        used_paths = self._load_used_paths_from_save_dir(self.save_dir)
        if not used_paths:
            raise FileNotFoundError(
                f"Missing used_path.txt next to pseudo-label dir: {os.path.dirname(self.save_dir)}"
            )
        self._validate_pseudo_labels(self.save_dir, used_paths)

    def has_valid_pseudo_labels(self, epoch: int) -> bool:
        save_dir = self._pl_epoch_dir(epoch)
        if not os.path.isdir(save_dir):
            return False
        used_paths = self._load_used_paths(epoch)
        if not used_paths:
            return False
        try:
            self._validate_pseudo_labels(save_dir, used_paths)
            return True
        except FileNotFoundError:
            return False

    def _ensure_pseudo_labels_ready(self, epoch: int):
        self.save_dir = self._pl_epoch_dir(epoch)
        if not os.path.isdir(self.save_dir):
            self.generate_pseudo_labels(epoch)
            return
        used_paths = self._load_used_paths(epoch)
        if not used_paths:
            if getattr(self.args, "auto_rebuild_pseudo_labels", True):
                print(f"Pseudo-label metadata missing for epoch {epoch + 1}, rebuilding...", flush=True)
                self.generate_pseudo_labels(epoch)
                return
            raise FileNotFoundError(f"Missing used_path.txt for epoch {epoch + 1}: {self.save_dir}")
        try:
            self._validate_pseudo_labels(self.save_dir, used_paths)
        except FileNotFoundError:
            if getattr(self.args, "auto_rebuild_pseudo_labels", True):
                print(f"Pseudo-labels incomplete for epoch {epoch + 1}, rebuilding...", flush=True)
                self.generate_pseudo_labels(epoch)
            else:
                raise

    def _on_train_iter_end(self, epoch: int, next_iter: int, global_step: int):
        self.global_step = global_step
        if self.resume_save_disabled:
            return
        try:
            self._save_resume_state(epoch, next_iter)
        except (RuntimeError, OSError) as exc:
            # Resume-state checkpoint is a robustness feature for long runs.
            # Do not abort the whole experiment if storage has a transient write failure.
            self.resume_save_disabled = True
            print(f"[warn] disable resume-state saving due to write failure: {exc}", flush=True)

    def get_pl_update_decision(self, epoch: int):
        """
        Return per-epoch pseudo-label update decision from controller.
        Also tracks update ratio for logging.
        """
        self.pl_total_epochs += 1
        decision = self.pl_update_controller.decide(epoch) if self.pl_update_controller is not None else None
        if decision is None:
            class _D:  # fallback object
                should_update = True
                reuse_epoch = epoch
                reason = "controller_none"
            decision = _D()
        if bool(decision.should_update):
            self.last_pl_stats["pseudo_label_updated_this_epoch"] = 1.0
        else:
            self.last_pl_stats["pseudo_label_updated_this_epoch"] = 0.0
        self.last_pl_stats["pseudo_label_update_ratio"] = float(self.pl_update_epochs / max(1, self.pl_total_epochs))
        return decision
