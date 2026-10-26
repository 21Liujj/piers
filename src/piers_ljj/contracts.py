"""Unified I/O contracts (protocol data entities). R1/R5.

Entities: ImageRecord, QueryRecord, FoVRecord, TextPrediction,
SemanticPrediction, RunManifest. Validators raise ValueError on violation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import hashlib, json


TaskType = Literal["category_union", "instance_referring"]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class ImageRecord:
    image_id: str
    path: str
    width: int
    height: int
    checksum: str = ""
    dataset: str = ""
    split: str = ""

    def validate(self):
        assert self.image_id and self.path, "image_id/path required"
        assert self.width > 0 and self.height > 0, "bad H/W"


@dataclass
class QueryRecord:
    query_id: str
    image_id: str
    raw_text: str
    norm_text: str
    task_type: str  # category_union | instance_referring
    ontology: Optional[dict] = None  # e.g. {"map13": 12, "relation": "exact|coarse_compatible|unmapped"}
    language: str = "en"
    query_set_version: str = "v0"

    def validate(self):
        assert self.query_id and self.image_id and self.raw_text, "query fields required"
        assert self.task_type in ("category_union", "instance_referring"), f"bad task_type {self.task_type}"


@dataclass
class FoVRecord:
    view_id: str
    image_id: str
    erp_w: int
    erp_h: int
    infer_w: int
    infer_h: int
    x_start: int  # may be negative or >= W (wrap allowed); canonical mod applied in geometry
    y_start: int
    win_w: int
    win_h: int
    wrap_x: bool = True
    scale: float = 1.0  # infer_erp_w / orig_erp_w
    pad: tuple = (0, 0, 0, 0)  # left,top,right,bottom letterbox on model input
    order: int = 0

    def validate(self):
        assert self.win_w > 0 and self.win_h > 0
        assert 0 <= self.y_start <= self.erp_h - self.win_h, "y_start out of range"


@dataclass
class TextPrediction:
    query_id: str
    image_id: str
    instance_id: str
    fov_id: str
    box_xyxy: Optional[tuple]  # global ERP coords, None if empty
    mask_path: str = ""  # or RLE string
    detector_score: float = 0.0
    sam_quality: float = 0.0
    phrase: str = ""
    assoc: dict = field(default_factory=dict)
    reject_reason: str = ""
    synthetic: bool = False

    def validate(self):
        assert self.query_id and self.image_id and self.instance_id
        if self.box_xyxy is not None:
            assert len(self.box_xyxy) == 4


@dataclass
class SemanticPrediction:
    image_id: str
    class_space: str  # e.g. "densepass13@v1"
    pred_path: str
    conf_path: str = ""
    entropy: float = 0.0
    coverage: float = 1.0
    ckpt_sha256: str = ""
    synthetic: bool = False


@dataclass
class RunManifest:
    run_id: str
    code_commit: str = "uncommitted"
    upstream_omni: str = ""
    upstream_grounded: str = ""
    config: dict = field(default_factory=dict)
    seed: int = 0
    model_hash: str = ""
    env: dict = field(default_factory=dict)
    device: str = "cpu"
    status: str = "not_run"  # not_run|passed_cpu|pending_gpu|passed_gpu_smoke|completed_real|blocked|failed
    synthetic: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
