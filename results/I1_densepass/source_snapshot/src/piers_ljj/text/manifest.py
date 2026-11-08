"""Query/image manifests: fixed ids, splits, GT isolation. R1/R5.

Worker side (text pipeline, omni worker) reads ONLY image + query text.
Evaluator side reads GT. Same manifest file carries both halves, but the
loader enforces the split: `worker_view()` strips every GT field.

Subsets: category_union vs instance_referring reported separately; negatives
require a verifiable missing-basis (taxonomy-missing != absent).
"""
from __future__ import annotations
import json


WORKER_OK_FIELDS = {"query_id", "image_id", "raw_text", "norm_text",
                    "task_type", "ontology", "language", "query_set_version"}


def make_manifest(images: list, queries: list, split: str,
                  query_set_version: str = "v0") -> dict:
    """images: [{image_id, path, width, height, ...}].
    queries: [{query_id, image_id, raw_text, task_type, ontology?, gt?...}].
    """
    for q in queries:
        assert q["task_type"] in ("category_union", "instance_referring"), q
    return {"split": split, "query_set_version": query_set_version,
            "images": images, "queries": queries}


def worker_view(manifest: dict) -> dict:
    """Strip GT/eval-only fields for inference workers."""
    wq = [{k: v for k, v in q.items() if k in WORKER_OK_FIELDS}
          for q in manifest["queries"]]
    return {"split": manifest["split"],
            "query_set_version": manifest["query_set_version"],
            "images": manifest["images"], "queries": wq}


def eval_view(manifest: dict) -> dict:
    return manifest


def subset_ids(manifest: dict, task_type: str) -> list:
    return [q["query_id"] for q in manifest["queries"]
            if q["task_type"] == task_type]


def negative_ids(manifest: dict) -> list:
    return [q["query_id"] for q in manifest["queries"]
            if q.get("negative") is True]


def save(path: str, manifest: dict) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
