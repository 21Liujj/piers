"""Regression tests for real-backend integration hazards found on 2026-10-29.

These guard two bugs that would only surface on a GPU host, where they are far
more expensive to debug:

1. `post_process_grounded_object_detection` renamed its box-threshold kwarg
   (`threshold` -> `box_threshold`) between transformers releases. Hard-coding
   either name breaks the worker on the other version.
2. The DINO backend may return 2 values (boxes, scores) or 3 (boxes, scores,
   phrases). The adapter must accept both without dropping the phrase.
"""
import numpy as np
from PIL import Image

from piers_ljj.text.grounded_adapter import GroundedConfig, run_fov_text_branch
from piers_ljj.workers.entry import _post_process_grounded


class _FakeProcessor:
    """Mimics a transformers processor with an EXPLICIT signature.

    Real transformers declares the box threshold as a named parameter, so the
    fake must too — otherwise `**kwargs` hides which name was actually used.
    """

    def __init__(self, param_name):
        self.param_name = param_name
        self.received = None

    def post_process_grounded_object_detection(self, outputs, input_ids,
                                               text_threshold=None,
                                               target_sizes=None,
                                               **extra):
        if self.param_name == "box_threshold":
            if "box_threshold" not in extra:
                raise TypeError("unexpected keyword argument 'box_threshold'")
            if "threshold" in extra:
                raise TypeError("unexpected keyword argument 'threshold'")
            box = extra["box_threshold"]
        else:
            if "threshold" not in extra:
                raise TypeError("unexpected keyword argument 'threshold'")
            if "box_threshold" in extra:
                raise TypeError("unexpected keyword argument 'box_threshold'")
            box = extra["threshold"]
        self.received = dict(extra)
        self.received["text_threshold"] = text_threshold
        self.received["target_sizes"] = target_sizes
        self.received["_box"] = box
        return [{"boxes": [], "scores": []}]


def _fake_outputs():
    return object()


def _fake_ids():
    return object()


def test_box_threshold_kwarg_resolved_for_new_transformers():
    """New transformers exposes `box_threshold`."""
    p = _FakeProcessor("box_threshold")
    _post_process_grounded(p, _fake_outputs(), _fake_ids(), 0.35, 0.25, [(10, 10)])
    assert p.received["_box"] == 0.35
    assert p.received["text_threshold"] == 0.25
    assert "threshold" not in p.received


def test_box_threshold_kwarg_resolved_for_old_transformers():
    """Old transformers exposed `threshold` — must still work."""
    p = _FakeProcessor("threshold")
    _post_process_grounded(p, _fake_outputs(), _fake_ids(), 0.4, 0.3, [(10, 10)])
    assert p.received["_box"] == 0.4
    assert "box_threshold" not in p.received


def test_unknown_transformers_signature_fails_loudly():
    """Never silently guess: an unsupported signature must raise, not mis-score."""
    class Weird:
        def post_process_grounded_object_detection(self, outputs, input_ids):
            return []

    import pytest
    with pytest.raises(TypeError):
        _post_process_grounded(Weird(), _fake_outputs(), _fake_ids(), 0.35, 0.25, [(1, 1)])


class _FakeSam:
    def set_image(self, img):
        return object()

    def predict_boxes(self, enc, boxes):
        m = np.zeros((64, 64), bool)
        m[10:30, 10:30] = True
        return [m], [0.95]


def test_adapter_accepts_backend_returning_phrases():
    """3-tuple backends keep the detected phrase (not just the raw query)."""
    class DinoWithPhrase:
        def __call__(self, img, text, bt, tt):
            return ([np.array([10, 10, 30, 30])], [0.9], ["red car"])

    r = run_fov_text_branch(Image.new("RGB", (64, 64)), "red car",
                            GroundedConfig(), backend=(DinoWithPhrase(), _FakeSam()))
    assert len(r["candidates"]) == 1
    assert r["candidates"][0].phrase == "red car"


def test_adapter_accepts_backend_returning_two_values():
    """Legacy 2-tuple backends still work and fall back to the normalized text."""
    class DinoTwo:
        def __call__(self, img, text, bt, tt):
            return ([np.array([10, 10, 30, 30])], [0.9])

    r = run_fov_text_branch(Image.new("RGB", (64, 64)), "red car",
                            GroundedConfig(), backend=(DinoTwo(), _FakeSam()))
    assert len(r["candidates"]) == 1
    assert r["candidates"][0].phrase == "red car."
