import numpy as np
from piers_ljj.text.grounded_adapter import normalize_text, run_fov_text_branch, GroundedConfig
from PIL import Image


def test_normalize_keeps_free_text():
    assert normalize_text("  Red Car ") == "red car."
    assert normalize_text("bus.") == "bus."


class FakeDino:
    def __call__(self, img, text, bt, tt):
        return [np.array([10, 10, 30, 30])], [0.9]


class FakeSam:
    def set_image(self, img):
        assert isinstance(img, Image.Image) and img.mode == "RGB"
        return object()

    def predict_boxes(self, enc, boxes):
        m = np.zeros((64, 64), bool)
        m[10:30, 10:30] = True
        return [m], [0.95]


def test_empty_is_legal():
    class EmptyDino(FakeDino):
        def __call__(self, img, text, bt, tt):
            return [], []

    img = Image.new("RGB", (64, 64))
    r = run_fov_text_branch(img, "nothing here", GroundedConfig(), backend=(EmptyDino(), FakeSam()))
    assert r["candidates"] == [] and r["meta"]["mask_decoder_calls"] == 0


def test_single_encode_per_fov():
    img = Image.new("RGB", (64, 64))
    r = run_fov_text_branch(img, "car", GroundedConfig(), backend=(FakeDino(), FakeSam()))
    assert r["meta"]["image_encodes"] == 1 and len(r["candidates"]) == 1
    c = r["candidates"][0]
    assert c.box_xyxy == (10, 10, 30, 30)


def test_no_backend_refuses_synthetic():
    import pytest
    with pytest.raises(RuntimeError):
        run_fov_text_branch(Image.new("RGB", (8, 8)), "car", GroundedConfig(), backend=None)
