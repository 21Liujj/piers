"""Indoor ERP geometry, isolated from the frozen outdoor modules.

Source GT (Stanford2D3D via 360BEV) is natively 1024x512. The outdoor pipeline
upscales 2048x400 -> 4096x1024 (2x blow-up). For indoor we keep the native
canvas and derive the window schedule from it:

  canonical : 1024 x 512   (native, no 2x blow-up)
  window    : 512 x 512    (full height, half width)
  stride    : 64
  starts    : 0,64,...,512  -> 9 positions covering [0,1024)
  passes    : 2 (forward + reversed) -> 18 base semantic forwards
  I3        : +2 adaptive cross-seam windows -> 20 forwards

Keeping 9 starts x 2 passes preserves the outdoor forward budget exactly
(18 base / 20 full) while costing 1/4 of the outdoor pixel throughput
(512x512 vs 1024x1024 windows).
"""
from __future__ import annotations

INDOOR_W = 1024
INDOOR_H = 512
INDOOR_WIN = 512
INDOOR_STRIDE = 64
INDOOR_PASSES = 2
INDOOR_BASE_FORWARDS = 18
INDOOR_FULL_EXTRA = 2
INDOOR_FULL_FORWARDS = 20

SPAN8_CLASSES = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
# 13-channel street head -> 8-class SPAN8 trainId (255 = ignore)
MAP13_TO_8 = [255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 5, 6, 7]

# Text branch: 4 horizontal wrap FoVs, full height, half width.
TEXT_FOVS = [
    {"view_id": "v000", "x_start": 0,    "y_start": 0, "win_w": 512, "win_h": 512},
    {"view_id": "v001", "x_start": 256,  "y_start": 0, "win_w": 512, "win_h": 512},
    {"view_id": "v002", "x_start": 512,  "y_start": 0, "win_w": 512, "win_h": 512},
    {"view_id": "v003", "x_start": 768,  "y_start": 0, "win_w": 512, "win_h": 512},
]


def semantic_starts(width: int = INDOOR_W, win: int = INDOOR_WIN,
                    stride: int = INDOOR_STRIDE) -> list:
    """Non-wrap positions covering [0, width) exactly.

    NOTE: the shared plan_fovs() helper uses ceil(W/stride) with wrap-around
    (16 for stride 128) and is only used for the text FoVs. Semantic positions
    are explicit here to match the 9x2=18 base-forward design.
    """
    if width < win:
        raise ValueError("ERP width smaller than semantic window")
    starts = list(range(0, width - win + 1, stride))
    if starts[-1] != width - win:
        starts.append(width - win)
    return starts


def i3_candidate_starts(width: int = INDOOR_W, win: int = INDOOR_WIN,
                        stride: int = INDOOR_STRIDE, n: int = 5) -> list:
    """Candidate extra cross-seam window starts, centred on the ERP seam (x=W).

    The outdoor variant uses [1664,1728,1792,1856,1920] on a 4096-wide tensor.
    For indoor we mirror that layout around the seam of the native canvas.
    """
    base = width - win // 2
    return [int(base + (i - (n - 1) / 2.0) * (win / 4.0)) for i in range(n)]


def i3_base_coverage(width: int = INDOOR_W, win: int = INDOOR_WIN,
                      stride: int = INDOOR_STRIDE) -> list:
    """Per-column base coverage counts, matching the outdoor merge schedule."""
    v = [0.0] * width
    for s in semantic_starts(width, win, stride):
        for c in range(s, s + win):
            v[c % width] += 2.0
    return v


def check_indoor_geometry() -> dict:
    starts = semantic_starts()
    assert len(starts) == 9, starts
    assert starts[0] == 0 and starts[-1] == INDOOR_W - INDOOR_WIN
    assert INDOOR_BASE_FORWARDS == len(starts) * INDOOR_PASSES
    assert INDOOR_FULL_FORWARDS == INDOOR_BASE_FORWARDS + INDOOR_FULL_EXTRA
    assert len(SPAN8_CLASSES) == 8 and len(MAP13_TO_8) == 13
    assert len(TEXT_FOVS) == 4
    for f in TEXT_FOVS:
        assert f["win_w"] == INDOOR_WIN and f["win_h"] == INDOOR_H and f["y_start"] == 0
    cov = i3_base_coverage()
    assert len(cov) == INDOOR_W and min(cov) > 0
    return {
        "canonical_WH": [INDOOR_W, INDOOR_H],
        "window": [INDOOR_WIN, INDOOR_H],
        "stride": INDOOR_STRIDE,
        "starts": starts,
        "passes": INDOOR_PASSES,
        "base_forwards": INDOOR_BASE_FORWARDS,
        "full_extra": INDOOR_FULL_EXTRA,
        "full_forwards": INDOOR_FULL_FORWARDS,
        "i3_candidate_starts": i3_candidate_starts(),
        "text_fovs": TEXT_FOVS,
        "classes": SPAN8_CLASSES,
        "map13_to_8": MAP13_TO_8,
        "base_coverage_len": len(cov),
        "base_coverage_min": min(cov),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(check_indoor_geometry(), indent=2))
