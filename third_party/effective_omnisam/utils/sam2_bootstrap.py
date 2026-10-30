"""
在导入 SAM2 相关模块之前执行：把 vendor 的 `sam/` 加入 sys.path，并创建 `sam/sam2 -> sam`
符号链接，使官方代码中的 `import sam2` 能解析（等价于 pip install SAM2 的布局）。

需在 train.py / validation.py / run_adaptation.py 等入口的最顶部 `import utils.sam2_bootstrap`。
"""
from __future__ import annotations

import os
import sys


def _bootstrap() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sam_vendor = os.path.join(repo_root, "sam")
    sam2_link = os.path.join(sam_vendor, "sam2")
    sam_inner = os.path.join(sam_vendor, "sam")
    if os.path.isdir(sam_inner) and not os.path.lexists(sam2_link):
        try:
            os.symlink("sam", sam2_link, target_is_directory=True)
        except OSError:
            pass
    if os.path.isdir(sam_vendor) and sam_vendor not in sys.path:
        sys.path.insert(0, sam_vendor)


_bootstrap()
