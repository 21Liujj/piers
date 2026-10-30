"""After wandb.init(), mirror stdout/stderr to <run.dir>/logs/console.log (still prints to terminal).

Disable with env: WANDB_CONSOLE_LOG=0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, TextIO


class _TeeTextIO(TextIO):
    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log = log_file

    def write(self, s: str) -> int:
        self._terminal.write(s)
        self._log.write(s)
        try:
            self._log.flush()
        except Exception:
            pass
        try:
            self._terminal.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._terminal.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass

    # minimal stubs so isinstance checks in some libs don't break badly
    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


_log_fp: Optional[TextIO] = None


def attach_wandb_console_log() -> Optional[Path]:
    """
    Call immediately after wandb.init(). Returns path to console.log or None.
    """
    global _log_fp
    if os.environ.get("WANDB_CONSOLE_LOG", "1").strip() in ("0", "false", "False", "no"):
        return None
    try:
        import wandb

        run = wandb.run
        if run is None:
            return None
        log_dir = Path(run.dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "console.log"
        _log_fp = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = _TeeTextIO(sys.__stdout__, _log_fp)
        sys.stderr = _TeeTextIO(sys.__stderr__, _log_fp)
        return path
    except Exception:
        return None
