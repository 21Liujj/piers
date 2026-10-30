from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class PluginRegistry:
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("plugin name is empty")
        self._builders[key] = builder

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        return self._builders.get(str(name).strip().lower(), None)


REGISTRY = PluginRegistry()

