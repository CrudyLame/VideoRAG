"""Compatibility layer preserving the public QueryParam dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class QueryParam:
    """Minimal query configuration used by legacy callers."""

    top_k: int = 5
    similarity_threshold: float | None = None
    response_type: str = "json"
    mode: Literal["cloud"] = "cloud"
    only_need_context: bool = False


__all__ = ["QueryParam"]
