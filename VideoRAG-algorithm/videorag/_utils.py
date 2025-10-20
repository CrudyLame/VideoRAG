"""Miscellaneous helpers for the cloud-native implementation."""

from __future__ import annotations

from typing import Iterable


def build_embedding_input(summary: dict[str, str], caption: str, transcript: str) -> str:
    raw_points = summary.get("key_points", [])
    if isinstance(raw_points, str):
        points_list = [raw_points]
    elif isinstance(raw_points, Iterable):
        points_list = list(raw_points)
    else:
        points_list = []

    key_points = "\n".join(points_list)
    return (
        f"Summary:\n{summary.get('summary', '')}\n"
        f"Key Points:\n{key_points}\n"
        f"Caption:\n{caption}\n"
        f"Transcript:\n{transcript}"
    )


__all__ = ["build_embedding_input"]
