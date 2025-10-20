"""High-level VideoRAG wrapper built on top of the decomposed cloud service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from ._op import CloudVideoRAGService


@dataclass
class VideoRAG:
    """Thin synchronous wrapper around :class:`CloudVideoRAGService`."""

    supabase_url: str
    supabase_key: str
    supabase_table: str
    supabase_match_fn: str
    segment_duration: int = 30
    frames_per_segment: int = 6
    embedding_model: str = "text-embedding-3-small"
    text_model: str = "gpt-5-nano"
    vision_model: str = "gpt-5-mini"
    whisper_model: str = "whisper-1"
    client: AsyncOpenAI = field(default_factory=AsyncOpenAI)

    def __post_init__(self) -> None:
        self._service = CloudVideoRAGService(
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key,
            supabase_table=self.supabase_table,
            supabase_match_fn=self.supabase_match_fn,
            segment_duration=self.segment_duration,
            frames_per_segment=self.frames_per_segment,
            embedding_model=self.embedding_model,
            text_model=self.text_model,
            vision_model=self.vision_model,
            whisper_model=self.whisper_model,
            client=self.client,
        )

    async def agingest_video(self, session_id: str, video_path: str) -> None:
        await self._service.ingest_video(session_id=session_id, video_path=video_path)

    def ingest_video(self, session_id: str, video_path: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.agingest_video(session_id=session_id, video_path=video_path))
        else:
            raise RuntimeError(
                "ingest_video cannot be called from within an existing event loop. "
                "Use `await agingest_video(...)` instead."
            )

    async def aquery(
        self,
        session_id: str,
        question: str,
        *,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        return await self._service.query(
            session_id=session_id,
            question=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def query(
        self,
        session_id: str,
        question: str,
        *,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aquery(
                    session_id=session_id,
                    question=question,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )
            )
        else:
            raise RuntimeError(
                "query cannot be called from within an existing event loop. "
                "Use `await aquery(...)` instead."
            )

    async def acleanup(self, session_id: str) -> None:
        await self._service.cleanup(session_id=session_id)

    def cleanup(self, session_id: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.acleanup(session_id=session_id))
        else:
            raise RuntimeError(
                "cleanup cannot be called from within an existing event loop. "
                "Use `await acleanup(...)` instead."
            )


__all__ = ["VideoRAG", "CloudVideoRAGService"]
