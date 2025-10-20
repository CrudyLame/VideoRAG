"""High level operations for the cloud-native VideoRAG pipeline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from ._llm import AnswerGenerator, EmbeddingClient
from ._storage import SupabaseVectorStore
from ._utils import build_embedding_input
from ._videoutil import (
    SegmentSummarizer,
    VideoSegmenter,
    VisionCaptioner,
    WhisperTranscriber,
)


@dataclass(slots=True)
class CloudVideoRAGService:
    """Coordinate ingestion, retrieval, and cleanup for VideoRAG sessions."""

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
        self.segmenter = VideoSegmenter(
            segment_duration=self.segment_duration,
            frames_per_segment=self.frames_per_segment,
        )
        self.transcriber = WhisperTranscriber(self.client, model=self.whisper_model)
        self.captioner = VisionCaptioner(self.client, model=self.vision_model)
        self.summarizer = SegmentSummarizer(self.client, model=self.text_model)
        self.vector_store = SupabaseVectorStore(
            url=self.supabase_url,
            key=self.supabase_key,
            table_name=self.supabase_table,
            match_rpc=self.supabase_match_fn,
        )
        self.embedding_client = EmbeddingClient(model=self.embedding_model, client=self.client)
        self.answer_generator = AnswerGenerator(model=self.text_model, client=self.client)

    async def ingest_video(self, session_id: str, video_path: str) -> None:
        segments, temp_dir = self.segmenter.segment(video_path)
        records: list[dict[str, Any]] = []

        try:
            for segment in segments:
                transcript = await self.transcriber.transcribe_path(segment.audio_path)
                caption = await self.captioner.describe(segment, transcript)
                summary = await self.summarizer.summarize(transcript, caption)

                embedding_input = build_embedding_input(summary, caption, transcript)
                embedding = await self.embedding_client.embed(embedding_input)

                records.append(
                    {
                        "session_id": session_id,
                        "segment_id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "summary": summary["summary"],
                        "key_points": summary["key_points"],
                        "caption": caption,
                        "transcript": transcript,
                        "embedding": embedding,
                    }
                )

            await self.vector_store.upsert_records(records)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def query(
        self,
        session_id: str,
        question: str,
        *,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        query_embedding = await self.embedding_client.embed(question)
        matches = await self.vector_store.similarity_search(
            session_id=session_id,
            embedding=query_embedding,
            match_count=top_k,
            similarity_threshold=similarity_threshold,
        )

        evidence = []
        for match in matches:
            data = match.get("metadata", match)
            evidence.append(
                {
                    "segment_id": data.get("segment_id"),
                    "start": data.get("start"),
                    "end": data.get("end"),
                    "summary": data.get("summary"),
                    "caption": data.get("caption"),
                    "transcript": data.get("transcript"),
                    "similarity": match.get("similarity") or data.get("similarity"),
                }
            )

        return await self.answer_generator.answer(question, evidence)

    async def cleanup(self, session_id: str) -> None:
        await self.vector_store.delete_session(session_id)


__all__ = ["CloudVideoRAGService"]
