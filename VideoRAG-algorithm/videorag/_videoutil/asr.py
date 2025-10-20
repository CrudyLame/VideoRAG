"""Async wrapper around OpenAI Whisper for speech-to-text."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI


class WhisperTranscriber:
    """OpenAI Whisper (cloud) wrapper for async audio transcription."""

    def __init__(self, client: Optional[AsyncOpenAI] = None, model: str = "whisper-1") -> None:
        self._client = client or AsyncOpenAI()
        self.model = model

    async def transcribe_path(self, audio_path: Optional[Path]) -> str:
        if audio_path is None or not audio_path.exists():
            return ""
        with audio_path.open("rb") as audio_file:
            response = await self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        return response.text or ""
