"""Core language model helpers for the cloud-native pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from openai import AsyncOpenAI


@dataclass(slots=True)
class EmbeddingClient:
    """Thin wrapper around the OpenAI embedding endpoint."""

    model: str = "text-embedding-3-small"
    client: AsyncOpenAI = field(default_factory=AsyncOpenAI)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


@dataclass(slots=True)
class AnswerGenerator:
    """Structured answer generation using ``gpt-5-nano``."""

    model: str = "gpt-5-nano"
    client: AsyncOpenAI = field(default_factory=AsyncOpenAI)

    def __post_init__(self) -> None:
        self._schema = {
            "name": "video_rag_answer",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "description": "0-1 confidence estimate for the answer.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment_id": {"type": "string"},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "summary": {"type": "string"},
                                "similarity": {"type": "number"},
                                "caption": {"type": "string"},
                                "transcript": {"type": "string"},
                            },
                            "required": ["segment_id", "start", "end", "summary"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["answer", "evidence"],
                "additionalProperties": False,
            },
        }

    async def answer(self, question: str, evidence: Sequence[dict[str, Any]]) -> dict[str, Any]:
        context_payload = json.dumps(
            {
                "question": question,
                "evidence": list(evidence),
            },
            ensure_ascii=False,
        )

        response = await self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You answer questions about videos using provided evidence."
                                " Respond with JSON that follows the schema."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "input_text", "text": context_payload}]},
            ],
            response_format={"type": "json_schema", "json_schema": self._schema},
        )

        payload = response.output_text if hasattr(response, "output_text") else response.output[0].content[0].text
        return json.loads(payload)


__all__ = ["EmbeddingClient", "AnswerGenerator"]
