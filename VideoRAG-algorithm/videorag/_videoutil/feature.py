"""Summarisation helpers for VideoRAG segments."""

from __future__ import annotations

import json
from typing import Any, Optional

from openai import AsyncOpenAI


class SegmentSummarizer:
    """Summarize transcript + caption using ``gpt-5-nano``."""

    def __init__(self, client: Optional[AsyncOpenAI] = None, model: str = "gpt-5-nano") -> None:
        self._client = client or AsyncOpenAI()
        self.model = model

        self._summary_schema = {
            "name": "segment_summary",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One paragraph summary of the segment.",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important bullet points extracted from the segment.",
                    },
                },
                "required": ["summary", "key_points"],
                "additionalProperties": False,
            },
        }

    async def summarize(self, transcript: str, caption: str) -> dict[str, Any]:
        response = await self._client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a senior editor summarizing video segments."
                                " Return structured JSON adhering to the provided schema."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Transcript:\n"
                                f"{transcript if transcript else 'N/A'}\n\n"
                                "Caption:\n"
                                f"{caption if caption else 'N/A'}"
                            ),
                        }
                    ],
                },
            ],
            response_format={"type": "json_schema", "json_schema": self._summary_schema},
        )

        payload = response.output_text if hasattr(response, "output_text") else response.output[0].content[0].text
        return json.loads(payload)
