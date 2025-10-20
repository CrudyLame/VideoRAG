"""Utilities for describing video segments with multimodal models."""

from __future__ import annotations

from typing import Any, Optional

from openai import AsyncOpenAI

from .split import VideoSegment


class VisionCaptioner:
    """Generate captions for segments using ``gpt-5-mini``."""

    def __init__(self, client: Optional[AsyncOpenAI] = None, model: str = "gpt-5-mini") -> None:
        self._client = client or AsyncOpenAI()
        self.model = model

    async def describe(self, segment: VideoSegment, transcript: str) -> str:
        if not segment.frames_base64:
            prompt = (
                "No frames were extracted for this segment."
                " Use the transcript (if any) to summarize the visuals."
            )
        else:
            prompt = "Analyze the provided video frames and transcript."

        user_content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"{prompt}\n\n"
                    f"Transcript:\n{transcript if transcript else 'N/A'}\n"
                    "Describe the scene, important actions and any visible text."
                ),
            }
        ]

        for frame in segment.frames_base64:
            user_content.append({"type": "input_image", "image_base64": frame})

        response = await self._client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are an assistant that describes short video segments."
                                " Provide concise but information dense captions."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": user_content},
            ],
        )

        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        first_choice = response.output[0].content[0]
        if isinstance(first_choice, dict):
            text_value = first_choice.get("text")
        else:
            text_value = getattr(first_choice, "text", None)

        return (text_value or str(first_choice)).strip()
