"""Video segmentation helpers for the cloud-native pipeline."""

from __future__ import annotations

import base64
import math
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image


@dataclass(slots=True)
class VideoSegment:
    """Metadata for a single video segment."""

    id: str
    start: float
    end: float
    frames_base64: List[str]
    audio_path: Optional[Path]


class VideoSegmenter:
    """Split videos into temporal chunks and export keyframes/audio."""

    def __init__(
        self,
        segment_duration: int = 30,
        frames_per_segment: int = 6,
    ) -> None:
        self.segment_duration = max(1, segment_duration)
        self.frames_per_segment = max(0, frames_per_segment)

    def segment(self, video_path: str) -> Tuple[list[VideoSegment], Path]:
        """Return ``VideoSegment`` objects and a temp directory for artifacts."""

        temp_dir = Path(tempfile.mkdtemp(prefix="videorag_segments_"))
        segments: list[VideoSegment] = []

        with VideoFileClip(video_path) as clip:
            duration = clip.duration or 0.0
            total_segments = int(math.ceil(duration / self.segment_duration))

            for index in range(total_segments):
                start = index * self.segment_duration
                end = min((index + 1) * self.segment_duration, duration)
                segment_id = f"seg-{index:04d}-{int(start * 1000)}-{int(end * 1000)}"

                subclip = clip.subclip(start, end)

                frame_offsets = self._frame_offsets(subclip.duration)
                frames_base64 = [
                    self._frame_to_base64(subclip.get_frame(offset))
                    for offset in frame_offsets
                ]

                audio_path: Optional[Path] = None
                if subclip.audio is not None:
                    audio_path = temp_dir / f"{segment_id}.mp3"
                    subclip.audio.write_audiofile(
                        audio_path.as_posix(),
                        codec="mp3",
                        verbose=False,
                        logger=None,
                    )

                segments.append(
                    VideoSegment(
                        id=segment_id,
                        start=float(start),
                        end=float(end),
                        frames_base64=frames_base64,
                        audio_path=audio_path,
                    )
                )

        return segments, temp_dir

    def _frame_offsets(self, duration: float) -> np.ndarray:
        if self.frames_per_segment <= 0 or duration <= 0:
            return np.array([], dtype=np.float32)
        return np.linspace(0, max(duration - 0.001, 0.0), self.frames_per_segment, endpoint=False)

    @staticmethod
    def _frame_to_base64(frame: np.ndarray) -> str:
        image = Image.fromarray(frame.astype("uint8"))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
