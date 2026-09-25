"""
File: metadata.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Audio metadata utilities for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path

from pulse.audio.torchcodec_compat import AudioDecoder


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Metadata extracted from an audio file."""

    path: str
    filename: str
    size_bytes: int
    duration_seconds: float | None
    duration_seconds_from_header: float | None
    begin_stream_seconds: float | None
    begin_stream_seconds_from_header: float | None
    bit_rate: float | None
    codec: str | None
    sample_rate: int | None
    num_channels: int | None
    sample_format: str | None
    stream_index: int


def read_audio_metadata(audio_path: str) -> AudioMetadata:
    """Read audio metadata using TorchCodec."""

    path = Path(audio_path)
    decoder = AudioDecoder(str(path))
    metadata = decoder.metadata

    return AudioMetadata(
        path=str(path),
        filename=path.name,
        size_bytes=path.stat().st_size,
        duration_seconds=metadata.duration_seconds,
        duration_seconds_from_header=metadata.duration_seconds_from_header,
        begin_stream_seconds=metadata.begin_stream_seconds,
        begin_stream_seconds_from_header=metadata.begin_stream_seconds_from_header,
        bit_rate=metadata.bit_rate,
        codec=metadata.codec,
        sample_rate=metadata.sample_rate,
        num_channels=metadata.num_channels,
        sample_format=metadata.sample_format,
        stream_index=metadata.stream_index,
    )
