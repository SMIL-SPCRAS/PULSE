"""
File: decoder.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Audio decoding utilities for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from torch import Tensor

from pulse.audio.torchcodec_compat import AudioDecoder
from pulse.settings.state import get_runtime_setting_by_flat_name


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Decoded audio data prepared for analysis."""

    path: str
    sample_rate: int
    num_channels: int
    duration_seconds: float
    waveform: Tensor

    @property
    def waveform_shape(self) -> tuple[int, ...]:
        """Return waveform tensor shape."""

        return tuple(int(dimension) for dimension in self.waveform.shape)


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer audio decoding config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def get_audio_samples_data(samples: Any) -> Tensor:
    """Return audio samples tensor from TorchCodec AudioSamples."""

    data = getattr(samples, "data", None)

    if data is None:
        msg = "TorchCodec AudioSamples object does not contain 'data'."
        raise AttributeError(msg)

    return cast(Tensor, data)


def get_audio_samples_sample_rate(samples: Any, fallback_sample_rate: int) -> int:
    """Return decoded audio sample rate."""

    sample_rate = getattr(samples, "sample_rate", None)

    if isinstance(sample_rate, int):
        return sample_rate

    return fallback_sample_rate


def get_audio_samples_duration(
    samples: Any,
    fallback_duration_seconds: float,
) -> float:
    """Return decoded audio duration."""

    duration_seconds = getattr(samples, "duration_seconds", None)

    if isinstance(duration_seconds, int | float):
        return float(duration_seconds)

    return fallback_duration_seconds


def decode_audio_for_analysis(audio_path: str) -> DecodedAudio:
    """Decode an audio file for future PULSE analysis."""

    path = Path(audio_path)

    if not path.exists():
        msg = f"Audio file was not found: {path}"
        raise FileNotFoundError(msg)

    target_sample_rate = get_config_int(
        "AudioDecoding_TARGET_SAMPLE_RATE",
        16000,
    )
    target_num_channels = get_config_int(
        "AudioDecoding_TARGET_NUM_CHANNELS",
        1,
    )

    decoder = AudioDecoder(
        str(path),
        sample_rate=target_sample_rate,
        num_channels=target_num_channels,
    )

    metadata = decoder.metadata
    samples = decoder.get_all_samples()
    waveform = get_audio_samples_data(samples)

    sample_rate = get_audio_samples_sample_rate(
        samples=samples,
        fallback_sample_rate=target_sample_rate,
    )
    duration_seconds = get_audio_samples_duration(
        samples=samples,
        fallback_duration_seconds=float(metadata.duration_seconds or 0.0),
    )

    return DecodedAudio(
        path=str(path),
        sample_rate=sample_rate,
        num_channels=target_num_channels,
        duration_seconds=duration_seconds,
        waveform=waveform,
    )
