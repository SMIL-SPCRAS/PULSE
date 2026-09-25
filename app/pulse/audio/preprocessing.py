"""
File: preprocessing.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Audio preprocessing utilities for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from pulse.audio.decoder import DecodedAudio
from pulse.settings.state import get_runtime_setting_by_flat_name


@dataclass(frozen=True, slots=True)
class PreprocessedAudio:
    """Preprocessed audio data prepared for model inference."""

    path: str
    sample_rate: int
    num_channels: int
    original_duration_seconds: float
    target_duration_seconds: float
    target_num_samples: int
    waveform: Tensor
    model_input: Tensor
    peak_amplitude: float
    was_peak_normalized: bool
    was_trimmed: bool
    was_padded: bool

    @property
    def waveform_shape(self) -> tuple[int, ...]:
        """Return preprocessed waveform tensor shape."""

        return tuple(int(dimension) for dimension in self.waveform.shape)

    @property
    def model_input_shape(self) -> tuple[int, ...]:
        """Return model input tensor shape."""

        return tuple(int(dimension) for dimension in self.model_input.shape)


def get_config_float(field_name: str, default_value: float) -> float:
    """Return float preprocessing config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_config_bool(field_name: str, default_value: bool) -> bool:
    """Return bool preprocessing config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def ensure_channel_first(waveform: Tensor) -> Tensor:
    """Ensure waveform shape is channel-first: (channels, samples)."""

    if waveform.ndim == 1:
        return waveform.unsqueeze(0)

    if waveform.ndim == 2:
        return waveform

    msg = f"Expected waveform with 1 or 2 dimensions, got shape={tuple(waveform.shape)}."
    raise ValueError(msg)


def ensure_float32(waveform: Tensor) -> Tensor:
    """Convert waveform to float32."""

    return waveform.to(dtype=torch.float32)


def peak_normalize_waveform(
    waveform: Tensor,
    eps: float,
) -> tuple[Tensor, float, bool]:
    """Normalize waveform by peak absolute amplitude."""

    peak = torch.amax(torch.abs(waveform))
    peak_amplitude = float(peak.detach().cpu().item())

    if peak_amplitude <= eps:
        return waveform, peak_amplitude, False

    return waveform / peak, peak_amplitude, True


def trim_or_pad_waveform(
    waveform: Tensor,
    target_num_samples: int,
) -> tuple[Tensor, bool, bool]:
    """Trim or right-pad waveform to the target number of samples."""

    current_num_samples = int(waveform.shape[-1])

    if current_num_samples == target_num_samples:
        return waveform, False, False

    if current_num_samples > target_num_samples:
        return waveform[..., :target_num_samples], True, False

    pad_samples = target_num_samples - current_num_samples
    padded_waveform = F.pad(waveform, (0, pad_samples))

    return padded_waveform, False, True


def preprocess_audio_for_analysis(decoded_audio: DecodedAudio) -> PreprocessedAudio:
    """Preprocess decoded audio for future PULSE model inference."""

    target_duration_seconds = get_config_float(
        "AudioPreprocessing_TARGET_DURATION_SECONDS",
        15.0,
    )
    enable_peak_normalization = get_config_bool(
        "AudioPreprocessing_ENABLE_PEAK_NORMALIZATION",
        True,
    )
    peak_normalization_eps = get_config_float(
        "AudioPreprocessing_PEAK_NORMALIZATION_EPS",
        1e-8,
    )

    target_num_samples = max(
        1,
        round(decoded_audio.sample_rate * target_duration_seconds),
    )

    waveform = ensure_channel_first(decoded_audio.waveform)
    waveform = ensure_float32(waveform)

    peak_amplitude = 0.0
    was_peak_normalized = False

    if enable_peak_normalization:
        waveform, peak_amplitude, was_peak_normalized = peak_normalize_waveform(
            waveform=waveform,
            eps=peak_normalization_eps,
        )

    waveform, was_trimmed, was_padded = trim_or_pad_waveform(
        waveform=waveform,
        target_num_samples=target_num_samples,
    )

    model_input = waveform.unsqueeze(0)

    return PreprocessedAudio(
        path=decoded_audio.path,
        sample_rate=decoded_audio.sample_rate,
        num_channels=int(waveform.shape[0]),
        original_duration_seconds=decoded_audio.duration_seconds,
        target_duration_seconds=target_duration_seconds,
        target_num_samples=target_num_samples,
        waveform=waveform,
        model_input=model_input,
        peak_amplitude=peak_amplitude,
        was_peak_normalized=was_peak_normalized,
        was_trimmed=was_trimmed,
        was_padded=was_padded,
    )
