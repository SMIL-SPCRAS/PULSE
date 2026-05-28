"""
File: waveform.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Audio waveform preparation for PULSE visualizations.
License: MIT License
"""

from pulse.audio.decoder import DecodedAudio
from pulse.inference.schemas import AudioWaveform
from pulse.settings.state import get_runtime_setting_by_flat_name


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer visualization config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def get_waveform_max_points() -> int:
    """Return maximum number of waveform points for visualization."""

    return max(
        100,
        get_config_int(
            "Visualization_WAVEFORM_MAX_POINTS",
            1200,
        ),
    )


def create_audio_waveform(
    decoded_audio: DecodedAudio,
) -> AudioWaveform:
    """Create a downsampled mono waveform for visualization."""

    waveform = decoded_audio.waveform.detach().cpu()

    if waveform.ndim == 1:
        mono_waveform = waveform
    elif waveform.ndim == 2:
        mono_waveform = waveform[0] if int(waveform.shape[0]) == 1 else waveform.mean(dim=0)
    else:
        msg = f"Expected decoded waveform with 1 or 2 dimensions, got shape={tuple(waveform.shape)}."
        raise ValueError(msg)

    sample_count = int(mono_waveform.shape[0])

    if sample_count <= 0:
        return AudioWaveform(
            timestamps=[],
            values=[],
        )

    max_points = get_waveform_max_points()

    if sample_count <= max_points:
        indices = list(range(sample_count))
    else:
        step = sample_count / max_points
        indices = [min(sample_count - 1, int(index * step)) for index in range(max_points)]

    timestamps = [index / decoded_audio.sample_rate for index in indices]
    values = [float(mono_waveform[index].item()) for index in indices]

    return AudioWaveform(
        timestamps=timestamps,
        values=values,
    )
