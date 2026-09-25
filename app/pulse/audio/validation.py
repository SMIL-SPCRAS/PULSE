"""
File: validation.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Audio validation utilities for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pulse.audio.metadata import AudioMetadata, read_audio_metadata
from pulse.settings.state import get_runtime_setting_by_flat_name

ERROR_SEVERITY: Final = "error"
WARNING_SEVERITY: Final = "warning"


@dataclass(frozen=True, slots=True)
class AudioValidationIssue:
    """One audio validation issue."""

    code: str
    severity: str
    actual: float | int | str | None = None
    expected: float | int | str | None = None


@dataclass(frozen=True, slots=True)
class AudioValidationResult:
    """Audio validation result."""

    metadata: AudioMetadata | None
    issues: list[AudioValidationIssue]

    @property
    def errors(self) -> list[AudioValidationIssue]:
        """Return validation errors."""

        return [issue for issue in self.issues if issue.severity == ERROR_SEVERITY]

    @property
    def warnings(self) -> list[AudioValidationIssue]:
        """Return validation warnings."""

        return [issue for issue in self.issues if issue.severity == WARNING_SEVERITY]

    @property
    def is_valid(self) -> bool:
        """Return whether the audio is valid for analysis."""

        return not self.errors


def get_config_float(field_name: str, default_value: float) -> float:
    """Return float validation config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer validation config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def create_issue(
    code: str,
    severity: str,
    actual: float | int | str | None = None,
    expected: float | int | str | None = None,
) -> AudioValidationIssue:
    """Create one validation issue."""

    return AudioValidationIssue(
        code=code,
        severity=severity,
        actual=actual,
        expected=expected,
    )


def validate_audio_metadata(metadata: AudioMetadata) -> AudioValidationResult:
    """Validate audio metadata."""

    issues: list[AudioValidationIssue] = []

    min_duration_seconds = get_config_float(
        "AudioValidation_MIN_DURATION_SECONDS",
        1.0,
    )
    max_duration_seconds = get_config_float(
        "AudioValidation_MAX_DURATION_SECONDS",
        60.0,
    )
    recommended_sample_rate = get_config_int(
        "AudioValidation_RECOMMENDED_SAMPLE_RATE",
        16000,
    )
    recommended_num_channels = get_config_int(
        "AudioValidation_RECOMMENDED_NUM_CHANNELS",
        1,
    )

    if metadata.size_bytes <= 0:
        issues.append(create_issue("EMPTY_FILE", ERROR_SEVERITY))

    if metadata.duration_seconds is None or metadata.duration_seconds <= 0:
        issues.append(create_issue("DURATION_UNAVAILABLE", ERROR_SEVERITY))
    elif metadata.duration_seconds < min_duration_seconds:
        issues.append(
            create_issue(
                code="TOO_SHORT",
                severity=ERROR_SEVERITY,
                actual=metadata.duration_seconds,
                expected=min_duration_seconds,
            ),
        )
    elif metadata.duration_seconds > max_duration_seconds:
        issues.append(
            create_issue(
                code="TOO_LONG",
                severity=ERROR_SEVERITY,
                actual=metadata.duration_seconds,
                expected=max_duration_seconds,
            ),
        )

    if metadata.sample_rate is None or metadata.sample_rate <= 0:
        issues.append(create_issue("SAMPLE_RATE_UNAVAILABLE", ERROR_SEVERITY))
    elif metadata.sample_rate != recommended_sample_rate:
        issues.append(
            create_issue(
                code="SAMPLE_RATE_WARNING",
                severity=WARNING_SEVERITY,
                actual=metadata.sample_rate,
                expected=recommended_sample_rate,
            ),
        )

    if metadata.num_channels is None or metadata.num_channels <= 0:
        issues.append(create_issue("CHANNELS_UNAVAILABLE", ERROR_SEVERITY))
    elif metadata.num_channels != recommended_num_channels:
        issues.append(
            create_issue(
                code="CHANNELS_WARNING",
                severity=WARNING_SEVERITY,
                actual=metadata.num_channels,
                expected=recommended_num_channels,
            ),
        )

    return AudioValidationResult(
        metadata=metadata,
        issues=issues,
    )


def validate_audio_file(audio_path: str) -> AudioValidationResult:
    """Validate an audio file by path."""

    path = Path(audio_path)

    if not path.exists():
        return AudioValidationResult(
            metadata=None,
            issues=[
                create_issue(
                    code="FILE_NOT_FOUND",
                    severity=ERROR_SEVERITY,
                    actual=str(path),
                ),
            ],
        )

    try:
        metadata = read_audio_metadata(str(path))
    except Exception as error:
        return AudioValidationResult(
            metadata=None,
            issues=[
                create_issue(
                    code="READ_FAILED",
                    severity=ERROR_SEVERITY,
                    actual=str(error),
                ),
            ],
        )

    return validate_audio_metadata(metadata)
