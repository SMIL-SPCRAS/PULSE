"""
File: schemas.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Typed schemas for PULSE inference requests and results.
License: MIT License
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Input request for PULSE analysis."""

    audio_path: str
    selected_task_keys: list[str]
    selected_task_labels: list[str]


@dataclass(frozen=True, slots=True)
class ClassProbability:
    """Probability assigned to a class label."""

    class_key: str
    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class TaskOutput:
    """Output produced for one selected analysis task."""

    task_key: str
    task_label: str
    values: list[ClassProbability]


@dataclass(frozen=True, slots=True)
class TemporalImportance:
    """Aggregated temporal importance values for explainability visualization."""

    timestamps: list[str]
    values: list[float]


@dataclass(frozen=True, slots=True)
class TemporalImportanceMap:
    """Temporal importance values for one attribution target."""

    task_key: str
    class_key: str
    label: str
    timestamps: list[str]
    values: list[float]


@dataclass(frozen=True, slots=True)
class SpeechActivity:
    """Speech activity mask diagnostics for explainability visualization."""

    timestamps: list[str]
    values: list[float]
    active_ratio: float
    min_value: float
    max_value: float
    mean_value: float


@dataclass(frozen=True, slots=True)
class AudioWaveform:
    """Downsampled audio waveform for visualization."""

    timestamps: list[float]
    values: list[float]


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One transcription segment."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class AudioTranscription:
    """Audio transcription result."""

    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    segments: list[TranscriptSegment]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """PULSE analysis result."""

    audio_path: str
    task_outputs: list[TaskOutput]
    temporal_importance: TemporalImportance
    temporal_importance_maps: list[TemporalImportanceMap]
    audio_waveform: AudioWaveform | None
    transcription: AudioTranscription | None
    speech_activity: SpeechActivity | None
    status: str
