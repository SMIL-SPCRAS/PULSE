"""
File: whisper.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Whisper transcription utilities for the PULSE inference pipeline.
License: MIT License
"""

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from pulse.config import PROJECT_ROOT, is_hugging_face_space
from pulse.inference.schemas import AudioTranscription, TranscriptSegment
from pulse.logger import get_logger
from pulse.settings.state import get_runtime_setting_by_flat_name

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WhisperRuntime:
    """Loaded Faster-Whisper runtime."""

    model_name: str
    device: str
    compute_type: str
    model: Any


_WHISPER_RUNTIME: WhisperRuntime | None = None


def is_whisper_runtime_loaded() -> bool:
    """Return whether Faster-Whisper runtime is already loaded."""

    return _WHISPER_RUNTIME is not None


def get_config_bool(field_name: str, default_value: bool) -> bool:
    """Return bool transcription config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def reset_whisper_runtime() -> None:
    """Reset cached Faster-Whisper runtime."""

    global _WHISPER_RUNTIME

    if _WHISPER_RUNTIME is None:
        LOGGER.info("Faster-Whisper runtime cache is already empty.")
        return

    runtime = _WHISPER_RUNTIME
    _WHISPER_RUNTIME = None

    del runtime
    gc.collect()

    LOGGER.info("Reset Faster-Whisper runtime cache.")


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer transcription config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def get_config_float(field_name: str, default_value: float) -> float:
    """Return float transcription config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_config_str(field_name: str, default_value: str) -> str:
    """Return string transcription config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, str):
        return value

    return default_value


def resolve_path_from_models_root(relative_path: str) -> Path:
    """Resolve a path relative to the configured models root."""

    path = Path(relative_path).expanduser()

    if path.is_absolute():
        return path

    models_root = Path(
        get_config_str(
            "StaticPaths_MODELS",
            "models",
        ),
    ).expanduser()

    if models_root.is_absolute():
        return models_root / path

    return PROJECT_ROOT / models_root / path


def resolve_whisper_hf_model_reference() -> str:
    """Return Faster-Whisper Hugging Face model reference."""

    hf_model_id = get_config_str(
        "Transcription_HF_MODEL_ID",
        "",
    ).strip()

    if hf_model_id:
        return hf_model_id

    model_name = get_config_str(
        "Transcription_MODEL_NAME",
        "large-v3",
    ).strip()

    return model_name or "large-v3"


def get_whisper_local_files_only() -> bool:
    """Return Faster-Whisper local-files-only mode."""

    if is_hugging_face_space():
        return False

    return get_config_bool(
        "Transcription_LOCAL_FILES_ONLY",
        False,
    )


def resolve_whisper_model_reference() -> str:
    """Return Faster-Whisper model reference from runtime settings."""

    if is_hugging_face_space():
        return resolve_whisper_hf_model_reference()

    model_dir = get_config_str(
        "Transcription_MODEL_DIR",
        "",
    ).strip()

    if model_dir:
        return str(resolve_path_from_models_root(model_dir))

    model_name = get_config_str(
        "Transcription_MODEL_NAME",
        "large-v3",
    ).strip()

    return model_name or "large-v3"


def is_transcription_enabled() -> bool:
    """Return whether Whisper transcription is enabled."""

    return get_config_bool(
        "Transcription_ENABLE",
        False,
    )


def should_fallback_to_empty_transcription() -> bool:
    """Return whether transcription errors should be converted to empty transcription."""

    return get_config_bool(
        "Transcription_FALLBACK_TO_EMPTY",
        True,
    )


def get_transcription_language() -> str | None:
    """Return configured Whisper language."""

    language = get_config_str(
        "Transcription_LANGUAGE",
        "auto",
    ).strip()

    if not language or language.lower() == "auto":
        return None

    return language


def get_transcription_initial_prompt() -> str | None:
    """Return configured Whisper initial prompt."""

    initial_prompt = get_config_str(
        "Transcription_INITIAL_PROMPT",
        "",
    ).strip()

    if not initial_prompt:
        return None

    return initial_prompt


def create_empty_transcription() -> AudioTranscription:
    """Create empty transcription result."""

    return AudioTranscription(
        text="",
        language=None,
        language_probability=None,
        duration_seconds=None,
        segments=[],
    )


def build_whisper_runtime() -> WhisperRuntime:
    """Build Faster-Whisper runtime."""

    model_reference = resolve_whisper_model_reference()
    device = get_config_str(
        "Transcription_DEVICE",
        "cpu",
    )
    compute_type = get_config_str(
        "Transcription_COMPUTE_TYPE",
        "int8",
    )
    cpu_threads = get_config_int(
        "Transcription_CPU_THREADS",
        8,
    )
    num_workers = get_config_int(
        "Transcription_NUM_WORKERS",
        1,
    )
    local_files_only = get_whisper_local_files_only()

    model = WhisperModel(
        model_reference,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        local_files_only=local_files_only,
    )

    LOGGER.info(
        ("Loaded Faster-Whisper model: model=%s, device=%s, compute_type=%s, local_files_only=%s"),
        model_reference,
        device,
        compute_type,
        local_files_only,
    )

    return WhisperRuntime(
        model_name=model_reference,
        device=device,
        compute_type=compute_type,
        model=model,
    )


def get_whisper_runtime() -> WhisperRuntime:
    """Return cached Faster-Whisper runtime."""

    global _WHISPER_RUNTIME

    if _WHISPER_RUNTIME is None:
        _WHISPER_RUNTIME = build_whisper_runtime()

    return _WHISPER_RUNTIME


def optional_float(value: Any) -> float | None:
    """Convert value to optional float."""

    if value is None:
        return None

    if isinstance(value, int | float):
        return float(value)

    return None


def get_float_attr(instance: Any, attr_name: str, default_value: float = 0.0) -> float:
    """Return float object attribute."""

    value = getattr(instance, attr_name, default_value)

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_str_attr(instance: Any, attr_name: str, default_value: str = "") -> str:
    """Return string object attribute."""

    value = getattr(instance, attr_name, default_value)

    if isinstance(value, str):
        return value

    return default_value


def transcribe_audio_file(audio_path: str | Path, runtime: WhisperRuntime | None = None) -> AudioTranscription:
    """Transcribe audio file with Faster-Whisper."""

    active_runtime = runtime or get_whisper_runtime()

    segments_generator, info = active_runtime.model.transcribe(
        str(audio_path),
        language=get_transcription_language(),
        beam_size=get_config_int("Transcription_BEAM_SIZE", 5),
        best_of=get_config_int("Transcription_BEST_OF", 5),
        temperature=get_config_float("Transcription_TEMPERATURE", 0.0),
        word_timestamps=get_config_bool("Transcription_WORD_TIMESTAMPS", True),
        vad_filter=get_config_bool("Transcription_VAD_FILTER", True),
        vad_parameters={
            "min_silence_duration_ms": get_config_int(
                "Transcription_VAD_MIN_SILENCE_DURATION_MS",
                500,
            ),
        },
        condition_on_previous_text=get_config_bool(
            "Transcription_CONDITION_ON_PREVIOUS_TEXT",
            True,
        ),
        initial_prompt=get_transcription_initial_prompt(),
    )

    raw_segments = list(segments_generator)

    segments = [
        TranscriptSegment(
            start=get_float_attr(segment, "start"),
            end=get_float_attr(segment, "end"),
            text=get_str_attr(segment, "text").strip(),
        )
        for segment in raw_segments
        if get_str_attr(segment, "text").strip()
    ]

    text = " ".join(segment.text for segment in segments).strip()

    return AudioTranscription(
        text=text,
        language=getattr(info, "language", None),
        language_probability=optional_float(
            getattr(info, "language_probability", None),
        ),
        duration_seconds=optional_float(
            getattr(info, "duration", None),
        ),
        segments=segments,
    )


def transcribe_audio_file_safely(
    audio_path: str | Path,
    runtime: WhisperRuntime | None = None,
) -> AudioTranscription | None:
    """Transcribe audio file and optionally fall back to empty result."""

    if not is_transcription_enabled():
        return None

    try:
        return transcribe_audio_file(
            audio_path=audio_path,
            runtime=runtime,
        )
    except Exception as error:
        if should_fallback_to_empty_transcription():
            LOGGER.warning("Transcription failed; using empty transcription: %s", error)
            return create_empty_transcription()

        raise
