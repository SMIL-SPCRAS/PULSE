"""
File: status.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Status text helpers for the PULSE Gradio application.
License: MIT License
"""

from typing import Any, Final

from pulse.localization import get_localized_text

PIPELINE_STAGE_LABEL_FIELDS: Final = {
    "validation": "Texts_PIPELINE_STAGE_VALIDATION",
    "decoding": "Texts_PIPELINE_STAGE_DECODING",
    "preprocessing": "Texts_PIPELINE_STAGE_PREPROCESSING",
    "waveform": "Texts_PIPELINE_STAGE_WAVEFORM",
    "whisper_loading": "Texts_PIPELINE_STAGE_WHISPER_LOADING",
    "transcription": "Texts_PIPELINE_STAGE_TRANSCRIPTION",
    "wav2vec_loading": "Texts_PIPELINE_STAGE_WAV2VEC_LOADING",
    "feature_extraction": "Texts_PIPELINE_STAGE_FEATURE_EXTRACTION",
    "pulse_model_loading": "Texts_PIPELINE_STAGE_PULSE_MODEL_LOADING",
    "model_inference": "Texts_PIPELINE_STAGE_MODEL_INFERENCE",
    "output_adaptation": "Texts_PIPELINE_STAGE_OUTPUT_ADAPTATION",
    "attribution": "Texts_PIPELINE_STAGE_ATTRIBUTION",
    "packaging": "Texts_PIPELINE_STAGE_PACKAGING",
    "visualization": "Texts_PIPELINE_STAGE_VISUALIZATION",
}


def get_pipeline_stage_label(
    stage_key: str,
    language_index: int,
) -> str:
    """Return localized pipeline stage label."""

    label_field = PIPELINE_STAGE_LABEL_FIELDS.get(stage_key)

    if label_field is None:
        return stage_key.replace("_", " ").title()

    return get_localized_text(label_field, language_index)


def get_stage_timing_value(
    stage_timing: Any,
    field_name: str,
) -> Any:
    """Return stage timing field from a dataclass or dictionary."""

    if isinstance(stage_timing, dict):
        return stage_timing.get(field_name)

    return getattr(stage_timing, field_name, None)


def create_stage_timings_markdown(
    stage_timings: tuple[Any, ...] | list[Any] | None,
    language_index: int,
) -> str:
    """Create localized Markdown table with pipeline stage timings."""

    if not stage_timings:
        return ""

    lines = [
        f"### {get_localized_text('Texts_STATUS_STAGE_TIMINGS_TITLE', language_index)}",
        "",
        ("| Stage | Time | Share |" if language_index == 0 else "| Этап | Время | Доля |"),
        "|---|---:|---:|",
    ]

    for stage_timing in stage_timings:
        stage_key = get_stage_timing_value(stage_timing, "stage_key")
        elapsed_seconds = get_stage_timing_value(stage_timing, "elapsed_seconds")
        percentage = get_stage_timing_value(stage_timing, "percentage")

        if not isinstance(stage_key, str):
            continue

        if not isinstance(elapsed_seconds, int | float):
            continue

        if not isinstance(percentage, int | float):
            continue

        lines.append(
            (
                f"| {get_pipeline_stage_label(stage_key, language_index)} "
                f"| {float(elapsed_seconds):.2f} s "
                f"| {float(percentage):.1f}% |"
            ),
        )

    if len(lines) <= 4:
        return ""

    return "\n".join(lines)


def create_processing_time_button_label(
    language_index: int,
    elapsed_seconds: float,
    device: str,
) -> str:
    """Create processing time button label."""

    return get_localized_text(
        "Texts_STATUS_ANALYSIS_SUMMARY",
        language_index,
    ).format(
        elapsed=elapsed_seconds,
        device=device,
    )


def create_analysis_status_text(
    language_index: int,
    elapsed_seconds: float,
    device: str,
    stage_timings: tuple[Any, ...] | list[Any] | None = None,
) -> str:
    """Create localized analysis completion status text."""

    return get_localized_text(
        "Texts_STATUS_ANALYSIS_COMPLETED",
        language_index,
    )


def create_analysis_status_text_from_cache(
    cache: dict[str, Any] | None,
    language_index: int,
) -> str | None:
    """Create localized analysis completion status text from cache."""

    if not cache:
        return None

    elapsed_seconds = cache.get("elapsed_seconds")
    device = cache.get("device")
    stage_timings = cache.get("stage_timings")

    if not isinstance(elapsed_seconds, int | float):
        return None

    if not isinstance(device, str):
        return None

    return create_analysis_status_text(
        language_index=language_index,
        elapsed_seconds=float(elapsed_seconds),
        device=device,
        stage_timings=stage_timings if isinstance(stage_timings, list) else None,
    )
