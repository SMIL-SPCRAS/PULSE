"""
File: settings.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Settings tab event handlers for the PULSE Gradio application.
License: MIT License
"""

from typing import Any

import gradio as gr

from pulse.inference import (
    is_embedding_extractor_loaded,
    is_pulse_runtime_loaded,
    reset_embedding_extractor,
    reset_inference_caches,
    reset_pulse_runtime,
)
from pulse.localization import get_language_index, get_localized_text
from pulse.settings.schema import (
    create_configuration_overview_markdown,
    get_localized_pair,
)
from pulse.settings.state import (
    get_runtime_setting,
    get_setting_spec,
    reset_runtime_settings_to_config,
    set_runtime_setting,
)
from pulse.transcription import is_whisper_runtime_loaded, reset_whisper_runtime


def get_loaded_status_text(is_loaded: bool, language_index: int) -> str:
    """Return localized loaded status text."""

    return get_localized_text(
        ("Texts_SETTINGS_STATUS_LOADED" if is_loaded else "Texts_SETTINGS_STATUS_NOT_LOADED"),
        language_index,
    )


def create_runtime_status_markdown(language_index: int) -> str:
    """Create runtime cache status markdown."""

    return "\n".join(
        [
            f"### {get_localized_text('Texts_SETTINGS_RUNTIME_CACHE_TITLE', language_index)}",
            "",
            "| Runtime | Status |" if language_index == 0 else "| Компонент | Статус |",
            "|---|---|",
            f"| Whisper | {get_loaded_status_text(is_whisper_runtime_loaded(), language_index)} |",
            f"| Wav2Vec2 | {get_loaded_status_text(is_embedding_extractor_loaded(), language_index)} |",
            f"| PULSE | {get_loaded_status_text(is_pulse_runtime_loaded(), language_index)} |",
        ],
    )


def handle_refresh_runtime_status(language: str) -> str:
    """Refresh runtime cache status."""

    language_index = get_language_index(language)

    return create_runtime_status_markdown(language_index)


def handle_reset_whisper_runtime(language: str) -> tuple[str, str]:
    """Reset Whisper runtime cache."""

    language_index = get_language_index(language)
    reset_whisper_runtime()

    return (
        create_runtime_status_markdown(language_index),
        get_localized_text("Texts_SETTINGS_ACTION_WHISPER_RESET", language_index),
    )


def handle_reset_wav2vec_runtime(language: str) -> tuple[str, str]:
    """Reset Wav2Vec2 runtime cache."""

    language_index = get_language_index(language)
    reset_embedding_extractor()

    return (
        create_runtime_status_markdown(language_index),
        get_localized_text("Texts_SETTINGS_ACTION_WAV2VEC_RESET", language_index),
    )


def handle_reset_pulse_runtime(language: str) -> tuple[str, str]:
    """Reset PULSE runtime cache."""

    language_index = get_language_index(language)
    reset_pulse_runtime()

    return (
        create_runtime_status_markdown(language_index),
        get_localized_text("Texts_SETTINGS_ACTION_PULSE_RESET", language_index),
    )


def handle_reset_all_runtime_caches(language: str) -> tuple[str, str]:
    """Reset all runtime caches."""

    language_index = get_language_index(language)
    reset_whisper_runtime()
    reset_inference_caches()

    return (
        create_runtime_status_markdown(language_index),
        get_localized_text("Texts_SETTINGS_ACTION_ALL_RESET", language_index),
    )


def get_runtime_setting_value(section: str, field: str, default_value: object) -> object:
    """Return runtime setting value."""

    return get_runtime_setting(
        section=section,
        field=field,
        default_value=default_value,
    )


def create_setting_component_update(
    section: str,
    field: str,
    language_index: int,
    default_value: object,
    interactive: bool | None = None,
) -> Any:
    """Create localized setting component update."""

    spec = get_setting_spec(section, field)
    label = get_localized_pair(spec.label, language_index)
    info = get_localized_pair(spec.description, language_index)
    value = get_runtime_setting_value(section, field, default_value)

    if interactive is None:
        return gr.update(
            label=label,
            info=info,
            value=value,
        )

    return gr.update(
        label=label,
        info=info,
        value=value,
        interactive=interactive,
    )


def create_attribution_settings_language_updates(
    language_index: int,
) -> tuple[Any, ...]:
    """Create attribution controls updates after language change."""

    return (
        gr.update(label="Attribution" if language_index == 0 else "Атрибуция"),
        gr.update(value=get_localized_text("Texts_SETTINGS_ATTRIBUTION_CONTROLS_DESCRIPTION", language_index)),
        *create_attribution_settings_value_updates(language_index),
        get_localized_text("Labels_SETTINGS_APPLY_ATTRIBUTION", language_index),
        get_localized_text("Labels_SETTINGS_RESET_SESSION", language_index),
    )


def is_smoothgrad_method(method: str) -> bool:
    """Return whether attribution method uses SmoothGrad controls."""

    return method == "smoothgrad_input_x_gradient"


def create_attribution_dependency_updates(
    enable_real_attribution: bool,
    method: str,
    enable_speech_mask: bool,
) -> tuple[Any, ...]:
    """Create interactivity updates for dependent attribution controls."""

    attribution_controls_enabled = bool(enable_real_attribution)
    smoothgrad_controls_enabled = attribution_controls_enabled and is_smoothgrad_method(method)
    speech_mask_controls_enabled = attribution_controls_enabled and bool(enable_speech_mask)

    return (
        gr.update(interactive=attribution_controls_enabled),  # method
        gr.update(interactive=attribution_controls_enabled),  # max_points
        gr.update(interactive=attribution_controls_enabled),  # enable_speech_mask
        gr.update(interactive=speech_mask_controls_enabled),  # speech_mask_threshold_ratio
        gr.update(interactive=speech_mask_controls_enabled),  # speech_mask_context_seconds
        gr.update(interactive=speech_mask_controls_enabled),  # speech_mask_silence_floor
        gr.update(interactive=smoothgrad_controls_enabled),  # smoothgrad_samples
        gr.update(interactive=smoothgrad_controls_enabled),  # smoothgrad_noise_std_ratio
    )


def handle_attribution_dependency_change(
    enable_real_attribution: bool,
    method: str,
    enable_speech_mask: bool,
) -> tuple[Any, ...]:
    """Update dependent attribution controls after UI changes."""

    return create_attribution_dependency_updates(
        enable_real_attribution=enable_real_attribution,
        method=method,
        enable_speech_mask=enable_speech_mask,
    )


def handle_apply_attribution_settings(
    language: str,
    enable_real_attribution: bool,
    method: str,
    max_points: int | float,
    enable_speech_mask: bool,
    speech_mask_threshold_ratio: float,
    speech_mask_context_seconds: float,
    speech_mask_silence_floor: float,
    smoothgrad_samples: int | float,
    smoothgrad_noise_std_ratio: float,
) -> tuple[str, str]:
    """Apply attribution settings to current session."""

    language_index = get_language_index(language)

    try:
        set_runtime_setting("Attribution", "ENABLE_REAL_ATTRIBUTION", enable_real_attribution)
        set_runtime_setting("Attribution", "METHOD", method)
        set_runtime_setting("Attribution", "MAX_POINTS", round(max_points))
        set_runtime_setting("Attribution", "ENABLE_SPEECH_MASK", enable_speech_mask)
        set_runtime_setting("Attribution", "SPEECH_MASK_THRESHOLD_RATIO", speech_mask_threshold_ratio)
        set_runtime_setting("Attribution", "SPEECH_MASK_CONTEXT_SECONDS", speech_mask_context_seconds)
        set_runtime_setting("Attribution", "SPEECH_MASK_SILENCE_FLOOR", speech_mask_silence_floor)
        set_runtime_setting("Attribution", "SMOOTHGRAD_SAMPLES", round(smoothgrad_samples))
        set_runtime_setting("Attribution", "SMOOTHGRAD_NOISE_STD_RATIO", smoothgrad_noise_std_ratio)
    except (TypeError, ValueError, KeyError) as error:
        error_prefix = get_localized_text("Texts_SETTINGS_ACTION_ERROR_PREFIX", language_index)

        return (
            create_configuration_overview_markdown(language_index),
            f"**{error_prefix}:** {error}",
        )

    return (
        create_configuration_overview_markdown(language_index),
        get_localized_text("Texts_SETTINGS_ACTION_ATTRIBUTION_APPLIED", language_index),
    )


def create_attribution_settings_value_updates(language_index: int) -> tuple[Any, ...]:
    """Create attribution controls value and interactivity updates."""

    enable_real_attribution = bool(
        get_runtime_setting_value(
            "Attribution",
            "ENABLE_REAL_ATTRIBUTION",
            True,
        ),
    )
    method = str(
        get_runtime_setting_value(
            "Attribution",
            "METHOD",
            "smoothgrad_input_x_gradient",
        ),
    )
    enable_speech_mask = bool(
        get_runtime_setting_value(
            "Attribution",
            "ENABLE_SPEECH_MASK",
            True,
        ),
    )

    attribution_controls_enabled = enable_real_attribution
    smoothgrad_controls_enabled = attribution_controls_enabled and is_smoothgrad_method(method)
    speech_mask_controls_enabled = attribution_controls_enabled and enable_speech_mask

    return (
        create_setting_component_update(
            "Attribution",
            "ENABLE_REAL_ATTRIBUTION",
            language_index,
            True,
            interactive=True,
        ),
        create_setting_component_update(
            "Attribution",
            "METHOD",
            language_index,
            "smoothgrad_input_x_gradient",
            interactive=attribution_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "MAX_POINTS",
            language_index,
            256,
            interactive=attribution_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "ENABLE_SPEECH_MASK",
            language_index,
            True,
            interactive=attribution_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "SPEECH_MASK_THRESHOLD_RATIO",
            language_index,
            0.08,
            interactive=speech_mask_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "SPEECH_MASK_CONTEXT_SECONDS",
            language_index,
            0.20,
            interactive=speech_mask_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "SPEECH_MASK_SILENCE_FLOOR",
            language_index,
            0.02,
            interactive=speech_mask_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "SMOOTHGRAD_SAMPLES",
            language_index,
            6,
            interactive=smoothgrad_controls_enabled,
        ),
        create_setting_component_update(
            "Attribution",
            "SMOOTHGRAD_NOISE_STD_RATIO",
            language_index,
            0.02,
            interactive=smoothgrad_controls_enabled,
        ),
    )


def handle_reset_session_settings_to_config(language: str) -> tuple[Any, ...]:
    """Reset session settings to config values."""

    language_index = get_language_index(language)
    reset_runtime_settings_to_config()

    return (
        *create_attribution_settings_value_updates(language_index),
        create_configuration_overview_markdown(language_index),
        get_localized_text("Texts_SETTINGS_ACTION_SESSION_RESET", language_index),
    )


def create_settings_language_updates(language_index: int) -> tuple[Any, ...]:
    """Create settings tab updates after language change."""

    return (
        f"### {get_localized_text('Texts_SETTINGS_TITLE', language_index)}",
        get_localized_text("Texts_SETTINGS_DESCRIPTION", language_index),
        create_runtime_status_markdown(language_index),
        create_configuration_overview_markdown(language_index),
        get_localized_text("Labels_SETTINGS_REFRESH_STATUS", language_index),
        get_localized_text("Labels_SETTINGS_RESET_WHISPER", language_index),
        get_localized_text("Labels_SETTINGS_RESET_WAV2VEC", language_index),
        get_localized_text("Labels_SETTINGS_RESET_PULSE", language_index),
        get_localized_text("Labels_SETTINGS_RESET_ALL", language_index),
        get_localized_text("Texts_SETTINGS_ACTION_READY", language_index),
        *create_attribution_settings_language_updates(language_index),
    )
