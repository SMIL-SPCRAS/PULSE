"""
File: settings.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Settings tab UI components for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass

import gradio as gr

from pulse.events.settings import create_runtime_status_markdown
from pulse.localization import get_localized_text
from pulse.settings.schema import (
    create_configuration_overview_markdown,
    get_localized_pair,
)
from pulse.settings.state import get_runtime_setting, get_setting_spec


@dataclass(frozen=True, slots=True)
class AttributionSettingsControls:
    """Editable attribution settings controls."""

    accordion: gr.Accordion
    description: gr.Markdown
    enable_real_attribution: gr.Checkbox
    method: gr.Dropdown
    max_points: gr.Slider
    enable_speech_mask: gr.Checkbox
    speech_mask_threshold_ratio: gr.Slider
    speech_mask_context_seconds: gr.Slider
    speech_mask_silence_floor: gr.Slider
    smoothgrad_samples: gr.Slider
    smoothgrad_noise_std_ratio: gr.Slider
    apply_button: gr.Button
    reset_session_button: gr.Button


@dataclass(frozen=True, slots=True)
class SettingsTabComponents:
    """Components created inside the settings tab."""

    title: gr.Markdown
    description: gr.Markdown
    runtime_status: gr.Markdown
    configuration_overview: gr.Markdown
    attribution_controls: AttributionSettingsControls
    refresh_button: gr.Button
    reset_whisper_button: gr.Button
    reset_wav2vec_button: gr.Button
    reset_pulse_button: gr.Button
    reset_all_button: gr.Button
    action_status: gr.Markdown


def get_bool_setting(section: str, field: str, default_value: bool) -> bool:
    """Return bool runtime setting value."""

    value = get_runtime_setting(section, field, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def get_str_setting(section: str, field: str, default_value: str) -> str:
    """Return string runtime setting value."""

    value = get_runtime_setting(section, field, default_value)

    if isinstance(value, str):
        return value

    return default_value


def get_int_setting(section: str, field: str, default_value: int) -> int:
    """Return integer runtime setting value."""

    value = get_runtime_setting(section, field, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return round(value)

    return default_value


def get_float_setting(section: str, field: str, default_value: float) -> float:
    """Return float runtime setting value."""

    value = get_runtime_setting(section, field, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_setting_label(section: str, field: str, language_index: int) -> str:
    """Return localized setting label."""

    return get_localized_pair(
        get_setting_spec(section, field).label,
        language_index,
    )


def get_setting_info(section: str, field: str, language_index: int) -> str:
    """Return localized setting info."""

    return get_localized_pair(
        get_setting_spec(section, field).description,
        language_index,
    )


def get_attribution_section_title(language_index: int) -> str:
    """Return localized attribution section title."""

    section = get_setting_spec("Attribution", "ENABLE_REAL_ATTRIBUTION").section

    if section == "Attribution":
        return "Attribution" if language_index == 0 else "Атрибуция"

    return section


def create_attribution_settings_controls(
    language_index: int = 0,
) -> AttributionSettingsControls:
    """Create editable attribution controls."""

    max_points_spec = get_setting_spec("Attribution", "MAX_POINTS")
    threshold_spec = get_setting_spec("Attribution", "SPEECH_MASK_THRESHOLD_RATIO")
    context_spec = get_setting_spec("Attribution", "SPEECH_MASK_CONTEXT_SECONDS")
    silence_floor_spec = get_setting_spec("Attribution", "SPEECH_MASK_SILENCE_FLOOR")
    smoothgrad_samples_spec = get_setting_spec("Attribution", "SMOOTHGRAD_SAMPLES")
    smoothgrad_noise_spec = get_setting_spec("Attribution", "SMOOTHGRAD_NOISE_STD_RATIO")
    method_spec = get_setting_spec("Attribution", "METHOD")

    enable_real_attribution_value = get_bool_setting(
        "Attribution",
        "ENABLE_REAL_ATTRIBUTION",
        True,
    )
    method_value = get_str_setting(
        "Attribution",
        "METHOD",
        "smoothgrad_input_x_gradient",
    )
    enable_speech_mask_value = get_bool_setting(
        "Attribution",
        "ENABLE_SPEECH_MASK",
        True,
    )

    attribution_controls_enabled = enable_real_attribution_value
    smoothgrad_controls_enabled = attribution_controls_enabled and method_value == "smoothgrad_input_x_gradient"
    speech_mask_controls_enabled = attribution_controls_enabled and enable_speech_mask_value

    with gr.Accordion(
        label=get_attribution_section_title(language_index),
        open=True,
        elem_classes="settings-editable-accordion",
    ) as accordion:
        description = gr.Markdown(
            value=get_localized_text("Texts_SETTINGS_ATTRIBUTION_CONTROLS_DESCRIPTION", language_index),
            elem_classes="settings-editable-description",
        )

        with gr.Row(elem_classes="settings-controls-row"):
            enable_real_attribution = gr.Checkbox(
                value=get_bool_setting("Attribution", "ENABLE_REAL_ATTRIBUTION", True),
                label=get_setting_label("Attribution", "ENABLE_REAL_ATTRIBUTION", language_index),
                info=get_setting_info("Attribution", "ENABLE_REAL_ATTRIBUTION", language_index),
                interactive=True,
                scale=1,
                min_width=260,
            )
            method = gr.Dropdown(
                choices=list(method_spec.choices),
                value=method_value,
                label=get_setting_label("Attribution", "METHOD", language_index),
                info=get_setting_info("Attribution", "METHOD", language_index),
                interactive=attribution_controls_enabled,
                allow_custom_value=False,
                filterable=False,
                scale=1,
                min_width=260,
            )

        with gr.Row(elem_classes="settings-controls-row"):
            max_points = gr.Slider(
                minimum=float(max_points_spec.minimum or 64),
                maximum=float(max_points_spec.maximum or 1024),
                value=get_int_setting("Attribution", "MAX_POINTS", 256),
                step=max_points_spec.step or 32,
                precision=0,
                label=get_setting_label("Attribution", "MAX_POINTS", language_index),
                info=get_setting_info("Attribution", "MAX_POINTS", language_index),
                interactive=attribution_controls_enabled,
                scale=1,
                min_width=260,
            )
            smoothgrad_samples = gr.Slider(
                minimum=float(smoothgrad_samples_spec.minimum or 1),
                maximum=float(smoothgrad_samples_spec.maximum or 32),
                value=get_int_setting("Attribution", "SMOOTHGRAD_SAMPLES", 6),
                step=smoothgrad_samples_spec.step or 1,
                precision=0,
                label=get_setting_label("Attribution", "SMOOTHGRAD_SAMPLES", language_index),
                info=get_setting_info("Attribution", "SMOOTHGRAD_SAMPLES", language_index),
                interactive=smoothgrad_controls_enabled,
                scale=1,
                min_width=260,
            )

        with gr.Row(elem_classes="settings-controls-row"):
            enable_speech_mask = gr.Checkbox(
                label=get_setting_label("Attribution", "ENABLE_SPEECH_MASK", language_index),
                info=get_setting_info("Attribution", "ENABLE_SPEECH_MASK", language_index),
                value=enable_speech_mask_value,
                interactive=attribution_controls_enabled,
                scale=1,
                min_width=260,
            )
            speech_mask_threshold_ratio = gr.Slider(
                minimum=float(threshold_spec.minimum or 0.0),
                maximum=float(threshold_spec.maximum or 0.5),
                value=get_float_setting("Attribution", "SPEECH_MASK_THRESHOLD_RATIO", 0.08),
                step=threshold_spec.step or 0.01,
                label=get_setting_label("Attribution", "SPEECH_MASK_THRESHOLD_RATIO", language_index),
                info=get_setting_info("Attribution", "SPEECH_MASK_THRESHOLD_RATIO", language_index),
                interactive=speech_mask_controls_enabled,
                scale=1,
                min_width=260,
            )

        with gr.Row(elem_classes="settings-controls-row"):
            speech_mask_context_seconds = gr.Slider(
                minimum=float(context_spec.minimum or 0.0),
                maximum=float(context_spec.maximum or 1.0),
                value=get_float_setting("Attribution", "SPEECH_MASK_CONTEXT_SECONDS", 0.20),
                step=context_spec.step or 0.05,
                label=get_setting_label("Attribution", "SPEECH_MASK_CONTEXT_SECONDS", language_index),
                info=get_setting_info("Attribution", "SPEECH_MASK_CONTEXT_SECONDS", language_index),
                interactive=speech_mask_controls_enabled,
                scale=1,
                min_width=260,
            )
            speech_mask_silence_floor = gr.Slider(
                minimum=float(silence_floor_spec.minimum or 0.0),
                maximum=float(silence_floor_spec.maximum or 0.3),
                value=get_float_setting("Attribution", "SPEECH_MASK_SILENCE_FLOOR", 0.02),
                step=silence_floor_spec.step or 0.01,
                label=get_setting_label("Attribution", "SPEECH_MASK_SILENCE_FLOOR", language_index),
                info=get_setting_info("Attribution", "SPEECH_MASK_SILENCE_FLOOR", language_index),
                interactive=speech_mask_controls_enabled,
                scale=1,
                min_width=260,
            )

        with gr.Row(elem_classes="settings-controls-row"):
            smoothgrad_noise_std_ratio = gr.Slider(
                minimum=float(smoothgrad_noise_spec.minimum or 0.0),
                maximum=float(smoothgrad_noise_spec.maximum or 0.2),
                value=get_float_setting("Attribution", "SMOOTHGRAD_NOISE_STD_RATIO", 0.02),
                step=smoothgrad_noise_spec.step or 0.005,
                label=get_setting_label("Attribution", "SMOOTHGRAD_NOISE_STD_RATIO", language_index),
                info=get_setting_info("Attribution", "SMOOTHGRAD_NOISE_STD_RATIO", language_index),
                interactive=smoothgrad_controls_enabled,
                scale=1,
                min_width=260,
            )

        with gr.Row(elem_classes="settings-controls-actions-row"):
            apply_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_APPLY_ATTRIBUTION", language_index),
                variant="primary",
                size="sm",
                elem_id="settings-apply-attribution-button",
            )
            reset_session_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_RESET_SESSION", language_index),
                variant="secondary",
                size="sm",
                elem_id="settings-reset-session-button",
            )

    return AttributionSettingsControls(
        accordion=accordion,
        description=description,
        enable_real_attribution=enable_real_attribution,
        method=method,
        max_points=max_points,
        enable_speech_mask=enable_speech_mask,
        speech_mask_threshold_ratio=speech_mask_threshold_ratio,
        speech_mask_context_seconds=speech_mask_context_seconds,
        speech_mask_silence_floor=speech_mask_silence_floor,
        smoothgrad_samples=smoothgrad_samples,
        smoothgrad_noise_std_ratio=smoothgrad_noise_std_ratio,
        apply_button=apply_button,
        reset_session_button=reset_session_button,
    )


def create_settings_tab(language_index: int = 0) -> SettingsTabComponents:
    """Create the settings tab."""

    title = gr.Markdown(
        f"### {get_localized_text('Texts_SETTINGS_TITLE', language_index)}",
        elem_classes="settings-title",
    )
    description = gr.Markdown(
        get_localized_text("Texts_SETTINGS_DESCRIPTION", language_index),
        elem_classes="settings-description",
    )

    with gr.Column(elem_classes="settings-card"):
        runtime_status = gr.Markdown(
            value=create_runtime_status_markdown(language_index),
            elem_classes="settings-runtime-status",
        )

        with gr.Row(elem_classes="settings-actions-row"):
            refresh_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_REFRESH_STATUS", language_index),
                variant="secondary",
                size="sm",
                elem_id="settings-refresh-button",
            )
            reset_whisper_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_RESET_WHISPER", language_index),
                variant="secondary",
                size="sm",
                elem_id="settings-reset-whisper-button",
            )
            reset_wav2vec_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_RESET_WAV2VEC", language_index),
                variant="secondary",
                size="sm",
                elem_id="settings-reset-wav2vec-button",
            )
            reset_pulse_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_RESET_PULSE", language_index),
                variant="secondary",
                size="sm",
                elem_id="settings-reset-pulse-button",
            )
            reset_all_button = gr.Button(
                value=get_localized_text("Labels_SETTINGS_RESET_ALL", language_index),
                variant="primary",
                size="sm",
                elem_id="settings-reset-all-button",
            )

        action_status = gr.Markdown(
            value=get_localized_text("Texts_SETTINGS_ACTION_READY", language_index),
            elem_classes="settings-action-status",
        )

        attribution_controls = create_attribution_settings_controls(language_index)

        configuration_overview = gr.Markdown(
            value=create_configuration_overview_markdown(language_index),
            elem_classes="settings-configuration-overview",
        )

    return SettingsTabComponents(
        title=title,
        description=description,
        runtime_status=runtime_status,
        configuration_overview=configuration_overview,
        attribution_controls=attribution_controls,
        refresh_button=refresh_button,
        reset_whisper_button=reset_whisper_button,
        reset_wav2vec_button=reset_wav2vec_button,
        reset_pulse_button=reset_pulse_button,
        reset_all_button=reset_all_button,
        action_status=action_status,
    )
