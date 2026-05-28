"""
File: language.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Language switching event handlers for the PULSE Gradio application.
License: MIT License
"""

from typing import Any, cast

import gradio as gr

from pulse.audio.formatting import create_audio_metadata_html
from pulse.config import config_data
from pulse.events.application import create_transcription_output_update
from pulse.events.settings import create_settings_language_updates
from pulse.events.status import create_analysis_status_text_from_cache
from pulse.inference.cache import restore_filtered_analysis_result_from_cache
from pulse.localization import (
    TASK_KEYS,
    get_language_flag_path,
    get_language_index,
    get_localized_text,
    get_task_keys_from_labels,
    get_task_labels_from_keys,
)
from pulse.ui.application import create_application_title_markdown
from pulse.visualization import (
    create_prediction_plot_html,
    create_temporal_importance_plot_html,
)

HIDE_AUDIO_INFO_MODAL_STYLE = """
<style>
#audio-info-modal {
    display: none !important;
    pointer-events: none !important;
}
</style>
"""

SHOW_AUDIO_INFO_MODAL_STYLE = """
<style>
#audio-info-modal {
    display: flex !important;
    pointer-events: auto !important;
}
</style>
"""


def create_audio_info_button_language_update(
    audio_path: str | None,
    language_index: int,
) -> Any:
    """Update audio metadata button after language change."""

    has_audio = bool(audio_path)

    return gr.update(
        value=get_localized_text("Labels_AUDIO_INFO_TITLE", language_index),
        visible=has_audio,
        interactive=has_audio,
    )


def create_audio_info_modal_language_updates(
    audio_path: str | None,
    language_index: int,
    is_modal_open: bool,
) -> tuple[Any, Any, Any]:
    """Update audio metadata button and modal after language change."""

    has_audio = bool(audio_path)

    button_update = gr.update(
        value=get_localized_text("Labels_AUDIO_INFO_TITLE", language_index),
        visible=has_audio,
        interactive=has_audio,
    )

    title_update = gr.update(
        value=f"### {get_localized_text('Labels_AUDIO_INFO_TITLE', language_index)}",
    )

    if not is_modal_open or not audio_path:
        return (
            button_update,
            HIDE_AUDIO_INFO_MODAL_STYLE,
            title_update,
        )

    return (
        button_update,
        SHOW_AUDIO_INFO_MODAL_STYLE,
        title_update,
    )


def get_localized_status_text(
    audio_path: str | None,
    selected_task_labels: list[str] | None,
    cache: dict[str, Any] | None,
    language_index: int,
) -> str:
    """Return localized status text for the current application state."""

    if cache:
        cache_status = cache.get("status")

        if cache_status:
            cached_status_text = create_analysis_status_text_from_cache(
                cache=cache,
                language_index=language_index,
            )

            if cached_status_text is not None:
                return cached_status_text

            return get_localized_text(
                "Texts_STATUS_ANALYSIS_COMPLETED",
                language_index,
            )

    if not audio_path:
        return get_localized_text("Texts_STATUS_READY", language_index)

    if not selected_task_labels:
        return get_localized_text("Texts_STATUS_NO_TASKS", language_index)

    return get_localized_text("Texts_STATUS_AUDIO_READY", language_index)


def handle_language_change(
    language: str,
    selected_task_labels: list[str] | None,
    audio_path: str | None,
    cache: dict[str, Any] | None,
    is_audio_info_modal_open: bool,
) -> tuple[Any, ...]:
    """Update UI components after language selection."""

    language_index = get_language_index(language)

    selected_task_keys = get_task_keys_from_labels(selected_task_labels)
    localized_selected_task_labels = get_task_labels_from_keys(
        selected_task_keys,
        language_index,
    )
    localized_task_labels = get_task_labels_from_keys(
        list(TASK_KEYS),
        language_index,
    )

    app_tab_labels = cast(list[str], config_data.Tabs_APP)
    settings_tab_labels = cast(list[str], config_data.Tabs_SETTINGS)
    about_app_tab_labels = cast(list[str], config_data.Tabs_ABOUT_APP)
    authors_tab_labels = cast(list[str], config_data.Tabs_AUTHORS)
    requirements_tab_labels = cast(list[str], config_data.Tabs_REQUIREMENTS)

    restored_result = restore_filtered_analysis_result_from_cache(
        cache=cache,
        selected_task_labels=selected_task_labels,
        language_index=language_index,
    )

    prediction_plot_update: Any = gr.update()
    heatmap_plot_update: Any = gr.update()

    if restored_result is not None:
        prediction_plot_update = gr.update(
            value=create_prediction_plot_html(
                result=restored_result,
                language_index=language_index,
            ),
        )
        heatmap_plot_update = gr.update(
            value=create_temporal_importance_plot_html(
                result=restored_result,
                language_index=language_index,
            ),
        )

    transcription_output_update = (
        create_transcription_output_update(
            result=restored_result,
            language_index=language_index,
        )
        if restored_result is not None
        else gr.update(value="", visible=False)
    )

    audio_info_button_update = create_audio_info_button_language_update(
        audio_path=audio_path,
        language_index=language_index,
    )

    (
        audio_info_button_update,
        audio_info_modal_update,
        audio_info_modal_title_update,
    ) = create_audio_info_modal_language_updates(
        audio_path=audio_path,
        language_index=language_index,
        is_modal_open=is_audio_info_modal_open,
    )

    audio_info_modal_content_update = gr.update()

    if is_audio_info_modal_open and audio_path:
        audio_info_modal_content_update = gr.update(
            value=create_audio_metadata_html(
                audio_path=audio_path,
                language_index=language_index,
            ),
        )

    return (
        gr.update(value=get_language_flag_path(language_index)),
        gr.update(label=app_tab_labels[language_index]),
        gr.update(label=settings_tab_labels[language_index]),
        gr.update(label=about_app_tab_labels[language_index]),
        gr.update(label=authors_tab_labels[language_index]),
        gr.update(label=requirements_tab_labels[language_index]),
        gr.update(
            value=create_application_title_markdown(language_index),
            visible=True,
        ),
        gr.update(
            value=get_localized_status_text(
                audio_path=audio_path,
                selected_task_labels=selected_task_labels,
                cache=cache,
                language_index=language_index,
            ),
        ),
        gr.update(label=get_localized_text("Labels_AUDIO_INPUT", language_index)),
        audio_info_button_update,
        audio_info_modal_update,
        audio_info_modal_title_update,
        audio_info_modal_content_update,
        gr.update(
            choices=localized_task_labels,
            value=localized_selected_task_labels,
            label=get_localized_text("Labels_TASKS", language_index),
        ),
        gr.update(value=f"### {get_localized_text('Labels_EXAMPLES', language_index)}"),
        gr.update(value=get_localized_text("Texts_EXAMPLES_PLACEHOLDER", language_index)),
        gr.update(value=get_localized_text("Labels_RUN_ANALYSIS", language_index)),
        gr.update(value=get_localized_text("Labels_CLEAR", language_index)),
        prediction_plot_update,
        heatmap_plot_update,
        transcription_output_update,
        gr.update(value=f"# {get_localized_text('Texts_ABOUT_TITLE', language_index)}"),
        gr.update(value=get_localized_text("Texts_ABOUT_DESCRIPTION", language_index)),
        gr.update(value=get_localized_text("Texts_ABOUT_PLACEHOLDER", language_index)),
        gr.update(value=f"### {get_localized_text('Texts_REQUIREMENTS_TITLE', language_index)}"),
        gr.update(value=get_localized_text("Texts_REQUIREMENTS_PLACEHOLDER", language_index)),
        *create_settings_language_updates(language_index),
    )
