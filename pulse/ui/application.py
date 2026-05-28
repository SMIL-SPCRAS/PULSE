"""
File: application.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Application tab UI components for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode

import gradio as gr

from pulse.config import PROJECT_ROOT, config_data
from pulse.localization import get_localized_text, get_task_labels


@dataclass(frozen=True, slots=True)
class ApplicationTabComponents:
    """Components created inside the main application tab."""

    title: gr.Markdown
    status: gr.Markdown
    audio_input: gr.Audio
    audio_info_button: gr.Button
    analysis_timing_button: gr.Button
    audio_info_button_column: gr.Column
    audio_info_modal_open_state: gr.State
    audio_info_modal_style: gr.HTML
    audio_info_modal: gr.Column
    audio_info_modal_title: gr.Markdown
    audio_info_modal_content: gr.HTML
    audio_info_modal_close_button: gr.Button
    task_selector: gr.CheckboxGroup
    examples_title: gr.Markdown
    examples_placeholder: gr.Markdown
    examples: Any
    run_button: gr.Button
    clear_button: gr.Button
    results_style: gr.HTML
    results_container: gr.Column
    prediction_plot: gr.HTML
    heatmap_plot: gr.HTML
    transcription_output: gr.HTML
    analysis_running_state: gr.State
    analysis_busy_overlay: gr.HTML
    cache_state: gr.State


def create_application_title_markdown(language_index: int) -> str:
    """Create application title markdown with a version badge."""

    title_text = get_localized_text(
        "Texts_APP_TITLE",
        language_index,
    )
    subtitle_text = get_localized_text(
        "Texts_APP_SUBTITLE",
        language_index,
    )
    app_version = str(
        getattr(
            config_data,
            "App_VERSION",
            "0.0.0",
        ),
    ).strip()
    version_label = f"v{app_version}"
    badge_query = urlencode(
        {
            "label": "version",
            "message": version_label,
            "color": "4FB823",
            "style": "flat",
        },
    )
    badge_url = f"https://img.shields.io/static/v1?{badge_query}"

    return (
        f"### {escape(title_text)}: {escape(subtitle_text)} "
        f'<img class="application-version-badge" '
        f'src="{badge_url}" '
        f'alt="Version {escape(version_label)}">'
    )


def get_example_audio_labels(example_audio_paths: list[list[str]]) -> list[str]:
    """Return display labels for example audio files."""

    labels: list[str] = []

    for example in example_audio_paths:
        stem = Path(example[0]).stem
        parts = stem.split("_", maxsplit=1)

        if len(parts) == 2:
            index, name = parts
            labels.append(f"{index} · {name.replace('_', ' ').title()}")
        else:
            labels.append(stem.replace("_", " ").title())

    return labels


def get_existing_example_audio_paths() -> list[list[str]]:
    """Return existing example audio files configured for the application."""

    examples_dir = cast(str, config_data.StaticPaths_EXAMPLES)
    example_files = cast(list[str], getattr(config_data, "Examples_AUDIO", []))

    existing_examples: list[list[str]] = []

    for example_file in example_files:
        path = PROJECT_ROOT / examples_dir / example_file

        if path.exists():
            existing_examples.append([str(path)])

    return existing_examples


def create_application_tab(language_index: int = 0) -> ApplicationTabComponents:
    """Create the main application tab."""

    task_labels = get_task_labels(language_index)

    title = gr.Markdown(
        create_application_title_markdown(language_index),
        elem_classes="application-compact-title",
    )

    cache_state = gr.State({})
    audio_info_modal_open_state = gr.State(False)
    analysis_running_state = gr.State(False)

    analysis_busy_overlay = gr.HTML(
        value="",
        visible=True,
        elem_classes="application-floating-html",
    )

    audio_info_modal_style = gr.HTML(
        value="""
<style>
#audio-info-modal {
    display: none !important;
    pointer-events: none !important;
}
</style>
""",
        visible=True,
        elem_classes="application-floating-html",
    )

    with gr.Row(elem_classes="application-input-row"):
        with gr.Column(
            scale=5,
            min_width=360,
            elem_classes="application-audio-column",
        ):
            audio_input = gr.Audio(
                label=get_localized_text("Labels_AUDIO_INPUT", language_index),
                type="filepath",
                sources=["upload", "microphone"],
                format="wav",
                buttons=["download"],
                elem_id="application-audio-input",
                elem_classes="application-audio-input",
            )

        with gr.Column(
            scale=3,
            min_width=260,
            elem_classes="application-examples-column",
        ):
            example_audio_paths = get_existing_example_audio_paths()

            examples_title = gr.Markdown(
                f"### {get_localized_text('Labels_EXAMPLES', language_index)}",
                elem_classes="application-examples-title",
            )

            if example_audio_paths:
                examples = gr.Examples(
                    examples=example_audio_paths,
                    inputs=[audio_input],
                    cache_examples=False,
                    examples_per_page=4,
                    label="",
                    example_labels=get_example_audio_labels(example_audio_paths),
                    elem_id="application-examples",
                )

                examples_placeholder = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes="application-examples-placeholder",
                )
            else:
                examples = None

                examples_placeholder = gr.Markdown(
                    get_localized_text("Texts_EXAMPLES_PLACEHOLDER", language_index),
                    visible=True,
                    elem_classes="application-examples-placeholder",
                )

        with gr.Column(
            scale=4,
            min_width=300,
            elem_classes="application-task-column",
        ):
            task_selector = gr.CheckboxGroup(
                choices=task_labels,
                value=task_labels,
                label=get_localized_text("Labels_TASKS", language_index),
                visible=False,
                elem_classes="application-task-selector",
            )

    with gr.Row(elem_classes="application-action-row"):
        run_button = gr.Button(
            value=get_localized_text("Labels_RUN_ANALYSIS", language_index),
            variant="primary",
            size="lg",
            interactive=False,
            elem_id="run-button",
            elem_classes="application-run-button",
        )

        clear_button = gr.Button(
            value=get_localized_text("Labels_CLEAR", language_index),
            variant="secondary",
            size="lg",
            interactive=False,
            elem_id="clear-button",
            elem_classes="application-clear-button",
        )

    with gr.Row(elem_classes="application-status-row"):
        with gr.Column(
            scale=7,
            min_width=360,
            elem_classes="application-status-column",
        ):
            status = gr.Markdown(
                get_localized_text("Texts_STATUS_READY", language_index),
                elem_classes="application-status",
            )

            analysis_timing_button = gr.Button(
                value="",
                variant="secondary",
                size="sm",
                visible=False,
                interactive=False,
                elem_id="analysis-timing-button",
                elem_classes="application-analysis-timing-button",
            )

        with gr.Column(
            scale=3,
            min_width=260,
            visible=False,
            elem_id="audio-info-button-column",
            elem_classes="application-audio-info-button-column",
        ) as audio_info_button_column:
            audio_info_button = gr.Button(
                value=get_localized_text("Labels_AUDIO_INFO_TITLE", language_index),
                variant="secondary",
                size="lg",
                interactive=False,
                visible=True,
                elem_id="audio-info-button",
                elem_classes="application-audio-info-button",
            )

    with (
        gr.Column(
            visible=True,
            elem_id="audio-info-modal",
            elem_classes="audio-info-modal-backdrop",
        ) as audio_info_modal,
        gr.Column(elem_classes="audio-info-modal-card"),
    ):
        with gr.Row(elem_classes="audio-info-modal-header"):
            with gr.Column(
                scale=1,
                min_width=0,
                elem_classes="audio-info-modal-title-column",
            ):
                audio_info_modal_title = gr.Markdown(
                    f"### {get_localized_text('Labels_AUDIO_INFO_TITLE', language_index)}",
                    elem_classes="audio-info-modal-title",
                )

            with gr.Column(
                scale=1,
                min_width=42,
                elem_classes="audio-info-modal-close-column",
            ):
                audio_info_modal_close_button = gr.Button(
                    value="Close",
                    variant="secondary",
                    size="sm",
                    elem_id="audio-info-modal-close-button",
                    elem_classes="audio-info-modal-close-button",
                )

        audio_info_modal_content = gr.HTML(
            value="",
            elem_classes="audio-info-modal-content",
        )

    results_style = gr.HTML(
        value="""
<style>
div.application-results-row {
    visibility: hidden !important;
    max-height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""",
        visible=True,
        elem_classes="application-floating-html",
    )

    with gr.Column(
        visible=True,
        elem_classes="application-results-row",
    ) as results_container:
        with gr.Column(
            elem_classes="application-prediction-column",
        ):
            prediction_plot = gr.HTML(
                value="",
                visible=True,
                elem_classes="application-prediction-plot",
            )

        with gr.Column(
            elem_classes="application-heatmap-column",
        ):
            heatmap_plot = gr.HTML(
                value="",
                visible=True,
                elem_classes="application-heatmap-plot",
            )
            transcription_output = gr.HTML(
                value="",
                visible=False,
                elem_classes="application-transcription",
            )

    return ApplicationTabComponents(
        title=title,
        status=status,
        audio_input=audio_input,
        audio_info_button=audio_info_button,
        analysis_timing_button=analysis_timing_button,
        audio_info_button_column=audio_info_button_column,
        audio_info_modal_open_state=audio_info_modal_open_state,
        audio_info_modal_style=audio_info_modal_style,
        audio_info_modal=audio_info_modal,
        audio_info_modal_title=audio_info_modal_title,
        audio_info_modal_content=audio_info_modal_content,
        audio_info_modal_close_button=audio_info_modal_close_button,
        task_selector=task_selector,
        examples_title=examples_title,
        examples_placeholder=examples_placeholder,
        examples=examples,
        run_button=run_button,
        clear_button=clear_button,
        results_style=results_style,
        results_container=results_container,
        prediction_plot=prediction_plot,
        heatmap_plot=heatmap_plot,
        transcription_output=transcription_output,
        analysis_running_state=analysis_running_state,
        analysis_busy_overlay=analysis_busy_overlay,
        cache_state=cache_state,
    )
