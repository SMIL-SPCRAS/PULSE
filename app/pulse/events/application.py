"""
File: application.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Application tab event handlers for the PULSE Gradio application.
License: MIT License
"""

from html import escape
import time
from typing import Any

import gradio as gr

from pulse.audio.formatting import (
    create_audio_metadata_html,
    create_unavailable_audio_metadata_html,
    format_audio_validation_error_markdown,
)
from pulse.audio.validation import validate_audio_file
from pulse.config import config_data
from pulse.events.status import (
    create_analysis_status_text,
    create_analysis_status_text_from_cache,
    create_processing_time_button_label,
    get_pipeline_stage_label,
)
from pulse.inference import (
    AnalysisPipelineError,
    AnalysisPipelineResult,
    AnalysisRequest,
    PipelineProgressEvent,
    create_analysis_cache,
    filter_analysis_result_by_task_keys,
    get_pipeline_progress_stage_keys,
    iter_analysis_pipeline,
    restore_filtered_analysis_result_from_cache,
)
from pulse.localization import (
    get_language_index,
    get_localized_text,
    get_plot_text,
    get_task_keys_from_labels,
    get_task_labels,
)
from pulse.logger import get_logger
from pulse.visualization import (
    create_prediction_plot_html,
    create_temporal_importance_plot_html,
)

LOGGER = get_logger(__name__)

HIDE_RESULTS_STYLE = """
<style>
div.application-results-row {
    display: flex !important;
    flex-direction: column !important;
    visibility: hidden !important;
    max-height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
"""

SHOW_RESULTS_STYLE = """
<style>
div.application-results-row {
    display: flex !important;
    flex-direction: column !important;
    visibility: visible !important;
    max-height: none !important;
    overflow: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
</style>
"""

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


def get_pipeline_overlay_stage_keys() -> tuple[str, ...]:
    """Return ordered stage keys for the progress overlay."""

    return get_pipeline_progress_stage_keys()


def handle_audio_change(
    audio_path: str | None,
    selected_tasks: list[str] | None,
    language: str,
) -> tuple[Any, ...]:
    """Update application controls when an audio sample is uploaded or removed."""

    language_index = get_language_index(language)

    if not audio_path:
        return (
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            get_localized_text("Texts_STATUS_READY", language_index),
            gr.update(visible=False),
            gr.update(
                value=get_localized_text("Labels_AUDIO_INFO_TITLE", language_index),
                interactive=False,
            ),
            HIDE_AUDIO_INFO_MODAL_STYLE,
            HIDE_RESULTS_STYLE,
            {},
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
            False,
        )

    validation_result = validate_audio_file(audio_path)
    is_audio_valid = validation_result.is_valid
    can_run_analysis = is_audio_valid and bool(selected_tasks)

    if not is_audio_valid:
        status_text = format_audio_validation_error_markdown(
            validation_result=validation_result,
            language_index=language_index,
        )
    elif selected_tasks:
        status_text = get_localized_text("Texts_STATUS_AUDIO_READY", language_index)
    else:
        status_text = get_localized_text("Texts_STATUS_NO_TASKS", language_index)

    return (
        gr.update(visible=True),
        gr.update(interactive=can_run_analysis),
        gr.update(interactive=True),
        status_text,
        gr.update(visible=True),
        gr.update(
            value=get_localized_text("Labels_AUDIO_INFO_TITLE", language_index),
            interactive=True,
        ),
        HIDE_AUDIO_INFO_MODAL_STYLE,
        HIDE_RESULTS_STYLE,
        {},
        gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        False,
    )


def create_audio_info_content(audio_path: str | None, language_index: int) -> str:
    """Create audio metadata modal content."""

    if not audio_path:
        return create_unavailable_audio_metadata_html(language_index)

    try:
        return create_audio_metadata_html(
            audio_path=audio_path,
            language_index=language_index,
        )
    except Exception as error:
        LOGGER.warning("Failed to read audio metadata: %s", error)
        return create_unavailable_audio_metadata_html(language_index)


def handle_show_audio_info(
    audio_path: str | None,
    language: str,
) -> tuple[Any, ...]:
    """Show audio metadata modal."""

    language_index = get_language_index(language)

    return (
        SHOW_AUDIO_INFO_MODAL_STYLE,
        gr.update(value=f"### {get_localized_text('Labels_AUDIO_INFO_TITLE', language_index)}"),
        gr.update(value=create_audio_info_content(audio_path, language_index)),
        True,
    )


def handle_hide_audio_info() -> tuple[Any, ...]:
    """Hide audio metadata modal."""

    return (
        HIDE_AUDIO_INFO_MODAL_STYLE,
        False,
    )


def handle_task_selection_change(
    selected_tasks: list[str] | None,
    audio_path: str | None,
    language: str,
    cache: dict[str, Any] | None,
    is_analysis_running: bool,
) -> tuple[Any, ...]:
    """Update controls and cached plots when selected tasks change."""

    language_index = get_language_index(language)

    if is_analysis_running:
        return (
            gr.update(interactive=False),  # run_button
            get_localized_text("Texts_STATUS_ANALYSIS_RUNNING", language_index),  # status
            gr.update(),  # results_style
            gr.update(),  # prediction_plot
            gr.update(),  # heatmap_plot
            gr.update(),  # transcription_output
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        )

    if not audio_path:
        return (
            gr.update(interactive=False),  # run_button
            get_localized_text("Texts_STATUS_READY", language_index),  # status
            HIDE_RESULTS_STYLE,  # results_style
            gr.update(),  # prediction_plot
            gr.update(),  # heatmap_plot
            gr.update(value="", visible=False),  # transcription_output
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        )

    validation_result = validate_audio_file(audio_path)

    if not validation_result.is_valid:
        return (
            gr.update(interactive=False),  # run_button
            format_audio_validation_error_markdown(
                validation_result=validation_result,
                language_index=language_index,
            ),  # status
            HIDE_RESULTS_STYLE,  # results_style
            gr.update(),  # prediction_plot
            gr.update(),  # heatmap_plot
            gr.update(value="", visible=False),  # transcription_output
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        )

    if not selected_tasks:
        return (
            gr.update(interactive=False),  # run_button
            get_localized_text("Texts_STATUS_NO_TASKS", language_index),  # status
            HIDE_RESULTS_STYLE,  # results_style
            gr.update(),  # prediction_plot
            gr.update(),  # heatmap_plot
            gr.update(value="", visible=False),  # transcription_output
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        )

    restored_result = restore_filtered_analysis_result_from_cache(
        cache=cache,
        selected_task_labels=selected_tasks,
        language_index=language_index,
    )

    if restored_result is not None:
        cached_status_text = create_analysis_status_text_from_cache(
            cache=cache,
            language_index=language_index,
        )

        if cached_status_text is None:
            cached_status_text = get_localized_text(
                "Texts_STATUS_ANALYSIS_COMPLETED",
                language_index,
            )

        return (
            gr.update(interactive=True),  # run_button
            cached_status_text,  # status
            SHOW_RESULTS_STYLE,  # results_style
            create_prediction_plot_html(
                result=restored_result,
                language_index=language_index,
            ),  # prediction_plot
            create_temporal_importance_plot_html(
                result=restored_result,
                language_index=language_index,
            ),  # heatmap_plot
            create_transcription_output_update(
                result=restored_result,
                language_index=language_index,
            ),  # transcription_output
            create_analysis_timing_button_update(
                cache=cache,
                language_index=language_index,
            ),  # analysis_timing_button
        )

    return (
        gr.update(interactive=True),  # run_button
        get_localized_text("Texts_STATUS_AUDIO_READY", language_index),  # status
        HIDE_RESULTS_STYLE,  # results_style
        gr.update(),  # prediction_plot
        gr.update(),  # heatmap_plot
        gr.update(value="", visible=False),  # transcription_output
        gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
    )


def create_running_analysis_update(
    progress_event: PipelineProgressEvent,
    language_index: int,
) -> tuple[Any, ...]:
    """Create UI update while analysis is running."""

    stage_label = get_pipeline_overlay_stage_label(
        progress_event.stage_key,
        language_index,
    )

    status_text = (
        f"{get_localized_text('Texts_STATUS_ANALYSIS_RUNNING', language_index)}\n\n"
        f"**{get_localized_text('Texts_ANALYSIS_PROGRESS_STAGE', language_index)}:** {stage_label}\n\n"
        f"**{get_localized_text('Texts_ANALYSIS_PROGRESS_PERCENT', language_index)}:** {progress_event.percentage:.0f}%\n\n"
        f"**{get_localized_text('Texts_ANALYSIS_PROGRESS_ELAPSED', language_index)}:** "
        f"{format_elapsed_seconds(progress_event.elapsed_seconds)}"
    )

    return (
        status_text,
        HIDE_RESULTS_STYLE,
        gr.update(),  # prediction_plot
        gr.update(),  # heatmap_plot
        gr.update(),  # transcription_output
        gr.update(),  # cache_state
        gr.update(),  # analysis_timing_button
        gr.update(
            value=get_localized_text("Labels_RUNNING_ANALYSIS", language_index),
            interactive=False,
        ),  # run_button
        gr.update(interactive=False),  # clear_button
        gr.update(interactive=False),  # task_selector
        gr.update(interactive=False),  # audio_input
        create_analysis_busy_overlay(
            language_index=language_index,
            progress_event=progress_event,
        ),  # analysis_busy_overlay
        True,  # analysis_running_state
    )


def handle_run_analysis(
    audio_path: str | None,
    selected_tasks: list[str] | None,
    language: str,
) -> Any:
    """Run PULSE analysis with live progress updates."""

    language_index = get_language_index(language)

    LOGGER.info("Run analysis clicked.")
    LOGGER.info("audio_path=%s", audio_path)
    LOGGER.info("selected_tasks=%s", selected_tasks)

    if not audio_path:
        raise gr.Error(get_localized_text("Texts_STATUS_READY", language_index))

    if not selected_tasks:
        raise gr.Error(get_localized_text("Texts_STATUS_NO_TASKS", language_index))

    selected_task_keys = get_task_keys_from_labels(selected_tasks)
    all_task_labels = get_task_labels(language_index)
    all_task_keys = get_task_keys_from_labels(all_task_labels)

    request = AnalysisRequest(
        audio_path=audio_path,
        selected_task_keys=all_task_keys,
        selected_task_labels=all_task_labels,
    )

    pipeline_result: AnalysisPipelineResult | None = None

    try:
        for pipeline_update in iter_analysis_pipeline(
            request=request,
            language_index=language_index,
        ):
            if isinstance(pipeline_update, PipelineProgressEvent):
                yield create_running_analysis_update(
                    progress_event=pipeline_update,
                    language_index=language_index,
                )
                continue

            pipeline_result = pipeline_update
    except AnalysisPipelineError as error:
        yield (
            str(error),
            HIDE_RESULTS_STYLE,
            gr.update(),
            gr.update(),
            gr.update(value="", visible=False),
            gr.update(),
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
            gr.update(
                value=get_localized_text("Labels_RUN_ANALYSIS", language_index),
                interactive=True,
            ),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "",
            False,
        )
        return

    if pipeline_result is None:
        yield (
            get_localized_text("Texts_STATUS_ANALYSIS_COMPLETED", language_index),
            HIDE_RESULTS_STYLE,
            gr.update(),
            gr.update(),
            gr.update(value="", visible=False),
            gr.update(),
            gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
            gr.update(
                value=get_localized_text("Labels_RUN_ANALYSIS", language_index),
                interactive=True,
            ),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "",
            False,
        )
        return

    overlay_stage_keys = get_pipeline_overlay_stage_keys()
    visualization_stage_index = (
        overlay_stage_keys.index("visualization") + 1
        if "visualization" in overlay_stage_keys
        else len(overlay_stage_keys)
    )

    visualization_progress_event = PipelineProgressEvent(
        stage_key="visualization",
        stage_index=visualization_stage_index,
        stage_count=len(overlay_stage_keys),
        elapsed_seconds=pipeline_result.elapsed_seconds,
        detail="",
    )

    yield create_running_analysis_update(
        progress_event=visualization_progress_event,
        language_index=language_index,
    )

    decoded_audio = pipeline_result.decoded_audio
    preprocessed_audio = pipeline_result.preprocessed_audio
    embeddings = pipeline_result.embeddings
    result = pipeline_result.result
    elapsed_seconds = pipeline_result.elapsed_seconds

    filtered_result = filter_analysis_result_by_task_keys(
        result=result,
        selected_task_keys=selected_task_keys,
    )

    LOGGER.info(
        "Decoded audio: sample_rate=%s, num_channels=%s, duration_seconds=%.3f, waveform_shape=%s",
        decoded_audio.sample_rate,
        decoded_audio.num_channels,
        decoded_audio.duration_seconds,
        decoded_audio.waveform_shape,
    )

    LOGGER.info(
        (
            "Preprocessed audio: target_duration_seconds=%.3f, target_num_samples=%s, "
            "waveform_shape=%s, model_input_shape=%s, peak_amplitude=%.6f, "
            "peak_normalized=%s, trimmed=%s, padded=%s"
        ),
        preprocessed_audio.target_duration_seconds,
        preprocessed_audio.target_num_samples,
        preprocessed_audio.waveform_shape,
        preprocessed_audio.model_input_shape,
        preprocessed_audio.peak_amplitude,
        preprocessed_audio.was_peak_normalized,
        preprocessed_audio.was_trimmed,
        preprocessed_audio.was_padded,
    )

    LOGGER.info(
        "Wav2Vec2 embeddings: model=%s, device=%s, sample_rate=%s, length=%s, embedding_dim=%s, shape=%s",
        embeddings.model_name,
        embeddings.device,
        embeddings.sample_rate,
        embeddings.length,
        embeddings.embedding_dim,
        embeddings.embeddings_shape,
    )

    LOGGER.info(
        "Attribution targets: %s",
        ", ".join(
            f"{target.task_key}:{target.class_key}[{target.target_index}]"
            for target in pipeline_result.attribution_targets
        ),
    )

    if pipeline_result.attribution_result is None:
        LOGGER.info("Temporal attribution: placeholder")
    else:
        LOGGER.info(
            "Temporal attribution: records=%s, points=%s",
            len(pipeline_result.attribution_result.records),
            len(pipeline_result.attribution_result.temporal_importance.values),
        )

    visualization_started_at = time.perf_counter()

    prediction_plot = create_prediction_plot_html(
        result=filtered_result,
        language_index=language_index,
    )
    heatmap_plot = create_temporal_importance_plot_html(
        result=filtered_result,
        language_index=language_index,
    )
    transcription_output_update = create_transcription_output_update(
        result=filtered_result,
        language_index=language_index,
    )

    visualization_elapsed_seconds = time.perf_counter() - visualization_started_at
    total_elapsed_seconds = elapsed_seconds + visualization_elapsed_seconds

    completed_stage_timings = append_visualization_stage_timing(
        stage_timings=pipeline_result.stage_timings,
        visualization_elapsed_seconds=visualization_elapsed_seconds,
    )

    status_text = create_analysis_status_text(
        language_index=language_index,
        elapsed_seconds=total_elapsed_seconds,
        device=embeddings.device,
        stage_timings=completed_stage_timings,
    )

    cache = create_analysis_cache(result)
    cache["elapsed_seconds"] = total_elapsed_seconds
    cache["device"] = embeddings.device
    cache["stage_timings"] = completed_stage_timings

    LOGGER.info("Analysis pipeline finished.")

    yield (
        status_text,
        SHOW_RESULTS_STYLE,
        prediction_plot,
        heatmap_plot,
        transcription_output_update,
        cache,
        gr.update(
            value=create_processing_time_button_label(
                language_index=language_index,
                elapsed_seconds=elapsed_seconds,
                device=embeddings.device,
            ),
            visible=True,
            interactive=True,
        ),  # analysis_timing_button
        gr.update(
            value=get_localized_text("Labels_RUN_ANALYSIS", language_index),
            interactive=True,
        ),  # run_button
        gr.update(interactive=True),  # clear_button
        gr.update(interactive=True),  # task_selector
        gr.update(interactive=True),  # audio_input
        "",  # analysis_busy_overlay
        False,  # analysis_running_state
    )


def handle_analysis_started(language: str) -> tuple[Any, ...]:
    """Lock UI immediately when analysis starts."""

    language_index = get_language_index(language)
    progress_event = create_initial_progress_event(language_index)

    return (
        gr.update(
            value=get_localized_text("Labels_RUNNING_ANALYSIS", language_index),
            interactive=False,
        ),  # run_button
        gr.update(interactive=False),  # clear_button
        gr.update(interactive=False),  # task_selector
        gr.update(interactive=False),  # audio_input
        gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        create_analysis_busy_overlay(
            language_index=language_index,
            progress_event=progress_event,
        ),  # analysis_busy_overlay
        get_localized_text("Texts_STATUS_ANALYSIS_RUNNING", language_index),  # status
        True,  # analysis_running_state
    )


def format_transcription_seconds(
    value: object,
    language_index: int,
) -> str:
    """Format transcription time value."""

    if not isinstance(value, int | float):
        return ""

    if value < 0:
        return ""

    value_text = f"{value:.2f}" if value < 10 else f"{value:.1f}"

    if language_index == 1:
        value_text = value_text.replace(".", ",")

    return f"{value_text} {get_plot_text('SECOND_SUFFIX', language_index)}"


def format_transcription_probability(value: object) -> str:
    """Format transcription language probability."""

    if not isinstance(value, int | float):
        return ""

    return f"{value:.2f}"


def get_transcription_language_display_name(
    language: object,
    language_index: int,
) -> str:
    """Return localized full transcription language name with safe fallback."""

    if not isinstance(language, str):
        return ""

    language_code = language.strip()

    if not language_code:
        return ""

    normalized_language_code = language_code.replace("-", "_").replace(" ", "_").upper()
    config_field = f"TranscriptionLanguageNames_{normalized_language_code}"
    localized_values = getattr(
        config_data,
        config_field,
        None,
    )

    if not isinstance(localized_values, list) or not localized_values:
        return language_code

    if language_index < 0 or language_index >= len(localized_values):
        language_index = 0

    localized_value = localized_values[language_index]

    if not isinstance(localized_value, str):
        return language_code

    localized_value = localized_value.strip()

    return localized_value or language_code


def create_transcription_meta_item(
    label_config_key: str,
    value: str,
    language_index: int,
) -> str:
    """Create one localized transcription metadata item."""

    label = escape(
        get_localized_text(
            label_config_key,
            language_index,
        ),
    )

    return f'<span class="application-transcription-meta-item">{label}: {escape(value)}</span>'


def create_transcription_meta_items(
    transcription: Any,
    language_index: int,
) -> str:
    """Create compact localized transcription metadata items."""

    items: list[str] = []

    language_display_name = get_transcription_language_display_name(
        language=getattr(transcription, "language", None),
        language_index=language_index,
    )

    if language_display_name:
        items.append(
            create_transcription_meta_item(
                label_config_key="Labels_TRANSCRIPTION_LANGUAGE",
                value=language_display_name,
                language_index=language_index,
            ),
        )

    language_probability = format_transcription_probability(
        getattr(transcription, "language_probability", None),
    )

    if language_probability:
        items.append(
            create_transcription_meta_item(
                label_config_key="Labels_TRANSCRIPTION_CONFIDENCE",
                value=language_probability,
                language_index=language_index,
            ),
        )

    duration = format_transcription_seconds(
        value=getattr(transcription, "duration_seconds", None),
        language_index=language_index,
    )

    if duration:
        items.append(
            create_transcription_meta_item(
                label_config_key="Labels_TRANSCRIPTION_DURATION",
                value=duration,
                language_index=language_index,
            ),
        )

    segments = getattr(transcription, "segments", [])

    if isinstance(segments, list) and segments:
        items.append(
            create_transcription_meta_item(
                label_config_key="Labels_TRANSCRIPTION_SEGMENTS",
                value=str(len(segments)),
                language_index=language_index,
            ),
        )

    return "".join(items)


def create_transcription_segments_html(
    transcription: Any,
    language_index: int,
) -> str:
    """Create collapsed localized transcription segments HTML."""

    segments = getattr(transcription, "segments", [])

    if not isinstance(segments, list) or not segments:
        return ""

    segment_rows: list[str] = []

    for segment in segments:
        start = format_transcription_seconds(
            value=getattr(segment, "start", None),
            language_index=language_index,
        )
        end = format_transcription_seconds(
            value=getattr(segment, "end", None),
            language_index=language_index,
        )
        text = escape(
            str(
                getattr(
                    segment,
                    "text",
                    "",
                ),
            ).strip(),
        )

        if not text:
            continue

        segment_rows.append(
            (
                '<div class="application-transcription-segment">'
                f'<span class="application-transcription-segment-time">{start} - {end}</span>'
                f'<span class="application-transcription-segment-text">{text}</span>'
                "</div>"
            ),
        )

    if not segment_rows:
        return ""

    summary = escape(
        get_localized_text(
            "Labels_TRANSCRIPTION_SEGMENTS",
            language_index,
        ),
    )

    return (
        '<details class="application-transcription-segments">'
        f'<summary class="application-transcription-segments-summary">{summary}</summary>'
        '<div class="application-transcription-segments-body">'
        f"{''.join(segment_rows)}"
        "</div>"
        "</details>"
    )


def create_transcription_output_html(
    transcription: Any,
    language_index: int,
) -> str:
    """Create compact transcription HTML card."""

    label = escape(
        get_localized_text(
            "Labels_TRANSCRIPTION",
            language_index,
        ),
    )
    transcript_text = escape(
        str(
            getattr(
                transcription,
                "text",
                "",
            ),
        ).strip(),
    )
    meta_html = create_transcription_meta_items(
        transcription=transcription,
        language_index=language_index,
    )
    segments_html = create_transcription_segments_html(
        transcription=transcription,
        language_index=language_index,
    )

    return (
        '<section class="application-transcription-card">'
        '<div class="application-transcription-header">'
        f'<div class="application-transcription-title">{label}</div>'
        f'<div class="application-transcription-meta">{meta_html}</div>'
        "</div>"
        f'<div class="application-transcription-text">{transcript_text}</div>'
        f"{segments_html}"
        "</section>"
    )


def create_transcription_output_update(
    result: Any,
    language_index: int,
) -> Any:
    """Create transcription HTML update."""

    transcription = getattr(result, "transcription", None)

    if transcription is None:
        return gr.update(value="", visible=False)

    transcript_text = str(
        getattr(
            transcription,
            "text",
            "",
        ),
    ).strip()

    if not transcript_text:
        return gr.update(value="", visible=False)

    return gr.update(
        value=create_transcription_output_html(
            transcription=transcription,
            language_index=language_index,
        ),
        visible=True,
    )


def get_pipeline_overlay_stage_label(
    stage_key: str,
    language_index: int,
) -> str:
    """Return localized label for one overlay stage."""

    return get_localized_text(
        f"Texts_PIPELINE_STAGE_{stage_key.upper()}",
        language_index,
    )


def serialize_pipeline_stage_timings(
    stage_timings: tuple[Any, ...],
) -> list[dict[str, float | str]]:
    """Serialize pipeline stage timings for Gradio cache."""

    serialized_timings: list[dict[str, float | str]] = []

    for stage_timing in stage_timings:
        stage_key = getattr(stage_timing, "stage_key", None)
        elapsed_seconds = getattr(stage_timing, "elapsed_seconds", None)
        percentage = getattr(stage_timing, "percentage", None)

        if not isinstance(stage_key, str):
            continue

        if not isinstance(elapsed_seconds, int | float):
            continue

        if not isinstance(percentage, int | float):
            continue

        serialized_timings.append(
            {
                "stage_key": stage_key,
                "elapsed_seconds": float(elapsed_seconds),
                "percentage": float(percentage),
            },
        )

    return serialized_timings


def append_visualization_stage_timing(
    stage_timings: tuple[Any, ...],
    visualization_elapsed_seconds: float,
) -> list[dict[str, float | str]]:
    """Append visualization timing and recompute percentages."""

    normalized_timings: list[dict[str, float | str]] = []

    total_elapsed_seconds = float(visualization_elapsed_seconds)

    for stage_timing in stage_timings:
        stage_key = getattr(stage_timing, "stage_key", None)
        elapsed_seconds = getattr(stage_timing, "elapsed_seconds", None)

        if not isinstance(stage_key, str):
            continue

        if not isinstance(elapsed_seconds, int | float):
            continue

        elapsed = float(elapsed_seconds)
        total_elapsed_seconds += elapsed

        normalized_timings.append(
            {
                "stage_key": stage_key,
                "elapsed_seconds": elapsed,
            },
        )

    normalized_timings.append(
        {
            "stage_key": "visualization",
            "elapsed_seconds": float(visualization_elapsed_seconds),
        },
    )

    if total_elapsed_seconds <= 0.0:
        for item in normalized_timings:
            item["percentage"] = 0.0
        return normalized_timings

    for item in normalized_timings:
        elapsed = float(item["elapsed_seconds"])
        item["percentage"] = 100.0 * elapsed / total_elapsed_seconds

    return normalized_timings


def create_analysis_timing_button_update(
    cache: dict[str, Any] | None,
    language_index: int,
) -> Any:
    """Create update for the last analysis timing button."""

    if not cache:
        return gr.update(value="", visible=False, interactive=False)

    elapsed_seconds = cache.get("elapsed_seconds")
    device = cache.get("device")

    if not isinstance(elapsed_seconds, int | float):
        return gr.update(value="", visible=False, interactive=False)

    if not isinstance(device, str):
        return gr.update(value="", visible=False, interactive=False)

    return gr.update(
        value=create_processing_time_button_label(
            language_index=language_index,
            elapsed_seconds=float(elapsed_seconds),
            device=device,
        ),
        visible=True,
        interactive=True,
    )


def create_initial_progress_event(language_index: int) -> PipelineProgressEvent:
    """Create initial queued progress event."""

    return PipelineProgressEvent(
        stage_key="queued",
        stage_index=1,
        stage_count=len(get_pipeline_overlay_stage_keys()),
        elapsed_seconds=0.0,
        detail=get_localized_text("Texts_STATUS_ANALYSIS_RUNNING", language_index),
    )


def format_elapsed_seconds(elapsed_seconds: float) -> str:
    """Format elapsed seconds for progress overlay."""

    if elapsed_seconds < 60.0:
        return f"{elapsed_seconds:.1f} s"

    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds - minutes * 60

    return f"{minutes} min {seconds:.0f} s"


def get_progress_stage_css_class(
    stage_key: str,
    progress_event: PipelineProgressEvent,
) -> str:
    """Return CSS class for a progress stage."""

    stage_keys = list(get_pipeline_overlay_stage_keys())

    if stage_key not in stage_keys:
        return "analysis-busy-stage-pending"

    current_stage_key = progress_event.stage_key

    if current_stage_key not in stage_keys:
        return "analysis-busy-stage-pending"

    stage_index = stage_keys.index(stage_key)
    current_index = stage_keys.index(current_stage_key)

    if stage_index < current_index:
        return "analysis-busy-stage-done"

    if stage_index == current_index:
        return "analysis-busy-stage-current"

    return "analysis-busy-stage-pending"


def create_analysis_busy_overlay(
    language_index: int,
    progress_event: PipelineProgressEvent | None = None,
) -> str:
    """Create analysis busy overlay HTML."""

    active_progress_event = progress_event or create_initial_progress_event(language_index)

    title = escape(get_localized_text("Labels_RUNNING_ANALYSIS", language_index))
    subtitle = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_SUBTITLE", language_index))
    note = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_NOTE", language_index))
    current_stage = escape(get_pipeline_overlay_stage_label(active_progress_event.stage_key, language_index))
    elapsed_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_ELAPSED", language_index))
    stage_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_STAGE", language_index))
    percent_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_PERCENT", language_index))
    elapsed_text = escape(format_elapsed_seconds(active_progress_event.elapsed_seconds))
    progress_percentage = active_progress_event.percentage

    stage_items = "".join(
        (
            f'<li class="analysis-busy-stage {get_progress_stage_css_class(stage_key, active_progress_event)}">'
            f'<span class="analysis-busy-stage-index">{stage_index}</span>'
            f'<span class="analysis-busy-stage-label">{escape(get_pipeline_overlay_stage_label(stage_key, language_index))}</span>'
            "</li>"
        )
        for stage_index, stage_key in enumerate(get_pipeline_overlay_stage_keys(), start=1)
    )

    if active_progress_event.detail in {"loading", "cached", "disabled"}:
        progress_detail_text = ""
    elif active_progress_event.detail:
        progress_detail_text = escape(active_progress_event.detail)
    else:
        progress_detail_text = ""

    return f"""
<div class="analysis-busy-overlay">
    <div class="analysis-busy-card">
        <div class="analysis-busy-header">
            <div class="analysis-busy-spinner"></div>
            <div class="analysis-busy-copy">
                <p class="analysis-busy-title">{title}</p>
                <p class="analysis-busy-text">{subtitle}</p>
            </div>
        </div>

        <div class="analysis-busy-metrics">
            <div class="analysis-busy-metric">
                <span>{stage_label}</span>
                <strong>{current_stage}</strong>
            </div>
            <div class="analysis-busy-metric">
                <span>{elapsed_label}</span>
                <strong>{elapsed_text}</strong>
            </div>
            <div class="analysis-busy-metric">
                <span>{percent_label}</span>
                <strong>{progress_percentage:.0f}%</strong>
            </div>
        </div>

        <div class="analysis-busy-progress">
            <div class="analysis-busy-progress-track">
                <div class="analysis-busy-progress-fill" style="width: {progress_percentage:.1f}%">
                    <div class="analysis-busy-progress-bar"></div>
                </div>
                <div class="analysis-busy-progress-label">{progress_detail_text}</div>
            </div>
        </div>

        <ol class="analysis-busy-stages">
            {stage_items}
        </ol>

        <p class="analysis-busy-note">{note}</p>
    </div>
</div>
"""


def get_stage_timing_field(
    stage_timing: Any,
    field_name: str,
) -> Any:
    """Return stage timing field from cache dictionary or dataclass."""

    if isinstance(stage_timing, dict):
        return stage_timing.get(field_name)

    return getattr(stage_timing, field_name, None)


def create_completed_stage_items(
    stage_timings: list[Any] | tuple[Any, ...],
    language_index: int,
) -> str:
    """Create completed stage cards for the shared analysis overlay."""

    items: list[str] = []

    for stage_index, stage_timing in enumerate(stage_timings, start=1):
        stage_key = get_stage_timing_field(stage_timing, "stage_key")
        elapsed_seconds = get_stage_timing_field(stage_timing, "elapsed_seconds")
        percentage = get_stage_timing_field(stage_timing, "percentage")

        if not isinstance(stage_key, str):
            continue

        if not isinstance(elapsed_seconds, int | float):
            continue

        if not isinstance(percentage, int | float):
            continue

        stage_label = escape(
            get_pipeline_stage_label(
                stage_key=stage_key,
                language_index=language_index,
            ),
        )

        items.append(
            (
                '<li class="analysis-busy-stage analysis-busy-stage-done analysis-completed-stage">'
                f'<span class="analysis-busy-stage-index">{stage_index}</span>'
                '<span class="analysis-completed-stage-body">'
                f'<span class="analysis-busy-stage-label">{stage_label}</span>'
                f'<span class="analysis-completed-stage-meta">{float(elapsed_seconds):.2f} s · {float(percentage):.1f}%</span>'
                "</span>"
                "</li>"
            ),
        )

    return "".join(items)


def create_completed_analysis_overlay(
    cache: dict[str, Any] | None,
    language_index: int,
) -> str:
    """Create completed analysis timing overlay using the shared busy modal shell."""

    if not cache:
        return ""

    elapsed_seconds = cache.get("elapsed_seconds")
    device = cache.get("device")
    stage_timings = cache.get("stage_timings")

    if not isinstance(elapsed_seconds, int | float):
        return ""

    if not isinstance(device, str):
        return ""

    if not isinstance(stage_timings, list):
        stage_timings = []

    title = escape(get_localized_text("Texts_STATUS_ANALYSIS_COMPLETED", language_index))
    timings_title = escape(get_localized_text("Texts_STATUS_STAGE_TIMINGS_TITLE", language_index))
    elapsed_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_ELAPSED", language_index))
    device_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_DEVICE", language_index))
    percent_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_PERCENT", language_index))
    completed_label = escape(get_localized_text("Texts_ANALYSIS_PROGRESS_COMPLETED", language_index))

    elapsed_text = escape(format_elapsed_seconds(float(elapsed_seconds)))
    device_text = escape(device)
    render_token = f"{time.perf_counter():.6f}"

    stage_items = create_completed_stage_items(
        stage_timings=stage_timings,
        language_index=language_index,
    )

    if not stage_items:
        stage_items = (
            '<li class="analysis-busy-stage analysis-busy-stage-pending analysis-completed-stage">'
            '<span class="analysis-busy-stage-index">!</span>'
            '<span class="analysis-completed-stage-body">'
            f'<span class="analysis-busy-stage-label">{escape(get_localized_text("Texts_STATUS_READY", language_index))}</span>'
            "</span>"
            "</li>"
        )

    return f"""
<div class="analysis-busy-overlay analysis-completed-overlay" data-render-token="{render_token}">
    <div class="analysis-busy-card analysis-completed-card">
        <div class="analysis-completed-header">
            <div class="analysis-completed-title">
                <p class="analysis-busy-title">{title}</p>
                <p class="analysis-busy-text">{timings_title}</p>
            </div>
            <button
                type="button"
                class="analysis-completed-close-button"
                onclick="const overlay=this.closest('.analysis-busy-overlay'); if (overlay) overlay.remove();"
                aria-label="Close"
            >
                &times;
            </button>
        </div>

        <div class="analysis-busy-metrics">
            <div class="analysis-busy-metric">
                <span>{elapsed_label}</span>
                <strong>{elapsed_text}</strong>
            </div>
            <div class="analysis-busy-metric">
                <span>{device_label}</span>
                <strong>{device_text}</strong>
            </div>
            <div class="analysis-busy-metric">
                <span>{percent_label}</span>
                <strong>100%</strong>
            </div>
        </div>

        <div class="analysis-busy-progress">
            <div class="analysis-busy-progress-track">
                <div class="analysis-busy-progress-fill" style="width: 100%">
                    <div class="analysis-busy-progress-bar"></div>
                </div>
                <div class="analysis-busy-progress-label">{completed_label}</div>
            </div>
        </div>

        <ol class="analysis-busy-stages analysis-completed-stages">
            {stage_items}
        </ol>
    </div>
</div>
"""


def handle_show_last_analysis_timing(
    cache: dict[str, Any] | None,
    language: str,
) -> str:
    """Show completed analysis timing overlay in the shared overlay component."""

    language_index = get_language_index(language)

    return create_completed_analysis_overlay(
        cache=cache,
        language_index=language_index,
    )


def handle_refresh_analysis_timing_button(
    cache: dict[str, Any] | None,
    language: str,
) -> Any:
    """Refresh last analysis timing button after analysis completion."""

    language_index = get_language_index(language)

    return create_analysis_timing_button_update(
        cache=cache,
        language_index=language_index,
    )


def handle_clear_application(language: str) -> tuple[Any, ...]:
    """Reset the application tab to its initial state."""

    language_index = get_language_index(language)
    task_labels = get_task_labels(language_index)

    return (
        None,  # audio_input
        gr.update(
            value=task_labels,
            visible=False,
        ),  # task_selector
        gr.update(interactive=False),  # run_button
        gr.update(interactive=False),  # clear_button
        get_localized_text("Texts_STATUS_CLEARED", language_index),  # status
        gr.update(visible=False),  # audio_info_button_column
        gr.update(  # audio_info_button
            value=get_localized_text("Labels_AUDIO_INFO_TITLE", language_index),
            interactive=False,
        ),
        HIDE_AUDIO_INFO_MODAL_STYLE,  # audio_info_modal_style
        HIDE_RESULTS_STYLE,  # results_style
        gr.update(value=""),  # prediction_plot
        gr.update(value=""),  # heatmap_plot
        gr.update(value="", visible=False),  # transcription_output
        {},  # cache_state
        gr.update(value="", visible=False, interactive=False),  # analysis_timing_button
        False,  # audio_info_modal_open_state
    )
