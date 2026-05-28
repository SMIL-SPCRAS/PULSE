"""
File: event_handlers.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Event handler registration for the PULSE Gradio application.
License: MIT License
"""

from pulse.events.application import (
    handle_analysis_started,
    handle_audio_change,
    handle_clear_application,
    handle_hide_audio_info,
    handle_refresh_analysis_timing_button,
    handle_run_analysis,
    handle_show_audio_info,
    handle_show_last_analysis_timing,
    handle_task_selection_change,
)
from pulse.events.language import handle_language_change
from pulse.events.settings import (
    handle_apply_attribution_settings,
    handle_attribution_dependency_change,
    handle_refresh_runtime_status,
    handle_reset_all_runtime_caches,
    handle_reset_pulse_runtime,
    handle_reset_session_settings_to_config,
    handle_reset_wav2vec_runtime,
    handle_reset_whisper_runtime,
)
from pulse.ui.language_selector import LanguageSelectorComponents
from pulse.ui.tabs import AppTabsComponents


def setup_app_event_handlers(
    language_selector: LanguageSelectorComponents,
    app_tabs: AppTabsComponents,
) -> None:
    """Register application event handlers."""

    language_selector.dropdown.change(
        fn=handle_language_change,
        inputs=[
            language_selector.dropdown,
            app_tabs.app_content.task_selector,
            app_tabs.app_content.audio_input,
            app_tabs.app_content.cache_state,
            app_tabs.app_content.audio_info_modal_open_state,
        ],
        outputs=[
            language_selector.flag,
            app_tabs.app,
            app_tabs.settings,
            app_tabs.about_app,
            app_tabs.authors,
            app_tabs.requirements,
            app_tabs.app_content.title,
            app_tabs.app_content.status,
            app_tabs.app_content.audio_input,
            app_tabs.app_content.audio_info_button,
            app_tabs.app_content.audio_info_modal_style,
            app_tabs.app_content.audio_info_modal_title,
            app_tabs.app_content.audio_info_modal_content,
            app_tabs.app_content.task_selector,
            app_tabs.app_content.examples_title,
            app_tabs.app_content.examples_placeholder,
            app_tabs.app_content.run_button,
            app_tabs.app_content.clear_button,
            app_tabs.app_content.prediction_plot,
            app_tabs.app_content.heatmap_plot,
            app_tabs.app_content.transcription_output,
            app_tabs.about_app_content.title,
            app_tabs.about_app_content.description,
            app_tabs.about_app_content.placeholder,
            app_tabs.requirements_content.title,
            app_tabs.requirements_content.placeholder,
            app_tabs.settings_content.title,
            app_tabs.settings_content.description,
            app_tabs.settings_content.runtime_status,
            app_tabs.settings_content.configuration_overview,
            app_tabs.settings_content.refresh_button,
            app_tabs.settings_content.reset_whisper_button,
            app_tabs.settings_content.reset_wav2vec_button,
            app_tabs.settings_content.reset_pulse_button,
            app_tabs.settings_content.reset_all_button,
            app_tabs.settings_content.action_status,
            app_tabs.settings_content.attribution_controls.accordion,
            app_tabs.settings_content.attribution_controls.description,
            app_tabs.settings_content.attribution_controls.enable_real_attribution,
            app_tabs.settings_content.attribution_controls.method,
            app_tabs.settings_content.attribution_controls.max_points,
            app_tabs.settings_content.attribution_controls.enable_speech_mask,
            app_tabs.settings_content.attribution_controls.speech_mask_threshold_ratio,
            app_tabs.settings_content.attribution_controls.speech_mask_context_seconds,
            app_tabs.settings_content.attribution_controls.speech_mask_silence_floor,
            app_tabs.settings_content.attribution_controls.smoothgrad_samples,
            app_tabs.settings_content.attribution_controls.smoothgrad_noise_std_ratio,
            app_tabs.settings_content.attribution_controls.apply_button,
            app_tabs.settings_content.attribution_controls.reset_session_button,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.app_content.audio_input.change(
        fn=handle_audio_change,
        inputs=[
            app_tabs.app_content.audio_input,
            app_tabs.app_content.task_selector,
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.task_selector,
            app_tabs.app_content.run_button,
            app_tabs.app_content.clear_button,
            app_tabs.app_content.status,
            app_tabs.app_content.audio_info_button_column,
            app_tabs.app_content.audio_info_button,
            app_tabs.app_content.audio_info_modal_style,
            app_tabs.app_content.results_style,
            app_tabs.app_content.cache_state,
            app_tabs.app_content.analysis_timing_button,
            app_tabs.app_content.audio_info_modal_open_state,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.app_content.audio_info_button.click(
        fn=handle_show_audio_info,
        inputs=[
            app_tabs.app_content.audio_input,
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.audio_info_modal_style,
            app_tabs.app_content.audio_info_modal_title,
            app_tabs.app_content.audio_info_modal_content,
            app_tabs.app_content.audio_info_modal_open_state,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.app_content.audio_info_modal_close_button.click(
        fn=handle_hide_audio_info,
        inputs=[],
        outputs=[
            app_tabs.app_content.audio_info_modal_style,
            app_tabs.app_content.audio_info_modal_open_state,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.app_content.task_selector.change(
        fn=handle_task_selection_change,
        inputs=[
            app_tabs.app_content.task_selector,
            app_tabs.app_content.audio_input,
            language_selector.dropdown,
            app_tabs.app_content.cache_state,
            app_tabs.app_content.analysis_running_state,
        ],
        outputs=[
            app_tabs.app_content.run_button,
            app_tabs.app_content.status,
            app_tabs.app_content.results_style,
            app_tabs.app_content.prediction_plot,
            app_tabs.app_content.heatmap_plot,
            app_tabs.app_content.transcription_output,
            app_tabs.app_content.analysis_timing_button,
        ],
        queue=False,
        show_progress="hidden",
    )

    run_analysis_event = app_tabs.app_content.run_button.click(
        fn=handle_analysis_started,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.run_button,
            app_tabs.app_content.clear_button,
            app_tabs.app_content.task_selector,
            app_tabs.app_content.audio_input,
            app_tabs.app_content.analysis_timing_button,
            app_tabs.app_content.analysis_busy_overlay,
            app_tabs.app_content.status,
            app_tabs.app_content.analysis_running_state,
        ],
        queue=False,
        show_progress="hidden",
    )

    run_analysis_result_event = run_analysis_event.then(
        fn=handle_run_analysis,
        inputs=[
            app_tabs.app_content.audio_input,
            app_tabs.app_content.task_selector,
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.status,
            app_tabs.app_content.results_style,
            app_tabs.app_content.prediction_plot,
            app_tabs.app_content.heatmap_plot,
            app_tabs.app_content.transcription_output,
            app_tabs.app_content.cache_state,
            app_tabs.app_content.analysis_timing_button,
            app_tabs.app_content.run_button,
            app_tabs.app_content.clear_button,
            app_tabs.app_content.task_selector,
            app_tabs.app_content.audio_input,
            app_tabs.app_content.analysis_busy_overlay,
            app_tabs.app_content.analysis_running_state,
        ],
        queue=True,
        show_progress="hidden",
    )

    run_analysis_result_event.then(
        fn=handle_refresh_analysis_timing_button,
        inputs=[
            app_tabs.app_content.cache_state,
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.analysis_timing_button,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.app_content.analysis_timing_button.click(
        fn=handle_show_last_analysis_timing,
        inputs=[
            app_tabs.app_content.cache_state,
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.analysis_busy_overlay,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.app_content.clear_button.click(
        fn=handle_clear_application,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.app_content.audio_input,
            app_tabs.app_content.task_selector,
            app_tabs.app_content.run_button,
            app_tabs.app_content.clear_button,
            app_tabs.app_content.status,
            app_tabs.app_content.audio_info_button_column,
            app_tabs.app_content.audio_info_button,
            app_tabs.app_content.audio_info_modal_style,
            app_tabs.app_content.results_style,
            app_tabs.app_content.prediction_plot,
            app_tabs.app_content.heatmap_plot,
            app_tabs.app_content.transcription_output,
            app_tabs.app_content.cache_state,
            app_tabs.app_content.analysis_timing_button,
            app_tabs.app_content.audio_info_modal_open_state,
        ],
        queue=False,
    )

    app_tabs.settings_content.refresh_button.click(
        fn=handle_refresh_runtime_status,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.settings_content.runtime_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.reset_whisper_button.click(
        fn=handle_reset_whisper_runtime,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.settings_content.runtime_status,
            app_tabs.settings_content.action_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.reset_wav2vec_button.click(
        fn=handle_reset_wav2vec_runtime,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.settings_content.runtime_status,
            app_tabs.settings_content.action_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.reset_pulse_button.click(
        fn=handle_reset_pulse_runtime,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.settings_content.runtime_status,
            app_tabs.settings_content.action_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.reset_all_button.click(
        fn=handle_reset_all_runtime_caches,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.settings_content.runtime_status,
            app_tabs.settings_content.action_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.attribution_controls.apply_button.click(
        fn=handle_apply_attribution_settings,
        inputs=[
            language_selector.dropdown,
            app_tabs.settings_content.attribution_controls.enable_real_attribution,
            app_tabs.settings_content.attribution_controls.method,
            app_tabs.settings_content.attribution_controls.max_points,
            app_tabs.settings_content.attribution_controls.enable_speech_mask,
            app_tabs.settings_content.attribution_controls.speech_mask_threshold_ratio,
            app_tabs.settings_content.attribution_controls.speech_mask_context_seconds,
            app_tabs.settings_content.attribution_controls.speech_mask_silence_floor,
            app_tabs.settings_content.attribution_controls.smoothgrad_samples,
            app_tabs.settings_content.attribution_controls.smoothgrad_noise_std_ratio,
        ],
        outputs=[
            app_tabs.settings_content.configuration_overview,
            app_tabs.settings_content.action_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.attribution_controls.reset_session_button.click(
        fn=handle_reset_session_settings_to_config,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            app_tabs.settings_content.attribution_controls.enable_real_attribution,
            app_tabs.settings_content.attribution_controls.method,
            app_tabs.settings_content.attribution_controls.max_points,
            app_tabs.settings_content.attribution_controls.enable_speech_mask,
            app_tabs.settings_content.attribution_controls.speech_mask_threshold_ratio,
            app_tabs.settings_content.attribution_controls.speech_mask_context_seconds,
            app_tabs.settings_content.attribution_controls.speech_mask_silence_floor,
            app_tabs.settings_content.attribution_controls.smoothgrad_samples,
            app_tabs.settings_content.attribution_controls.smoothgrad_noise_std_ratio,
            app_tabs.settings_content.configuration_overview,
            app_tabs.settings_content.action_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    attribution_dependency_inputs = [
        app_tabs.settings_content.attribution_controls.enable_real_attribution,
        app_tabs.settings_content.attribution_controls.method,
        app_tabs.settings_content.attribution_controls.enable_speech_mask,
    ]

    attribution_dependency_outputs = [
        app_tabs.settings_content.attribution_controls.method,
        app_tabs.settings_content.attribution_controls.max_points,
        app_tabs.settings_content.attribution_controls.enable_speech_mask,
        app_tabs.settings_content.attribution_controls.speech_mask_threshold_ratio,
        app_tabs.settings_content.attribution_controls.speech_mask_context_seconds,
        app_tabs.settings_content.attribution_controls.speech_mask_silence_floor,
        app_tabs.settings_content.attribution_controls.smoothgrad_samples,
        app_tabs.settings_content.attribution_controls.smoothgrad_noise_std_ratio,
    ]

    app_tabs.settings_content.attribution_controls.enable_real_attribution.change(
        fn=handle_attribution_dependency_change,
        inputs=attribution_dependency_inputs,
        outputs=attribution_dependency_outputs,
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.attribution_controls.method.change(
        fn=handle_attribution_dependency_change,
        inputs=attribution_dependency_inputs,
        outputs=attribution_dependency_outputs,
        queue=False,
        show_progress="hidden",
    )

    app_tabs.settings_content.attribution_controls.enable_speech_mask.change(
        fn=handle_attribution_dependency_change,
        inputs=attribution_dependency_inputs,
        outputs=attribution_dependency_outputs,
        queue=False,
        show_progress="hidden",
    )
