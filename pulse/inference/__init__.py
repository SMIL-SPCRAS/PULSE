"""
File: __init__.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Inference package exports for the PULSE Gradio application.
License: MIT License
"""

from pulse.inference.adapters import (
    TASK_CLASS_KEYS,
    AttributionTarget,
    build_attribution_targets,
    build_selected_task_outputs,
    build_task_output,
    create_placeholder_temporal_importance,
    get_predicted_target_index,
)
from pulse.inference.attribution import (
    AttributionRecord,
    TemporalAttributionResult,
    compute_temporal_attribution,
)
from pulse.inference.cache import (
    create_analysis_cache,
    filter_analysis_result_by_task_keys,
    restore_analysis_result_from_cache,
    restore_filtered_analysis_result_from_cache,
)
from pulse.inference.features import (
    EmbeddingExtractionResult,
    EmbeddingExtractor,
    extract_wav2vec2_embeddings,
    get_embedding_extractor,
    is_embedding_extractor_loaded,
    reset_embedding_extractor,
)
from pulse.inference.multitask import (
    MultitaskRawOutputs,
    run_placeholder_multitask_model,
)
from pulse.inference.pipeline import (
    AnalysisPipelineError,
    AnalysisPipelineResult,
    PipelineProgressEvent,
    get_pipeline_progress_stage_keys,
    get_pipeline_progress_stages,
    iter_analysis_pipeline,
    run_analysis_pipeline,
)
from pulse.inference.runtime import (
    PulseRuntime,
    get_pulse_runtime,
    is_pulse_runtime_loaded,
    is_real_model_enabled,
    reset_pulse_runtime,
    run_real_multitask_model,
)
from pulse.inference.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AudioTranscription,
    AudioWaveform,
    ClassProbability,
    TaskOutput,
    TemporalImportance,
    TemporalImportanceMap,
    TranscriptSegment,
)


def reset_inference_caches() -> None:
    """Reset cached inference runtimes."""

    reset_embedding_extractor()
    reset_pulse_runtime()


__all__ = [
    "TASK_CLASS_KEYS",
    "AnalysisPipelineError",
    "AnalysisPipelineResult",
    "AnalysisRequest",
    "AnalysisResult",
    "AttributionRecord",
    "AttributionTarget",
    "AudioTranscription",
    "AudioWaveform",
    "ClassProbability",
    "EmbeddingExtractionResult",
    "EmbeddingExtractor",
    "MultitaskRawOutputs",
    "PipelineProgressEvent",
    "PulseRuntime",
    "TaskOutput",
    "TemporalAttributionResult",
    "TemporalImportance",
    "TemporalImportanceMap",
    "TranscriptSegment",
    "build_attribution_targets",
    "build_selected_task_outputs",
    "build_task_output",
    "compute_temporal_attribution",
    "create_analysis_cache",
    "create_placeholder_temporal_importance",
    "extract_wav2vec2_embeddings",
    "filter_analysis_result_by_task_keys",
    "get_embedding_extractor",
    "get_pipeline_progress_stage_keys",
    "get_pipeline_progress_stages",
    "get_predicted_target_index",
    "get_pulse_runtime",
    "is_embedding_extractor_loaded",
    "is_pulse_runtime_loaded",
    "is_real_model_enabled",
    "iter_analysis_pipeline",
    "reset_embedding_extractor",
    "reset_inference_caches",
    "reset_pulse_runtime",
    "restore_analysis_result_from_cache",
    "restore_filtered_analysis_result_from_cache",
    "run_analysis_pipeline",
    "run_placeholder_multitask_model",
    "run_real_multitask_model",
]
