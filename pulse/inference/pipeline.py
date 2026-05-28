"""
File: pipeline.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Inference pipeline orchestration for the PULSE Gradio application.
License: MIT License
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import time
from typing import Final

from pulse.audio.decoder import DecodedAudio, decode_audio_for_analysis
from pulse.audio.formatting import format_audio_validation_error_plain
from pulse.audio.preprocessing import PreprocessedAudio, preprocess_audio_for_analysis
from pulse.audio.validation import AudioValidationResult, validate_audio_file
from pulse.config import config_data
from pulse.inference.adapters import (
    AttributionTarget,
    build_attribution_targets,
    build_selected_task_outputs,
    create_placeholder_temporal_importance,
)
from pulse.inference.attribution import (
    AttributionProgress,
    TemporalAttributionResult,
    create_attribution_method_summary,
    is_real_attribution_enabled,
    iter_temporal_attribution,
    should_fallback_to_placeholder_attribution,
    should_log_speech_mask,
)
from pulse.inference.features import (
    EmbeddingExtractionResult,
    extract_wav2vec2_embeddings,
    get_embedding_extractor,
    is_embedding_extractor_loaded,
)
from pulse.inference.multitask import (
    MultitaskRawOutputs,
    run_placeholder_multitask_model,
)
from pulse.inference.runtime import (
    get_pulse_runtime,
    is_pulse_runtime_loaded,
    is_real_model_enabled,
    run_real_multitask_model,
)
from pulse.inference.schemas import (
    AnalysisRequest,
    AnalysisResult,
    ClassProbability,
    TaskOutput,
    TemporalImportanceMap,
)
from pulse.inference.waveform import create_audio_waveform
from pulse.localization import get_class_label
from pulse.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_PIPELINE_PROGRESS_STAGES: Final[tuple[tuple[str, float], ...]] = (
    ("validation", 1.0),
    ("decoding", 1.0),
    ("preprocessing", 1.0),
    ("waveform", 1.0),
    ("whisper_loading", 10.0),
    ("transcription", 20.0),
    ("wav2vec_loading", 8.0),
    ("feature_extraction", 8.0),
    ("pulse_model_loading", 8.0),
    ("model_inference", 2.0),
    ("output_adaptation", 1.0),
    ("attribution", 36.0),
    ("packaging", 2.0),
    ("visualization", 1.0),
)


def normalize_pipeline_progress_stages(
    stages_value: object,
) -> tuple[tuple[str, float], ...]:
    """Normalize pipeline progress stages from config."""

    if not isinstance(stages_value, list):
        return DEFAULT_PIPELINE_PROGRESS_STAGES

    normalized_stages: list[tuple[str, float]] = []
    seen_stage_keys: set[str] = set()

    for stage_value in stages_value:
        if not isinstance(stage_value, Mapping):
            continue

        key_value = stage_value.get("key")
        weight_value = stage_value.get("weight")

        if not isinstance(key_value, str):
            continue

        if not isinstance(weight_value, int | float):
            continue

        weight = float(weight_value)

        if weight <= 0.0:
            continue

        if key_value in seen_stage_keys:
            continue

        normalized_stages.append((key_value, weight))
        seen_stage_keys.add(key_value)

    if not normalized_stages:
        return DEFAULT_PIPELINE_PROGRESS_STAGES

    return tuple(normalized_stages)


def get_pipeline_progress_stages() -> tuple[tuple[str, float], ...]:
    """Return ordered pipeline progress stages with weights."""

    return normalize_pipeline_progress_stages(
        getattr(
            config_data,
            "Progress_STAGES",
            None,
        ),
    )


def get_pipeline_progress_stage_keys() -> tuple[str, ...]:
    """Return ordered pipeline progress stage keys."""

    return tuple(stage_key for stage_key, _stage_weight in get_pipeline_progress_stages())


def get_pipeline_progress_stage_weight(
    stage_key: str,
) -> float:
    """Return configured progress weight for one stage."""

    for current_stage_key, stage_weight in get_pipeline_progress_stages():
        if current_stage_key == stage_key:
            return stage_weight

    return 0.0


def clamp_progress_fraction(value: float) -> float:
    """Clamp progress fraction to [0, 1]."""

    return min(
        1.0,
        max(
            0.0,
            value,
        ),
    )


def calculate_weighted_progress_percentage(
    stage_key: str,
    stage_progress: float,
) -> float:
    """Calculate weighted overall progress percentage."""

    progress_stages = get_pipeline_progress_stages()
    total_weight = sum(stage_weight for _stage_key, stage_weight in progress_stages)

    if total_weight <= 0.0:
        return 0.0

    completed_weight = 0.0

    for current_stage_key, current_stage_weight in progress_stages:
        if current_stage_key == stage_key:
            weighted_percentage = (
                100.0
                * (completed_weight + current_stage_weight * clamp_progress_fraction(stage_progress))
                / total_weight
            )

            return min(
                99.0,
                max(
                    0.0,
                    weighted_percentage,
                ),
            )

        completed_weight += current_stage_weight

    return 0.0


class AnalysisPipelineError(RuntimeError):
    """Error raised when the PULSE analysis pipeline cannot be completed."""


@dataclass(frozen=True, slots=True)
class PipelineStageTiming:
    """Timing information for one analysis pipeline stage."""

    stage_key: str
    elapsed_seconds: float
    percentage: float


@dataclass(frozen=True, slots=True)
class PipelineProgressEvent:
    """Live progress event emitted by the analysis pipeline."""

    stage_key: str
    stage_index: int
    stage_count: int
    elapsed_seconds: float
    detail: str = ""
    stage_progress: float = 0.0

    @property
    def percentage(self) -> float:
        """Return weighted overall progress percentage."""

        return calculate_weighted_progress_percentage(
            stage_key=self.stage_key,
            stage_progress=self.stage_progress,
        )


@dataclass(frozen=True, slots=True)
class AnalysisPipelineResult:
    """Complete output of the PULSE analysis pipeline."""

    result: AnalysisResult
    validation_result: AudioValidationResult
    decoded_audio: DecodedAudio
    preprocessed_audio: PreprocessedAudio
    embeddings: EmbeddingExtractionResult
    raw_outputs: MultitaskRawOutputs
    selected_task_keys: tuple[str, ...]
    selected_task_labels: tuple[str, ...]
    attribution_targets: tuple[AttributionTarget, ...]
    attribution_result: TemporalAttributionResult | None
    elapsed_seconds: float
    stage_timings: tuple[PipelineStageTiming, ...]


def record_stage_timing(
    raw_stage_timings: list[tuple[str, float]],
    stage_key: str,
    stage_started_at: float,
) -> None:
    """Record elapsed time for one pipeline stage."""

    raw_stage_timings.append(
        (
            stage_key,
            time.perf_counter() - stage_started_at,
        ),
    )


def create_pipeline_stage_timings(
    raw_stage_timings: list[tuple[str, float]],
    total_elapsed_seconds: float,
) -> tuple[PipelineStageTiming, ...]:
    """Create normalized pipeline stage timing records."""

    if total_elapsed_seconds <= 0.0:
        return tuple(
            PipelineStageTiming(
                stage_key=stage_key,
                elapsed_seconds=elapsed_seconds,
                percentage=0.0,
            )
            for stage_key, elapsed_seconds in raw_stage_timings
        )

    return tuple(
        PipelineStageTiming(
            stage_key=stage_key,
            elapsed_seconds=elapsed_seconds,
            percentage=100.0 * elapsed_seconds / total_elapsed_seconds,
        )
        for stage_key, elapsed_seconds in raw_stage_timings
    )


def log_pipeline_stage_timings(
    stage_timings: tuple[PipelineStageTiming, ...],
    total_elapsed_seconds: float,
) -> None:
    """Log compact pipeline timing summary."""

    timing_summary = " · ".join(
        f"{stage.stage_key}={stage.elapsed_seconds:.3f}s/{stage.percentage:.1f}%" for stage in stage_timings
    )

    LOGGER.info(
        "Pipeline timings: total=%.3fs · %s",
        total_elapsed_seconds,
        timing_summary,
    )


def create_pipeline_progress_event(
    stage_key: str,
    stage_index: int,
    stage_count: int,
    started_at: float,
    detail: str = "",
    stage_progress: float = 0.0,
) -> PipelineProgressEvent:
    """Create one live pipeline progress event."""

    return PipelineProgressEvent(
        stage_key=stage_key,
        stage_index=stage_index,
        stage_count=stage_count,
        elapsed_seconds=time.perf_counter() - started_at,
        detail=detail,
        stage_progress=stage_progress,
    )


def create_temporal_importance_maps(
    attribution_result: TemporalAttributionResult | None,
    temporal_importance_values: list[float],
    temporal_importance_timestamps: list[str],
) -> list[TemporalImportanceMap]:
    """Create temporal importance maps for visualization."""

    if attribution_result is None:
        return [
            TemporalImportanceMap(
                task_key="aggregate",
                class_key="temporal_importance",
                label="Temporal importance",
                timestamps=temporal_importance_timestamps,
                values=temporal_importance_values,
            ),
        ]

    timestamps = attribution_result.temporal_importance.timestamps

    return [
        TemporalImportanceMap(
            task_key=record.task_key,
            class_key=record.class_key,
            label=record.class_key,
            timestamps=timestamps,
            values=list(record.values),
        )
        for record in attribution_result.records
    ]


def localize_analysis_result(
    result: AnalysisResult,
    language_index: int,
) -> AnalysisResult:
    """Return analysis result with localized task and class labels."""

    localized_task_outputs: list[TaskOutput] = []

    for task_output in result.task_outputs:
        localized_values = [
            ClassProbability(
                class_key=value.class_key,
                label=get_class_label(
                    task_key=task_output.task_key,
                    class_key=value.class_key,
                    language_index=language_index,
                ),
                probability=value.probability,
            )
            for value in task_output.values
        ]

        localized_task_outputs.append(
            TaskOutput(
                task_key=task_output.task_key,
                task_label=task_output.task_label,
                values=localized_values,
            ),
        )

    localized_temporal_importance_maps: list[TemporalImportanceMap] = []

    for temporal_map in result.temporal_importance_maps:
        if temporal_map.task_key == "aggregate":
            localized_label = temporal_map.label
        else:
            localized_label = get_class_label(
                task_key=temporal_map.task_key,
                class_key=temporal_map.class_key,
                language_index=language_index,
            )

        localized_temporal_importance_maps.append(
            TemporalImportanceMap(
                task_key=temporal_map.task_key,
                class_key=temporal_map.class_key,
                label=localized_label,
                timestamps=temporal_map.timestamps,
                values=temporal_map.values,
            ),
        )

    return AnalysisResult(
        audio_path=result.audio_path,
        task_outputs=localized_task_outputs,
        temporal_importance=result.temporal_importance,
        temporal_importance_maps=localized_temporal_importance_maps,
        audio_waveform=result.audio_waveform,
        transcription=result.transcription,
        speech_activity=result.speech_activity,
        status=result.status,
    )


def create_attribution_progress_detail(
    progress: AttributionProgress,
) -> str:
    """Create compact attribution progress detail."""

    return (
        f"target {progress.target_index}/{progress.target_count} · "
        f"sample {progress.sample_index}/{progress.sample_count} · "
        f"{progress.target_task_key}:{progress.target_class_key} · "
        f"{progress.percentage:.0f}%"
    )


def iter_analysis_pipeline(
    request: AnalysisRequest,
    language_index: int,
) -> Iterator[PipelineProgressEvent | AnalysisPipelineResult]:
    """Run the complete PULSE analysis pipeline and emit live progress events."""

    from pulse.transcription import (
        get_whisper_runtime,
        is_transcription_enabled,
        is_whisper_runtime_loaded,
        transcribe_audio_file_safely,
    )

    started_at = time.perf_counter()
    raw_stage_timings: list[tuple[str, float]] = []
    stage_keys = get_pipeline_progress_stage_keys()
    stage_count = len(stage_keys)

    def emit(
        stage_key: str,
        detail: str = "",
        stage_progress: float = 0.0,
    ) -> PipelineProgressEvent:
        stage_index = stage_keys.index(stage_key) + 1 if stage_key in stage_keys else stage_count

        return create_pipeline_progress_event(
            stage_key=stage_key,
            stage_index=stage_index,
            stage_count=stage_count,
            started_at=started_at,
            detail=detail,
            stage_progress=stage_progress,
        )

    yield emit("validation")
    stage_started_at = time.perf_counter()
    validation_result = validate_audio_file(request.audio_path)
    record_stage_timing(raw_stage_timings, "validation", stage_started_at)

    if not validation_result.is_valid:
        raise AnalysisPipelineError(
            format_audio_validation_error_plain(
                validation_result=validation_result,
                language_index=language_index,
            ),
        )

    yield emit("decoding")
    stage_started_at = time.perf_counter()
    decoded_audio = decode_audio_for_analysis(request.audio_path)
    record_stage_timing(raw_stage_timings, "decoding", stage_started_at)

    yield emit("preprocessing")
    stage_started_at = time.perf_counter()
    preprocessed_audio = preprocess_audio_for_analysis(decoded_audio)
    record_stage_timing(raw_stage_timings, "preprocessing", stage_started_at)

    yield emit("waveform")
    stage_started_at = time.perf_counter()
    audio_waveform = create_audio_waveform(decoded_audio)
    record_stage_timing(raw_stage_timings, "waveform", stage_started_at)

    yield emit(
        "whisper_loading",
        detail="cached" if is_whisper_runtime_loaded() else "loading",
    )
    stage_started_at = time.perf_counter()
    whisper_runtime = get_whisper_runtime() if is_transcription_enabled() else None
    record_stage_timing(raw_stage_timings, "whisper_loading", stage_started_at)

    yield emit("transcription")
    stage_started_at = time.perf_counter()
    transcription = transcribe_audio_file_safely(
        audio_path=request.audio_path,
        runtime=whisper_runtime,
    )
    record_stage_timing(raw_stage_timings, "transcription", stage_started_at)

    yield emit(
        "wav2vec_loading",
        detail="cached" if is_embedding_extractor_loaded() else "loading",
    )
    stage_started_at = time.perf_counter()
    embedding_extractor = get_embedding_extractor()
    record_stage_timing(raw_stage_timings, "wav2vec_loading", stage_started_at)

    yield emit("feature_extraction")
    stage_started_at = time.perf_counter()
    embeddings = extract_wav2vec2_embeddings(
        decoded_audio=decoded_audio,
        extractor=embedding_extractor,
    )
    record_stage_timing(raw_stage_timings, "feature_extraction", stage_started_at)

    pulse_runtime = None

    if is_real_model_enabled():
        yield emit(
            "pulse_model_loading",
            detail="cached" if is_pulse_runtime_loaded() else "loading",
        )
        stage_started_at = time.perf_counter()
        pulse_runtime = get_pulse_runtime()
        record_stage_timing(raw_stage_timings, "pulse_model_loading", stage_started_at)
    else:
        yield emit("pulse_model_loading", detail="disabled")
        raw_stage_timings.append(("pulse_model_loading", 0.0))

    yield emit("model_inference")
    stage_started_at = time.perf_counter()

    if is_real_model_enabled():
        raw_outputs = run_real_multitask_model(
            embeddings=embeddings,
            runtime=pulse_runtime,
        )
    else:
        raw_outputs = run_placeholder_multitask_model(preprocessed_audio)

    record_stage_timing(raw_stage_timings, "model_inference", stage_started_at)

    yield emit("output_adaptation")
    stage_started_at = time.perf_counter()
    task_outputs = build_selected_task_outputs(
        raw_outputs=raw_outputs,
        selected_task_keys=request.selected_task_keys,
        selected_task_labels=request.selected_task_labels,
    )
    attribution_targets = build_attribution_targets(
        raw_outputs=raw_outputs,
        selected_task_keys=request.selected_task_keys,
    )
    record_stage_timing(raw_stage_timings, "output_adaptation", stage_started_at)

    yield emit("attribution", detail=create_attribution_method_summary())
    stage_started_at = time.perf_counter()
    attribution_result: TemporalAttributionResult | None = None
    temporal_importance = create_placeholder_temporal_importance()

    if is_real_model_enabled() and is_real_attribution_enabled():
        try:
            LOGGER.info(
                "Temporal attribution method: %s",
                create_attribution_method_summary(),
            )
            for attribution_update in iter_temporal_attribution(
                embeddings=embeddings,
                targets=tuple(attribution_targets),
                duration_seconds=decoded_audio.duration_seconds,
                waveform=decoded_audio.waveform,
                sample_rate=decoded_audio.sample_rate,
            ):
                if isinstance(attribution_update, AttributionProgress):
                    yield emit(
                        "attribution",
                        detail=create_attribution_progress_detail(attribution_update),
                        stage_progress=attribution_update.percentage / 100.0,
                    )
                    continue

                attribution_result = attribution_update

            if attribution_result is None:
                msg = "Temporal attribution did not produce a final result."
                raise RuntimeError(msg)

            temporal_importance = attribution_result.temporal_importance
            speech_activity = attribution_result.speech_activity

            if speech_activity is not None and should_log_speech_mask():
                LOGGER.info(
                    "Speech mask: active_ratio=%.3f, min=%.3f, max=%.3f, mean=%.3f, points=%s",
                    speech_activity.active_ratio,
                    speech_activity.min_value,
                    speech_activity.max_value,
                    speech_activity.mean_value,
                    len(speech_activity.values),
                )
        except Exception as error:
            if should_fallback_to_placeholder_attribution():
                LOGGER.warning("Real attribution failed; using placeholder heatmap: %s", error)
            else:
                raise

    record_stage_timing(raw_stage_timings, "attribution", stage_started_at)

    yield emit("packaging")
    stage_started_at = time.perf_counter()
    temporal_importance_maps = create_temporal_importance_maps(
        attribution_result=attribution_result,
        temporal_importance_values=temporal_importance.values,
        temporal_importance_timestamps=temporal_importance.timestamps,
    )

    speech_activity = attribution_result.speech_activity if attribution_result is not None else None

    result = AnalysisResult(
        audio_path=request.audio_path,
        task_outputs=task_outputs,
        temporal_importance=temporal_importance,
        temporal_importance_maps=temporal_importance_maps,
        audio_waveform=audio_waveform,
        transcription=transcription,
        speech_activity=speech_activity,
        status="real" if is_real_model_enabled() else "placeholder",
    )

    result = localize_analysis_result(
        result=result,
        language_index=language_index,
    )
    record_stage_timing(raw_stage_timings, "packaging", stage_started_at)

    elapsed_seconds = time.perf_counter() - started_at
    stage_timings = create_pipeline_stage_timings(
        raw_stage_timings=raw_stage_timings,
        total_elapsed_seconds=elapsed_seconds,
    )
    log_pipeline_stage_timings(
        stage_timings=stage_timings,
        total_elapsed_seconds=elapsed_seconds,
    )

    yield AnalysisPipelineResult(
        result=result,
        validation_result=validation_result,
        decoded_audio=decoded_audio,
        preprocessed_audio=preprocessed_audio,
        embeddings=embeddings,
        raw_outputs=raw_outputs,
        selected_task_keys=tuple(request.selected_task_keys),
        selected_task_labels=tuple(request.selected_task_labels),
        attribution_targets=tuple(attribution_targets),
        attribution_result=attribution_result,
        elapsed_seconds=elapsed_seconds,
        stage_timings=stage_timings,
    )


def run_analysis_pipeline(
    request: AnalysisRequest,
    language_index: int,
) -> AnalysisPipelineResult:
    """Run the complete PULSE analysis pipeline."""

    final_result: AnalysisPipelineResult | None = None

    for pipeline_update in iter_analysis_pipeline(
        request=request,
        language_index=language_index,
    ):
        if isinstance(pipeline_update, AnalysisPipelineResult):
            final_result = pipeline_update

    if final_result is None:
        msg = "Analysis pipeline did not produce a final result."
        raise AnalysisPipelineError(msg)

    return final_result
