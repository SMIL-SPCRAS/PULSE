"""
File: cache.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Analysis result cache helpers for the PULSE inference pipeline.
License: MIT License
"""

from typing import Any, cast

from pulse.inference.schemas import (
    AnalysisResult,
    AudioTranscription,
    AudioWaveform,
    ClassProbability,
    SpeechActivity,
    TaskOutput,
    TemporalImportance,
    TemporalImportanceMap,
    TranscriptSegment,
)
from pulse.localization import (
    get_class_label,
    get_localized_text,
    get_task_keys_from_labels,
    get_task_labels_from_keys,
)


def serialize_task_outputs(task_outputs: list[TaskOutput]) -> list[dict[str, Any]]:
    """Serialize task outputs for Gradio state."""

    return [
        {
            "task_key": task_output.task_key,
            "task_label": task_output.task_label,
            "values": [
                {
                    "class_key": value.class_key,
                    "label": value.label,
                    "probability": value.probability,
                }
                for value in task_output.values
            ],
        }
        for task_output in task_outputs
    ]


def serialize_temporal_importance_maps(
    temporal_maps: list[TemporalImportanceMap],
) -> list[dict[str, Any]]:
    """Serialize temporal importance maps for Gradio state."""

    return [
        {
            "task_key": temporal_map.task_key,
            "class_key": temporal_map.class_key,
            "label": temporal_map.label,
            "timestamps": temporal_map.timestamps,
            "values": temporal_map.values,
        }
        for temporal_map in temporal_maps
    ]


def serialize_transcription(
    transcription: AudioTranscription | None,
) -> dict[str, Any] | None:
    """Serialize transcription for Gradio state."""

    if transcription is None:
        return None

    return {
        "text": transcription.text,
        "language": transcription.language,
        "language_probability": transcription.language_probability,
        "duration_seconds": transcription.duration_seconds,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in transcription.segments
        ],
    }


def serialize_speech_activity(
    speech_activity: SpeechActivity | None,
) -> dict[str, Any] | None:
    """Serialize speech activity diagnostics for Gradio state."""

    if speech_activity is None:
        return None

    return {
        "timestamps": speech_activity.timestamps,
        "values": speech_activity.values,
        "active_ratio": speech_activity.active_ratio,
        "min_value": speech_activity.min_value,
        "max_value": speech_activity.max_value,
        "mean_value": speech_activity.mean_value,
    }


def restore_speech_activity_from_cache(
    speech_activity_data: Any,
) -> SpeechActivity | None:
    """Restore speech activity diagnostics from Gradio state."""

    if not isinstance(speech_activity_data, dict):
        return None

    timestamps = speech_activity_data.get("timestamps")
    values = speech_activity_data.get("values")
    active_ratio = speech_activity_data.get("active_ratio")
    min_value = speech_activity_data.get("min_value")
    max_value = speech_activity_data.get("max_value")
    mean_value = speech_activity_data.get("mean_value")

    if not isinstance(timestamps, list):
        return None

    if not isinstance(values, list):
        return None

    if not isinstance(active_ratio, int | float):
        return None

    if not isinstance(min_value, int | float):
        return None

    if not isinstance(max_value, int | float):
        return None

    if not isinstance(mean_value, int | float):
        return None

    return SpeechActivity(
        timestamps=[str(timestamp) for timestamp in timestamps],
        values=[float(value) for value in values if isinstance(value, int | float)],
        active_ratio=float(active_ratio),
        min_value=float(min_value),
        max_value=float(max_value),
        mean_value=float(mean_value),
    )


def restore_transcription_from_cache(
    transcription_data: Any,
) -> AudioTranscription | None:
    """Restore transcription from Gradio state."""

    if not isinstance(transcription_data, dict):
        return None

    text = transcription_data.get("text")
    language = transcription_data.get("language")
    language_probability = transcription_data.get("language_probability")
    duration_seconds = transcription_data.get("duration_seconds")
    segments_data = transcription_data.get("segments")

    if not isinstance(text, str):
        return None

    segments: list[TranscriptSegment] = []

    if isinstance(segments_data, list):
        for segment_data in segments_data:
            if not isinstance(segment_data, dict):
                continue

            start = segment_data.get("start")
            end = segment_data.get("end")
            segment_text = segment_data.get("text")

            if not isinstance(start, int | float):
                continue

            if not isinstance(end, int | float):
                continue

            if not isinstance(segment_text, str):
                continue

            segments.append(
                TranscriptSegment(
                    start=float(start),
                    end=float(end),
                    text=segment_text,
                ),
            )

    return AudioTranscription(
        text=text,
        language=language if isinstance(language, str) else None,
        language_probability=(float(language_probability) if isinstance(language_probability, int | float) else None),
        duration_seconds=(float(duration_seconds) if isinstance(duration_seconds, int | float) else None),
        segments=segments,
    )


def create_analysis_cache(result: AnalysisResult) -> dict[str, Any]:
    """Create a serializable cache from a full analysis result."""

    return {
        "audio_path": result.audio_path,
        "selected_task_keys": [task_output.task_key for task_output in result.task_outputs],
        "task_outputs": serialize_task_outputs(result.task_outputs),
        "temporal_importance": {
            "timestamps": result.temporal_importance.timestamps,
            "values": result.temporal_importance.values,
        },
        "temporal_importance_maps": serialize_temporal_importance_maps(
            result.temporal_importance_maps,
        ),
        "audio_waveform": (
            {
                "timestamps": result.audio_waveform.timestamps,
                "values": result.audio_waveform.values,
            }
            if result.audio_waveform is not None
            else None
        ),
        "transcription": serialize_transcription(result.transcription),
        "speech_activity": serialize_speech_activity(result.speech_activity),
        "status": result.status,
    }


def restore_task_outputs_from_cache(
    task_outputs_data: list[Any],
    selected_task_keys: list[str],
    language_index: int,
) -> list[TaskOutput]:
    """Restore localized task outputs from cache."""

    localized_task_labels = get_task_labels_from_keys(
        selected_task_keys,
        language_index,
    )
    task_outputs: list[TaskOutput] = []

    for index, task_key in enumerate(selected_task_keys):
        if index >= len(task_outputs_data):
            continue

        task_output_data = task_outputs_data[index]

        if not isinstance(task_output_data, dict):
            continue

        values_data = task_output_data.get("values")

        if not isinstance(values_data, list):
            continue

        values: list[ClassProbability] = []

        for value_data in values_data:
            if not isinstance(value_data, dict):
                continue

            class_key = value_data.get("class_key")
            label = value_data.get("label")
            probability = value_data.get("probability")

            if not isinstance(class_key, str):
                if isinstance(label, str):
                    class_key = label.lower()
                else:
                    continue

            if not isinstance(probability, int | float):
                continue

            values.append(
                ClassProbability(
                    class_key=class_key,
                    label=get_class_label(
                        task_key=task_key,
                        class_key=class_key,
                        language_index=language_index,
                    ),
                    probability=float(probability),
                ),
            )

        task_label = localized_task_labels[index] if index < len(localized_task_labels) else task_key

        task_outputs.append(
            TaskOutput(
                task_key=task_key,
                task_label=task_label,
                values=values,
            ),
        )

    return task_outputs


def restore_audio_waveform_from_cache(
    audio_waveform_data: Any,
) -> AudioWaveform | None:
    """Restore audio waveform from cache."""

    if not isinstance(audio_waveform_data, dict):
        return None

    timestamps = audio_waveform_data.get("timestamps")
    values = audio_waveform_data.get("values")

    if not isinstance(timestamps, list):
        return None

    if not isinstance(values, list):
        return None

    return AudioWaveform(
        timestamps=[float(timestamp) for timestamp in timestamps if isinstance(timestamp, int | float)],
        values=[float(value) for value in values if isinstance(value, int | float)],
    )


def restore_temporal_importance_from_cache(
    temporal_importance_data: dict[str, Any],
) -> TemporalImportance | None:
    """Restore aggregate temporal importance from cache."""

    timestamps = temporal_importance_data.get("timestamps")
    importance_values = temporal_importance_data.get("values")

    if not isinstance(timestamps, list):
        return None

    if not isinstance(importance_values, list):
        return None

    return TemporalImportance(
        timestamps=[str(timestamp) for timestamp in timestamps],
        values=[float(value) for value in importance_values if isinstance(value, int | float)],
    )


def restore_temporal_importance_maps_from_cache(
    temporal_importance_maps_data: Any,
    fallback_temporal_importance: TemporalImportance,
    language_index: int,
) -> list[TemporalImportanceMap]:
    """Restore localized temporal importance maps from cache."""

    temporal_maps: list[TemporalImportanceMap] = []

    if isinstance(temporal_importance_maps_data, list):
        for map_data in temporal_importance_maps_data:
            if not isinstance(map_data, dict):
                continue

            map_task_key = map_data.get("task_key")
            map_class_key = map_data.get("class_key")
            timestamps_data = map_data.get("timestamps")
            values_data = map_data.get("values")

            if not isinstance(map_task_key, str):
                continue

            if not isinstance(map_class_key, str):
                continue

            if not isinstance(timestamps_data, list):
                continue

            if not isinstance(values_data, list):
                continue

            label = (
                get_localized_text(
                    "Texts_PLOT_TEMPORAL_IMPORTANCE_TITLE",
                    language_index,
                )
                if map_task_key == "aggregate"
                else get_class_label(
                    task_key=map_task_key,
                    class_key=map_class_key,
                    language_index=language_index,
                )
            )

            temporal_maps.append(
                TemporalImportanceMap(
                    task_key=map_task_key,
                    class_key=map_class_key,
                    label=label,
                    timestamps=[str(timestamp) for timestamp in timestamps_data],
                    values=[float(value) for value in values_data if isinstance(value, int | float)],
                ),
            )

    if temporal_maps:
        return temporal_maps

    return [
        TemporalImportanceMap(
            task_key="aggregate",
            class_key="temporal_importance",
            label=get_localized_text(
                "Texts_PLOT_TEMPORAL_IMPORTANCE_TITLE",
                language_index,
            ),
            timestamps=fallback_temporal_importance.timestamps,
            values=fallback_temporal_importance.values,
        ),
    ]


def restore_analysis_result_from_cache(
    cache: dict[str, Any] | None,
    language_index: int,
) -> AnalysisResult | None:
    """Restore full cached analysis result with localized labels."""

    if not cache:
        return None

    audio_path = cache.get("audio_path")
    selected_task_keys = cache.get("selected_task_keys")
    task_outputs_data = cache.get("task_outputs")
    temporal_importance_data = cache.get("temporal_importance")
    temporal_importance_maps_data = cache.get("temporal_importance_maps")
    audio_waveform_data = cache.get("audio_waveform")
    transcription_data = cache.get("transcription")
    speech_activity_data = cache.get("speech_activity")
    status = cache.get("status", "cached")

    if not isinstance(audio_path, str):
        return None

    if not isinstance(selected_task_keys, list):
        return None

    if not isinstance(task_outputs_data, list):
        return None

    if not isinstance(temporal_importance_data, dict):
        return None

    task_keys = cast(list[str], selected_task_keys)

    temporal_importance = restore_temporal_importance_from_cache(
        temporal_importance_data,
    )

    if temporal_importance is None:
        return None

    task_outputs = restore_task_outputs_from_cache(
        task_outputs_data=task_outputs_data,
        selected_task_keys=task_keys,
        language_index=language_index,
    )
    temporal_maps = restore_temporal_importance_maps_from_cache(
        temporal_importance_maps_data=temporal_importance_maps_data,
        fallback_temporal_importance=temporal_importance,
        language_index=language_index,
    )

    audio_waveform = restore_audio_waveform_from_cache(audio_waveform_data)
    transcription = restore_transcription_from_cache(transcription_data)

    speech_activity = restore_speech_activity_from_cache(speech_activity_data)

    return AnalysisResult(
        audio_path=audio_path,
        task_outputs=task_outputs,
        temporal_importance=temporal_importance,
        temporal_importance_maps=temporal_maps,
        audio_waveform=audio_waveform,
        transcription=transcription,
        speech_activity=speech_activity,
        status=str(status),
    )


def create_aggregate_temporal_importance(
    temporal_maps: list[TemporalImportanceMap],
    fallback_temporal_importance: TemporalImportance,
) -> TemporalImportance:
    """Create aggregate temporal importance from selected maps."""

    if not temporal_maps:
        return fallback_temporal_importance

    first_map = temporal_maps[0]
    point_count = len(first_map.values)

    if point_count == 0:
        return fallback_temporal_importance

    compatible_maps = [temporal_map for temporal_map in temporal_maps if len(temporal_map.values) == point_count]

    if not compatible_maps:
        return fallback_temporal_importance

    values = [
        sum(temporal_map.values[index] for temporal_map in compatible_maps) / len(compatible_maps)
        for index in range(point_count)
    ]

    return TemporalImportance(
        timestamps=first_map.timestamps,
        values=values,
    )


def filter_analysis_result_by_task_keys(
    result: AnalysisResult,
    selected_task_keys: list[str],
) -> AnalysisResult:
    """Filter cached analysis result by selected task keys."""

    if not selected_task_keys:
        return AnalysisResult(
            audio_path=result.audio_path,
            task_outputs=[],
            temporal_importance=TemporalImportance(
                timestamps=[],
                values=[],
            ),
            temporal_importance_maps=[],
            audio_waveform=result.audio_waveform,
            transcription=result.transcription,
            speech_activity=result.speech_activity,
            status=result.status,
        )

    task_output_by_key = {task_output.task_key: task_output for task_output in result.task_outputs}

    filtered_task_outputs = [
        task_output_by_key[task_key] for task_key in selected_task_keys if task_key in task_output_by_key
    ]

    has_task_specific_maps = any(
        temporal_map.task_key != "aggregate" for temporal_map in result.temporal_importance_maps
    )

    if has_task_specific_maps:
        filtered_temporal_maps = [
            temporal_map
            for task_key in selected_task_keys
            for temporal_map in result.temporal_importance_maps
            if temporal_map.task_key == task_key
        ]
    else:
        filtered_temporal_maps = result.temporal_importance_maps

    filtered_temporal_importance = create_aggregate_temporal_importance(
        temporal_maps=filtered_temporal_maps,
        fallback_temporal_importance=result.temporal_importance,
    )

    return AnalysisResult(
        audio_path=result.audio_path,
        task_outputs=filtered_task_outputs,
        temporal_importance=filtered_temporal_importance,
        temporal_importance_maps=filtered_temporal_maps,
        audio_waveform=result.audio_waveform,
        transcription=result.transcription,
        speech_activity=result.speech_activity,
        status=result.status,
    )


def restore_filtered_analysis_result_from_cache(
    cache: dict[str, Any] | None,
    selected_task_labels: list[str] | None,
    language_index: int,
) -> AnalysisResult | None:
    """Restore and filter cached result for selected localized task labels."""

    if not selected_task_labels:
        return None

    result = restore_analysis_result_from_cache(
        cache=cache,
        language_index=language_index,
    )

    if result is None:
        return None

    selected_task_keys = get_task_keys_from_labels(selected_task_labels)

    return filter_analysis_result_by_task_keys(
        result=result,
        selected_task_keys=selected_task_keys,
    )
