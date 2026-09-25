"""
File: adapters.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Adapters converting shared multitask model outputs into task outputs.
License: MIT License
"""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Final

from pulse.inference.multitask import MultitaskRawOutputs
from pulse.inference.schemas import ClassProbability, TaskOutput, TemporalImportance

TASK_CLASS_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "resd": (
        "anger",
        "disgust",
        "fear",
        "happiness",
        "neutral",
        "sadness",
        "enthusiasm",
    ),
    "mosei": (
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ),
    "bah": (
        "absence",
        "presence",
    ),
    "fiv2": (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ),
}


@dataclass(frozen=True, slots=True)
class AttributionTarget:
    """Target output used for future temporal attribution."""

    task_key: str
    class_key: str
    target_index: int
    group_key: str


def clip_score(value: float) -> float:
    """Clip score to the [0, 1] range."""

    if not isfinite(value):
        return 0.0

    return min(max(value, 0.0), 1.0)


def softmax_values(values: list[float]) -> list[float]:
    """Apply numerically stable softmax to raw logits."""

    if not values:
        return []

    max_value = max(values)
    exp_values = [exp(value - max_value) for value in values]
    denominator = sum(exp_values)

    if denominator <= 0.0:
        return [0.0 for _ in values]

    return [value / denominator for value in exp_values]


def get_probability_values_for_task(
    raw_outputs: MultitaskRawOutputs,
    task_key: str,
) -> list[float]:
    """Return display-ready probabilities or scores for one task."""

    if task_key == "resd":
        return softmax_values(raw_outputs.resd_logits)

    if task_key == "mosei":
        return softmax_values(raw_outputs.mosei_logits)

    if task_key == "bah":
        return softmax_values(raw_outputs.bah_logits)

    if task_key == "fiv2":
        if len(raw_outputs.personality_scores) != 5:
            msg = f"Personality output length mismatch: expected 5 values, got {len(raw_outputs.personality_scores)}."
            raise ValueError(msg)

        openness = clip_score(raw_outputs.personality_scores[0])
        conscientiousness = clip_score(raw_outputs.personality_scores[1])
        extraversion = clip_score(raw_outputs.personality_scores[2])
        agreeableness = clip_score(raw_outputs.personality_scores[3])
        non_neuroticism = clip_score(raw_outputs.personality_scores[4])
        neuroticism = 1.0 - non_neuroticism

        return [
            openness,
            conscientiousness,
            extraversion,
            agreeableness,
            neuroticism,
        ]

    supported_tasks = ", ".join(TASK_CLASS_KEYS)
    msg = f"Unsupported task key: {task_key}. Supported tasks: {supported_tasks}."
    raise KeyError(msg)


def get_predicted_target_index(
    raw_outputs: MultitaskRawOutputs,
    task_key: str,
) -> int:
    """Return the predicted target index for a task.

    For MOSEI, neutral is skipped when selecting the attribution target,
    following the previous PULSE prototype behavior.
    """

    values = get_probability_values_for_task(
        raw_outputs=raw_outputs,
        task_key=task_key,
    )

    if not values:
        msg = f"Cannot determine predicted target for empty task output: {task_key}."
        raise ValueError(msg)

    if task_key == "mosei" and len(values) > 1:
        non_neutral_values = values[1:]
        return 1 + max(
            range(len(non_neutral_values)),
            key=lambda index: non_neutral_values[index],
        )

    return max(
        range(len(values)),
        key=lambda index: values[index],
    )


def build_task_output(
    raw_outputs: MultitaskRawOutputs,
    task_key: str,
    task_label: str,
) -> TaskOutput:
    """Build a task output from shared multitask raw outputs."""

    class_keys = TASK_CLASS_KEYS.get(task_key)

    if class_keys is None:
        supported_tasks = ", ".join(TASK_CLASS_KEYS)
        msg = f"Unsupported task key: {task_key}. Supported tasks: {supported_tasks}."
        raise KeyError(msg)

    values = get_probability_values_for_task(
        raw_outputs=raw_outputs,
        task_key=task_key,
    )

    if len(values) != len(class_keys):
        msg = f"Output length mismatch for task '{task_key}': {len(values)} values for {len(class_keys)} classes."
        raise ValueError(msg)

    return TaskOutput(
        task_key=task_key,
        task_label=task_label,
        values=[
            ClassProbability(
                class_key=class_key,
                label=class_key,
                probability=float(probability),
            )
            for class_key, probability in zip(class_keys, values, strict=True)
        ],
    )


def build_selected_task_outputs(
    raw_outputs: MultitaskRawOutputs,
    selected_task_keys: list[str],
    selected_task_labels: list[str],
) -> list[TaskOutput]:
    """Build task outputs for selected tasks."""

    if len(selected_task_keys) != len(selected_task_labels):
        msg = (
            "Selected task keys and labels must have the same length: "
            f"{len(selected_task_keys)} != {len(selected_task_labels)}."
        )
        raise ValueError(msg)

    return [
        build_task_output(
            raw_outputs=raw_outputs,
            task_key=task_key,
            task_label=task_label,
        )
        for task_key, task_label in zip(
            selected_task_keys,
            selected_task_labels,
            strict=True,
        )
    ]


def build_attribution_targets(
    raw_outputs: MultitaskRawOutputs,
    selected_task_keys: list[str],
) -> list[AttributionTarget]:
    """Build attribution targets for selected tasks.

    Classification tasks get one target: the predicted class.
    Personality gets one target per trait, matching the previous prototype.
    """

    targets: list[AttributionTarget] = []

    for task_key in selected_task_keys:
        class_keys = TASK_CLASS_KEYS.get(task_key)

        if class_keys is None:
            supported_tasks = ", ".join(TASK_CLASS_KEYS)
            msg = f"Unsupported task key: {task_key}. Supported tasks: {supported_tasks}."
            raise KeyError(msg)

        if task_key == "fiv2":
            targets.extend(
                AttributionTarget(
                    task_key=task_key,
                    class_key=class_key,
                    target_index=target_index,
                    group_key=task_key,
                )
                for target_index, class_key in enumerate(class_keys)
            )
            continue

        target_index = get_predicted_target_index(
            raw_outputs=raw_outputs,
            task_key=task_key,
        )

        targets.append(
            AttributionTarget(
                task_key=task_key,
                class_key=class_keys[target_index],
                target_index=target_index,
                group_key=task_key,
            ),
        )

    return targets


def create_placeholder_temporal_importance() -> TemporalImportance:
    """Create placeholder temporal importance values."""

    return TemporalImportance(
        timestamps=["0%", "15%", "30%", "45%", "60%", "75%", "100%"],
        values=[0.1, 0.25, 0.4, 0.8, 0.6, 0.35, 0.2],
    )
