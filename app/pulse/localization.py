"""
File: localization.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Localization helpers for the PULSE Gradio application.
License: MIT License
"""

from typing import Final, cast

from pulse.config import PROJECT_ROOT, config_data

DEFAULT_LANGUAGE_INDEX: Final = 0
TASK_KEYS: Final = ("resd", "mosei", "bah", "fiv2")

TASK_CONFIG_FIELDS: Final = {
    "resd": "Tasks_RESD",
    "mosei": "Tasks_MOSEI",
    "bah": "Tasks_BAH",
    "fiv2": "Tasks_FIV2",
}

CLASS_CONFIG_FIELDS: Final = {
    "resd": {
        "anger": "Classes_RESD_ANGER",
        "disgust": "Classes_RESD_DISGUST",
        "fear": "Classes_RESD_FEAR",
        "happiness": "Classes_RESD_HAPPINESS",
        "neutral": "Classes_RESD_NEUTRAL",
        "sadness": "Classes_RESD_SADNESS",
        "enthusiasm": "Classes_RESD_ENTHUSIASM",
    },
    "mosei": {
        "neutral": "Classes_MOSEI_NEUTRAL",
        "happiness": "Classes_MOSEI_HAPPINESS",
        "sadness": "Classes_MOSEI_SADNESS",
        "anger": "Classes_MOSEI_ANGER",
        "fear": "Classes_MOSEI_FEAR",
        "disgust": "Classes_MOSEI_DISGUST",
        "surprise": "Classes_MOSEI_SURPRISE",
    },
    "bah": {
        "absence": "Classes_BAH_ABSENCE",
        "presence": "Classes_BAH_PRESENCE",
    },
    "fiv2": {
        "extraversion": "Classes_FIV2_EXTRAVERSION",
        "agreeableness": "Classes_FIV2_AGREEABLENESS",
        "conscientiousness": "Classes_FIV2_CONSCIENTIOUSNESS",
        "neuroticism": "Classes_FIV2_NEUROTICISM",
        "openness": "Classes_FIV2_OPENNESS",
    },
}


def get_class_label(
    task_key: str,
    class_key: str,
    language_index: int,
) -> str:
    """Return a localized class label."""

    task_class_fields = CLASS_CONFIG_FIELDS.get(task_key)

    if task_class_fields is None:
        return class_key

    config_field = task_class_fields.get(class_key)

    if config_field is None:
        return class_key

    return get_localized_text(config_field, language_index)


def get_localized_values(config_field: str) -> list[str]:
    """Return localized values from the flattened configuration."""

    return cast(list[str], getattr(config_data, config_field))


def get_localized_text(config_field: str, language_index: int) -> str:
    """Return a localized string by configuration field and language index."""

    values = get_localized_values(config_field)

    if language_index < 0 or language_index >= len(values):
        return values[DEFAULT_LANGUAGE_INDEX]

    return values[language_index]


def get_language_index(language: str) -> int:
    """Return the index of the selected language."""

    languages = get_localized_values("Languages_CHOICES")

    try:
        return languages.index(language)
    except ValueError:
        return DEFAULT_LANGUAGE_INDEX


def get_language_flag_path(language_index: int) -> str:
    """Return the flag image path for the selected language."""

    images_dir = cast(str, config_data.StaticPaths_IMAGES)
    language_images = get_localized_values("Images_LANGUAGES")

    if language_index < 0 or language_index >= len(language_images):
        language_index = DEFAULT_LANGUAGE_INDEX

    return str(PROJECT_ROOT / images_dir / language_images[language_index])


def get_task_label(task_key: str, language_index: int) -> str:
    """Return a localized task label by task key."""

    config_field = TASK_CONFIG_FIELDS[task_key]
    return get_localized_text(config_field, language_index)


def get_task_labels(language_index: int) -> list[str]:
    """Return localized task labels in the canonical task order."""

    return [get_task_label(task_key, language_index) for task_key in TASK_KEYS]


def get_task_keys_from_labels(task_labels: list[str] | None) -> list[str]:
    """Resolve task keys from labels in any supported language."""

    if not task_labels:
        return list(TASK_KEYS)

    resolved_keys: list[str] = []

    for task_key in TASK_KEYS:
        all_localized_labels = get_localized_values(TASK_CONFIG_FIELDS[task_key])

        if any(label in task_labels for label in all_localized_labels):
            resolved_keys.append(task_key)

    return resolved_keys or list(TASK_KEYS)


def get_task_labels_from_keys(task_keys: list[str], language_index: int) -> list[str]:
    """Return localized task labels for task keys."""

    return [get_task_label(task_key, language_index) for task_key in task_keys]


def get_plot_text(config_key: str, language_index: int) -> str:
    """Return localized plot text by short plot key."""

    return get_localized_text(f"Texts_PLOT_{config_key}", language_index)
