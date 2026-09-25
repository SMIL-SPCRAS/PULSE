"""
File: state.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Runtime settings state for the PULSE Gradio application.
License: MIT License
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from pulse.config import config_data

if TYPE_CHECKING:
    from pulse.settings.schema import SettingSpec


_RUNTIME_SETTINGS: dict[str, object] = {}
_SETTINGS_LOCK = RLock()


def make_config_field_name(section: str, field: str) -> str:
    """Return flattened config field name."""

    return f"{section}_{field}"


def get_runtime_setting_by_flat_name(
    field_name: str,
    default_value: object = None,
) -> object:
    """Return runtime setting by flattened config field name."""

    with _SETTINGS_LOCK:
        if field_name in _RUNTIME_SETTINGS:
            return _RUNTIME_SETTINGS[field_name]

    return getattr(config_data, field_name, default_value)


def get_runtime_setting(
    section: str,
    field: str,
    default_value: object = None,
) -> object:
    """Return runtime setting by section and field."""

    return get_runtime_setting_by_flat_name(
        field_name=make_config_field_name(
            section=section,
            field=field,
        ),
        default_value=default_value,
    )


def get_setting_spec(section: str, field: str) -> SettingSpec:
    """Return setting schema item."""

    from pulse.settings.schema import SETTINGS_SCHEMA

    for section_spec in SETTINGS_SCHEMA:
        for setting_spec in section_spec.settings:
            if setting_spec.section == section and setting_spec.field == field:
                return setting_spec

    msg = f"Unknown runtime setting: {section}.{field}"
    raise KeyError(msg)


def coerce_bool_setting(value: object) -> bool:
    """Coerce a setting value to bool."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized_value = value.strip().lower()

        if normalized_value in {"true", "1", "yes", "on"}:
            return True

        if normalized_value in {"false", "0", "no", "off"}:
            return False

    msg = f"Expected bool setting value, got {type(value).__name__}."
    raise TypeError(msg)


def coerce_int_setting(
    value: object,
    spec: SettingSpec,
) -> int:
    """Coerce and validate an integer setting value."""

    if isinstance(value, bool):
        msg = "Boolean value cannot be used as integer setting."
        raise TypeError(msg)

    if isinstance(value, int):
        coerced_value = value
    elif isinstance(value, str):
        coerced_value = int(value.strip())
    else:
        msg = f"Expected integer setting value, got {type(value).__name__}."
        raise TypeError(msg)

    validate_numeric_range(
        value=float(coerced_value),
        spec=spec,
    )

    return coerced_value


def coerce_float_setting(
    value: object,
    spec: SettingSpec,
) -> float:
    """Coerce and validate a float setting value."""

    if isinstance(value, bool):
        msg = "Boolean value cannot be used as float setting."
        raise TypeError(msg)

    if isinstance(value, int | float):
        coerced_value = float(value)
    elif isinstance(value, str):
        coerced_value = float(value.strip())
    else:
        msg = f"Expected float setting value, got {type(value).__name__}."
        raise TypeError(msg)

    validate_numeric_range(
        value=coerced_value,
        spec=spec,
    )

    return coerced_value


def coerce_choice_setting(
    value: object,
    spec: SettingSpec,
) -> str:
    """Coerce and validate a choice setting value."""

    if not isinstance(value, str):
        msg = f"Expected choice setting value as string, got {type(value).__name__}."
        raise TypeError(msg)

    if value not in spec.choices:
        choices = ", ".join(spec.choices)
        msg = f"Invalid value for {spec.section}.{spec.field}: {value}. Allowed values: {choices}."
        raise ValueError(msg)

    return value


def validate_numeric_range(
    value: float,
    spec: SettingSpec,
) -> None:
    """Validate numeric setting range."""

    if spec.minimum is not None and value < spec.minimum:
        msg = f"Value for {spec.section}.{spec.field} is below minimum {spec.minimum:g}: {value:g}."
        raise ValueError(msg)

    if spec.maximum is not None and value > spec.maximum:
        msg = f"Value for {spec.section}.{spec.field} is above maximum {spec.maximum:g}: {value:g}."
        raise ValueError(msg)


def coerce_setting_value(
    value: object,
    spec: SettingSpec,
) -> object:
    """Coerce a runtime setting value according to schema."""

    if spec.value_type == "bool":
        return coerce_bool_setting(value)

    if spec.value_type == "int":
        return coerce_int_setting(
            value=value,
            spec=spec,
        )

    if spec.value_type == "float":
        return coerce_float_setting(
            value=value,
            spec=spec,
        )

    if spec.value_type == "choice":
        return coerce_choice_setting(
            value=value,
            spec=spec,
        )

    if spec.value_type == "str":
        if isinstance(value, str):
            return value

        return str(value)

    msg = f"Unsupported setting type: {spec.value_type}"
    raise ValueError(msg)


def set_runtime_setting(
    section: str,
    field: str,
    value: object,
) -> str:
    """Set one runtime setting and return required reset scope."""

    spec = get_setting_spec(
        section=section,
        field=field,
    )
    coerced_value = coerce_setting_value(
        value=value,
        spec=spec,
    )

    with _SETTINGS_LOCK:
        _RUNTIME_SETTINGS[
            make_config_field_name(
                section=section,
                field=field,
            )
        ] = coerced_value

    return str(spec.reset_scope)


def reset_runtime_setting(
    section: str,
    field: str,
) -> None:
    """Reset one runtime setting to config value."""

    with _SETTINGS_LOCK:
        _RUNTIME_SETTINGS.pop(
            make_config_field_name(
                section=section,
                field=field,
            ),
            None,
        )


def reset_runtime_settings_to_config() -> None:
    """Reset all runtime settings to config values."""

    with _SETTINGS_LOCK:
        _RUNTIME_SETTINGS.clear()


def get_runtime_settings_snapshot() -> dict[str, object]:
    """Return current runtime setting overrides."""

    with _SETTINGS_LOCK:
        return dict(_RUNTIME_SETTINGS)
