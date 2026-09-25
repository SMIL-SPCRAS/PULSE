"""
File: config.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Configuration utilities for loading PULSE application settings.
License: MIT License
"""

from collections.abc import Mapping
import os
from pathlib import Path
import tomllib
from types import SimpleNamespace
from typing import Any, Final, Protocol

CONFIG_NAME: Final = "config.toml"
PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
CONFIG_PATH: Final = PROJECT_ROOT / CONFIG_NAME


def is_hugging_face_space() -> bool:
    """Return whether the app is running inside a Hugging Face Space."""

    return bool(
        os.getenv("SPACE_ID")
        or os.getenv("SPACE_HOST")
        or os.getenv("SPACE_REPO_NAME")
        or os.getenv("SPACE_AUTHOR_NAME"),
    )


class TabCreator(Protocol):
    """Callable protocol for Gradio tab factory functions."""

    def __call__(self) -> Any: ...


def flatten_dict(
    data: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested mapping using underscore-separated keys."""

    flattened: dict[str, Any] = {}

    for key, value in data.items():
        flattened_key = f"{prefix}{key}"

        if isinstance(value, Mapping):
            flattened.update(flatten_dict(value, prefix=f"{flattened_key}_"))
        else:
            flattened[flattened_key] = value

    return flattened


def load_toml(file_path: str | Path) -> dict[str, Any]:
    """Load a TOML file."""

    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as file:
        return tomllib.load(file)


def load_config(file_path: str | Path = CONFIG_PATH) -> SimpleNamespace:
    """Load application configuration as a flat namespace."""

    config = load_toml(file_path)
    flattened_config = flatten_dict(config)

    return SimpleNamespace(**flattened_config)


def load_tab_creators(
    file_path: str | Path,
    available_functions: Mapping[str, TabCreator],
) -> dict[str, TabCreator]:
    """Load tab creator functions from the application configuration."""

    config = load_toml(file_path)
    tab_creators_data = config.get("TabCreators", {})

    if not isinstance(tab_creators_data, Mapping):
        msg = "The 'TabCreators' section must be a TOML table."
        raise TypeError(msg)

    tab_creators: dict[str, TabCreator] = {}

    for tab_name, function_name in tab_creators_data.items():
        if not isinstance(function_name, str):
            msg = f"Tab creator for '{tab_name}' must be a string."
            raise TypeError(msg)

        tab_creator = available_functions.get(function_name)

        if tab_creator is None:
            available = ", ".join(sorted(available_functions)) or "none"
            msg = (
                f"Tab creator function '{function_name}' for tab '{tab_name}' was not found. "
                f"Available functions: {available}."
            )
            raise KeyError(msg)

        tab_creators[str(tab_name)] = tab_creator

    return tab_creators


config_data: SimpleNamespace = load_config()
