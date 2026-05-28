"""
File: language_selector.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Language selector UI components for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from typing import cast

from gradio.components import Dropdown, Image
from gradio.layouts import Column, Row

from pulse.config import config_data
from pulse.localization import DEFAULT_LANGUAGE_INDEX, get_language_flag_path


@dataclass(frozen=True, slots=True)
class LanguageSelectorComponents:
    """Components created by the language selector UI."""

    wrapper: Column
    row: Row
    flag: Image
    dropdown: Dropdown


def create_language_selector() -> LanguageSelectorComponents:
    """Create the language selector section."""

    language_choices = cast(list[str], config_data.Languages_CHOICES)
    default_language = cast(str, config_data.Languages_DEFAULT)

    with (
        Column(
            visible=True,
            render=True,
            variant="default",
            min_width=0,
            elem_classes="language-selector-wrapper",
        ) as wrapper,
        Row(
            visible=True,
            render=True,
            variant="default",
            elem_classes="language-selector",
        ) as row,
    ):
        flag = Image(
            value=get_language_flag_path(DEFAULT_LANGUAGE_INDEX),
            container=False,
            interactive=False,
            show_label=False,
            visible=True,
            buttons=[],
            elem_classes="language-selector-flag",
            height=32,
            width=32,
        )

        dropdown = Dropdown(
            label=None,
            info=None,
            choices=language_choices,
            value=default_language,
            visible=True,
            show_label=False,
            elem_classes="language-selector-dropdown",
            interactive=True,
            filterable=False,
            allow_custom_value=False,
            min_width=140,
        )

    return LanguageSelectorComponents(
        wrapper=wrapper,
        row=row,
        flag=flag,
        dropdown=dropdown,
    )
