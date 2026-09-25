"""
File: tabs.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Tab layout components for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from typing import cast

import gradio as gr

from pulse.config import config_data
from pulse.ui.application import ApplicationTabComponents, create_application_tab
from pulse.ui.settings import SettingsTabComponents, create_settings_tab

AUTHORS_MARKDOWN = """
### Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
"""


@dataclass(frozen=True, slots=True)
class AboutAppTabComponents:
    """Components created inside the about application tab."""

    title: gr.Markdown
    description: gr.Markdown
    placeholder: gr.Markdown


@dataclass(frozen=True, slots=True)
class AuthorsTabComponents:
    """Components created inside the authors tab."""

    content: gr.Markdown


@dataclass(frozen=True, slots=True)
class RequirementsTabComponents:
    """Components created inside the requirements tab."""

    title: gr.Markdown
    placeholder: gr.Markdown


@dataclass(frozen=True, slots=True)
class AppTabsComponents:
    """Components created by the application tabs UI."""

    tabs: gr.Tabs
    app: gr.Tab
    settings: gr.Tab
    about_app: gr.Tab
    authors: gr.Tab
    requirements: gr.Tab
    app_content: ApplicationTabComponents
    settings_content: SettingsTabComponents
    about_app_content: AboutAppTabComponents
    authors_content: AuthorsTabComponents
    requirements_content: RequirementsTabComponents


def create_about_app_tab(language_index: int = 0) -> AboutAppTabComponents:
    """Create the about application tab."""

    about_titles = cast(list[str], config_data.Texts_ABOUT_TITLE)
    about_descriptions = cast(list[str], config_data.Texts_ABOUT_DESCRIPTION)
    about_placeholders = cast(list[str], config_data.Texts_ABOUT_PLACEHOLDER)

    title = gr.Markdown(f"# {about_titles[language_index]}")
    description = gr.Markdown(about_descriptions[language_index])
    placeholder = gr.Markdown(about_placeholders[language_index])

    return AboutAppTabComponents(
        title=title,
        description=description,
        placeholder=placeholder,
    )


def create_authors_tab() -> AuthorsTabComponents:
    """Create the authors tab."""

    content = gr.Markdown(AUTHORS_MARKDOWN)

    return AuthorsTabComponents(content=content)


def create_requirements_tab(language_index: int = 0) -> RequirementsTabComponents:
    """Create the requirements tab."""

    requirements_titles = cast(list[str], config_data.Texts_REQUIREMENTS_TITLE)
    requirements_placeholders = cast(list[str], config_data.Texts_REQUIREMENTS_PLACEHOLDER)

    title = gr.Markdown(f"### {requirements_titles[language_index]}")
    placeholder = gr.Markdown(requirements_placeholders[language_index])

    return RequirementsTabComponents(
        title=title,
        placeholder=placeholder,
    )


def create_app_tabs(language_index: int = 0) -> AppTabsComponents:
    """Create all application tabs."""

    app_tab_labels = cast(list[str], config_data.Tabs_APP)
    about_app_tab_labels = cast(list[str], config_data.Tabs_ABOUT_APP)
    authors_tab_labels = cast(list[str], config_data.Tabs_AUTHORS)
    requirements_tab_labels = cast(list[str], config_data.Tabs_REQUIREMENTS)
    settings_tab_labels = cast(list[str], config_data.Tabs_SETTINGS)

    with gr.Tabs(elem_classes="app-tabs") as tabs:
        with gr.Tab(
            label=app_tab_labels[language_index],
            id="app",
            elem_id="tab-app",
            elem_classes="app-tab",
        ) as app:
            app_content = create_application_tab(language_index)

        with gr.Tab(
            label=settings_tab_labels[language_index],
            id="settings",
            elem_id="tab-settings",
            elem_classes="settings-tab",
        ) as settings:
            settings_content = create_settings_tab(language_index)

        with gr.Tab(
            label=about_app_tab_labels[language_index],
            id="about-app",
            elem_id="tab-about-app",
            elem_classes="about-app-tab",
        ) as about_app:
            about_app_content = create_about_app_tab(language_index)

        with gr.Tab(
            label=authors_tab_labels[language_index],
            id="authors",
            elem_id="tab-authors",
            elem_classes="authors-tab",
        ) as authors:
            authors_content = create_authors_tab()

        with gr.Tab(
            label=requirements_tab_labels[language_index],
            id="requirements",
            elem_id="tab-requirements",
            elem_classes="requirements-tab",
        ) as requirements:
            requirements_content = create_requirements_tab(language_index)

    return AppTabsComponents(
        tabs=tabs,
        app=app,
        settings=settings,
        settings_content=settings_content,
        about_app=about_app,
        authors=authors,
        requirements=requirements,
        app_content=app_content,
        about_app_content=about_app_content,
        authors_content=authors_content,
        requirements_content=requirements_content,
    )
