"""
File: app.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: PULSE Gradio application entry point.
License: MIT License
"""

import os
from typing import cast

import gradio as gr

from pulse.config import config_data, is_hugging_face_space
from pulse.events.event_handlers import setup_app_event_handlers
from pulse.logger import setup_logging
from pulse.port import ensure_port_available
from pulse.ui.language_selector import create_language_selector
from pulse.ui.tabs import create_app_tabs


def create_gradio_app() -> gr.Blocks:
    """Create the PULSE Gradio application."""

    static_images_path = cast(str, config_data.StaticPaths_IMAGES)
    app_title = cast(str, config_data.App_TITLE)

    gr.set_static_paths(paths=[static_images_path])

    with (
        gr.Blocks(
            title=app_title,
            fill_width=True,
        ) as gradio_app,
        gr.Column(elem_classes="app-shell"),
    ):
        language_selector = create_language_selector()
        app_tabs = create_app_tabs()
        setup_app_event_handlers(
            language_selector=language_selector,
            app_tabs=app_tabs,
        )

    return cast("gr.Blocks", gradio_app)


def main() -> None:
    """Run the PULSE Gradio application."""

    configured_server_name = cast(str, config_data.Server_NAME)
    configured_server_port = cast(int, config_data.Server_PORT)

    server_name = "0.0.0.0" if is_hugging_face_space() else configured_server_name
    server_port = int(os.getenv("PORT", str(configured_server_port)))
    app_css_path = cast(str, config_data.App_CSS_PATH)

    ensure_port_available(
        host=server_name,
        port=server_port,
    )

    create_gradio_app().queue(api_open=False).launch(
        theme="default",
        css_paths=app_css_path,
        share=False,
        server_name=server_name,
        server_port=server_port,
        footer_links=[],
    )


if __name__ == "__main__":
    setup_logging()
    main()
