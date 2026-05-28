"""
File: visualization.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Interactive visualization utilities for the PULSE Gradio application.
License: MIT License
"""

import html
import json
import math
from typing import Any, cast
import uuid

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from pulse.inference.schemas import (
    AnalysisResult,
    ClassProbability,
    TaskOutput,
    TemporalImportanceMap,
)
from pulse.localization import get_plot_text, get_task_labels_from_keys
from pulse.settings.state import get_runtime_setting_by_flat_name

EMOTION_CLASS_ORDER = [
    "neutral",
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "enthusiasm",
    "surprise",
]

BAH_CLASS_ORDER = [
    "absence",
    "presence",
]

FIV2_CLASS_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

PREDICTION_CLASS_ORDERS = {
    "resd": EMOTION_CLASS_ORDER,
    "mosei": EMOTION_CLASS_ORDER,
    "bah": BAH_CLASS_ORDER,
    "fiv2": FIV2_CLASS_ORDER,
}

EMOTION_CLASS_COLORS = {
    "neutral": "#64748B",
    "anger": "#EF4444",
    "disgust": "#84CC16",
    "fear": "#8B5CF6",
    "happiness": "#F59E0B",
    "sadness": "#3B82F6",
    "enthusiasm": "#EC4899",
    "surprise": "#EC4899",
}

BAH_CLASS_COLORS = {
    "absence": "#94A3B8",
    "presence": "#10B981",
}

FIV2_CLASS_COLORS = {
    "openness": "#A855F7",
    "conscientiousness": "#6366F1",
    "extraversion": "#F97316",
    "agreeableness": "#14B8A6",
    "neuroticism": "#EF4444",
}

PREDICTION_FALLBACK_COLORS = [
    "#64748B",
    "#EF4444",
    "#84CC16",
    "#8B5CF6",
    "#F59E0B",
    "#3B82F6",
    "#EC4899",
]

PREDICTION_BAR_WIDTH = 0.56
PREDICTION_HORIZONTAL_SPACING = 0.04
PREDICTION_FIGURE_HEIGHT = 270
PREDICTION_MARGIN_LEFT = 46
PREDICTION_MARGIN_RIGHT = 10
PREDICTION_MARGIN_TOP = 34
PREDICTION_MARGIN_BOTTOM = 82
PREDICTION_Y_RANGE = [0.0, 1.06]
PREDICTION_Y_TICK_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PREDICTION_DYNAMIC_Y_RANGE = True
PREDICTION_MIN_Y_AXIS_TOP = 0.2
PREDICTION_Y_HEADROOM_ABSOLUTE = 0.045
PREDICTION_Y_HEADROOM_RATIO = 0.10
PREDICTION_FULL_SCALE_RANGE_TOP = 1.06
PREDICTION_DYNAMIC_Y_AXIS_TOPS = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]
PREDICTION_TITLE_Y = 1.018
PREDICTION_TITLE_FONT_SIZE = 13
PREDICTION_X_TICK_ANGLE = -38
PREDICTION_X_TICK_FONT_SIZE = 10
PREDICTION_VALUE_FONT_SIZE = 10
HEATMAP_FIGURE_MIN_HEIGHT = 300
HEATMAP_FIGURE_MAX_HEIGHT = 640
HEATMAP_BASE_HEIGHT = 112
HEATMAP_MARGIN_LEFT_MIN = 230
HEATMAP_MARGIN_LEFT_MAX = 430
HEATMAP_MARGIN_LEFT_PADDING = 30
HEATMAP_MARGIN_RIGHT = 54
HEATMAP_MARGIN_TOP = 4
HEATMAP_MARGIN_BOTTOM = 38
HEATMAP_WAVEFORM_ROW_HEIGHT = 0.16
HEATMAP_HEATMAP_ROW_HEIGHT = 0.84
HEATMAP_VERTICAL_SPACING = 0.006
HEATMAP_COLORBAR_THICKNESS = 14
HEATMAP_LABEL_WRAP_MIN = 22
HEATMAP_LABEL_WRAP_MAX = 38
HEATMAP_LABEL_WRAP_STEP = 2
HEATMAP_LABEL_FONT_SIZE = 12
HEATMAP_LABEL_LINE_HEIGHT = 14
HEATMAP_ROW_VERTICAL_PADDING = 4
HEATMAP_MIN_ROW_HEIGHT = 28


def wrap_text(text: str, max_line_length: int = 24) -> str:
    """Wrap long text for compact Plotly labels."""

    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        candidate = " ".join([*current_line, word])

        if len(candidate) <= max_line_length:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)


def get_max_output_count(result: AnalysisResult) -> int:
    """Return maximum number of output values across selected tasks."""

    if not result.task_outputs:
        return 1

    return max(len(task_output.values) for task_output in result.task_outputs)


def get_centered_x_positions(value_count: int, max_value_count: int) -> list[float]:
    """Return centered numeric x positions for bars inside a shared subplot domain."""

    offset = (max_value_count - value_count) / 2

    return [offset + index for index in range(value_count)]


def get_prediction_axis_reference(
    axis_letter: str,
    axis_index: int,
) -> str:
    """Return Plotly trace axis reference."""

    if axis_index == 1:
        return axis_letter

    return f"{axis_letter}{axis_index}"


def get_prediction_layout_axis_name(
    axis_letter: str,
    axis_index: int,
) -> str:
    """Return Plotly layout axis name."""

    if axis_index == 1:
        return f"{axis_letter}axis"

    return f"{axis_letter}axis{axis_index}"


def get_prediction_axis_domain(
    axis_index: int,
    axis_count: int,
) -> list[float]:
    """Return horizontal domain for one prediction panel."""

    if axis_count <= 1:
        return [0.0, 1.0]

    total_spacing = PREDICTION_HORIZONTAL_SPACING * (axis_count - 1)
    domain_width = (1.0 - total_spacing) / axis_count
    domain_start = (axis_index - 1) * (domain_width + PREDICTION_HORIZONTAL_SPACING)

    return [
        domain_start,
        domain_start + domain_width,
    ]


def get_ordered_prediction_values(task_output: TaskOutput) -> list[ClassProbability]:
    """Return prediction values in a stable task-specific order."""

    class_order = PREDICTION_CLASS_ORDERS.get(task_output.task_key)

    if class_order is None:
        return list(task_output.values)

    order_by_class_key = {class_key: index for index, class_key in enumerate(class_order)}

    return sorted(
        task_output.values,
        key=lambda value: (
            order_by_class_key.get(value.class_key.lower(), len(order_by_class_key)),
            value.class_key,
        ),
    )


def get_prediction_bar_colors(task_output: TaskOutput) -> list[str]:
    """Return per-class colors for one task output."""

    if task_output.task_key in {"resd", "mosei"}:
        color_map = EMOTION_CLASS_COLORS
    elif task_output.task_key == "bah":
        color_map = BAH_CLASS_COLORS
    elif task_output.task_key == "fiv2":
        color_map = FIV2_CLASS_COLORS
    else:
        color_map = {}

    ordered_values = get_ordered_prediction_values(task_output)
    colors: list[str] = []

    for index, value in enumerate(ordered_values):
        class_key = value.class_key.lower()
        colors.append(
            color_map.get(
                class_key,
                PREDICTION_FALLBACK_COLORS[index % len(PREDICTION_FALLBACK_COLORS)],
            ),
        )

    return colors


def get_prediction_axis_top(task_output: TaskOutput) -> float:
    """Return a compact nice y-axis top for one prediction panel."""

    if not PREDICTION_DYNAMIC_Y_RANGE:
        return 1.0

    probabilities = [value.probability for value in task_output.values]

    if not probabilities:
        return 1.0

    max_probability = max(probabilities)
    headroom = max(
        PREDICTION_Y_HEADROOM_ABSOLUTE,
        max_probability * PREDICTION_Y_HEADROOM_RATIO,
    )
    required_top = min(
        1.0,
        max_probability + headroom,
    )
    required_top = max(
        PREDICTION_MIN_Y_AXIS_TOP,
        required_top,
    )

    for candidate_top in PREDICTION_DYNAMIC_Y_AXIS_TOPS:
        if required_top <= candidate_top:
            return candidate_top

    return 1.0


def get_prediction_y_range(task_output: TaskOutput) -> list[float]:
    """Return y-axis range for one prediction panel."""

    axis_top = get_prediction_axis_top(task_output)

    if axis_top >= 1.0:
        return [
            0.0,
            PREDICTION_FULL_SCALE_RANGE_TOP,
        ]

    return [
        0.0,
        axis_top,
    ]


def get_prediction_y_tick_values(task_output: TaskOutput) -> list[float]:
    """Return compact y-axis tick values for one prediction panel."""

    axis_top = get_prediction_axis_top(task_output)
    tick_step = 0.1 if axis_top <= 0.5 else 0.2
    tick_values: list[float] = []
    current_value = 0.0

    while current_value < axis_top:
        tick_values.append(round(current_value, 2))
        current_value += tick_step

    if not tick_values or tick_values[-1] != axis_top:
        tick_values.append(round(axis_top, 2))

    return tick_values


def create_task_bar_trace(
    task_output: TaskOutput,
    max_value_count: int,
    language_index: int,
) -> go.Bar:
    """Create a compact bar trace for one task output."""

    ordered_values = get_ordered_prediction_values(task_output)
    labels = [value.label for value in ordered_values]
    probabilities = [value.probability for value in ordered_values]
    x_positions = get_centered_x_positions(
        value_count=len(probabilities),
        max_value_count=max_value_count,
    )

    return go.Bar(
        x=x_positions,
        y=probabilities,
        width=PREDICTION_BAR_WIDTH,
        name=task_output.task_label,
        marker={
            "color": get_prediction_bar_colors(task_output),
            "line": {
                "width": 0.0,
            },
        },
        text=[f"{probability:.2f}" for probability in probabilities],
        textposition="outside",
        textfont={
            "size": PREDICTION_VALUE_FONT_SIZE,
            "color": "#2A3F5F",
        },
        cliponaxis=False,
        customdata=labels,
        hovertemplate=(
            f"<b>{task_output.task_label}</b><br>"
            f"{get_plot_text('LABEL', language_index)}: %{{customdata}}<br>"
            f"{get_plot_text('SCORE', language_index)}: %{{y:.2f}}<extra></extra>"
        ),
    )


def create_empty_prediction_plot(language_index: int) -> go.Figure:
    """Create an empty prediction plot."""

    no_task_selected = get_plot_text("NO_TASK_SELECTED", language_index)

    figure = go.Figure(
        data=[
            go.Bar(
                x=[0],
                y=[0.0],
                width=PREDICTION_BAR_WIDTH,
                name=no_task_selected,
                text=["0.00"],
                textposition="outside",
                cliponaxis=False,
            ),
        ],
    )

    figure.update_xaxes(
        tickvals=[0],
        ticktext=[no_task_selected],
        range=[-0.5, 0.5],
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )

    figure.update_yaxes(
        title_text=get_plot_text("SCORE", language_index),
        range=PREDICTION_Y_RANGE,
        tickmode="array",
        tickvals=PREDICTION_Y_TICK_VALUES,
        ticktext=[f"{value:g}" for value in PREDICTION_Y_TICK_VALUES],
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.90)",
        zeroline=False,
        fixedrange=True,
    )

    figure.update_layout(
        title=None,
        height=PREDICTION_FIGURE_HEIGHT,
        autosize=False,
        margin={
            "l": PREDICTION_MARGIN_LEFT,
            "r": PREDICTION_MARGIN_RIGHT,
            "t": PREDICTION_MARGIN_TOP,
            "b": PREDICTION_MARGIN_BOTTOM,
            "autoexpand": False,
        },
        showlegend=False,
        bargap=0.28,
        plot_bgcolor="rgba(245, 247, 250, 1)",
        paper_bgcolor="white",
        font={
            "size": 11,
            "color": "#2A3F5F",
        },
    )

    return figure


def add_prediction_panel_title(
    figure: go.Figure,
    task_output: TaskOutput,
    axis_index: int,
    axis_count: int,
) -> None:
    """Add one-line title above a prediction panel."""

    xaxis_reference = get_prediction_axis_reference("x", axis_index)
    yaxis_reference = get_prediction_axis_reference("y", axis_index)

    figure.add_annotation(
        x=0.5,
        y=PREDICTION_TITLE_Y,
        xref=f"{xaxis_reference} domain",
        yref=f"{yaxis_reference} domain",
        text=task_output.task_label,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        align="center",
        font={
            "size": PREDICTION_TITLE_FONT_SIZE,
            "color": "#2A3F5F",
        },
    )


def create_prediction_plot(
    result: AnalysisResult,
    language_index: int,
) -> go.Figure:
    """Create a compact multi-task prediction plot with manual Plotly axes."""

    if not result.task_outputs:
        return create_empty_prediction_plot(language_index)

    task_count = len(result.task_outputs)
    max_value_count = get_max_output_count(result)
    figure = go.Figure()
    axis_layout: dict[str, Any] = {}

    for axis_index, task_output in enumerate(result.task_outputs, start=1):
        xaxis_reference = get_prediction_axis_reference("x", axis_index)
        yaxis_reference = get_prediction_axis_reference("y", axis_index)
        xaxis_name = get_prediction_layout_axis_name("x", axis_index)
        yaxis_name = get_prediction_layout_axis_name("y", axis_index)

        ordered_values = get_ordered_prediction_values(task_output)
        x_positions = get_centered_x_positions(
            value_count=len(ordered_values),
            max_value_count=max_value_count,
        )
        tick_labels = [value.label for value in ordered_values]
        y_tick_values = get_prediction_y_tick_values(task_output)

        trace = create_task_bar_trace(
            task_output=task_output,
            max_value_count=max_value_count,
            language_index=language_index,
        )
        trace.update(
            xaxis=xaxis_reference,
            yaxis=yaxis_reference,
        )
        figure.add_trace(trace)

        axis_layout[xaxis_name] = {
            "anchor": yaxis_reference,
            "domain": get_prediction_axis_domain(
                axis_index=axis_index,
                axis_count=task_count,
            ),
            "tickmode": "array",
            "tickvals": x_positions,
            "ticktext": tick_labels,
            "range": [-0.5, max_value_count - 0.5],
            "tickangle": PREDICTION_X_TICK_ANGLE,
            "tickfont": {
                "size": PREDICTION_X_TICK_FONT_SIZE,
                "color": "#2A3F5F",
            },
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": "rgba(0, 0, 0, 0.18)",
            "ticks": "",
            "automargin": False,
            "fixedrange": True,
        }

        axis_layout[yaxis_name] = {
            "anchor": xaxis_reference,
            "title": {
                "text": (get_plot_text("SCORE", language_index) if axis_index == 1 else ""),
                "standoff": 6,
            },
            "range": get_prediction_y_range(task_output),
            "tickmode": "array",
            "tickvals": y_tick_values,
            "ticktext": [f"{value:g}" for value in y_tick_values],
            "showticklabels": True,
            "ticks": "outside",
            "ticklen": 4,
            "tickfont": {
                "size": 10,
                "color": "#2A3F5F",
            },
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.90)",
            "gridwidth": 1,
            "zeroline": False,
            "showline": False,
            "automargin": False,
            "fixedrange": True,
        }

        add_prediction_panel_title(
            figure=figure,
            task_output=task_output,
            axis_index=axis_index,
            axis_count=task_count,
        )

    figure.update_layout(
        **axis_layout,
        title=None,
        height=PREDICTION_FIGURE_HEIGHT,
        autosize=False,
        margin={
            "l": PREDICTION_MARGIN_LEFT,
            "r": PREDICTION_MARGIN_RIGHT,
            "t": PREDICTION_MARGIN_TOP,
            "b": PREDICTION_MARGIN_BOTTOM,
            "autoexpand": False,
        },
        showlegend=False,
        bargap=0.28,
        uniformtext={
            "mode": "show",
            "minsize": 9,
        },
        plot_bgcolor="rgba(245, 247, 250, 1)",
        paper_bgcolor="white",
        font={
            "size": 11,
            "color": "#2A3F5F",
        },
    )

    return figure


PREDICTION_HTML_CONFIG: dict[str, object] = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def create_prediction_plot_html(
    result: AnalysisResult,
    language_index: int,
) -> str:
    """Create an interactive Plotly prediction plot inside an iframe."""

    figure = create_prediction_plot(
        result=result,
        language_index=language_index,
    )

    figure.update_layout(
        width=None,
        height=PREDICTION_FIGURE_HEIGHT,
        autosize=True,
        margin_l=PREDICTION_MARGIN_LEFT,
        margin_r=PREDICTION_MARGIN_RIGHT,
        margin_t=PREDICTION_MARGIN_TOP,
        margin_b=PREDICTION_MARGIN_BOTTOM,
        margin_pad=0,
        margin_autoexpand=False,
        title_text=None,
    )

    div_id = f"pulse-prediction-{uuid.uuid4().hex}"
    figure_json = json.dumps(figure.to_plotly_json(), ensure_ascii=False)
    config_json = json.dumps(PREDICTION_HTML_CONFIG, ensure_ascii=False)

    plotly_loader_html = cast(str, get_plotlyjs())

    iframe_document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,
body {{
    width: 100%;
    height: {PREDICTION_FIGURE_HEIGHT}px;
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: white;
}}

body {{
    opacity: 0;
    transition: opacity 0.01s linear;
}}

body.pulse-ready {{
    opacity: 1;
}}

#{div_id} {{
    width: 100%;
    height: {PREDICTION_FIGURE_HEIGHT}px;
}}
</style>
<script>
{plotly_loader_html}
</script>
</head>
<body>
<div id="{div_id}"></div>
<script>
(function() {{
    const figure = {figure_json};
    const config = {config_json};
    const plot = document.getElementById("{div_id}");

    function drawPredictionPlot() {{
        const width = Math.max(document.documentElement.clientWidth, 320);

        figure.layout = figure.layout || {{}};
        figure.layout.width = width;
        figure.layout.height = {PREDICTION_FIGURE_HEIGHT};
        figure.layout.autosize = false;

        window.Plotly.newPlot(
            plot,
            figure.data,
            figure.layout,
            config
        ).then(function() {{
            document.body.classList.add("pulse-ready");
        }});
    }}

    if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", drawPredictionPlot);
    }} else {{
        drawPredictionPlot();
    }}

    window.addEventListener("resize", function() {{
        if (!window.Plotly || !plot) {{
            return;
        }}

        const width = Math.max(document.documentElement.clientWidth, 320);

        window.Plotly.relayout(plot, {{
            width: width,
            height: {PREDICTION_FIGURE_HEIGHT}
        }});
    }});
}})();
</script>
</body>
</html>"""

    escaped_iframe_document = html.escape(
        iframe_document,
        quote=True,
    )

    shell_class = f"pulse-prediction-shell-{uuid.uuid4().hex}"
    iframe_class = f"pulse-prediction-iframe-{uuid.uuid4().hex}"
    loader_class = f"pulse-prediction-loader-{uuid.uuid4().hex}"
    fade_in_animation_name = f"pulsePredictionFadeIn{uuid.uuid4().hex}"
    fade_out_animation_name = f"pulsePredictionFadeOut{uuid.uuid4().hex}"

    return (
        f"<style>"
        f"@keyframes {fade_in_animation_name} {{"
        f"0% {{ opacity: 0; }}"
        f"100% {{ opacity: 1; }}"
        f"}}"
        f"@keyframes {fade_out_animation_name} {{"
        f"0% {{ opacity: 1; visibility: visible; }}"
        f"99% {{ opacity: 0; visibility: visible; }}"
        f"100% {{ opacity: 0; visibility: hidden; }}"
        f"}}"
        f".{shell_class} {{"
        f"position: relative;"
        f"width: 100%;"
        f"height: {PREDICTION_FIGURE_HEIGHT}px;"
        f"overflow: hidden;"
        f"background: white;"
        f"}}"
        f".{shell_class} .{loader_class} {{"
        f"position: absolute;"
        f"inset: 0;"
        f"z-index: 2;"
        f"background: white;"
        f"opacity: 1;"
        f"animation: {fade_out_animation_name} 0.16s ease-out 0.56s forwards;"
        f"}}"
        f".{shell_class} .{iframe_class} {{"
        f"position: absolute;"
        f"inset: 0;"
        f"z-index: 1;"
        f"width: 100%;"
        f"height: {PREDICTION_FIGURE_HEIGHT}px;"
        f"min-height: {PREDICTION_FIGURE_HEIGHT}px;"
        f"max-height: {PREDICTION_FIGURE_HEIGHT}px;"
        f"border: 0;"
        f"display: block;"
        f"overflow: hidden;"
        f"background: white;"
        f"opacity: 0;"
        f"animation: {fade_in_animation_name} 0.18s ease-out 0.60s forwards;"
        f"}}"
        f"</style>"
        f'<div class="pulse-prediction-shell {shell_class}">'
        f'<div class="{loader_class}" aria-hidden="true"></div>'
        f"<iframe "
        f'class="pulse-prediction-iframe {iframe_class}" '
        f'srcdoc="{escaped_iframe_document}" '
        f'scrolling="no" '
        f'loading="eager">'
        f"</iframe>"
        f"</div>"
    )


def get_score_by_task_and_class(result: AnalysisResult) -> dict[tuple[str, str], float]:
    """Return mapping from task/class keys to displayed scores."""

    score_by_target: dict[tuple[str, str], float] = {}

    for task_output in result.task_outputs:
        for value in task_output.values:
            score_by_target[(task_output.task_key, value.class_key)] = value.probability

    return score_by_target


def get_task_label_by_key(
    result: AnalysisResult,
    language_index: int,
) -> dict[str, str]:
    """Return localized task labels by task key."""

    task_keys = [task_output.task_key for task_output in result.task_outputs]
    task_labels = get_task_labels_from_keys(
        task_keys,
        language_index,
    )

    return dict(zip(task_keys, task_labels, strict=True))


def sort_temporal_maps(
    temporal_maps: list[TemporalImportanceMap],
) -> list[TemporalImportanceMap]:
    """Sort temporal maps in the same order as prediction plots."""

    task_order = {
        "resd": 0,
        "mosei": 1,
        "bah": 2,
        "fiv2": 3,
    }
    fiv2_order = {
        "openness": 0,
        "conscientiousness": 1,
        "extraversion": 2,
        "agreeableness": 3,
        "neuroticism": 4,
    }

    def get_sort_key(temporal_map: TemporalImportanceMap) -> tuple[int, int]:
        task_rank = task_order.get(temporal_map.task_key, 999)
        class_rank = fiv2_order.get(temporal_map.class_key, 999) if temporal_map.task_key == "fiv2" else 0

        return task_rank, class_rank

    return sorted(temporal_maps, key=get_sort_key)


def estimate_text_visual_width(text: str) -> float:
    """Estimate rendered text width in pixels for Plotly tick labels."""

    width = 0.0
    punctuation_characters = {
        ".",
        ",",
        ":",
        ";",
        "-",
        "/",
        chr(0x00B7),  # middle dot
        chr(0x2013),  # en dash
        chr(0x2014),  # em dash
    }

    for character in text:
        character_codepoint = ord(character)

        if character == " ":
            width += 3.8
        elif character.isdigit():
            width += 6.2
        elif character in punctuation_characters:
            width += 4.5
        elif 0x0410 <= character_codepoint <= 0x044F or character_codepoint in {
            0x0401,
            0x0451,
        }:
            width += 7.3
        elif character.isupper():
            width += 7.4
        else:
            width += 6.7

    return width


def get_wrapped_label_width(label: str) -> float:
    """Return maximum visual width across wrapped label lines."""

    lines = label.split("<br>")

    if not lines:
        return 0.0

    return max(estimate_text_visual_width(line) for line in lines)


def get_wrapped_label_line_count(label: str) -> int:
    """Return number of visual lines in a Plotly HTML label."""

    return max(
        1,
        label.count("<br>") + 1,
    )


def create_heatmap_row_label_layout(
    raw_labels: list[str],
) -> tuple[list[str], int, int]:
    """Create wrapped labels, dynamic left margin, and dynamic row height."""

    best_wrapped_labels: list[str] = []
    best_margin_left = HEATMAP_MARGIN_LEFT_MAX
    best_max_line_count = 1

    for wrap_length in range(
        HEATMAP_LABEL_WRAP_MAX,
        HEATMAP_LABEL_WRAP_MIN - 1,
        -HEATMAP_LABEL_WRAP_STEP,
    ):
        wrapped_labels = [
            wrap_text(
                label,
                max_line_length=wrap_length,
            )
            for label in raw_labels
        ]

        max_label_width = max(
            (get_wrapped_label_width(label) for label in wrapped_labels),
            default=0.0,
        )
        margin_left = math.ceil(
            max(
                HEATMAP_MARGIN_LEFT_MIN,
                min(
                    HEATMAP_MARGIN_LEFT_MAX,
                    max_label_width + HEATMAP_MARGIN_LEFT_PADDING,
                ),
            ),
        )
        max_line_count = max(
            (get_wrapped_label_line_count(label) for label in wrapped_labels),
            default=1,
        )

        best_wrapped_labels = wrapped_labels
        best_margin_left = margin_left
        best_max_line_count = max_line_count

        if max_label_width + HEATMAP_MARGIN_LEFT_PADDING <= HEATMAP_MARGIN_LEFT_MAX:
            break

    row_height = max(
        HEATMAP_MIN_ROW_HEIGHT,
        HEATMAP_LABEL_LINE_HEIGHT * best_max_line_count + HEATMAP_ROW_VERTICAL_PADDING,
    )

    return (
        best_wrapped_labels,
        best_margin_left,
        row_height,
    )


def get_temporal_importance_plot_height(
    row_count: int,
    row_height: int,
) -> int:
    """Return compact temporal-importance plot height."""

    return max(
        HEATMAP_FIGURE_MIN_HEIGHT,
        min(
            HEATMAP_FIGURE_MAX_HEIGHT,
            HEATMAP_BASE_HEIGHT + row_height * row_count,
        ),
    )


def create_heatmap_row_label(
    temporal_map: TemporalImportanceMap,
    task_label_by_key: dict[str, str],
    score_by_target: dict[tuple[str, str], float],
    language_index: int,
) -> str:
    """Create unique localized heatmap row label."""

    if temporal_map.task_key == "aggregate":
        return get_plot_text("TEMPORAL_IMPORTANCE_TITLE", language_index)

    task_label = task_label_by_key.get(
        temporal_map.task_key,
        temporal_map.task_key,
    )
    score = score_by_target.get(
        (
            temporal_map.task_key,
            temporal_map.class_key,
        ),
    )

    if score is None:
        return f"{task_label} · {temporal_map.label}"

    return f"{task_label} · {temporal_map.label}: {score:.2f}"


def parse_timestamp_seconds(
    timestamp: object,
    fallback_index: int,
) -> float:
    """Parse timestamp value to seconds."""

    if isinstance(timestamp, int | float):
        return float(timestamp)

    timestamp_text = str(timestamp).strip().replace(",", ".")

    if not timestamp_text:
        return float(fallback_index)

    first_token = timestamp_text.split()[0]

    try:
        return float(first_token)
    except ValueError:
        return float(fallback_index)


def get_second_suffix(language_index: int) -> str:
    """Return localized second suffix."""

    return get_plot_text("SECOND_SUFFIX", language_index)


def format_second_tick_label(
    seconds: float,
    language_index: int,
) -> str:
    """Format one second-based time tick."""

    second_suffix = get_second_suffix(language_index)

    if math.isclose(
        seconds,
        round(seconds),
        abs_tol=1e-2,
    ):
        value = f"{round(seconds):.0f}"
    else:
        value = f"{seconds:.2f}".rstrip("0").rstrip(".")

    if language_index == 1:
        value = value.replace(".", ",")

    return f"{value} {second_suffix}"


def create_second_time_ticks(
    duration_seconds: float,
    language_index: int,
    max_ticks: int = 12,
) -> tuple[list[float], list[str]]:
    """Create readable second-based time-axis ticks with exact final timestamp."""

    duration_seconds = max(
        0.0,
        duration_seconds,
    )

    if duration_seconds <= 0.0:
        return [
            0.0,
        ], [
            format_second_tick_label(
                seconds=0.0,
                language_index=language_index,
            ),
        ]

    duration_floor = math.floor(duration_seconds)

    if duration_seconds <= max_ticks:
        tick_values = [float(second) for second in range(duration_floor + 1)]
    else:
        tick_step = max(
            1,
            math.ceil(duration_seconds / max_ticks),
        )
        tick_values = [
            float(second)
            for second in range(
                0,
                duration_floor + 1,
                tick_step,
            )
        ]

    final_tick = duration_seconds

    if math.isclose(
        final_tick,
        round(final_tick),
        abs_tol=1e-2,
    ):
        final_tick = float(round(final_tick))

    if tick_values:
        last_tick = tick_values[-1]

        if math.isclose(
            last_tick,
            final_tick,
            abs_tol=1e-2,
        ):
            tick_values[-1] = final_tick
        else:
            if final_tick - last_tick < 0.45:
                tick_values.pop()

            tick_values.append(final_tick)
    else:
        tick_values.append(final_tick)

    unique_tick_values: list[float] = []

    for tick_value in tick_values:
        if not unique_tick_values or not math.isclose(
            unique_tick_values[-1],
            tick_value,
            abs_tol=1e-2,
        ):
            unique_tick_values.append(tick_value)

    tick_text = [
        format_second_tick_label(
            seconds=tick_value,
            language_index=language_index,
        )
        for tick_value in unique_tick_values
    ]

    return (
        unique_tick_values,
        tick_text,
    )


def get_visualization_config_bool(field_name: str, default_value: bool) -> bool:
    """Return bool visualization config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def should_show_speech_activity() -> bool:
    """Return whether speech activity row should be shown in temporal heatmap."""

    return get_visualization_config_bool(
        "Visualization_SHOW_SPEECH_ACTIVITY",
        True,
    )


def interpolate_float_list(
    values: list[float],
    target_length: int,
) -> list[float]:
    """Interpolate a float list to target length."""

    source_length = len(values)

    if target_length <= 0:
        return []

    if source_length == target_length:
        return values

    if source_length <= 0:
        return [0.0 for _ in range(target_length)]

    if source_length == 1:
        return [values[0] for _ in range(target_length)]

    interpolated_values: list[float] = []

    for target_index in range(target_length):
        source_position = target_index * (source_length - 1) / (target_length - 1)
        left_index = int(source_position)
        right_index = min(source_length - 1, left_index + 1)
        fraction = source_position - left_index
        interpolated_values.append(
            values[left_index] * (1.0 - fraction) + values[right_index] * fraction,
        )

    return interpolated_values


def create_temporal_importance_plot(
    result: AnalysisResult,
    language_index: int,
) -> go.Figure:
    """Create an interactive waveform + multi-row temporal-importance heatmap."""

    temporal_maps = list(result.temporal_importance_maps)

    if not temporal_maps:
        temporal_maps = [
            TemporalImportanceMap(
                task_key="aggregate",
                class_key="temporal_importance",
                label=get_plot_text("TEMPORAL_IMPORTANCE_TITLE", language_index),
                timestamps=result.temporal_importance.timestamps,
                values=result.temporal_importance.values,
            ),
        ]

    temporal_maps = sort_temporal_maps(temporal_maps)

    score_by_target = get_score_by_task_and_class(result)
    task_label_by_key = get_task_label_by_key(
        result=result,
        language_index=language_index,
    )

    raw_row_labels = [
        create_heatmap_row_label(
            temporal_map=temporal_map,
            task_label_by_key=task_label_by_key,
            score_by_target=score_by_target,
            language_index=language_index,
        )
        for temporal_map in temporal_maps
    ]

    z_values = [temporal_map.values for temporal_map in temporal_maps]
    raw_heatmap_timestamps = temporal_maps[0].timestamps if temporal_maps else []
    heatmap_timestamps = [
        parse_timestamp_seconds(
            timestamp=timestamp,
            fallback_index=index,
        )
        for index, timestamp in enumerate(raw_heatmap_timestamps)
    ]

    if (
        should_show_speech_activity()
        and result.speech_activity is not None
        and result.speech_activity.values
        and heatmap_timestamps
    ):
        raw_row_labels.insert(
            0,
            get_plot_text("SPEECH_ACTIVITY", language_index),
        )
        z_values.insert(
            0,
            interpolate_float_list(
                values=result.speech_activity.values,
                target_length=len(heatmap_timestamps),
            ),
        )

    row_labels, heatmap_margin_left, heatmap_row_height = create_heatmap_row_label_layout(
        raw_row_labels,
    )

    row_count = max(1, len(row_labels))
    height = get_temporal_importance_plot_height(
        row_count=row_count,
        row_height=heatmap_row_height,
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=HEATMAP_VERTICAL_SPACING,
        row_heights=[
            HEATMAP_WAVEFORM_ROW_HEIGHT,
            HEATMAP_HEATMAP_ROW_HEIGHT,
        ],
    )

    if result.audio_waveform is not None and result.audio_waveform.timestamps and result.audio_waveform.values:
        waveform_timestamps = result.audio_waveform.timestamps
        waveform_values = result.audio_waveform.values

        figure.add_trace(
            go.Scatter(
                x=waveform_timestamps,
                y=waveform_values,
                mode="lines",
                name="Waveform",
                line={
                    "width": 1.2,
                    "color": "rgba(31, 119, 180, 0.95)",
                },
                hovertemplate=(
                    f"{get_plot_text('TIME', language_index)}: %{{x:.2f}} "
                    f"{get_second_suffix(language_index)}<br>"
                    f"{get_plot_text('AMPLITUDE', language_index)}: %{{y:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        x_range = [
            min(waveform_timestamps),
            max(waveform_timestamps),
        ]
    else:
        figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Waveform",
            ),
            row=1,
            col=1,
        )
        x_range = [
            min(heatmap_timestamps) if heatmap_timestamps else 0.0,
            max(heatmap_timestamps) if heatmap_timestamps else 1.0,
        ]

    duration_seconds = max(x_range[1], 0.0)
    tick_values, tick_text = create_second_time_ticks(
        duration_seconds=duration_seconds,
        language_index=language_index,
        max_ticks=12,
    )

    heatmap_axis = cast(Any, figure.layout)["yaxis2"]
    heatmap_domain = heatmap_axis["domain"]
    heatmap_colorbar_y = (float(heatmap_domain[0]) + float(heatmap_domain[1])) / 2
    heatmap_colorbar_len = float(heatmap_domain[1]) - float(heatmap_domain[0])

    figure.add_trace(
        go.Heatmap(
            z=z_values,
            x=heatmap_timestamps,
            y=row_labels,
            zmin=0.0,
            zmax=1.0,
            colorscale="rdbu_r",
            hovertemplate=(
                f"{get_plot_text('TIME', language_index)}: %{{x:.2f}} "
                f"{get_second_suffix(language_index)}<br>"
                f"{get_plot_text('LABEL', language_index)}: %{{y}}<br>"
                f"{get_plot_text('IMPORTANCE', language_index)}: %{{z:.2f}}"
                "<extra></extra>"
            ),
            colorbar={
                "title": {
                    "text": get_plot_text("IMPORTANCE", language_index),
                    "side": "right",
                    "font": {
                        "size": 12,
                        "color": "#2A3F5F",
                    },
                },
                "len": heatmap_colorbar_len,
                "y": heatmap_colorbar_y,
                "yanchor": "middle",
                "thickness": HEATMAP_COLORBAR_THICKNESS,
                "x": 1.004,
                "xanchor": "left",
                "xpad": 0,
                "tickvals": [0.0, 1.0],
                "ticktext": ["0", "1"],
                "tickfont": {
                    "size": 12,
                    "color": "#2A3F5F",
                },
                "ticks": "outside",
                "ticklen": 3,
                "outlinewidth": 0,
            },
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title=None,
        height=height,
        autosize=False,
        margin={
            "l": heatmap_margin_left,
            "r": HEATMAP_MARGIN_RIGHT,
            "t": HEATMAP_MARGIN_TOP,
            "b": HEATMAP_MARGIN_BOTTOM,
            "autoexpand": False,
        },
        plot_bgcolor="rgba(245, 247, 250, 1)",
        paper_bgcolor="white",
        font={
            "size": 12,
            "color": "#2A3F5F",
        },
        showlegend=False,
    )

    figure.update_xaxes(
        range=x_range,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )

    figure.update_xaxes(
        range=x_range,
        title_text=get_plot_text("TIME", language_index),
        title_standoff=6,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        tickangle=0,
        automargin=True,
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )

    figure.update_yaxes(
        automargin=False,
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        tickfont={
            "size": HEATMAP_LABEL_FONT_SIZE,
            "color": "#2A3F5F",
        },
        ticklabeloverflow="allow",
        row=2,
        col=1,
    )

    return figure


HEATMAP_HTML_CONFIG: dict[str, object] = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def create_temporal_importance_plot_html(
    result: AnalysisResult,
    language_index: int,
) -> str:
    """Create temporal-importance Plotly figure inside a stable iframe."""

    figure = create_temporal_importance_plot(
        result=result,
        language_index=language_index,
    )

    layout_height = figure.layout.height
    figure_height = int(layout_height) if isinstance(layout_height, int | float) else HEATMAP_FIGURE_MIN_HEIGHT

    current_margin_left = figure.layout.margin.l

    figure.update_layout(
        width=None,
        height=figure_height,
        autosize=True,
        margin_l=(
            int(current_margin_left) if isinstance(current_margin_left, int | float) else HEATMAP_MARGIN_LEFT_MIN
        ),
        margin_r=HEATMAP_MARGIN_RIGHT,
        margin_t=HEATMAP_MARGIN_TOP,
        margin_b=HEATMAP_MARGIN_BOTTOM,
        margin_pad=0,
        margin_autoexpand=False,
        title_text=None,
    )

    div_id = f"pulse-heatmap-{uuid.uuid4().hex}"
    figure_json = json.dumps(
        figure.to_plotly_json(),
        ensure_ascii=False,
    )
    config_json = json.dumps(
        HEATMAP_HTML_CONFIG,
        ensure_ascii=False,
    )
    plotly_loader_html = cast(str, get_plotlyjs())

    iframe_document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,
body {{
    width: 100%;
    height: {figure_height}px;
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: white;
}}

body {{
    opacity: 0;
    transition: opacity 0.01s linear;
}}

body.pulse-ready {{
    opacity: 1;
}}

#{div_id} {{
    width: 100%;
    height: {figure_height}px;
}}
</style>
<script>
{plotly_loader_html}
</script>
</head>
<body>
<div id="{div_id}"></div>
<script>
(function() {{
    const figure = {figure_json};
    const config = {config_json};
    const plot = document.getElementById("{div_id}");

    function drawHeatmapPlot() {{
        const width = Math.max(document.documentElement.clientWidth, 420);

        figure.layout = figure.layout || {{}};
        figure.layout.width = width;
        figure.layout.height = {figure_height};
        figure.layout.autosize = false;

        window.Plotly.newPlot(
            plot,
            figure.data,
            figure.layout,
            config
        ).then(function() {{
            document.body.classList.add("pulse-ready");
        }});
    }}

    if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", drawHeatmapPlot);
    }} else {{
        drawHeatmapPlot();
    }}

    window.addEventListener("resize", function() {{
        if (!window.Plotly || !plot) {{
            return;
        }}

        const width = Math.max(document.documentElement.clientWidth, 420);

        window.Plotly.relayout(plot, {{
            width: width,
            height: {figure_height}
        }});
    }});
}})();
</script>
</body>
</html>"""

    escaped_iframe_document = html.escape(
        iframe_document,
        quote=True,
    )

    shell_class = f"pulse-heatmap-shell-{uuid.uuid4().hex}"
    iframe_class = f"pulse-heatmap-iframe-{uuid.uuid4().hex}"
    loader_class = f"pulse-heatmap-loader-{uuid.uuid4().hex}"
    fade_in_animation_name = f"pulseHeatmapFadeIn{uuid.uuid4().hex}"
    fade_out_animation_name = f"pulseHeatmapFadeOut{uuid.uuid4().hex}"

    return (
        f"<style>"
        f"@keyframes {fade_in_animation_name} {{"
        f"0% {{ opacity: 0; }}"
        f"100% {{ opacity: 1; }}"
        f"}}"
        f"@keyframes {fade_out_animation_name} {{"
        f"0% {{ opacity: 1; visibility: visible; }}"
        f"99% {{ opacity: 0; visibility: visible; }}"
        f"100% {{ opacity: 0; visibility: hidden; }}"
        f"}}"
        f".{shell_class} {{"
        f"position: relative;"
        f"width: 100%;"
        f"height: {figure_height}px;"
        f"overflow: hidden;"
        f"background: white;"
        f"}}"
        f".{shell_class} .{loader_class} {{"
        f"position: absolute;"
        f"inset: 0;"
        f"z-index: 2;"
        f"background: white;"
        f"opacity: 1;"
        f"animation: {fade_out_animation_name} 0.16s ease-out 0.56s forwards;"
        f"}}"
        f".{shell_class} .{iframe_class} {{"
        f"position: absolute;"
        f"inset: 0;"
        f"z-index: 1;"
        f"width: 100%;"
        f"height: {figure_height}px;"
        f"min-height: {figure_height}px;"
        f"max-height: {figure_height}px;"
        f"border: 0;"
        f"display: block;"
        f"overflow: hidden;"
        f"background: white;"
        f"opacity: 0;"
        f"animation: {fade_in_animation_name} 0.18s ease-out 0.60s forwards;"
        f"}}"
        f"</style>"
        f'<div class="pulse-heatmap-shell {shell_class}">'
        f'<div class="{loader_class}" aria-hidden="true"></div>'
        f"<iframe "
        f'class="pulse-heatmap-iframe {iframe_class}" '
        f'srcdoc="{escaped_iframe_document}" '
        f'scrolling="no" '
        f'loading="eager">'
        f"</iframe>"
        f"</div>"
    )
