"""
File: formatting.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Formatting utilities for audio metadata in the PULSE Gradio application.
License: MIT License
"""

from html import escape

from pulse.audio.metadata import AudioMetadata, read_audio_metadata
from pulse.audio.validation import (
    AudioValidationIssue,
    AudioValidationResult,
    validate_audio_metadata,
)
from pulse.localization import get_localized_text


def format_optional_value(value: object | None) -> str:
    """Format an optional metadata value."""

    return "—" if value is None else str(value)


def format_duration(seconds: float | None) -> str:
    """Format audio duration."""

    if seconds is None:
        return "—"

    return f"{seconds:.2f} s"


def format_sample_rate(sample_rate: int | None) -> str:
    """Format audio sample rate."""

    if sample_rate is None:
        return "—"

    return f"{sample_rate:,} Hz"


def format_bit_rate(bit_rate: float | None) -> str:
    """Format audio bit rate."""

    if bit_rate is None:
        return "—"

    return f"{bit_rate / 1000:.0f} kbps"


def format_file_size(size_bytes: int) -> str:
    """Format file size."""

    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / 1024**2:.2f} MB"


def create_audio_info_row(label: str, value: str, *, code: bool = False) -> str:
    """Create one compact HTML row for audio metadata."""

    safe_label = escape(label)
    safe_value = escape(value)
    value_html = f"<code>{safe_value}</code>" if code else safe_value

    return f"""
<div class="audio-info-row">
    <div class="audio-info-label">{safe_label}</div>
    <div class="audio-info-value">{value_html}</div>
</div>
"""


def get_audio_validation_issue_message(
    issue: AudioValidationIssue,
    language_index: int,
) -> str:
    """Return localized validation issue message."""

    template = get_localized_text(
        f"Texts_AUDIO_VALIDATION_{issue.code}",
        language_index,
    )

    actual = issue.actual if issue.actual is not None else "—"
    expected = issue.expected if issue.expected is not None else "—"

    try:
        return template.format(
            actual=actual,
            expected=expected,
        )
    except TypeError, ValueError:
        return template


def format_audio_validation_issue_html(
    issue: AudioValidationIssue,
    language_index: int,
) -> str:
    """Format one validation issue as HTML."""

    issue_class = "audio-validation-error" if issue.severity == "error" else "audio-validation-warning"
    message = escape(
        get_audio_validation_issue_message(
            issue=issue,
            language_index=language_index,
        ),
    )

    return f'<li class="{issue_class}">{message}</li>'


def format_audio_validation_html(
    validation_result: AudioValidationResult,
    language_index: int,
) -> str:
    """Format audio validation result as HTML."""

    title = escape(get_localized_text("Labels_AUDIO_VALIDATION_TITLE", language_index))

    if validation_result.is_valid and not validation_result.warnings:
        message = escape(get_localized_text("Texts_AUDIO_VALIDATION_OK", language_index))

        return f"""
<section class="audio-validation-card audio-validation-card-ok">
    <h4>{title}</h4>
    <p class="audio-validation-ok">{message}</p>
</section>
"""

    sections: list[str] = []

    if validation_result.errors:
        errors_title = escape(
            get_localized_text("Labels_AUDIO_VALIDATION_ERRORS", language_index),
        )
        error_items = "".join(
            format_audio_validation_issue_html(issue, language_index) for issue in validation_result.errors
        )
        sections.append(f"<h5>{errors_title}</h5><ul>{error_items}</ul>")

    if validation_result.warnings:
        warnings_title = escape(
            get_localized_text("Labels_AUDIO_VALIDATION_WARNINGS", language_index),
        )
        warning_items = "".join(
            format_audio_validation_issue_html(issue, language_index) for issue in validation_result.warnings
        )
        sections.append(f"<h5>{warnings_title}</h5><ul>{warning_items}</ul>")

    return f"""
<section class="audio-validation-card">
    <h4>{title}</h4>
    {"".join(sections)}
</section>
"""


def format_audio_validation_error_markdown(
    validation_result: AudioValidationResult,
    language_index: int,
) -> str:
    """Format validation errors for the status block."""

    if validation_result.is_valid:
        return get_localized_text("Texts_AUDIO_VALIDATION_OK", language_index)

    messages = [get_audio_validation_issue_message(issue, language_index) for issue in validation_result.errors]

    if not messages:
        messages = [get_audio_validation_issue_message(issue, language_index) for issue in validation_result.warnings]

    message_lines = "\n".join(f"- {message}" for message in messages)

    return f"{get_localized_text('Texts_STATUS_AUDIO_INVALID', language_index)}\n\n{message_lines}"


def format_audio_validation_error_plain(
    validation_result: AudioValidationResult,
    language_index: int,
) -> str:
    """Format validation errors as plain text."""

    return format_audio_validation_error_markdown(
        validation_result=validation_result,
        language_index=language_index,
    ).replace("\n\n", "\n")


def format_audio_metadata_html(
    metadata: AudioMetadata,
    language_index: int,
    validation_result: AudioValidationResult | None = None,
) -> str:
    """Format audio metadata as compact HTML."""

    rows = [
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_FILENAME", language_index),
            metadata.filename,
            code=True,
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_DURATION", language_index),
            format_duration(metadata.duration_seconds),
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_SAMPLE_RATE", language_index),
            format_sample_rate(metadata.sample_rate),
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_CHANNELS", language_index),
            format_optional_value(metadata.num_channels),
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_CODEC", language_index),
            format_optional_value(metadata.codec),
            code=True,
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_BIT_RATE", language_index),
            format_bit_rate(metadata.bit_rate),
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_FORMAT", language_index),
            format_optional_value(metadata.sample_format),
            code=True,
        ),
        create_audio_info_row(
            get_localized_text("Labels_AUDIO_INFO_SIZE", language_index),
            format_file_size(metadata.size_bytes),
        ),
    ]

    validation_html = (
        format_audio_validation_html(
            validation_result=validation_result,
            language_index=language_index,
        )
        if validation_result is not None
        else ""
    )

    return f"""
<section class="audio-info-card">
    {"".join(rows)}
</section>
{validation_html}
"""


def create_audio_metadata_html(
    audio_path: str,
    language_index: int,
) -> str:
    """Read and format audio metadata as HTML."""

    metadata = read_audio_metadata(audio_path)
    validation_result = validate_audio_metadata(metadata)

    return format_audio_metadata_html(
        metadata=metadata,
        language_index=language_index,
        validation_result=validation_result,
    )


def create_unavailable_audio_metadata_html(language_index: int) -> str:
    """Return localized unavailable metadata message as HTML."""

    message = escape(get_localized_text("Texts_AUDIO_INFO_UNAVAILABLE", language_index))

    return f"""
<section class="audio-info-card">
    <p>{message}</p>
</section>
"""
