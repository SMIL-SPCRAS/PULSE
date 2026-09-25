"""
File: schema.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Runtime-editable settings schema for the PULSE Gradio application.
License: MIT License
"""

from dataclasses import dataclass
from typing import Final, Literal

from pulse.settings.state import get_runtime_setting

SettingValueType = Literal["bool", "int", "float", "str", "choice"]
ResetScope = Literal["none", "whisper", "wav2vec", "pulse", "all"]


@dataclass(frozen=True, slots=True)
class SettingSpec:
    """One configurable application setting."""

    section: str
    field: str
    value_type: SettingValueType
    label: tuple[str, str]
    description: tuple[str, str]
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    reset_scope: ResetScope = "none"
    editable: bool = True


@dataclass(frozen=True, slots=True)
class SettingsSectionSpec:
    """One settings section."""

    title: tuple[str, str]
    description: tuple[str, str]
    settings: tuple[SettingSpec, ...]


SETTINGS_SCHEMA: Final[tuple[SettingsSectionSpec, ...]] = (
    SettingsSectionSpec(
        title=("Transcription", "Транскрипция"),
        description=(
            "Whisper transcription settings.",
            "Настройки Whisper-транскрипции.",
        ),
        settings=(
            SettingSpec(
                section="Transcription",
                field="ENABLE",
                value_type="bool",
                label=("Enable transcription", "Включить транскрипцию"),
                description=(
                    "Run Faster-Whisper before PULSE inference.",
                    "Запускать Faster-Whisper перед инференсом PULSE.",
                ),
                reset_scope="none",
            ),
            SettingSpec(
                section="Transcription",
                field="MODEL_NAME",
                value_type="str",
                label=("Whisper model", "Модель Whisper"),
                description=(
                    "Faster-Whisper model name.",
                    "Имя модели Faster-Whisper.",
                ),
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="MODEL_DIR",
                value_type="str",
                label=("Whisper local directory", "Локальная директория Whisper"),
                description=(
                    "Path relative to StaticPaths.MODELS. If set, it is used instead of MODEL_NAME.",
                    "Путь относительно StaticPaths.MODELS. Если задан, используется вместо MODEL_NAME.",
                ),
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="LOCAL_FILES_ONLY",
                value_type="bool",
                label=(
                    "Whisper local files only",
                    "Whisper только из локальных файлов",
                ),
                description=(
                    "Disable downloads from Hugging Face Hub for Faster-Whisper.",
                    "Отключить загрузку Faster-Whisper из Hugging Face Hub.",
                ),
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="DEVICE",
                value_type="choice",
                label=("Whisper device", "Устройство Whisper"),
                description=(
                    "Device used by Faster-Whisper.",
                    "Устройство для Faster-Whisper.",
                ),
                choices=("cpu", "cuda", "auto"),
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="COMPUTE_TYPE",
                value_type="choice",
                label=("Compute type", "Тип вычислений"),
                description=(
                    "Faster-Whisper compute precision.",
                    "Точность вычислений Faster-Whisper.",
                ),
                choices=("int8", "int8_float16", "float16", "float32"),
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="CPU_THREADS",
                value_type="int",
                label=("CPU threads", "Потоки CPU"),
                description=(
                    "Number of CPU threads for transcription.",
                    "Количество CPU-потоков для транскрипции.",
                ),
                minimum=1,
                maximum=32,
                step=1,
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="NUM_WORKERS",
                value_type="int",
                label=("Workers", "Рабочие процессы"),
                description=(
                    "Number of Faster-Whisper workers.",
                    "Количество рабочих процессов Faster-Whisper.",
                ),
                minimum=1,
                maximum=8,
                step=1,
                reset_scope="whisper",
            ),
            SettingSpec(
                section="Transcription",
                field="BEAM_SIZE",
                value_type="int",
                label=("Beam size", "Размер beam"),
                description=("Beam search size.", "Размер beam search."),
                minimum=1,
                maximum=10,
                step=1,
                reset_scope="none",
            ),
            SettingSpec(
                section="Transcription",
                field="BEST_OF",
                value_type="int",
                label=("Best of", "Best of"),
                description=(
                    "Number of candidates for decoding.",
                    "Количество кандидатов при декодировании.",
                ),
                minimum=1,
                maximum=10,
                step=1,
                reset_scope="none",
            ),
            SettingSpec(
                section="Transcription",
                field="TEMPERATURE",
                value_type="float",
                label=("Temperature", "Температура"),
                description=(
                    "Whisper decoding temperature.",
                    "Температура декодирования Whisper.",
                ),
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                reset_scope="none",
            ),
            SettingSpec(
                section="Transcription",
                field="WORD_TIMESTAMPS",
                value_type="bool",
                label=("Word timestamps", "Временные метки слов"),
                description=(
                    "Enable word-level timestamps.",
                    "Включить временные метки на уровне слов.",
                ),
                reset_scope="none",
            ),
            SettingSpec(
                section="Transcription",
                field="VAD_FILTER",
                value_type="bool",
                label=("VAD filter", "VAD-фильтр"),
                description=(
                    "Use voice activity filtering.",
                    "Использовать фильтрацию речевой активности.",
                ),
                reset_scope="none",
            ),
        ),
    ),
    SettingsSectionSpec(
        title=("Feature extraction", "Извлечение признаков"),
        description=(
            "Wav2Vec2 embedding extraction settings.",
            "Настройки извлечения признаков Wav2Vec2.",
        ),
        settings=(
            SettingSpec(
                section="FeatureExtraction",
                field="WAV2VEC_MODEL_NAME",
                value_type="str",
                label=("Wav2Vec2 model", "Модель Wav2Vec2"),
                description=(
                    "Hugging Face model name for embeddings.",
                    "Имя модели Hugging Face для эмбеддингов.",
                ),
                reset_scope="wav2vec",
            ),
            SettingSpec(
                section="FeatureExtraction",
                field="WAV2VEC_MODEL_DIR",
                value_type="str",
                label=("Wav2Vec2 local directory", "Локальная директория Wav2Vec2"),
                description=(
                    "Path relative to StaticPaths.MODELS. If set, it is used instead of WAV2VEC_MODEL_NAME.",
                    "Путь относительно StaticPaths.MODELS. Если задан, используется вместо WAV2VEC_MODEL_NAME.",
                ),
                reset_scope="wav2vec",
            ),
            SettingSpec(
                section="FeatureExtraction",
                field="SAMPLE_RATE",
                value_type="int",
                label=("Sample rate", "Частота дискретизации"),
                description=(
                    "Expected audio sample rate for feature extraction.",
                    "Ожидаемая частота дискретизации для признаков.",
                ),
                minimum=8000,
                maximum=48000,
                step=1000,
                reset_scope="wav2vec",
            ),
            SettingSpec(
                section="FeatureExtraction",
                field="DEVICE",
                value_type="choice",
                label=("Feature device", "Устройство признаков"),
                description=("Device used by Wav2Vec2.", "Устройство для Wav2Vec2."),
                choices=("auto", "cpu", "mps", "cuda"),
                reset_scope="wav2vec",
            ),
            SettingSpec(
                section="FeatureExtraction",
                field="LOCAL_FILES_ONLY",
                value_type="bool",
                label=("Local files only", "Только локальные файлы"),
                description=(
                    "Disable downloads from Hugging Face Hub.",
                    "Отключить загрузку из Hugging Face Hub.",
                ),
                reset_scope="wav2vec",
            ),
        ),
    ),
    SettingsSectionSpec(
        title=("PULSE model", "Модель PULSE"),
        description=(
            "Multitask PULSE model runtime settings.",
            "Runtime-настройки многозадачной модели PULSE.",
        ),
        settings=(
            SettingSpec(
                section="Model",
                field="USE_REAL_MODEL",
                value_type="bool",
                label=("Use real model", "Использовать реальную модель"),
                description=(
                    "Use checkpoint-based PULSE inference.",
                    "Использовать инференс PULSE на основе checkpoint.",
                ),
                reset_scope="pulse",
            ),
            SettingSpec(
                section="Model",
                field="CHECKPOINT_FILE",
                value_type="str",
                label=("Checkpoint file", "Файл checkpoint"),
                description=(
                    "Path to the PULSE checkpoint.",
                    "Путь к checkpoint модели PULSE.",
                ),
                reset_scope="pulse",
            ),
            SettingSpec(
                section="Model",
                field="DEVICE",
                value_type="choice",
                label=("Model device", "Устройство модели"),
                description=(
                    "Device used by the PULSE model.",
                    "Устройство для модели PULSE.",
                ),
                choices=("auto", "cpu", "mps", "cuda"),
                reset_scope="pulse",
            ),
            SettingSpec(
                section="Model",
                field="D_MODEL",
                value_type="int",
                label=("Model dimension", "Размерность модели"),
                description=(
                    "Hidden dimension of the multitask model.",
                    "Скрытая размерность многозадачной модели.",
                ),
                minimum=64,
                maximum=1024,
                step=32,
                reset_scope="pulse",
            ),
            SettingSpec(
                section="Model",
                field="NUM_LAYERS",
                value_type="int",
                label=("Layers", "Слои"),
                description=("Number of model layers.", "Количество слоёв модели."),
                minimum=1,
                maximum=16,
                step=1,
                reset_scope="pulse",
            ),
            SettingSpec(
                section="Model",
                field="DROPOUT",
                value_type="float",
                label=("Dropout", "Dropout"),
                description=("Model dropout rate.", "Вероятность dropout в модели."),
                minimum=0.0,
                maximum=0.8,
                step=0.05,
                reset_scope="pulse",
            ),
        ),
    ),
    SettingsSectionSpec(
        title=("Attribution", "Атрибуция"),
        description=(
            "Temporal attribution and speech-mask settings.",
            "Настройки временной атрибуции и речевой маски.",
        ),
        settings=(
            SettingSpec(
                section="Attribution",
                field="ENABLE_REAL_ATTRIBUTION",
                value_type="bool",
                label=("Enable real attribution", "Включить реальную атрибуцию"),
                description=(
                    "Compute gradient-based temporal attribution.",
                    "Вычислять gradient-based временную атрибуцию.",
                ),
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="METHOD",
                value_type="choice",
                label=("Attribution method", "Метод атрибуции"),
                description=(
                    "Temporal attribution method.",
                    "Метод временной атрибуции.",
                ),
                choices=("input_x_gradient", "smoothgrad_input_x_gradient"),
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="MAX_POINTS",
                value_type="int",
                label=("Max points", "Максимум точек"),
                description=(
                    "Number of points in temporal attribution curves.",
                    "Количество точек во временных кривых атрибуции.",
                ),
                minimum=64,
                maximum=1024,
                step=32,
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="ENABLE_SPEECH_MASK",
                value_type="bool",
                label=("Speech mask", "Речевая маска"),
                description=(
                    "Suppress attribution in silent regions.",
                    "Подавлять атрибуцию в областях тишины.",
                ),
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="SPEECH_MASK_THRESHOLD_RATIO",
                value_type="float",
                label=("Speech threshold", "Порог речи"),
                description=(
                    "Energy threshold ratio for speech activity.",
                    "Относительный энергетический порог речевой активности.",
                ),
                minimum=0.0,
                maximum=0.5,
                step=0.01,
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="SPEECH_MASK_CONTEXT_SECONDS",
                value_type="float",
                label=("Speech context", "Контекст речи"),
                description=(
                    "Context window around active speech.",
                    "Контекстное окно вокруг активной речи.",
                ),
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="SPEECH_MASK_SILENCE_FLOOR",
                value_type="float",
                label=("Silence floor", "Нижний уровень тишины"),
                description=(
                    "Minimum attribution weight in silent regions.",
                    "Минимальный вес атрибуции в областях тишины.",
                ),
                minimum=0.0,
                maximum=0.3,
                step=0.01,
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="SMOOTHGRAD_SAMPLES",
                value_type="int",
                label=("SmoothGrad samples", "Сэмплы SmoothGrad"),
                description=(
                    "Number of noisy samples for SmoothGrad.",
                    "Количество зашумлённых сэмплов для SmoothGrad.",
                ),
                minimum=1,
                maximum=32,
                step=1,
                reset_scope="none",
            ),
            SettingSpec(
                section="Attribution",
                field="SMOOTHGRAD_NOISE_STD_RATIO",
                value_type="float",
                label=("SmoothGrad noise", "Шум SmoothGrad"),
                description=(
                    "Noise standard deviation ratio.",
                    "Относительное стандартное отклонение шума.",
                ),
                minimum=0.0,
                maximum=0.2,
                step=0.005,
                reset_scope="none",
            ),
        ),
    ),
    SettingsSectionSpec(
        title=("Visualization", "Визуализация"),
        description=(
            "Plot and transcription display settings.",
            "Настройки отображения графиков и транскрипции.",
        ),
        settings=(
            SettingSpec(
                section="Visualization",
                field="WAVEFORM_MAX_POINTS",
                value_type="int",
                label=("Waveform points", "Точки осциллограммы"),
                description=(
                    "Maximum number of waveform points.",
                    "Максимальное количество точек осциллограммы.",
                ),
                minimum=200,
                maximum=5000,
                step=100,
                reset_scope="none",
            ),
            SettingSpec(
                section="Visualization",
                field="TRANSCRIPT_MAX_CHARS",
                value_type="int",
                label=("Transcript max chars", "Максимум символов транскрипции"),
                description=(
                    "Maximum displayed transcript length.",
                    "Максимальная отображаемая длина транскрипции.",
                ),
                minimum=0,
                maximum=20000,
                step=100,
                reset_scope="none",
            ),
            SettingSpec(
                section="Visualization",
                field="TRANSCRIPT_WRAP",
                value_type="int",
                label=("Transcript wrap", "Перенос транскрипции"),
                description=(
                    "Soft wrap length for transcript text.",
                    "Мягкая длина переноса текста транскрипции.",
                ),
                minimum=40,
                maximum=600,
                step=20,
                reset_scope="none",
            ),
            SettingSpec(
                section="Visualization",
                field="SHOW_SPEECH_ACTIVITY",
                value_type="bool",
                label=("Show speech activity", "Показывать речевую активность"),
                description=(
                    "Display speech activity on temporal plots.",
                    "Показывать речевую активность на временных графиках.",
                ),
                reset_scope="none",
            ),
        ),
    ),
)


def get_localized_pair(value: tuple[str, str], language_index: int) -> str:
    """Return localized pair item."""

    if language_index <= 0:
        return value[0]

    return value[1]


def get_config_field_name(spec: SettingSpec) -> str:
    """Return flattened config_data field name."""

    return f"{spec.section}_{spec.field}"


def get_current_setting_value(spec: SettingSpec) -> object:
    """Return current setting value from runtime state."""

    return get_runtime_setting(section=spec.section, field=spec.field)


def format_setting_value(value: object) -> str:
    """Format setting value for Markdown."""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        return f"{value:g}"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, list):
        return f"{len(value)} items"

    if value is None:
        return "—"

    return str(value)


def format_setting_range(spec: SettingSpec, language_index: int) -> str:
    """Format setting type and allowed range."""

    if spec.value_type == "bool":
        return "`true / false`"

    if spec.value_type == "choice":
        return " / ".join(f"`{choice}`" for choice in spec.choices)

    if spec.value_type in {"int", "float"}:
        type_label = "integer" if spec.value_type == "int" else "float"
        if language_index == 1:
            type_label = "целое" if spec.value_type == "int" else "число"

        if spec.minimum is None or spec.maximum is None:
            return type_label

        step_text = f", step {spec.step:g}" if spec.step is not None else ""

        if language_index == 1:
            step_text = f", шаг {spec.step:g}" if spec.step is not None else ""

        return f"{type_label}: {spec.minimum:g}-{spec.maximum:g}{step_text}"

    return "text" if language_index == 0 else "текст"


def format_reset_scope(reset_scope: ResetScope, language_index: int) -> str:
    """Format required reset scope."""

    english_labels = {
        "none": "Immediate",
        "whisper": "Reset Whisper",
        "wav2vec": "Reset Wav2Vec2",
        "pulse": "Reset PULSE",
        "all": "Reset all caches",
    }
    russian_labels = {
        "none": "\u0421\u0440\u0430\u0437\u0443",
        "whisper": "\u0421\u0431\u0440\u043e\u0441 Whisper",
        "wav2vec": "\u0421\u0431\u0440\u043e\u0441 Wav2Vec2",
        "pulse": "\u0421\u0431\u0440\u043e\u0441 PULSE",
        "all": "\u0421\u0431\u0440\u043e\u0441 \u0432\u0441\u0435\u0445 \u043a\u044d\u0448\u0435\u0439",
    }

    labels = english_labels if language_index == 0 else russian_labels

    return labels[reset_scope]


def create_configuration_overview_markdown(language_index: int) -> str:
    """Create read-only configuration overview markdown."""

    title = "Configuration overview" if language_index == 0 else "Обзор конфигурации"
    description = (
        "The table below is generated from the same schema that will later drive editable controls."
        if language_index == 0
        else "Таблица ниже формируется из той же схемы, которая затем будет использоваться для редактируемых элементов управления."
    )

    header = (
        "| Setting | Current value | Allowed value / range | Applies after |"
        if language_index == 0
        else "| Параметр | Текущее значение | Допустимое значение / диапазон | Применяется после |"
    )

    lines = [
        f"### {title}",
        "",
        description,
        "",
    ]

    for section in SETTINGS_SCHEMA:
        lines.extend(
            [
                f"#### {get_localized_pair(section.title, language_index)}",
                "",
                get_localized_pair(section.description, language_index),
                "",
                header,
                "|---|---:|---|---|",
            ],
        )

        for setting in section.settings:
            label = get_localized_pair(setting.label, language_index)
            value = format_setting_value(get_current_setting_value(setting))
            allowed_range = format_setting_range(setting, language_index)
            reset_scope = format_reset_scope(setting.reset_scope, language_index)

            lines.append(
                f"| {label} | `{value}` | {allowed_range} | {reset_scope} |",
            )

        lines.append("")

    return "\n".join(lines)
