"""
File: attribution.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Temporal attribution utilities for the PULSE inference pipeline.
License: MIT License
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
import torch.nn.functional as F

from pulse.inference.adapters import AttributionTarget
from pulse.inference.features import EmbeddingExtractionResult
from pulse.inference.runtime import get_pulse_runtime
from pulse.inference.schemas import SpeechActivity, TemporalImportance
from pulse.settings.state import get_runtime_setting_by_flat_name


@dataclass(frozen=True, slots=True)
class AttributionRecord:
    """Temporal attribution values for one target."""

    task_key: str
    class_key: str
    target_index: int
    group_key: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TemporalAttributionResult:
    """Temporal attribution result for selected targets."""

    records: tuple[AttributionRecord, ...]
    temporal_importance: TemporalImportance
    speech_activity: SpeechActivity | None


@dataclass(frozen=True, slots=True)
class AttributionProgress:
    """Fine-grained attribution progress."""

    target_index: int

    target_count: int

    sample_index: int

    sample_count: int

    target_task_key: str

    target_class_key: str

    @property
    def percentage(self) -> float:
        """Return attribution-local progress percentage."""

        if self.target_count <= 0 or self.sample_count <= 0:
            return 0.0

        completed_samples = (self.target_index - 1) * self.sample_count + self.sample_index

        total_samples = self.target_count * self.sample_count

        return min(
            100.0,
            max(
                0.0,
                100.0 * completed_samples / total_samples,
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetAttributionResult:
    """Temporal attribution values for one target before post-processing."""

    values: Tensor


def get_config_bool(field_name: str, default_value: bool) -> bool:
    """Return bool attribution config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer attribution config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def get_config_float(field_name: str, default_value: float) -> float:
    """Return float attribution config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_config_str(field_name: str, default_value: str) -> str:
    """Return string attribution config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, str):
        return value

    return default_value


def is_real_attribution_enabled() -> bool:
    """Return whether real temporal attribution is enabled."""

    return get_config_bool(
        "Attribution_ENABLE_REAL_ATTRIBUTION",
        True,
    )


def should_fallback_to_placeholder_attribution() -> bool:
    """Return whether attribution errors should fall back to placeholder heatmap."""

    return get_config_bool(
        "Attribution_FALLBACK_TO_PLACEHOLDER",
        True,
    )


def get_attribution_reduction() -> str:
    """Return attribution temporal reduction mode."""

    return get_config_str(
        "Attribution_REDUCTION",
        "sum_abs",
    )


def get_attribution_method() -> str:
    """Return temporal attribution method."""

    return get_config_str(
        "Attribution_METHOD",
        "input_x_gradient",
    )


def get_smoothgrad_samples() -> int:
    """Return number of SmoothGrad samples."""

    return max(
        1,
        get_config_int(
            "Attribution_SMOOTHGRAD_SAMPLES",
            6,
        ),
    )


def get_smoothgrad_noise_std_ratio() -> float:
    """Return SmoothGrad noise standard deviation ratio."""

    return max(
        0.0,
        get_config_float(
            "Attribution_SMOOTHGRAD_NOISE_STD_RATIO",
            0.02,
        ),
    )


def should_include_clean_smoothgrad_sample() -> bool:
    """Return whether SmoothGrad should include one clean sample."""

    return get_config_bool(
        "Attribution_SMOOTHGRAD_INCLUDE_CLEAN_SAMPLE",
        True,
    )


def get_attribution_max_points() -> int:
    """Return maximum number of points in displayed attribution."""

    return max(
        2,
        get_config_int(
            "Attribution_MAX_POINTS",
            256,
        ),
    )


def get_attribution_eps() -> float:
    """Return attribution normalization epsilon."""

    return get_config_float(
        "Attribution_EPS",
        1e-12,
    )


def is_speech_mask_enabled() -> bool:
    """Return whether speech-aware attribution masking is enabled."""

    return get_config_bool(
        "Attribution_ENABLE_SPEECH_MASK",
        True,
    )


def get_speech_mask_frame_ms() -> float:
    """Return speech-mask frame size in milliseconds."""

    return get_config_float(
        "Attribution_SPEECH_MASK_FRAME_MS",
        25.0,
    )


def get_speech_mask_hop_ms() -> float:
    """Return speech-mask hop size in milliseconds."""

    return get_config_float(
        "Attribution_SPEECH_MASK_HOP_MS",
        10.0,
    )


def get_speech_mask_threshold_ratio() -> float:
    """Return relative energy threshold for speech masking."""

    return max(
        0.0,
        get_config_float(
            "Attribution_SPEECH_MASK_THRESHOLD_RATIO",
            0.08,
        ),
    )


def get_speech_mask_min_active_ratio() -> float:
    """Return minimum active-mask ratio before falling back to no mask."""

    return max(
        0.0,
        get_config_float(
            "Attribution_SPEECH_MASK_MIN_ACTIVE_RATIO",
            0.02,
        ),
    )


def get_speech_mask_context_seconds() -> float:
    """Return speech-mask temporal context in seconds."""

    return max(
        0.0,
        get_config_float(
            "Attribution_SPEECH_MASK_CONTEXT_SECONDS",
            0.20,
        ),
    )


def get_speech_mask_smoothing_seconds() -> float:
    """Return speech-mask smoothing window in seconds."""

    return max(
        0.0,
        get_config_float(
            "Attribution_SPEECH_MASK_SMOOTHING_SECONDS",
            0.12,
        ),
    )


def get_speech_mask_silence_floor() -> float:
    """Return lower attribution multiplier for silence regions."""

    return min(
        1.0,
        max(
            0.0,
            get_config_float(
                "Attribution_SPEECH_MASK_SILENCE_FLOOR",
                0.02,
            ),
        ),
    )


def waveform_to_mono_float32(waveform: Tensor) -> Tensor:
    """Convert waveform tensor to mono float32 CPU tensor."""

    cpu_waveform = waveform.detach().cpu().to(dtype=torch.float32)

    if cpu_waveform.ndim == 1:
        return cpu_waveform

    if cpu_waveform.ndim == 2:
        return cpu_waveform[0] if int(cpu_waveform.shape[0]) == 1 else cpu_waveform.mean(dim=0)

    msg = f"Expected waveform with 1 or 2 dimensions, got shape={tuple(cpu_waveform.shape)}."
    raise ValueError(msg)


def interpolate_1d(
    values: Tensor,
    target_length: int,
) -> Tensor:
    """Interpolate a 1D tensor to target length."""

    source_length = int(values.shape[0])

    if target_length <= 0:
        return torch.empty(0, dtype=torch.float32)

    if source_length == target_length:
        return values.to(dtype=torch.float32)

    if source_length <= 1:
        fill_value = values[0] if source_length == 1 else torch.tensor(0.0)
        return torch.full(
            (target_length,),
            float(fill_value.detach().cpu().item()),
            dtype=torch.float32,
        )

    interpolated = F.interpolate(
        values.view(1, 1, source_length).to(dtype=torch.float32),
        size=target_length,
        mode="linear",
        align_corners=True,
    )

    return interpolated.flatten()


def make_odd_kernel_size(kernel_size: int) -> int:
    """Return a positive odd kernel size."""

    if kernel_size <= 1:
        return 1

    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def max_pool_1d_same(
    values: Tensor,
    kernel_size: int,
) -> Tensor:
    """Apply same-length 1D max pooling."""

    kernel_size = make_odd_kernel_size(kernel_size)

    if kernel_size <= 1:
        return values

    padding = kernel_size // 2
    pooled = F.max_pool1d(
        values.view(1, 1, -1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    return pooled.flatten()


def avg_pool_1d_same(
    values: Tensor,
    kernel_size: int,
) -> Tensor:
    """Apply same-length 1D average pooling."""

    kernel_size = make_odd_kernel_size(kernel_size)

    if kernel_size <= 1:
        return values

    padding = kernel_size // 2
    padded_values = F.pad(
        values.view(1, 1, -1),
        pad=(padding, padding),
        mode="replicate",
    )
    pooled = F.avg_pool1d(
        padded_values,
        kernel_size=kernel_size,
        stride=1,
    )

    return pooled.flatten()


def seconds_to_temporal_frames(
    seconds: float,
    duration_seconds: float,
    target_length: int,
) -> int:
    """Convert seconds to temporal mask frames."""

    if seconds <= 0.0 or duration_seconds <= 0.0 or target_length <= 1:
        return 1

    return max(
        1,
        round(seconds * target_length / duration_seconds),
    )


def create_energy_envelope(
    waveform: Tensor,
    sample_rate: int,
) -> Tensor:
    """Create short-time RMS energy envelope from waveform."""

    mono_waveform = waveform_to_mono_float32(waveform)
    sample_count = int(mono_waveform.shape[0])

    if sample_count <= 0 or sample_rate <= 0:
        return torch.zeros(1, dtype=torch.float32)

    frame_samples = max(
        1,
        round(sample_rate * get_speech_mask_frame_ms() / 1000.0),
    )
    hop_samples = max(
        1,
        round(sample_rate * get_speech_mask_hop_ms() / 1000.0),
    )

    squared_waveform = mono_waveform.square().view(1, 1, -1)

    if sample_count < frame_samples:
        rms = torch.sqrt(squared_waveform.mean(dim=-1).flatten() + get_attribution_eps())
        return rms.to(dtype=torch.float32)

    pooled_energy = F.avg_pool1d(
        squared_waveform,
        kernel_size=frame_samples,
        stride=hop_samples,
        ceil_mode=True,
    )
    rms = torch.sqrt(pooled_energy.flatten() + get_attribution_eps())

    return rms.to(dtype=torch.float32)


def create_speech_activity_mask(
    waveform: Tensor | None,
    sample_rate: int | None,
    target_length: int,
    duration_seconds: float,
) -> Tensor | None:
    """Create a soft speech-activity mask aligned with attribution frames."""

    if not is_speech_mask_enabled():
        return None

    if waveform is None or sample_rate is None or sample_rate <= 0 or target_length <= 0:
        return None

    envelope = create_energy_envelope(
        waveform=waveform,
        sample_rate=sample_rate,
    )
    envelope = interpolate_1d(
        values=envelope,
        target_length=target_length,
    )

    max_energy = torch.max(envelope)

    if float(max_energy.detach().cpu().item()) <= get_attribution_eps():
        return torch.ones(target_length, dtype=torch.float32)

    threshold = max_energy * get_speech_mask_threshold_ratio()
    binary_mask = (envelope >= threshold).to(dtype=torch.float32)

    active_ratio = float(binary_mask.mean().detach().cpu().item())

    if active_ratio < get_speech_mask_min_active_ratio():
        return torch.ones(target_length, dtype=torch.float32)

    context_frames = seconds_to_temporal_frames(
        seconds=get_speech_mask_context_seconds(),
        duration_seconds=duration_seconds,
        target_length=target_length,
    )
    smoothing_frames = seconds_to_temporal_frames(
        seconds=get_speech_mask_smoothing_seconds(),
        duration_seconds=duration_seconds,
        target_length=target_length,
    )

    expanded_mask = max_pool_1d_same(
        values=binary_mask,
        kernel_size=2 * context_frames + 1,
    )
    smooth_mask = avg_pool_1d_same(
        values=expanded_mask,
        kernel_size=smoothing_frames,
    )
    smooth_mask = torch.clamp(smooth_mask, min=0.0, max=1.0)

    silence_floor = get_speech_mask_silence_floor()

    return silence_floor + (1.0 - silence_floor) * smooth_mask


def apply_speech_mask(
    values: Tensor,
    speech_mask: Tensor | None,
) -> Tensor:
    """Apply speech mask to temporal attribution values."""

    if speech_mask is None:
        return values

    aligned_mask = interpolate_1d(
        values=speech_mask,
        target_length=int(values.shape[0]),
    )

    return values * aligned_mask.to(dtype=values.dtype)


def should_log_speech_mask() -> bool:
    """Return whether speech-mask diagnostics should be logged by the pipeline."""

    return get_config_bool(
        "Attribution_LOG_SPEECH_MASK",
        True,
    )


def calculate_soft_mask_active_ratio(mask_values: Tensor) -> float:
    """Calculate active ratio from a soft speech mask."""

    if int(mask_values.shape[0]) <= 0:
        return 0.0

    silence_floor = get_speech_mask_silence_floor()
    threshold = silence_floor + (1.0 - silence_floor) * 0.5
    active_values = (mask_values >= threshold).to(dtype=torch.float32)

    return float(active_values.mean().detach().cpu().item())


def create_speech_activity_diagnostics(
    speech_mask: Tensor | None,
    max_points: int,
    duration_seconds: float,
) -> SpeechActivity | None:
    """Create serializable speech-activity diagnostics from a soft mask."""

    if speech_mask is None:
        return None

    sampled_mask = sample_temporal_values(
        values=speech_mask.to(dtype=torch.float32),
        max_points=max_points,
    )
    sampled_mask = torch.clamp(
        torch.nan_to_num(
            sampled_mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        min=0.0,
        max=1.0,
    )

    values = list(tensor_to_float_tuple(sampled_mask))

    if not values:
        return None

    return SpeechActivity(
        timestamps=create_temporal_timestamps(
            point_count=len(values),
            duration_seconds=duration_seconds,
        ),
        values=values,
        active_ratio=calculate_soft_mask_active_ratio(sampled_mask),
        min_value=min(values),
        max_value=max(values),
        mean_value=sum(values) / len(values),
    )


def scalar_from_outputs(
    outputs: dict[str, Tensor],
    target: AttributionTarget,
) -> Tensor:
    """Select scalar model output for one attribution target."""

    if target.task_key == "mosei":
        return outputs["emotion_mosei_pred"][0, target.target_index]

    if target.task_key == "resd":
        return outputs["emotion_resd_logits"][0, target.target_index]

    if target.task_key == "bah":
        return outputs["ah_logits"][0, target.target_index]

    if target.task_key == "fiv2":
        score = outputs["personality_preds"][0, target.target_index]

        if target.class_key == "neuroticism":
            return -score

        return score

    msg = f"Unsupported attribution target task: {target.task_key}."
    raise KeyError(msg)


def reduce_temporal_attribution(
    attribution: Tensor,
    reduction: str,
) -> Tensor:
    """Reduce feature-wise attribution to one temporal curve."""

    if reduction == "sum_abs":
        return torch.abs(attribution).sum(dim=-1)

    if reduction == "l2":
        return torch.sqrt(torch.sum(attribution * attribution, dim=-1))

    msg = f"Unsupported attribution reduction mode: {reduction}."
    raise ValueError(msg)


def normalize_temporal_values(
    values: Tensor,
    eps: float,
) -> Tensor:
    """Normalize temporal attribution values to [0, 1]."""

    finite_values = torch.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    max_value = torch.max(finite_values)

    if float(max_value.detach().cpu().item()) <= eps:
        return torch.zeros_like(finite_values)

    return torch.clamp(finite_values / max_value, min=0.0, max=1.0)


def sample_temporal_values(
    values: Tensor,
    max_points: int,
) -> Tensor:
    """Sample temporal values to a maximum number of points."""

    point_count = int(values.shape[0])

    if point_count <= max_points:
        return values

    indices = torch.linspace(
        0,
        point_count - 1,
        steps=max_points,
    ).round()
    indices = indices.to(dtype=torch.long)

    return values[indices]


def tensor_to_float_tuple(values: Tensor) -> tuple[float, ...]:
    """Convert 1D tensor to tuple of floats."""

    return tuple(float(value) for value in values.detach().cpu().tolist())


def create_temporal_timestamps(
    point_count: int,
    duration_seconds: float,
) -> list[str]:
    """Create display timestamps for temporal attribution."""

    if point_count <= 1:
        return ["0.00 s"]

    return [f"{duration_seconds * index / (point_count - 1):.2f} s" for index in range(point_count)]


def compute_input_x_gradient_temporal_attribution(
    input_embeddings: Tensor,
    target: AttributionTarget,
) -> Tensor:
    """Compute one input x gradient temporal attribution pass."""

    runtime = get_pulse_runtime()
    device = runtime.device

    x = input_embeddings.detach().clone().to(device=device, dtype=torch.float32)
    length = int(x.shape[0])

    if length <= 0:
        msg = "Empty embedding sequence."
        raise RuntimeError(msg)

    x_var = x.unsqueeze(0).requires_grad_(True)
    lengths = torch.tensor(
        [length],
        dtype=torch.long,
        device=device,
    )

    outputs = cast(
        dict[str, Tensor],
        runtime.model(
            {
                "x": x_var,
                "lengths": lengths,
            },
        ),
    )

    score = scalar_from_outputs(
        outputs=outputs,
        target=target,
    )

    gradient = torch.autograd.grad(
        score,
        x_var,
        retain_graph=False,
        create_graph=False,
    )[0]

    attribution = x_var[0, :length, :] * gradient[0, :length, :]

    return (
        reduce_temporal_attribution(
            attribution=attribution,
            reduction=get_attribution_reduction(),
        )
        .detach()
        .cpu()
    )


def calculate_embedding_noise_std(
    embeddings: Tensor,
) -> float:
    """Calculate SmoothGrad noise standard deviation."""

    noise_std_ratio = get_smoothgrad_noise_std_ratio()

    if noise_std_ratio <= 0.0:
        return 0.0

    embedding_std = torch.std(embeddings.detach().to(dtype=torch.float32))
    embedding_std_value = float(embedding_std.detach().cpu().item())

    if embedding_std_value <= get_attribution_eps():
        return 0.0

    return embedding_std_value * noise_std_ratio


def iter_smoothgrad_temporal_attribution(
    embeddings: EmbeddingExtractionResult,
    target: AttributionTarget,
    target_index: int,
    target_count: int,
) -> Iterator[AttributionProgress | TargetAttributionResult]:
    """Compute SmoothGrad input x gradient temporal attribution with progress events."""

    base_embeddings = embeddings.embeddings.detach().to(dtype=torch.float32)
    sample_count = get_smoothgrad_samples()
    noise_std = calculate_embedding_noise_std(base_embeddings)
    include_clean_sample = should_include_clean_smoothgrad_sample()

    accumulated_values: Tensor | None = None

    for sample_index in range(sample_count):
        yield AttributionProgress(
            target_index=target_index,
            target_count=target_count,
            sample_index=sample_index + 1,
            sample_count=sample_count,
            target_task_key=target.task_key,
            target_class_key=target.class_key,
        )

        if (sample_index == 0 and include_clean_sample) or noise_std <= 0.0:
            sampled_embeddings = base_embeddings
        else:
            sampled_embeddings = base_embeddings + torch.randn_like(base_embeddings) * noise_std

        temporal_values = compute_input_x_gradient_temporal_attribution(
            input_embeddings=sampled_embeddings,
            target=target,
        )

        accumulated_values = temporal_values if accumulated_values is None else accumulated_values + temporal_values

    if accumulated_values is None:
        msg = "SmoothGrad did not produce attribution values."
        raise RuntimeError(msg)

    yield TargetAttributionResult(
        values=accumulated_values / sample_count,
    )


def compute_smoothgrad_temporal_attribution(
    embeddings: EmbeddingExtractionResult,
    target: AttributionTarget,
) -> Tensor:
    """Compute SmoothGrad input x gradient temporal attribution."""

    final_result: TargetAttributionResult | None = None

    for update in iter_smoothgrad_temporal_attribution(
        embeddings=embeddings,
        target=target,
        target_index=1,
        target_count=1,
    ):
        if isinstance(update, TargetAttributionResult):
            final_result = update

    if final_result is None:
        msg = "SmoothGrad did not produce attribution values."
        raise RuntimeError(msg)

    return final_result.values


def iter_target_temporal_attribution(
    embeddings: EmbeddingExtractionResult,
    target: AttributionTarget,
    target_index: int,
    target_count: int,
) -> Iterator[AttributionProgress | TargetAttributionResult]:
    """Compute temporal attribution for one target with progress events."""

    attribution_method = get_attribution_method()

    if attribution_method == "input_x_gradient":
        yield AttributionProgress(
            target_index=target_index,
            target_count=target_count,
            sample_index=1,
            sample_count=1,
            target_task_key=target.task_key,
            target_class_key=target.class_key,
        )

        yield TargetAttributionResult(
            values=compute_input_x_gradient_temporal_attribution(
                input_embeddings=embeddings.embeddings,
                target=target,
            ),
        )
        return

    if attribution_method == "smoothgrad_input_x_gradient":
        yield from iter_smoothgrad_temporal_attribution(
            embeddings=embeddings,
            target=target,
            target_index=target_index,
            target_count=target_count,
        )
        return

    msg = f"Unsupported attribution method: {attribution_method}."
    raise ValueError(msg)


def compute_target_temporal_attribution(
    embeddings: EmbeddingExtractionResult,
    target: AttributionTarget,
) -> Tensor:
    """Compute temporal attribution for one target."""

    final_result: TargetAttributionResult | None = None

    for update in iter_target_temporal_attribution(
        embeddings=embeddings,
        target=target,
        target_index=1,
        target_count=1,
    ):
        if isinstance(update, TargetAttributionResult):
            final_result = update

    if final_result is None:
        msg = "Target attribution did not produce values."
        raise RuntimeError(msg)

    return final_result.values


def create_attribution_method_summary() -> str:
    """Create attribution method summary for logs."""

    attribution_method = get_attribution_method()

    if attribution_method != "smoothgrad_input_x_gradient":
        return attribution_method

    return (
        f"{attribution_method}"
        f"(samples={get_smoothgrad_samples()}, "
        f"noise_std_ratio={get_smoothgrad_noise_std_ratio():.3f}, "
        f"include_clean={should_include_clean_smoothgrad_sample()})"
    )


def iter_temporal_attribution(
    embeddings: EmbeddingExtractionResult,
    targets: tuple[AttributionTarget, ...],
    duration_seconds: float,
    waveform: Tensor | None = None,
    sample_rate: int | None = None,
) -> Iterator[AttributionProgress | TemporalAttributionResult]:
    """Compute temporal attribution for selected targets with fine-grained progress events."""

    if not targets:
        yield TemporalAttributionResult(
            records=(),
            temporal_importance=TemporalImportance(
                timestamps=[],
                values=[],
            ),
            speech_activity=None,
        )
        return

    eps = get_attribution_eps()
    max_points = get_attribution_max_points()
    embedding_length = int(embeddings.length)

    speech_mask = create_speech_activity_mask(
        waveform=waveform,
        sample_rate=sample_rate,
        target_length=embedding_length,
        duration_seconds=duration_seconds,
    )
    speech_activity = create_speech_activity_diagnostics(
        speech_mask=speech_mask,
        max_points=max_points,
        duration_seconds=duration_seconds,
    )

    normalized_maps: list[Tensor] = []
    records: list[AttributionRecord] = []
    target_count = len(targets)

    for target_index, target in enumerate(targets, start=1):
        target_result: TargetAttributionResult | None = None

        for attribution_update in iter_target_temporal_attribution(
            embeddings=embeddings,
            target=target,
            target_index=target_index,
            target_count=target_count,
        ):
            if isinstance(attribution_update, AttributionProgress):
                yield attribution_update
            else:
                target_result = attribution_update

        if target_result is None:
            msg = f"Attribution did not produce values for target {target.task_key}:{target.class_key}."
            raise RuntimeError(msg)

        temporal_values = apply_speech_mask(
            values=target_result.values,
            speech_mask=speech_mask,
        )
        temporal_values = sample_temporal_values(
            values=temporal_values,
            max_points=max_points,
        )
        temporal_values = normalize_temporal_values(
            values=temporal_values,
            eps=eps,
        )

        normalized_maps.append(temporal_values)

        records.append(
            AttributionRecord(
                task_key=target.task_key,
                class_key=target.class_key,
                target_index=target.target_index,
                group_key=target.group_key,
                values=tensor_to_float_tuple(temporal_values),
            ),
        )

    stacked_maps = torch.stack(normalized_maps, dim=0)
    aggregate_values = normalize_temporal_values(
        values=stacked_maps.mean(dim=0),
        eps=eps,
    )

    aggregate_tuple = tensor_to_float_tuple(aggregate_values)
    timestamps = create_temporal_timestamps(
        point_count=len(aggregate_tuple),
        duration_seconds=duration_seconds,
    )

    yield TemporalAttributionResult(
        records=tuple(records),
        temporal_importance=TemporalImportance(
            timestamps=timestamps,
            values=list(aggregate_tuple),
        ),
        speech_activity=speech_activity,
    )


def compute_temporal_attribution(
    embeddings: EmbeddingExtractionResult,
    targets: tuple[AttributionTarget, ...],
    duration_seconds: float,
    waveform: Tensor | None = None,
    sample_rate: int | None = None,
) -> TemporalAttributionResult:
    """Compute temporal attribution for selected targets."""

    final_result: TemporalAttributionResult | None = None

    for update in iter_temporal_attribution(
        embeddings=embeddings,
        targets=targets,
        duration_seconds=duration_seconds,
        waveform=waveform,
        sample_rate=sample_rate,
    ):
        if isinstance(update, TemporalAttributionResult):
            final_result = update

    if final_result is None:
        msg = "Temporal attribution did not produce a final result."
        raise RuntimeError(msg)

    return final_result
