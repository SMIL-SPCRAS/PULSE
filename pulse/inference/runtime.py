"""
File: runtime.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Runtime management for real PULSE multitask inference.
License: MIT License
"""

from dataclasses import dataclass
import gc
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch
from torch import Tensor
import torch.nn as nn

from pulse.config import PROJECT_ROOT, is_hugging_face_space
from pulse.inference.features import EmbeddingExtractionResult
from pulse.inference.models import (
    MTLConfig,
    MultiTaskMambaModel,
    build_multitask_mamba_model,
    load_multitask_mamba_weights,
)
from pulse.inference.multitask import MultitaskRawOutputs
from pulse.logger import get_logger
from pulse.settings.state import get_runtime_setting_by_flat_name

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PulseRuntime:
    """Loaded real PULSE runtime."""

    device: torch.device
    model: MultiTaskMambaModel
    checkpoint_path: Path


_RUNTIME: PulseRuntime | None = None


def is_pulse_runtime_loaded() -> bool:
    """Return whether real PULSE runtime is already loaded."""

    return _RUNTIME is not None


def clear_torch_memory_cache() -> None:
    """Clear best-effort torch memory caches."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def reset_pulse_runtime() -> None:
    """Reset cached real PULSE runtime."""

    global _RUNTIME

    if _RUNTIME is None:
        LOGGER.info("PULSE runtime cache is already empty.")
        return

    runtime = _RUNTIME
    _RUNTIME = None

    del runtime
    gc.collect()
    clear_torch_memory_cache()

    LOGGER.info("Reset PULSE runtime cache.")


def get_config_bool(field_name: str, default_value: bool) -> bool:
    """Return bool model config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer model config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def get_config_float(field_name: str, default_value: float) -> float:
    """Return float model config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int | float):
        return float(value)

    return default_value


def get_config_str(field_name: str, default_value: str) -> str:
    """Return string model config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, str):
        return value

    return default_value


def get_model_device() -> torch.device:
    """Return device for the real multitask model."""

    requested_device = get_config_str(
        "Model_DEVICE",
        "auto",
    ).lower()

    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested_device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(requested_device)


def get_hf_checkpoint_path() -> Path | None:
    """Return downloaded PULSE checkpoint path from Hugging Face Hub if configured."""

    repo_id = get_config_str(
        "Model_HF_CHECKPOINT_REPO_ID",
        "",
    ).strip()
    filename = get_config_str(
        "Model_HF_CHECKPOINT_FILENAME",
        "",
    ).strip()

    if not repo_id or not filename:
        return None

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        ),
    )


def get_checkpoint_path() -> Path:
    """Return resolved checkpoint path."""

    if is_hugging_face_space():
        hf_checkpoint_path = get_hf_checkpoint_path()

        if hf_checkpoint_path is not None:
            return hf_checkpoint_path

    models_dir = Path(
        get_config_str(
            "StaticPaths_MODELS",
            "models",
        ),
    ).expanduser()

    checkpoint_file = Path(
        get_config_str(
            "Model_CHECKPOINT_FILE",
            "pulse/best_model.pt",
        ),
    ).expanduser()

    if checkpoint_file.is_absolute():
        return checkpoint_file

    if models_dir.is_absolute():
        return models_dir / checkpoint_file

    return PROJECT_ROOT / models_dir / checkpoint_file


def build_mtl_config() -> MTLConfig:
    """Build multitask model config from application config."""

    return MTLConfig(
        in_dim=get_config_int("Model_INPUT_DIM", 1024),
        d_model=get_config_int("Model_D_MODEL", 256),
        num_layers=get_config_int("Model_NUM_LAYERS", 4),
        dropout=get_config_float("Model_DROPOUT", 0.1),
        max_len=get_config_int("Model_MAX_LEN", 15000),
        mamba_d_state=get_config_int("Model_MAMBA_D_STATE", 8),
        mamba_d_conv=get_config_int("Model_MAMBA_D_CONV", 3),
        mamba_expand=get_config_int("Model_MAMBA_EXPAND", 2),
        emo_mosei_out_dim=7,
        emo_resd_out_dim=7,
        pers_out_dim=5,
        ah_out_dim=2,
    )


def is_real_model_enabled() -> bool:
    """Return whether real model inference is enabled."""

    return get_config_bool(
        "Model_USE_REAL_MODEL",
        False,
    )


def load_pulse_runtime() -> PulseRuntime:
    """Load real PULSE runtime."""

    checkpoint_path = get_checkpoint_path()

    if not checkpoint_path.exists():
        msg = f"Model checkpoint was not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    device = get_model_device()
    model_config = build_mtl_config()
    model = build_multitask_mamba_model(
        config=model_config,
        device=device,
    )
    load_multitask_mamba_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    nn.Module.eval(model)

    LOGGER.info("Loaded real multitask model: device=%s, checkpoint=%s", device, checkpoint_path)

    return PulseRuntime(
        device=device,
        model=model,
        checkpoint_path=checkpoint_path,
    )


def get_pulse_runtime() -> PulseRuntime:
    """Return cached real PULSE runtime."""

    global _RUNTIME

    if _RUNTIME is None:
        _RUNTIME = load_pulse_runtime()

    return _RUNTIME


def tensor_to_float_list(tensor: Tensor) -> list[float]:
    """Convert a 1D tensor to a list of floats."""

    values = tensor.detach().cpu().tolist()

    if not isinstance(values, list):
        return [float(values)]

    return [float(value) for value in values]


@torch.no_grad()
def run_real_multitask_model(
    embeddings: EmbeddingExtractionResult,
    runtime: PulseRuntime | None = None,
) -> MultitaskRawOutputs:
    """Run the real shared multitask PULSE model."""

    active_runtime = runtime or get_pulse_runtime()
    device = active_runtime.device

    x = embeddings.embeddings.to(device=device, dtype=torch.float32)
    length = int(x.shape[0])

    if length <= 0:
        msg = "Empty embedding sequence."
        raise RuntimeError(msg)

    batch = {
        "x": x.unsqueeze(0),
        "lengths": torch.tensor(
            [length],
            dtype=torch.long,
            device=device,
        ),
    }

    outputs = active_runtime.model(batch)

    return MultitaskRawOutputs(
        mosei_logits=tensor_to_float_list(outputs["emotion_mosei_pred"][0]),
        resd_logits=tensor_to_float_list(outputs["emotion_resd_logits"][0]),
        bah_logits=tensor_to_float_list(outputs["ah_logits"][0]),
        personality_scores=tensor_to_float_list(outputs["personality_preds"][0]),
    )
