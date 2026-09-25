"""
File: features.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Wav2Vec2 feature extraction utilities for the PULSE inference pipeline.
License: MIT License
"""

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, cast

from huggingface_hub import hf_hub_download
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from pulse.audio.decoder import DecodedAudio
from pulse.config import PROJECT_ROOT, is_hugging_face_space
from pulse.settings.state import get_runtime_setting_by_flat_name


@dataclass(frozen=True, slots=True)
class EmbeddingExtractor:
    """Loaded Wav2Vec2 embedding extractor."""

    model_name: str
    sample_rate: int
    device: torch.device
    feature_extractor: Any
    model: Any


@dataclass(frozen=True, slots=True)
class EmbeddingExtractionResult:
    """Extracted Wav2Vec2 embeddings."""

    embeddings: Tensor
    length: int
    sample_rate: int
    model_name: str
    device: str

    @property
    def embeddings_shape(self) -> tuple[int, ...]:
        """Return embeddings tensor shape."""

        return tuple(int(dimension) for dimension in self.embeddings.shape)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""

        if self.embeddings.ndim != 2:
            return 0

        return int(self.embeddings.shape[-1])


_EMBEDDING_EXTRACTOR: EmbeddingExtractor | None = None


def is_embedding_extractor_loaded() -> bool:
    """Return whether Wav2Vec2 embedding extractor is already loaded."""

    return _EMBEDDING_EXTRACTOR is not None


def clear_torch_memory_cache() -> None:
    """Clear best-effort torch memory caches."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def reset_embedding_extractor() -> None:
    """Reset cached Wav2Vec2 embedding extractor."""

    global _EMBEDDING_EXTRACTOR

    if _EMBEDDING_EXTRACTOR is None:
        return

    extractor = _EMBEDDING_EXTRACTOR
    _EMBEDDING_EXTRACTOR = None

    del extractor
    gc.collect()
    clear_torch_memory_cache()


def get_config_str(field_name: str, default_value: str) -> str:
    """Return string feature extraction config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, str):
        return value

    return default_value


def get_config_int(field_name: str, default_value: int) -> int:
    """Return integer feature extraction config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return default_value

    if isinstance(value, int):
        return value

    return default_value


def get_config_bool(field_name: str, default_value: bool) -> bool:
    """Return bool feature extraction config value."""

    value = get_runtime_setting_by_flat_name(field_name, default_value)

    if isinstance(value, bool):
        return value

    return default_value


def resolve_path_from_models_root(relative_path: str) -> Path:
    """Resolve a path relative to the configured models root."""

    path = Path(relative_path).expanduser()

    if path.is_absolute():
        return path

    models_root = Path(
        get_config_str(
            "StaticPaths_MODELS",
            "models",
        ),
    ).expanduser()

    if models_root.is_absolute():
        return models_root / path

    return PROJECT_ROOT / models_root / path


def resolve_wav2vec2_hf_model_reference() -> str:
    """Return Wav2Vec2 Hugging Face model reference."""

    hf_model_id = get_config_str(
        "FeatureExtraction_WAV2VEC_HF_MODEL_ID",
        "",
    ).strip()

    if hf_model_id:
        return hf_model_id

    model_name = get_config_str(
        "FeatureExtraction_WAV2VEC_MODEL_NAME",
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ).strip()

    return model_name or "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


def get_wav2vec2_local_files_only() -> bool:
    """Return Wav2Vec2 local-files-only mode."""

    if is_hugging_face_space():
        return False

    return get_config_bool(
        "FeatureExtraction_LOCAL_FILES_ONLY",
        False,
    )


def resolve_wav2vec2_model_reference() -> str:
    """Return Wav2Vec2 model reference from runtime settings."""

    if is_hugging_face_space():
        return resolve_wav2vec2_hf_model_reference()

    model_dir = get_config_str(
        "FeatureExtraction_WAV2VEC_MODEL_DIR",
        "",
    ).strip()

    if model_dir:
        return str(resolve_path_from_models_root(model_dir))

    model_name = get_config_str(
        "FeatureExtraction_WAV2VEC_MODEL_NAME",
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ).strip()

    return model_name or "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


def get_feature_extraction_device() -> torch.device:
    """Return device for Wav2Vec2 feature extraction."""

    requested_device = get_config_str(
        "FeatureExtraction_DEVICE",
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


def load_wav2vec2_config(
    model_reference: str,
    local_files_only: bool,
) -> Any:
    """Load Wav2Vec2 config and patch missing vocab size if needed."""

    model_path = Path(model_reference).expanduser()

    if model_path.is_absolute():
        config_path = model_path / "config.json"

        if not config_path.exists():
            msg = f"Wav2Vec2 config.json was not found: {config_path}"
            raise FileNotFoundError(msg)
    else:
        config_path = Path(
            str(
                hf_hub_download(
                    repo_id=model_reference,
                    filename="config.json",
                    local_files_only=local_files_only,
                ),
            ),
        )

    with config_path.open(encoding="utf-8") as file:
        config_data_json = json.load(file)

    if not isinstance(config_data_json, dict):
        msg = f"Expected config.json to contain a JSON object: {config_path}"
        raise ValueError(msg)

    config_dict = cast(dict[str, Any], config_data_json)

    if config_dict.get("vocab_size") is None:
        config_dict["vocab_size"] = 32

    return Wav2Vec2Config.from_dict(config_dict)


def build_embedding_extractor() -> EmbeddingExtractor:
    """Build Wav2Vec2 embedding extractor."""

    model_reference = resolve_wav2vec2_model_reference()
    sample_rate = get_config_int(
        "FeatureExtraction_SAMPLE_RATE",
        16000,
    )
    local_files_only = get_wav2vec2_local_files_only()
    device = get_feature_extraction_device()

    feature_extractor = cast(
        Any,
        Wav2Vec2FeatureExtractor.from_pretrained(
            model_reference,
            local_files_only=local_files_only,
        ),
    )
    wav2vec2_config = load_wav2vec2_config(
        model_reference=model_reference,
        local_files_only=local_files_only,
    )
    model = cast(
        Any,
        Wav2Vec2Model.from_pretrained(
            model_reference,
            config=wav2vec2_config,
            local_files_only=local_files_only,
        ),
    )

    model_module = cast(nn.Module, model)
    nn.Module.to(model_module, device)
    nn.Module.eval(model_module)

    return EmbeddingExtractor(
        model_name=model_reference,
        sample_rate=sample_rate,
        device=device,
        feature_extractor=feature_extractor,
        model=model,
    )


def get_embedding_extractor() -> EmbeddingExtractor:
    """Return cached Wav2Vec2 embedding extractor."""

    global _EMBEDDING_EXTRACTOR

    if _EMBEDDING_EXTRACTOR is None:
        _EMBEDDING_EXTRACTOR = build_embedding_extractor()

    return _EMBEDDING_EXTRACTOR


def decoded_audio_to_numpy(decoded_audio: DecodedAudio) -> npt.NDArray[np.float32]:
    """Convert decoded channel-first audio tensor to mono NumPy waveform."""

    waveform = decoded_audio.waveform.detach().cpu()

    if waveform.ndim == 1:
        signal = waveform
    elif waveform.ndim == 2:
        signal = waveform[0] if int(waveform.shape[0]) == 1 else waveform.mean(dim=0)
    else:
        msg = f"Expected decoded waveform with 1 or 2 dimensions, got shape={tuple(waveform.shape)}."
        raise ValueError(msg)

    return cast(
        npt.NDArray[np.float32],
        signal.numpy().astype(np.float32, copy=False),
    )


@torch.inference_mode()
def extract_wav2vec2_embeddings(
    decoded_audio: DecodedAudio,
    extractor: EmbeddingExtractor | None = None,
) -> EmbeddingExtractionResult:
    """Extract Wav2Vec2 embeddings from decoded audio."""

    active_extractor = extractor or get_embedding_extractor()
    signal = decoded_audio_to_numpy(decoded_audio)

    inputs = active_extractor.feature_extractor(
        signal,
        sampling_rate=active_extractor.sample_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = cast(Tensor, inputs["input_values"]).to(active_extractor.device)
    model_inputs: dict[str, Tensor] = {
        "input_values": input_values,
    }

    attention_mask = inputs.get("attention_mask")

    if isinstance(attention_mask, Tensor):
        model_inputs["attention_mask"] = attention_mask.to(active_extractor.device)

    outputs = active_extractor.model(**model_inputs)
    last_hidden_state = cast(Tensor, outputs.last_hidden_state)

    embeddings = last_hidden_state.detach().cpu()[0].to(dtype=torch.float32)
    length = int(embeddings.shape[0])

    if length <= 0:
        msg = "Empty Wav2Vec2 embedding sequence."
        raise RuntimeError(msg)

    return EmbeddingExtractionResult(
        embeddings=embeddings,
        length=length,
        sample_rate=active_extractor.sample_rate,
        model_name=active_extractor.model_name,
        device=str(active_extractor.device),
    )
