"""
File: mamba.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Mamba-based shared multitask model for the PULSE inference pipeline.
License: MIT License
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
import torch.nn as nn

from pulse.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MTLConfig:
    """Configuration of the shared multitask Mamba model."""

    in_dim: int
    d_model: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    max_len: int = 15000
    mamba_d_state: int = 8
    mamba_d_conv: int = 3
    mamba_expand: int = 2
    emo_mosei_out_dim: int = 7
    emo_resd_out_dim: int = 7
    pers_out_dim: int = 5
    ah_out_dim: int = 2


class MambaBlock(nn.Module):
    """Residual Mamba block for temporal audio embeddings."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        mamba_module = importlib.import_module("mamba_ssm")
        mamba_cls = cast(Any, mamba_module.Mamba)

        self.norm = nn.LayerNorm(d_model)
        self.mamba = cast(
            nn.Module,
            mamba_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Run residual Mamba block."""

        residual = x
        normalized = cast(Tensor, self.norm(x))
        mamba_output = cast(Tensor, self.mamba(normalized))
        dropped_output = cast(Tensor, self.dropout(mamba_output))

        return residual + dropped_output


class MambaAudioEncoder(nn.Module):
    """Shared Mamba encoder for Wav2Vec2 embeddings."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_layers: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ],
        )

    @staticmethod
    def make_src_key_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
        """Create a source padding mask from sequence lengths."""

        indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return indices >= lengths.unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode embedding sequence and return sequence and pooled outputs."""

        sequence_length = int(x.size(1))

        x = self.input_proj(x)

        for layer in self.layers:
            x = cast(Tensor, layer(x))

        sequence_output = x

        if lengths is None:
            pooled_output = sequence_output.mean(dim=1)
        else:
            pad_mask = self.make_src_key_padding_mask(
                lengths=lengths,
                max_len=sequence_length,
            )
            mask = (~pad_mask).unsqueeze(-1).float()
            pooled_output = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1.0,
            )

        return sequence_output, pooled_output


class MultiTaskMambaModel(nn.Module):
    """Shared multitask Mamba model with task-specific heads."""

    def __init__(self, config: MTLConfig) -> None:
        super().__init__()

        self.config = config
        self.encoder = MambaAudioEncoder(
            in_dim=config.in_dim,
            d_model=config.d_model,
            num_layers=config.num_layers,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            dropout=config.dropout,
        )

        hidden_dim = config.d_model

        self.emo_mosei_head = nn.Linear(hidden_dim, config.emo_mosei_out_dim)
        self.emo_resd_head = nn.Linear(hidden_dim, config.emo_resd_out_dim)
        self.personality_head = nn.Linear(hidden_dim, config.pers_out_dim)
        self.ah_head = nn.Linear(hidden_dim, config.ah_out_dim)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run shared encoder and all task-specific heads."""

        _, pooled_output = self.encoder(
            batch["x"],
            batch["lengths"],
        )

        return {
            "pooled": pooled_output,
            "emotion_mosei_pred": self.emo_mosei_head(pooled_output),
            "emotion_resd_logits": self.emo_resd_head(pooled_output),
            "personality_preds": self.personality_head(pooled_output),
            "ah_logits": self.ah_head(pooled_output),
        }


def extract_state_dict(raw_state: Any) -> tuple[Any, str]:
    """Extract model state dictionary from a checkpoint object."""

    if not isinstance(raw_state, dict):
        return raw_state, "plain_state_dict"

    if "model_state" in raw_state:
        return raw_state["model_state"], "model_state"

    if "state_dict" in raw_state:
        return raw_state["state_dict"], "state_dict"

    return raw_state, "plain_state_dict"


def strip_module_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remove DataParallel 'module.' prefix from checkpoint keys if needed."""

    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    return state_dict


def build_multitask_mamba_model(
    config: MTLConfig,
    device: torch.device,
) -> MultiTaskMambaModel:
    """Build multitask Mamba model."""

    model = MultiTaskMambaModel(config)
    nn.Module.to(model, device)
    return model


def load_multitask_mamba_weights(
    model: MultiTaskMambaModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load multitask Mamba model weights."""

    raw_state = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    state_dict_raw, source_name = extract_state_dict(raw_state)

    if not isinstance(state_dict_raw, dict):
        msg = f"Unsupported checkpoint state type: {type(state_dict_raw)!r}"
        raise TypeError(msg)

    state_dict = strip_module_prefix(cast(dict[str, Tensor], state_dict_raw))

    LOGGER.info("Checkpoint format: %s", source_name)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    nn.Module.to(model, device)
    nn.Module.eval(model)

    if missing_keys:
        LOGGER.warning("Missing checkpoint keys: %s", list(missing_keys[:50]))

    if unexpected_keys:
        LOGGER.warning("Unexpected checkpoint keys: %s", list(unexpected_keys[:50]))

    if not missing_keys and not unexpected_keys:
        LOGGER.info("State dict loaded cleanly.")

    return raw_state if isinstance(raw_state, dict) else {}
