import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_encoder import PositionalEncoding

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class MHLA1DAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int,
        max_len: int,
        dropout: float = 0.1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        transform: str = "linear",
        local_thres: float = 1.5,
        exp_sigma: float = 3.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = int(chunk_size)
        self.max_blocks = math.ceil(max_len / self.chunk_size)
        self.eps = float(eps)
        self.transform = transform
        self.local_thres = float(local_thres)
        self.exp_sigma = float(exp_sigma)

        self.norm = nn.LayerNorm(d_model)
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.q_norm = RMSNorm(d_model, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(d_model, eps=eps) if qk_norm else nn.Identity()
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

        init_mix = self._build_initial_mixing(self.max_blocks)
        self.mixing = nn.Parameter(init_mix)

    def _build_initial_mixing(self, n_blocks: int) -> torch.Tensor:
        idx = torch.arange(n_blocks, dtype=torch.float32)
        dist = (idx[:, None] - idx[None, :]).abs()

        if self.transform == "linear":
            row_max = dist.max(dim=1, keepdim=True).values.clamp_min(1.0)
            mat = 1.0 - dist / row_max
        elif self.transform == "cos":
            row_max = dist.max(dim=1, keepdim=True).values.clamp_min(1.0)
            mat = torch.cos((dist / row_max) * (math.pi / 4.0))
        elif self.transform == "exp":
            mat = torch.exp(-dist / max(self.exp_sigma, self.eps))
        elif self.transform == "gaussian":
            sigma = max(float(dist.max().item()) / 3.0, self.eps)
            mat = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        elif self.transform == "local":
            mat = (dist <= self.local_thres).float()
        else:
            raise ValueError(f"Unknown MHLA transform: {self.transform}")

        mat = mat / mat.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return mat

    def _get_mixing_matrix(self, n_blocks: int) -> torch.Tensor:
        mix = self.mixing[:n_blocks, :n_blocks]
        mix = mix.clamp(min=0.0, max=1.0)
        mix = mix / mix.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return mix

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        bsz, seq_len, _ = x.shape
        chunk = self.chunk_size
        n_blocks = math.ceil(seq_len / chunk)
        padded_len = n_blocks * chunk
        pad_len = padded_len - seq_len

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        if valid_mask is None:
            valid_mask = torch.ones(bsz, seq_len, device=x.device, dtype=x.dtype)
        else:
            valid_mask = valid_mask.to(dtype=x.dtype)

        if pad_len > 0:
            valid_mask = F.pad(valid_mask, (0, pad_len))

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = F.relu(q) + self.eps
        k = F.relu(k) + self.eps

        q = q.view(bsz, n_blocks, chunk, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, n_blocks, chunk, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(bsz, n_blocks, chunk, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        token_mask = valid_mask.view(bsz, n_blocks, chunk).unsqueeze(1).unsqueeze(-1)
        q = q * token_mask
        k = k * token_mask
        v = v * token_mask

        q = q.reshape(bsz * self.n_heads, n_blocks, chunk, self.head_dim)
        k = k.reshape(bsz * self.n_heads, n_blocks, chunk, self.head_dim)
        v = v.reshape(bsz * self.n_heads, n_blocks, chunk, self.head_dim)

        kv = torch.matmul(k.transpose(-2, -1), v)
        mix = self._get_mixing_matrix(n_blocks)
        mixed_kv = torch.einsum("ij,bjkl->bikl", mix, kv)

        k_sum = k.sum(dim=-2, keepdim=True)
        normalizer = torch.matmul(q, k_sum.transpose(-2, -1))
        mixed_normalizer = torch.einsum("ij,bjwk->biwk", mix, normalizer)

        out = torch.matmul(q, mixed_kv) / (mixed_normalizer + self.eps)
        out = out.view(bsz, self.n_heads, n_blocks, chunk, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(bsz, padded_len, self.d_model)
        out = out[:, :seq_len]
        return self.to_out(out)

class MHLABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        chunk_size: int,
        max_len: int,
        qk_norm: bool,
        transform: str,
        local_thres: float,
        exp_sigma: float,
    ):
        super().__init__()
        self.attn = MHLA1DAttention(
            d_model=d_model,
            n_heads=n_heads,
            chunk_size=chunk_size,
            max_len=max_len,
            dropout=dropout,
            qk_norm=qk_norm,
            transform=transform,
            local_thres=local_thres,
            exp_sigma=exp_sigma,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(x, valid_mask=valid_mask)
        x = x + self.ff(self.norm_ff(x))
        return x

class MHLAAudioEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        mhla_chunk_size: int = 64,
        mhla_qk_norm: bool = True,
        mhla_transform: str = "linear",
        mhla_local_thres: float = 1.5,
        mhla_exp_sigma: float = 3.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                MHLABlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    chunk_size=mhla_chunk_size,
                    max_len=max_len,
                    qk_norm=mhla_qk_norm,
                    transform=mhla_transform,
                    local_thres=mhla_local_thres,
                    exp_sigma=mhla_exp_sigma,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _make_valid_mask(lengths: Optional[torch.Tensor], max_len: int) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        idxs = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idxs < lengths.unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        valid_mask = self._make_valid_mask(lengths, seq_len)

        for layer in self.layers:
            x = layer(x, valid_mask=valid_mask)

        seq_out = self.out_norm(x)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).to(seq_out.dtype)
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = seq_out.mean(dim=1)

        return seq_out, pooled
