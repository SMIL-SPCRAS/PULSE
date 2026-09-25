from __future__ import annotations

from typing import Optional

import torch

class STCH:

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        mu: float = 1.0,
        warmup_epoch: int = 4,
        eps: float = 1e-20,
    ):
        self.n_tasks = int(n_tasks)
        self.device = device
        self.mu = float(mu)
        self.warmup_epoch = int(warmup_epoch)
        self.eps = float(eps)

        if self.n_tasks <= 0:
            raise ValueError(f"STCH: n_tasks must be positive, got {self.n_tasks}")
        if self.mu <= 0.0:
            raise ValueError(f"STCH: mu must be > 0, got {self.mu}")
        if self.warmup_epoch < 0:
            raise ValueError(
                f"STCH: warmup_epoch must be >= 0, got {self.warmup_epoch}"
            )

        self.reference_sum = torch.zeros(
            self.n_tasks, dtype=torch.float32, device=self.device
        )
        self.reference_count = torch.zeros(
            self.n_tasks, dtype=torch.long, device=self.device
        )
        self.nadir_vector: Optional[torch.Tensor] = None

        self.last_mode = "init"
        self.last_loss_value: Optional[float] = None
        self.last_softmax_weights = torch.zeros(
            self.n_tasks, dtype=torch.float32, device=self.device
        )

    def _validate_inputs(self, losses: torch.Tensor, active_mask: torch.Tensor):
        if losses.dim() != 1 or losses.numel() != self.n_tasks:
            raise ValueError(
                f"STCH expects losses shape ({self.n_tasks},), got {tuple(losses.shape)}"
            )
        if active_mask.dim() != 1 or active_mask.numel() != self.n_tasks:
            raise ValueError(
                "STCH expects active_mask with the same number of tasks as losses, "
                f"got {tuple(active_mask.shape)}"
            )

    def _accumulate_reference(self, losses: torch.Tensor, active_mask: torch.Tensor):
        with torch.no_grad():
            idx = torch.nonzero(active_mask, as_tuple=False).view(-1)
            if idx.numel() == 0:
                return
            vals = losses.detach().index_select(0, idx).to(torch.float32)
            self.reference_sum.index_add_(0, idx, vals)
            self.reference_count.index_add_(
                0,
                idx,
                torch.ones(idx.numel(), dtype=torch.long, device=self.device),
            )

    def _ensure_nadir(self, losses: torch.Tensor, active_mask: torch.Tensor):
        if self.nadir_vector is None:
            counts = self.reference_count.clamp_min(1).to(torch.float32)
            nadir = self.reference_sum / counts
            self.nadir_vector = nadir.clamp_min(self.eps)

        with torch.no_grad():
            missing = (self.reference_count == 0) & active_mask
            if missing.any():
                self.nadir_vector[missing] = losses.detach()[missing].clamp_min(self.eps)
                self.reference_count[missing] = 1

    def backward(
        self,
        losses: torch.Tensor,
        active_mask: torch.Tensor,
        epoch: int,
    ):
        self._validate_inputs(losses, active_mask)
        active_mask = active_mask.to(device=losses.device, dtype=torch.bool)
        active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)

        if active_idx.numel() == 0:
            self.last_mode = "no_active_tasks"
            self.last_loss_value = 0.0
            self.last_softmax_weights.zero_()
            return {
                "loss": 0.0,
                "mode": self.last_mode,
                "weights": self.last_softmax_weights.detach().cpu().numpy(),
            }

        active_losses = losses.index_select(0, active_idx)
        if torch.any(active_losses < 0):
            raise ValueError(
                "STCH expects non-negative task losses before its internal log transform."
            )

        if epoch < self.warmup_epoch:
            scalar_loss = torch.log(active_losses + self.eps).sum()
            mode = "warmup"
            softmax_weights = torch.full(
                (active_idx.numel(),),
                1.0 / float(active_idx.numel()),
                dtype=losses.dtype,
                device=losses.device,
            )

        elif epoch == self.warmup_epoch:
            self._accumulate_reference(losses, active_mask)
            scalar_loss = torch.log(active_losses + self.eps).sum()
            mode = "reference"
            softmax_weights = torch.full(
                (active_idx.numel(),),
                1.0 / float(active_idx.numel()),
                dtype=losses.dtype,
                device=losses.device,
            )

        else:
            self._ensure_nadir(losses, active_mask)
            active_ref = self.nadir_vector.index_select(0, active_idx).to(losses.dtype)

            normalized = torch.log(active_losses / active_ref + self.eps)

            max_term = normalized.detach().max()
            regularized = normalized - max_term

            active_task_num = int(active_idx.numel())
            scalar_loss = (
                self.mu
                * torch.logsumexp(regularized / self.mu, dim=0)
                * float(active_task_num)
            )
            mode = "stch"
            softmax_weights = torch.softmax(regularized.detach() / self.mu, dim=0)

        scalar_loss.backward()

        self.last_mode = mode
        self.last_loss_value = float(scalar_loss.detach().item())
        self.last_softmax_weights.zero_()
        self.last_softmax_weights[active_idx] = softmax_weights.to(torch.float32)

        return {
            "loss": self.last_loss_value,
            "mode": self.last_mode,
            "weights": self.last_softmax_weights.detach().cpu().numpy(),
        }

    def debug_string(self, task_names=None) -> str:
        if task_names is None:
            task_names = [f"task_{i}" for i in range(self.n_tasks)]
        weights = self.last_softmax_weights.detach().cpu().tolist()
        pairs = ", ".join(
            f"{name}={weights[i]:.3f}" for i, name in enumerate(task_names)
        )
        ref = "unset"
        if self.nadir_vector is not None:
            ref_vals = self.nadir_vector.detach().cpu().tolist()
            ref = "[" + ", ".join(f"{v:.4g}" for v in ref_vals) + "]"
        return (
            f"mode={self.last_mode} | mu={self.mu:g} | "
            f"warmup_epoch={self.warmup_epoch} | weights: {pairs} | ref={ref}"
        )
