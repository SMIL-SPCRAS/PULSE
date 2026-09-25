from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

@dataclass
class AutoLambdaStepInfo:
    meta_loss: float
    train_loss: float
    meta_weights: np.ndarray
    virtual_lr: float
    hessian_eps: float

class AutoLambda:

    def __init__(
        self,
        model: nn.Module,
        n_tasks: int,
        device: torch.device,
        weight_init: float = 0.1,
        meta_lr: float = 1e-4,
        virtual_lr: float = 0.0,
        hessian_eps_scale: float = 0.01,
        finite_eps: float = 1e-12,
    ):
        if n_tasks <= 0:
            raise ValueError("Auto-Lambda requires at least one active task.")
        if meta_lr <= 0:
            raise ValueError("autolambda_lr must be positive.")
        if virtual_lr < 0:
            raise ValueError("autolambda_virtual_lr must be >= 0 (0 = current model LR).")
        if hessian_eps_scale <= 0:
            raise ValueError("autolambda_hessian_eps must be positive.")

        self.model = model
        self.model_ = copy.deepcopy(model)
        self.n_tasks = int(n_tasks)
        self.device = device
        self.virtual_lr = float(virtual_lr)
        self.hessian_eps_scale = float(hessian_eps_scale)
        self.finite_eps = float(finite_eps)

        self.meta_weights = nn.Parameter(
            torch.full(
                (self.n_tasks,),
                float(weight_init),
                dtype=torch.float32,
                device=device,
            )
        )
        self.meta_optimizer = torch.optim.Adam([self.meta_weights], lr=float(meta_lr))
        self.last_info: Optional[AutoLambdaStepInfo] = None

    def weights_tensor(self) -> torch.Tensor:
        return self.meta_weights

    def weights_numpy(self) -> np.ndarray:
        return self.meta_weights.detach().cpu().numpy().copy()

    def _weighted_loss(self, losses: torch.Tensor) -> torch.Tensor:
        if losses.ndim != 1 or losses.numel() != self.n_tasks:
            raise ValueError(
                f"Auto-Lambda expected a {self.n_tasks}-element loss vector, "
                f"got shape={tuple(losses.shape)}."
            )
        return torch.sum(self.meta_weights * losses)

    @staticmethod
    def _sync_buffers(source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src, dst in zip(source.buffers(), target.buffers()):
                dst.copy_(src)

    def _resolve_virtual_lr(self, model_optimizer: torch.optim.Optimizer) -> float:
        if self.virtual_lr > 0:
            return self.virtual_lr
        if not model_optimizer.param_groups:
            raise RuntimeError("Auto-Lambda received an optimizer without parameter groups.")
        alpha = float(model_optimizer.param_groups[0].get("lr", 0.0))
        if alpha <= 0:
            raise RuntimeError(f"Auto-Lambda virtual learning rate is non-positive: {alpha}.")
        return alpha

    def virtual_step(
        self,
        train_loss_fn: Callable[[nn.Module], torch.Tensor],
        alpha: float,
        model_optimizer: torch.optim.Optimizer,
    ) -> None:

        self.model.train()
        self.model_.train()

        train_losses = train_loss_fn(self.model)
        weighted_loss = self._weighted_loss(train_losses)
        if not torch.isfinite(train_losses).all() or not torch.isfinite(weighted_loss):
            raise RuntimeError(
                "Auto-Lambda train loss became non-finite before the virtual step: "
                f"{train_losses.detach().cpu().tolist()}"
            )

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        virtual_params = [p for p in self.model_.parameters() if p.requires_grad]
        if len(model_params) != len(virtual_params):
            raise RuntimeError("Auto-Lambda model/model_ parameter layouts do not match.")

        grads = torch.autograd.grad(
            weighted_loss,
            model_params,
            allow_unused=True,
        )

        param_to_group = {}
        for group in model_optimizer.param_groups:
            for p in group["params"]:
                param_to_group[id(p)] = group

        self._sync_buffers(self.model, self.model_)
        with torch.no_grad():
            for weight, weight_, grad in zip(model_params, virtual_params, grads):
                if grad is None:
                    weight_.copy_(weight)
                    continue

                group = param_to_group.get(id(weight), {})
                weight_decay = float(group.get("weight_decay", 0.0))
                momentum = 0.0
                if "momentum" in group:
                    state = model_optimizer.state.get(weight, {})
                    momentum_buffer = state.get("momentum_buffer", 0.0)
                    momentum = momentum_buffer * float(group.get("momentum", 0.0))

                update = grad + weight_decay * weight
                if isinstance(momentum, torch.Tensor):
                    update = update + momentum
                elif momentum != 0.0:
                    update = update + momentum
                weight_.copy_(weight - alpha * update)

    def compute_hessian(
        self,
        d_model: Sequence[Optional[torch.Tensor]],
        train_loss_fn: Callable[[nn.Module], torch.Tensor],
    ) -> tuple[torch.Tensor, float]:

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(model_params) != len(d_model):
            raise RuntimeError("Auto-Lambda Hessian direction has the wrong length.")

        directions = []
        for p, d in zip(model_params, d_model):
            directions.append(torch.zeros_like(p) if d is None else d.detach())

        flat_nonempty = [d.reshape(-1) for d in directions if d.numel() > 0]
        if not flat_nonempty:
            return torch.zeros_like(self.meta_weights), 0.0

        norm = torch.cat(flat_nonempty).norm()
        if not torch.isfinite(norm):
            raise RuntimeError("Auto-Lambda validation-gradient norm became NaN/Inf.")
        norm_value = float(norm.detach().item())
        if norm_value <= self.finite_eps:
            return torch.zeros_like(self.meta_weights), 0.0

        eps = self.hessian_eps_scale / norm_value

        def meta_grad_at_current_params() -> torch.Tensor:
            train_losses = train_loss_fn(self.model)
            weighted_loss = self._weighted_loss(train_losses)
            if not torch.isfinite(weighted_loss):
                raise RuntimeError(
                    "Auto-Lambda finite-difference train loss became non-finite: "
                    f"{train_losses.detach().cpu().tolist()}"
                )
            (grad_w,) = torch.autograd.grad(weighted_loss, self.meta_weights)
            return grad_w.detach().clone()

        shift_units = 0.0
        try:
            with torch.no_grad():
                for p, d in zip(model_params, directions):
                    p.add_(eps * d)
            shift_units = 1.0
            d_weight_p = meta_grad_at_current_params()

            with torch.no_grad():
                for p, d in zip(model_params, directions):
                    p.add_(-2.0 * eps * d)
            shift_units = -1.0
            d_weight_n = meta_grad_at_current_params()
        finally:
            if shift_units != 0.0:
                with torch.no_grad():
                    for p, d in zip(model_params, directions):
                        p.add_(-shift_units * eps * d)

        hessian = (d_weight_p - d_weight_n) / (2.0 * eps)
        if not torch.isfinite(hessian).all():
            raise RuntimeError(
                "Auto-Lambda finite-difference Hessian became NaN/Inf. "
                "Try a smaller model LR or autolambda_lr."
            )
        return hessian, float(eps)

    def unrolled_backward(
        self,
        train_loss_fn: Callable[[nn.Module], torch.Tensor],
        val_loss_fn: Callable[[nn.Module], torch.Tensor],
        model_optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, float]:

        alpha = self._resolve_virtual_lr(model_optimizer)
        self.virtual_step(train_loss_fn, alpha, model_optimizer)

        val_losses = val_loss_fn(self.model_)
        if val_losses.ndim != 1 or val_losses.numel() != self.n_tasks:
            raise ValueError(
                f"Auto-Lambda validation loss vector must contain {self.n_tasks} values."
            )
        meta_loss = val_losses.sum()
        if not torch.isfinite(val_losses).all() or not torch.isfinite(meta_loss):
            raise RuntimeError(
                "Auto-Lambda validation meta-loss became non-finite: "
                f"{val_losses.detach().cpu().tolist()}"
            )

        virtual_params = [p for p in self.model_.parameters() if p.requires_grad]
        d_model = torch.autograd.grad(
            meta_loss,
            virtual_params,
            allow_unused=True,
        )
        hessian, fd_eps = self.compute_hessian(d_model, train_loss_fn)

        self.meta_weights.grad = (-alpha * hessian).detach().clone()
        if not torch.isfinite(self.meta_weights.grad).all():
            raise RuntimeError("Auto-Lambda meta-weight gradient became NaN/Inf.")

        return float(meta_loss.detach().item()), float(alpha), float(fd_eps)

    def meta_step(self) -> None:
        self.meta_optimizer.step()

    def zero_meta_grad(self) -> None:
        self.meta_optimizer.zero_grad(set_to_none=True)

    def record_step(
        self,
        meta_loss: float,
        train_loss: float,
        virtual_lr: float,
        hessian_eps: float,
    ) -> AutoLambdaStepInfo:
        info = AutoLambdaStepInfo(
            meta_loss=float(meta_loss),
            train_loss=float(train_loss),
            meta_weights=self.weights_numpy(),
            virtual_lr=float(virtual_lr),
            hessian_eps=float(hessian_eps),
        )
        self.last_info = info
        return info

    def debug_string(self, task_names: Sequence[str]) -> str:
        weights = self.weights_numpy()
        pairs = ", ".join(
            f"{name}={weights[i]:.4f}" for i, name in enumerate(task_names)
        )
        return f"weights: {pairs}"

    def state_dict(self) -> dict:
        return {
            "meta_weights": self.meta_weights.detach().cpu(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "virtual_lr": self.virtual_lr,
            "hessian_eps_scale": self.hessian_eps_scale,
        }
