from __future__ import annotations

from typing import List, Tuple, Dict, Any
import time

import numpy as np
import torch
from scipy.optimize import least_squares

class FairGrad:

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        alpha: float = 1.0,
        max_norm: float = 1.0,
        eps: float = 1e-8,

        alpha_mode: str = "fixed",  # "fixed" | "adaptive"
        alpha_min: float = 0.25,
        alpha_max: float = 20.0,
        alpha_lr: float = 0.05,
        alpha_target_imbalance: float = 0.7,
        alpha_warmup_steps: int = 0,
        alpha_momentum: float = 0.0,
        alpha_adapt_metric: str = "progress",  # "progress" | "gradnorm"

        update_every: int = 1,
        time_every: int = 0,
        time_sync_cuda: bool = True,
        reuse_weight_min: float = 1e-6,
    ):
        self.n_tasks = int(n_tasks)
        self.device = device
        self.alpha = float(alpha)
        self.max_norm = float(max_norm)
        self.eps = float(eps)

        self.alpha_mode = str(alpha_mode)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_lr = float(alpha_lr)
        self.alpha_target_imbalance = float(alpha_target_imbalance)
        self.alpha_warmup_steps = int(alpha_warmup_steps)
        self.alpha_momentum = float(alpha_momentum)
        self.alpha_adapt_metric = str(alpha_adapt_metric)

        self.update_every = int(update_every)
        self.time_every = int(time_every)
        self.time_sync_cuda = bool(time_sync_cuda)
        self.reuse_weight_min = float(reuse_weight_min)

        self._step = 0
        self._w_full_cache: np.ndarray | None = None

        if self.alpha_mode not in ("fixed", "adaptive"):
            raise ValueError(f"FairGrad: alpha_mode must be 'fixed' or 'adaptive', got {self.alpha_mode!r}")
        if self.alpha_adapt_metric not in ("progress", "gradnorm"):
            raise ValueError(
                f"FairGrad: alpha_adapt_metric must be 'progress' or 'gradnorm', got {self.alpha_adapt_metric!r}"
            )
        if not (self.alpha_min > 0 and self.alpha_max >= self.alpha_min):
            raise ValueError(f"FairGrad: invalid alpha_min/alpha_max: {self.alpha_min}, {self.alpha_max}")

        if self.update_every < 1:
            raise ValueError(f"FairGrad: update_every must be >=1, got {self.update_every}")
        if self.time_every < 0:
            raise ValueError(f"FairGrad: time_every must be >=0, got {self.time_every}")
        if not (0.0 < self.reuse_weight_min <= 1e-2):
            raise ValueError(f"FairGrad: reuse_weight_min should be in (0, 1e-2], got {self.reuse_weight_min}")

        self.alpha = float(np.clip(self.alpha, self.alpha_min, self.alpha_max))

    def _maybe_sync(self):
        if self.time_sync_cuda and isinstance(self.device, torch.device) and self.device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    @staticmethod
    def _grad2vec(shared_params: List[torch.nn.Parameter], grads: torch.Tensor, grad_dims: List[int], task: int):
        grads[:, task].fill_(0.0)
        cnt = 0
        for p in shared_params:
            g = p.grad
            if g is not None:
                g = g.detach().view(-1)
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                end = sum(grad_dims[: (cnt + 1)])
                grads[beg:end, task].copy_(g)
            cnt += 1

    def _overwrite_grad(self, shared_params: List[torch.nn.Parameter], newgrad: torch.Tensor, grad_dims: List[int]):

        newgrad = newgrad * self.n_tasks
        cnt = 0
        for p in shared_params:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            end = sum(grad_dims[: (cnt + 1)])
            p.grad = newgrad[beg:end].contiguous().view_as(p.data).detach().clone()
            cnt += 1

    def _solve_weights(self, GG: np.ndarray) -> np.ndarray:

        K = GG.shape[0]
        x0 = np.ones(K, dtype=np.float64) / K
        alpha = float(self.alpha)
        eps = float(self.eps)

        def objfn(x):
            x = np.maximum(x, eps)
            return GG.dot(x) - np.power(1.0 / x, 1.0 / alpha)

        res = least_squares(objfn, x0, bounds=(0.0, np.inf))
        x = np.maximum(res.x, eps)
        return x

    def _imbalance_from_diag(self, GG: np.ndarray) -> float:
        d = np.diag(GG).astype(np.float64)
        d = np.maximum(d, float(self.eps))
        return float(np.log(d.max() / d.min()))

    def _imbalance_from_progress(self, progress: np.ndarray) -> float:
        p = np.asarray(progress, dtype=np.float64)
        p = np.maximum(p, float(self.eps))
        return float(np.log(p.max() / p.min()))

    def _maybe_update_alpha(self, imbalance: float) -> Tuple[float, float]:

        a_used = float(self.alpha)
        if self.alpha_mode != "adaptive":
            return a_used, a_used

        if self._step < self.alpha_warmup_steps:
            return a_used, a_used

        log_a = float(np.log(max(a_used, float(self.eps))))
        log_a_new = log_a + float(self.alpha_lr) * (float(imbalance) - float(self.alpha_target_imbalance))
        a_new = float(np.exp(log_a_new))
        a_new = float(np.clip(a_new, self.alpha_min, self.alpha_max))

        if self.alpha_momentum > 0:
            m = float(np.clip(self.alpha_momentum, 0.0, 0.999))
            a_next = m * a_used + (1.0 - m) * a_new
        else:
            a_next = a_new

        self.alpha = float(a_next)
        return a_used, float(self.alpha)

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: List[torch.nn.Parameter],
        task_specific_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ) -> Tuple[None, Dict[str, Any]]:
        assert losses.dim() == 1 and losses.numel() == self.n_tasks, \
            f"FairGrad expects losses shape ({self.n_tasks},), got {tuple(losses.shape)}"

        shared_params = list(shared_parameters)
        grad_dims = [p.data.numel() for p in shared_params]
        D = int(sum(grad_dims))

        with torch.no_grad():
            active_mask = (losses.detach() > 0).to(torch.bool)
        active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)

        do_update = (self.update_every <= 1) or (self._w_full_cache is None) or ((self._step % self.update_every) == 0)
        do_time = bool(self.time_every) and ((self._step % int(self.time_every)) == 0)

        if active_idx.numel() <= 1:
            losses.sum().backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(shared_params, self.max_norm)

            w_full = np.zeros(self.n_tasks, dtype=np.float64)
            active_tasks = active_idx.tolist()
            if active_idx.numel() == 1:
                w_full[int(active_idx.item())] = 1.0

            extra = {
                "weights": w_full,
                "GTG": None,
                "active_tasks": active_tasks,
                "alpha": float(self.alpha),
                "alpha_used": float(self.alpha),
                "alpha_before": float(self.alpha),
                "alpha_after": float(self.alpha),
                "alpha_next": float(self.alpha),
                "imbalance": None,
                "imbalance_progress": None,
                "imbalance_gradnorm": None,
                "progress": None,
                "grad_norms": None,
                "w_active_min": None,
                "w_active_max": None,
                "did_update": False,
                "update_every": int(self.update_every),
                "time_backward": None,
                "time_copy": None,
                "time_solve": None,
                "time_total": None,
            }
            self._step += 1
            return None, extra

        if not do_update and self._w_full_cache is not None:
            active_tasks = active_idx.tolist()
            w_full = np.asarray(self._w_full_cache, dtype=np.float64).reshape(-1)
            idx_cpu = active_idx.detach().cpu().numpy()
            w_active = w_full[idx_cpu]

            pos = w_active[w_active > 0]
            fill = float(pos.mean()) if pos.size > 0 else 1.0
            w_active = np.where(w_active > 0, w_active, fill)
            w_active = np.maximum(w_active, self.reuse_weight_min)

            ww = torch.as_tensor(w_active, device=losses.device, dtype=losses.dtype)

            loss_w = (losses[active_idx] * ww).sum() * float(self.n_tasks)

            t0 = None
            if do_time:
                self._maybe_sync()
                t0 = time.perf_counter()

            loss_w.backward()

            if isinstance(task_specific_parameters, dict):
                scale = w_active * float(self.n_tasks)
                for j, ti in enumerate(active_tasks):
                    s = float(scale[j])
                    if s <= float(self.eps):
                        continue
                    for p in task_specific_parameters.get(int(ti), []):
                        if p.grad is not None:
                            p.grad.data.div_(s)

            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(shared_params, self.max_norm)

            t_total = None
            t_backward = None
            if do_time and t0 is not None:
                self._maybe_sync()
                t_total = float(time.perf_counter() - t0)
                t_backward = t_total

            extra = {
                "weights": w_full,
                "GTG": None,
                "active_tasks": active_tasks,
                "alpha": float(self.alpha),
                "alpha_used": float(self.alpha),
                "alpha_before": float(self.alpha),
                "alpha_after": float(self.alpha),
                "alpha_next": float(self.alpha),
                "imbalance": None,
                "imbalance_progress": None,
                "imbalance_gradnorm": None,
                "progress": None,
                "grad_norms": None,
                "w_active_min": float(np.min(w_active)),
                "w_active_max": float(np.max(w_active)),
                "did_update": False,
                "update_every": int(self.update_every),
                "time_backward": t_backward,
                "time_copy": None,
                "time_solve": None,
                "time_total": t_total,
            }

            self._step += 1
            return None, extra

        K = int(active_idx.numel())
        grads = torch.zeros(D, K, device=self.device, dtype=torch.float32)

        t0 = t1 = t2 = t3 = None
        if do_time:
            self._maybe_sync()
            t0 = time.perf_counter()

        for j, ti in enumerate(active_idx.tolist()):
            retain = (j < K - 1)
            losses[ti].backward(retain_graph=retain)
            self._grad2vec(shared_params, grads, grad_dims, j)
            for p in shared_params:
                p.grad = None

        if do_time:
            self._maybe_sync()
            t1 = time.perf_counter()

        GG_t = grads.t().mm(grads).detach().cpu().numpy()

        if do_time:
            t2 = time.perf_counter()

        alpha_used = float(self.alpha)
        w_active = self._solve_weights(GG_t)

        if do_time:
            t3 = time.perf_counter()

        progress = GG_t.dot(w_active)
        grad_norms = np.diag(GG_t).astype(np.float64)

        imb_prog = self._imbalance_from_progress(progress)
        imb_gn = self._imbalance_from_diag(GG_t)
        imb_for_alpha = float(imb_prog) if self.alpha_adapt_metric == "progress" else float(imb_gn)

        alpha_before, alpha_after = self._maybe_update_alpha(imb_for_alpha)

        ww = torch.tensor(w_active, device=grads.device, dtype=grads.dtype)
        g = (grads * ww.view(1, -1)).sum(dim=1)
        self._overwrite_grad(shared_params, g, grad_dims)

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_params, self.max_norm)

        w_full = np.zeros(self.n_tasks, dtype=np.float64)
        active_tasks = active_idx.tolist()
        for j, ti in enumerate(active_tasks):
            w_full[int(ti)] = float(w_active[j])

        self._w_full_cache = w_full.copy()

        extra = {
            "weights": w_full,
            "GTG": GG_t,
            "active_tasks": active_tasks,
            "alpha": float(alpha_used),
            "alpha_used": float(alpha_used),
            "alpha_before": float(alpha_before),
            "alpha_after": float(alpha_after),
            "alpha_next": float(alpha_after),
            "imbalance": float(imb_for_alpha),
            "imbalance_progress": float(imb_prog),
            "imbalance_gradnorm": float(imb_gn),
            "progress": progress.astype(np.float64),
            "grad_norms": grad_norms.astype(np.float64),
            "w_active_min": float(np.min(w_active)),
            "w_active_max": float(np.max(w_active)),
            "did_update": True,
            "update_every": int(self.update_every),
            "time_backward": (float(t1 - t0) if (do_time and t0 is not None and t1 is not None) else None),
            "time_copy": (float(t2 - t1) if (do_time and t1 is not None and t2 is not None) else None),
            "time_solve": (float(t3 - t2) if (do_time and t2 is not None and t3 is not None) else None),
            "time_total": (float(t3 - t0) if (do_time and t0 is not None and t3 is not None) else None),
        }

        self._step += 1
        return None, extra
