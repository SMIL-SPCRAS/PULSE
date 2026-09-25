import os
import random
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm

from models.multitask_model import MultiTaskTransformerModel, TransformerMTLConfig
from mtl.ntkmtl import NTKMTL
from mtl.fairgrad import FairGrad
from mtl.stch import STCH
from mtl.autolambda import AutoLambda
from data.dataloaders import make_dataloader

from metrics.metrics import (
    evaluate_dataset,
    aggregate_multidataset,
    process_emotions,
    mf1, uar,
    acc_func, ccc,
    mf1_ah, uar_ah
)

def _fmt(v, ndigits=4):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return str(v)

def pretty_print_train(epoch, num_epochs, loss, metrics: dict):
    print()
    print(f"Train loss: {_fmt(loss)}")

    mF1 = metrics.get("mF1")
    mUAR = metrics.get("mUAR")
    mF1_resd = metrics.get("mF1_RESD")
    mUAR_resd = metrics.get("mUAR_RESD")
    acc = metrics.get("ACC")
    ccc_val = metrics.get("CCC")
    mf1_ah_val = metrics.get("MF1_AH")
    uar_ah_val = metrics.get("UAR_AH")

    print(f"    • CMU-MOSEI: mF1={_fmt(mF1)}, mUAR={_fmt(mUAR)}")
    print(f"    • RESD     : mF1={_fmt(mF1_resd)}, mUAR={_fmt(mUAR_resd)}")
    print(f"    • FIV2     : ACC={_fmt(acc)}, CCC={_fmt(ccc_val)}")
    print(f"    • BAH      : MF1_AH={_fmt(mf1_ah_val)}, UAR_AH={_fmt(uar_ah_val)}")
    print()

def pretty_print_eval(metrics: dict, split_name: str = "Test"):
    print(f"{split_name} metrics:")

    mean_all = metrics.get("mean_all")

    print("  ─ Aggregated:")
    print(f"    • mean_all = {_fmt(mean_all)}")

    if "by_dataset" in metrics:
        print("  ─ Per-dataset raw:")
        for ds in metrics["by_dataset"]:
            name = ds.get("name", "unknown")
            kv = ", ".join(f"{k}={_fmt(v)}" for k, v in ds.items() if k != "name")
            print(f"    • {name}: {kv}")
    print()

def pretty_print_test(metrics: dict):

    pretty_print_eval(metrics, "Test")

def _task_order():
    return ["mosei", "resd", "fiv2", "bah"]

def _active_task_indices_from_loaders(test_loaders: Dict[str, DataLoader]):
    order = _task_order()
    idx = []
    for i, name in enumerate(order):
        if name in test_loaders:
            idx.append(i)
    if len(idx) == 0:
        raise ValueError("No active tasks inferred from test_loaders.")
    return idx

def _cfg_fingerprint(cfg, active_datasets):
    keys = [
        "encoder_type",
        "zeros_use_norm",
        "d_model",
        "n_heads",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "max_len",
        "mamba_d_state",
        "mamba_d_conv",
        "mamba_expand",
        "mhla_chunk_size",
        "mhla_qk_norm",
        "mhla_transform",
        "mhla_local_thres",
        "mhla_exp_sigma",
        "lr",
        "encoder_lr_mult",
        "heads_lr_mult",
        "label_smoothing",
        "regression_loss_type",
        "mt_weight_method",
        "mt_max_norm",
        "ntk_exp",
        "group_decay",
        "group_regroup_interval",
        "group_affinity_threshold",
        "fairgrad_alpha",
        "stch_mu",
        "stch_warmup_epoch",
        "autolambda_init",
        "autolambda_lr",
        "autolambda_virtual_lr",
        "autolambda_hessian_eps",
        "autolambda_batch_size",
        "autolambda_steps_per_epoch",
        "trace_warmup_ratio",
        "trace_conflict_ratio",
        "trace_refine_ratio",
        "trace_recovery_shared_mult",
        "trace_recovery_head_mult",
        "trace_constraint_eta",
        "trace_constraint_delta",
        "trace_constraint_max_lambda",
        "trace_constraint_weight_scale",
        "trace_constraint_trigger_lambda",
        "trace_constraint_decay",
        "trace_constraint_normalize_weights",
        "batch_size",
        "eval_bs",
        "num_epochs",
        "selection_metric",
    ]
    d = {"active_datasets": list(active_datasets)}
    for k in keys:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            try:
                json.dumps(v)
                d[k] = v
            except Exception:
                d[k] = str(v)
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _get_run_dir(cfg, active_datasets):
    base = Path(getattr(cfg, "checkpoint_dir", "./checkpoints"))
    base.mkdir(parents=True, exist_ok=True)

    existing = getattr(cfg, "_run_dir", None)
    if existing:
        p = Path(existing)
        p.mkdir(parents=True, exist_ok=True)
        return p

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = _cfg_fingerprint(cfg, active_datasets)
    name = f"{ts}_{fp}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg._run_dir = str(run_dir)

    meta = {"run_name": name, "active_datasets": list(active_datasets), "fingerprint": fp}
    meta_path = run_dir / "run_meta.json"
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return run_dir

class _CriterionView(nn.Module):
    def __init__(self, base: nn.Module, task_idx: List[int]):
        super().__init__()
        self.base = base
        self.task_idx = list(task_idx)

    def forward(self, outputs, batch, return_per_task: bool = False):
        losses = self.base(outputs, batch, return_per_task=True)
        losses = losses[self.task_idx]
        if return_per_task:
            return losses
        return losses.sum()

def _batch_active_mask(batch, device, task_idx: Optional[List[int]] = None):
    flags = [
        bool(batch["has_emo_mosei"].any().item()),
        bool(batch["has_emo_resd"].any().item()),
        bool(batch["has_pers"].any().item()),
        bool(batch["has_ah"].any().item()),
    ]
    mask = torch.tensor(flags, dtype=torch.bool, device=device)
    if task_idx is not None:
        idx = torch.tensor(task_idx, dtype=torch.long, device=device)
        mask = mask.index_select(0, idx)
    return mask

def _dataset_scalar_from_entry(name: str, entry: dict):
    if name == "mosei":
        vals = [entry.get("mF1"), entry.get("mUAR")]
    elif name == "resd":
        vals = [entry.get("mF1_RESD"), entry.get("mUAR_RESD")]
    elif name == "fiv2":
        vals = [entry.get("ACC"), entry.get("CCC")]
    elif name == "bah":
        vals = [entry.get("MF1_AH"), entry.get("UAR_AH")]
    else:
        vals = []
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))

class PhaseController:
    def __init__(self, cfg):
        self.enabled = getattr(cfg, "mt_weight_method", "none") == "trace"
        self.warmup_ratio = float(getattr(cfg, "trace_warmup_ratio", 0.2))
        self.conflict_ratio = float(getattr(cfg, "trace_conflict_ratio", 0.7))
        self.refine_ratio = float(getattr(cfg, "trace_refine_ratio", 0.9))
        self.recovery_shared_mult = float(getattr(cfg, "trace_recovery_shared_mult", 0.7))
        self.recovery_head_mult = float(getattr(cfg, "trace_recovery_head_mult", 1.15))

        self.constraint_eta = float(getattr(cfg, "trace_constraint_eta", 0.5))
        self.constraint_delta = float(getattr(cfg, "trace_constraint_delta", 0.02))
        self.constraint_max_lambda = float(getattr(cfg, "trace_constraint_max_lambda", 2.0))
        self.constraint_weight_scale = float(getattr(cfg, "trace_constraint_weight_scale", 1.0))
        self.constraint_trigger_lambda = float(getattr(cfg, "trace_constraint_trigger_lambda", 0.1))
        self.constraint_decay = float(getattr(cfg, "trace_constraint_decay", 0.95))
        self.constraint_normalize_weights = bool(getattr(cfg, "trace_constraint_normalize_weights", True))

        self.best_task_scores = {}
        self.cur_task_scores = {}
        self.lambdas = {name: 0.0 for name in _task_order()}
        self._last_phase_state = None

    def _base_phase(self, epoch: int, num_epochs: int) -> str:
        progress = float(epoch) / max(1, num_epochs)
        if progress < self.warmup_ratio:
            return "warmup_shared"
        if progress < self.conflict_ratio:
            return "conflict_aware"
        if progress < self.refine_ratio:
            return "refinement"
        return "late_refinement"

    def _base_multipliers(self, base_phase: str):
        if base_phase == "warmup_shared":
            return 1.0, 1.0
        if base_phase == "conflict_aware":
            return 0.9, 1.0
        if base_phase == "refinement":
            return 0.5, 1.1
        return 0.35, 1.15

    def _constrained_task_weights(self):
        weights = {}
        for name in _task_order():
            lam = max(0.0, float(self.lambdas.get(name, 0.0)))
            weights[name] = 1.0 + self.constraint_weight_scale * lam
        if self.constraint_normalize_weights and weights:
            mean_w = float(np.mean(list(weights.values())))
            if np.isfinite(mean_w) and mean_w > 1e-12:
                weights = {name: float(w / mean_w) for name, w in weights.items()}
        return weights

    def get_phase_state(self, epoch: int, num_epochs: int, early_stop_cnt: int):
        base_phase = self._base_phase(epoch, num_epochs)
        base_shared_mult, base_heads_mult = self._base_multipliers(base_phase)

        if not self.enabled:
            phase = {
                "name": "static",
                "base_name": "static",
                "recovery_active": False,
                "shared_lr_mult": 1.0,
                "heads_lr_mult": 1.0,
                "task_loss_weights": {name: 1.0 for name in _task_order()},
                "recovery_strength": 0.0,
            }
            self._last_phase_state = phase
            return phase

        positive = [max(0.0, float(v)) for v in self.lambdas.values()]
        max_lambda = max(positive) if positive else 0.0
        recovery_active = base_phase != "warmup_shared" and max_lambda >= self.constraint_trigger_lambda
        recovery_strength = 0.0
        task_weights = {name: 1.0 for name in _task_order()}
        if recovery_active:
            recovery_strength = min(1.0, max_lambda / max(self.constraint_max_lambda, 1e-8))
            task_weights = self._constrained_task_weights()

        shared_mult = base_shared_mult
        heads_mult = base_heads_mult
        phase_name = base_phase
        if recovery_active:
            phase_name = f"{base_phase}+recovery"
            shared_mult = base_shared_mult * (1.0 - recovery_strength * (1.0 - self.recovery_shared_mult))
            heads_mult = base_heads_mult * (1.0 + recovery_strength * (self.recovery_head_mult - 1.0))

        phase = {
            "name": phase_name,
            "base_name": base_phase,
            "recovery_active": bool(recovery_active),
            "shared_lr_mult": float(shared_mult),
            "heads_lr_mult": float(heads_mult),
            "task_loss_weights": task_weights,
            "recovery_strength": float(recovery_strength),
        }
        self._last_phase_state = phase
        return phase

    def update_after_epoch(self, control_final: dict, epoch: int, num_epochs: int):
        by_dataset = {d.get("name"): d for d in control_final.get("by_dataset", [])}
        self.cur_task_scores = {}
        for name in _task_order():
            entry = by_dataset.get(name)
            if not entry:
                continue
            score = _dataset_scalar_from_entry(name, entry)
            if score is None:
                continue
            self.cur_task_scores[name] = float(score)
            self.best_task_scores[name] = max(float(score), float(self.best_task_scores.get(name, -float("inf"))))

        if self.enabled:
            for name in _task_order():
                cur = self.cur_task_scores.get(name)
                best = self.best_task_scores.get(name)
                if cur is None or best is None or not np.isfinite(best):
                    continue
                violation = max(0.0, float(best) - self.constraint_delta - float(cur))
                lam = self.lambdas.get(name, 0.0)
                lam = max(0.0, min(self.constraint_max_lambda,
                                   self.constraint_decay * lam + self.constraint_eta * violation))
                self.lambdas[name] = float(lam)

    def debug_string(self):
        if not self.enabled:
            return "disabled"
        lam_s = ", ".join(f"{k}={self.lambdas.get(k, 0.0):.3f}" for k in _task_order())
        w_map = self._constrained_task_weights()
        w_s = ", ".join(f"{k}={w_map.get(k, 1.0):.3f}" for k in _task_order())
        return f"lambdas: {lam_s} | weights: {w_s} | normalize={self.constraint_normalize_weights}"

def _apply_optimizer_phase_lrs(optimizer, base_encoder_lr: float, base_heads_lr: float, phase_state: dict):
    if len(optimizer.param_groups) == 0:
        return
    shared_mult = float(phase_state.get("shared_lr_mult", 1.0))
    heads_mult = float(phase_state.get("heads_lr_mult", 1.0))
    optimizer.param_groups[0]["lr"] = base_encoder_lr * shared_mult
    for pg in optimizer.param_groups[1:]:
        pg["lr"] = base_heads_lr * heads_mult

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        diff = pred - tgt
        return torch.log(torch.cosh(diff)).mean()

class MultiTaskLoss(nn.Module):

    def __init__(
            self,
            label_smoothing: float = 0.0,
            regression_loss_type: str = "mse",
    ):
        super().__init__()

        rtype = regression_loss_type.lower()
        if rtype == "mse":
            self.regression_loss = nn.MSELoss(reduction="mean")
        elif rtype == "smoothl1":
            self.regression_loss = nn.SmoothL1Loss(beta=1.0, reduction="mean")
        elif rtype == "logcosh":
            self.regression_loss = LogCoshLoss()
        else:
            raise ValueError(f"Unknown regression_loss_type: {regression_loss_type}")

        self.ce_resd = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)
        self.ce_ah = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(self, outputs, batch, return_per_task: bool = False):
        device = outputs["emotion_mosei_pred"].device

        loss_emo_m = torch.tensor(0.0, device=device)
        loss_emo_r = torch.tensor(0.0, device=device)
        loss_pers = torch.tensor(0.0, device=device)
        loss_ah = torch.tensor(0.0, device=device)

        if batch["has_emo_mosei"].any():
            pred = outputs["emotion_mosei_pred"][batch["has_emo_mosei"]]
            tgt = batch["emotion_mosei"][batch["has_emo_mosei"]]
            loss_emo_m = self.regression_loss(pred, tgt)

        if batch["has_emo_resd"].any():
            pred = outputs["emotion_resd_logits"][batch["has_emo_resd"]]
            tgt = batch["emotion_resd"][batch["has_emo_resd"]]
            loss_emo_r = self.ce_resd(pred, tgt)

        if batch["has_pers"].any():
            pred = outputs["personality_preds"][batch["has_pers"]]
            tgt = batch["personality"][batch["has_pers"]]
            loss_pers = self.regression_loss(pred, tgt)

        if batch["has_ah"].any():
            pred = outputs["ah_logits"][batch["has_ah"]]
            tgt = batch["ah"][batch["has_ah"]]
            loss_ah = self.ce_ah(pred, tgt)

        losses_vec = torch.stack([loss_emo_m, loss_emo_r, loss_pers, loss_ah])

        if return_per_task:
            return losses_vec

        return losses_vec.sum()

class SelectiveTaskGroupUpdater:

    def __init__(
            self,
            n_tasks: int,
            beta: float = 1e-3,
            regroup_interval: int = 50,
            affinity_threshold: float = 0.0,
            device: torch.device = torch.device("cpu"),
    ):
        self.n_tasks = n_tasks
        self.beta = beta
        self.regroup_interval = regroup_interval
        self.affinity_threshold = affinity_threshold
        self.device = device

        self.B = torch.zeros(n_tasks, n_tasks, dtype=torch.float32, device=device)
        self.groups: List[List[int]] = [[i] for i in range(n_tasks)]
        self.step_idx = 0

    def _update_affinity(self, group: List[int], base_losses: torch.Tensor, new_losses: torch.Tensor):
        eps = 1e-8
        K = self.n_tasks
        with torch.no_grad():
            for j in range(K):
                if base_losses[j].item() <= 0.0:
                    continue
                ratio = new_losses[j] / (base_losses[j] + eps)
                b_val = 1.0 - ratio
                for i in group:
                    old = self.B[i, j]
                    self.B[i, j] = (1.0 - self.beta) * old + self.beta * b_val

    def _recompute_groups(self):
        K = self.n_tasks
        visited = [False] * K
        new_groups: List[List[int]] = []

        with torch.no_grad():
            for i in range(K):
                if visited[i]:
                    continue
                group = [i]
                visited[i] = True
                stack = [i]
                while stack:
                    u = stack.pop()
                    for v in range(K):
                        if visited[v]:
                            continue
                        if self.B[u, v] > self.affinity_threshold and self.B[v, u] > self.affinity_threshold:
                            visited[v] = True
                            stack.append(v)
                            group.append(v)
                new_groups.append(group)

        if len(new_groups) == 0:
            new_groups = [[i] for i in range(K)]

        self.groups = new_groups

    def step(self, model, optimizer, criterion, batch, device, task_weights: Optional[torch.Tensor] = None):
        self.step_idx += 1
        model.train()

        has_any = (
                batch["has_emo_mosei"].any()
                or batch["has_emo_resd"].any()
                or batch["has_pers"].any()
                or batch["has_ah"].any()
        )
        if not has_any:
            with torch.no_grad():
                out = model(batch)
                losses_vec = criterion(out, batch, return_per_task=True)
                if task_weights is not None and task_weights.numel() == losses_vec.numel():
                    losses_vec = losses_vec * task_weights
                total_loss_scalar = float(losses_vec.sum().item())
            return total_loss_scalar, out

        group_order = list(range(len(self.groups)))
        random.shuffle(group_order)

        last_out = None
        last_total_loss = 0.0

        for g_idx in group_order:
            group = self.groups[g_idx]

            optimizer.zero_grad()
            out = model(batch)
            losses_vec = criterion(out, batch, return_per_task=True)
            if task_weights is not None and task_weights.numel() == losses_vec.numel():
                losses_vec = losses_vec * task_weights
            base_losses = losses_vec.detach().clone()

            group_indices = torch.tensor(group, dtype=torch.long, device=losses_vec.device)
            group_loss = losses_vec.index_select(0, group_indices).sum()

            if group_loss.item() == 0.0:
                continue

            group_loss.backward()
            optimizer.step()

            with torch.no_grad():
                out_new = model(batch)
                new_losses_vec = criterion(out_new, batch, return_per_task=True)
                if task_weights is not None and task_weights.numel() == new_losses_vec.numel():
                    new_losses_vec = new_losses_vec * task_weights

            self._update_affinity(group, base_losses, new_losses_vec.detach().clone())

            last_out = out_new
            last_total_loss = float(new_losses_vec.sum().item())

        if last_out is None:
            with torch.no_grad():
                out = model(batch)
                losses_vec = criterion(out, batch, return_per_task=True)
                if task_weights is not None and task_weights.numel() == losses_vec.numel():
                    losses_vec = losses_vec * task_weights
                last_total_loss = float(losses_vec.sum().item())
            last_out = out

        if self.regroup_interval > 0 and (self.step_idx % self.regroup_interval == 0):
            self._recompute_groups()

        return last_total_loss, last_out

_TASK_TO_GLOBAL_IDX = {
    "mosei": 0,
    "resd": 1,
    "fiv2": 2,
    "bah": 3,
}

def _canonical_dataset_name_from_dataset(dataset) -> str:
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    raw = str(getattr(base, "dataset_name", "")).lower()
    aliases = {
        "cmu_mosei": "mosei",
        "mosei": "mosei",
        "resd": "resd",
        "fiv2": "fiv2",
        "bah": "bah",
    }
    if raw not in aliases:
        raise ValueError(f"Cannot infer task name from dataset_name={raw!r}.")
    return aliases[raw]

def _build_autolambda_train_loaders(train_loader: DataLoader, task_names: List[str], cfg):

    concat = train_loader.dataset
    if not isinstance(concat, ConcatDataset):
        raise TypeError(
            "Auto-Lambda expects the PULSE train loader to wrap a ConcatDataset. "
            f"Got {type(concat).__name__}."
        )

    dataset_map = {}
    for dataset in concat.datasets:
        name = _canonical_dataset_name_from_dataset(dataset)
        dataset_map[name] = dataset

    missing = [name for name in task_names if name not in dataset_map]
    if missing:
        raise ValueError(f"Auto-Lambda could not find train datasets for tasks: {missing}")

    batch_size = int(getattr(cfg, "autolambda_batch_size", 0) or getattr(cfg, "batch_size", 16))
    if batch_size <= 0:
        raise ValueError("autolambda_batch_size must be > 0, or 0 to reuse cfg.batch_size.")

    loaders = {}
    for name in task_names:
        loaders[name] = make_dataloader(
            dataset_map[name],
            batch_size=batch_size,
            shuffle=True,
            num_workers=getattr(train_loader, "num_workers", 4),
            pin_memory=getattr(train_loader, "pin_memory", True),
        )
    return loaders

class _CyclingLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

def _task_loss_vector(
    model: nn.Module,
    batches: Dict[str, dict],
    task_names: List[str],
    criterion_full: MultiTaskLoss,
    capture_outputs: Optional[dict] = None,
) -> torch.Tensor:
    losses = []
    for name in task_names:
        batch = batches[name]
        out = model(batch)
        full_losses = criterion_full(out, batch, return_per_task=True)
        losses.append(full_losses[_TASK_TO_GLOBAL_IDX[name]])
        if capture_outputs is not None:
            capture_outputs[name] = out
    return torch.stack(losses)

def _collect_task_train_metrics(
    task_names: List[str],
    batches: Dict[str, dict],
    outputs: Dict[str, dict],
    accum: dict,
):
    if "mosei" in task_names:
        batch = batches["mosei"]
        out = outputs["mosei"]
        pred = out["emotion_mosei_pred"][batch["has_emo_mosei"]]
        tgt = batch["emotion_mosei"][batch["has_emo_mosei"]]
        p, t = process_emotions(pred, tgt)
        accum["emo_preds"].extend(p)
        accum["emo_tgts"].extend(t)

    if "resd" in task_names:
        batch = batches["resd"]
        out = outputs["resd"]
        pred = out["emotion_resd_logits"][batch["has_emo_resd"]].argmax(dim=1)
        tgt = batch["emotion_resd"][batch["has_emo_resd"]]
        accum["resd_preds"].extend(pred.detach().cpu().numpy())
        accum["resd_tgts"].extend(tgt.detach().cpu().numpy())

    if "fiv2" in task_names:
        batch = batches["fiv2"]
        out = outputs["fiv2"]
        pred = out["personality_preds"][batch["has_pers"]].detach().cpu().numpy()
        tgt = batch["personality"][batch["has_pers"]].detach().cpu().numpy()
        accum["pkl_preds"].append(pred)
        accum["pkl_tgts"].append(tgt)

    if "bah" in task_names:
        batch = batches["bah"]
        out = outputs["bah"]
        pred = out["ah_logits"][batch["has_ah"]].argmax(dim=1)
        tgt = batch["ah"][batch["has_ah"]]
        accum["ah_preds"].extend(pred.detach().cpu().numpy())
        accum["ah_tgts"].extend(tgt.detach().cpu().numpy())

def _finalize_task_train_metrics(accum: dict) -> dict:
    results = {}
    if accum["emo_tgts"]:
        emo_tgt = np.asarray(accum["emo_tgts"])
        emo_prd = np.asarray(accum["emo_preds"])
        results["mF1"] = mf1(emo_tgt, emo_prd)
        results["mUAR"] = uar(emo_tgt, emo_prd)

    if accum["resd_tgts"]:
        results["mF1_RESD"] = mf1_ah(
            np.asarray(accum["resd_tgts"]), np.asarray(accum["resd_preds"])
        )
        results["mUAR_RESD"] = uar_ah(
            np.asarray(accum["resd_tgts"]), np.asarray(accum["resd_preds"])
        )

    if accum["pkl_tgts"]:
        tgt = np.vstack(accum["pkl_tgts"])
        prd = np.vstack(accum["pkl_preds"])
        results["ACC"] = acc_func(tgt, prd)
        ccs = []
        for i in range(tgt.shape[1]):
            mask = ~np.isnan(tgt[:, i])
            if mask.sum() > 0:
                ccs.append(ccc(tgt[mask, i], prd[mask, i]))
        if ccs:
            results["CCC"] = float(np.mean(ccs))

    if accum["ah_tgts"]:
        results["MF1_AH"] = mf1_ah(
            np.asarray(accum["ah_tgts"]), np.asarray(accum["ah_preds"])
        )
        results["UAR_AH"] = uar_ah(
            np.asarray(accum["ah_tgts"]), np.asarray(accum["ah_preds"])
        )
    return results

def train_one_epoch_autolambda(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion_full: MultiTaskLoss,
    device: torch.device,
    autolambda_method: AutoLambda,
    train_task_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    task_names: List[str],
    steps_per_epoch: int = 0,
):

    model.train()

    missing_val = [name for name in task_names if name not in val_loaders]
    if missing_val:
        raise ValueError(
            "Auto-Lambda requires validation/control data for every active task. "
            f"Missing: {missing_val}."
        )

    train_cycles = {name: _CyclingLoader(train_task_loaders[name]) for name in task_names}
    val_cycles = {name: _CyclingLoader(val_loaders[name]) for name in task_names}

    official_like_steps = max(len(train_task_loaders[name]) for name in task_names)
    num_steps = int(steps_per_epoch) if int(steps_per_epoch) > 0 else official_like_steps

    total_loss = 0.0
    accum = {
        "emo_preds": [], "emo_tgts": [],
        "resd_preds": [], "resd_tgts": [],
        "pkl_preds": [], "pkl_tgts": [],
        "ah_preds": [], "ah_tgts": [],
    }

    for _ in tqdm(range(num_steps), desc="Train/AutoLambda", leave=False):
        train_batches = {
            name: _move_batch_to_device(train_cycles[name].next(), device)
            for name in task_names
        }
        val_batches = {
            name: _move_batch_to_device(val_cycles[name].next(), device)
            for name in task_names
        }

        def train_loss_fn(current_model):
            return _task_loss_vector(
                current_model,
                train_batches,
                task_names,
                criterion_full,
            )

        def val_loss_fn(current_model):
            return _task_loss_vector(
                current_model,
                val_batches,
                task_names,
                criterion_full,
            )

        autolambda_method.zero_meta_grad()
        meta_loss, virtual_lr, fd_eps = autolambda_method.unrolled_backward(
            train_loss_fn=train_loss_fn,
            val_loss_fn=val_loss_fn,
            model_optimizer=optimizer,
        )
        autolambda_method.meta_step()

        optimizer.zero_grad(set_to_none=True)
        captured_train_outputs = {}
        train_losses = _task_loss_vector(
            model,
            train_batches,
            task_names,
            criterion_full,
            capture_outputs=captured_train_outputs,
        )
        weighted_train_loss = torch.sum(
            autolambda_method.weights_tensor() * train_losses
        )
        if not torch.isfinite(train_losses).all() or not torch.isfinite(weighted_train_loss):
            raise RuntimeError(
                "Auto-Lambda real train loss became non-finite: "
                f"losses={train_losses.detach().cpu().tolist()}, "
                f"weights={autolambda_method.weights_numpy().tolist()}"
            )
        weighted_train_loss.backward()
        optimizer.step()

        train_loss_scalar = float(weighted_train_loss.detach().item())
        total_loss += train_loss_scalar
        autolambda_method.record_step(
            meta_loss=meta_loss,
            train_loss=train_loss_scalar,
            virtual_lr=virtual_lr,
            hessian_eps=fd_eps,
        )

        _collect_task_train_metrics(
            task_names=task_names,
            batches=train_batches,
            outputs=captured_train_outputs,
            accum=accum,
        )

    return total_loss / max(1, num_steps), _finalize_task_train_metrics(accum)

def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        weight_method=None,
        group_method: Optional[SelectiveTaskGroupUpdater] = None,
        active_task_idx: Optional[List[int]] = None,
        phase_state: Optional[dict] = None,
        epoch: int = 0,
):
    model.train()

    total_loss = 0.0
    n_samples = 0

    emo_preds, emo_tgts = [], []
    resd_preds, resd_tgts = [], []
    pkl_preds, pkl_tgts = [], []
    ah_preds, ah_tgts = [], []

    shared_params = list(model.encoder.parameters())

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        task_weights_map = (phase_state or {}).get("task_loss_weights", {name: 1.0 for name in _task_order()})
        if active_task_idx is None:
            active_task_names = _task_order()
        else:
            active_task_names = [_task_order()[i] for i in active_task_idx]
        task_weights = torch.tensor(
            [float(task_weights_map.get(name, 1.0)) for name in active_task_names],
            device=device,
            dtype=torch.float32,
        )

        if group_method is not None:
            loss_scalar, out = group_method.step(model, optimizer, criterion, batch, device, task_weights=task_weights)
        else:
            optimizer.zero_grad()
            out = model(batch)
            losses_vec = criterion(out, batch, return_per_task=True)
            scaled_losses_vec = losses_vec * task_weights.to(losses_vec.dtype)

            if weight_method is None:
                loss = scaled_losses_vec.sum()
                loss.backward()
                optimizer.step()
                loss_scalar = float(loss.item())
            elif isinstance(weight_method, STCH):
                active_mask = _batch_active_mask(
                    batch,
                    device=device,
                    task_idx=active_task_idx,
                )
                ret = weight_method.backward(
                    losses=scaled_losses_vec,
                    active_mask=active_mask,
                    epoch=epoch,
                )
                optimizer.step()
                loss_scalar = float(ret["loss"])
            else:
                heads = [
                    model.emo_mosei_head,
                    model.emo_resd_head,
                    model.personality_head,
                    model.ah_head,
                ]
                task_params = []
                if active_task_idx is None:
                    active_task_idx_local = list(range(len(heads)))
                else:
                    active_task_idx_local = list(active_task_idx)
                for i in active_task_idx_local:
                    task_params.extend(list(heads[i].parameters()))

                ret = weight_method.backward(
                    losses=scaled_losses_vec,
                    shared_parameters=shared_params,
                    task_specific_parameters=task_params,
                    last_shared_parameters=None,
                    representation=None,
                )

                with torch.no_grad():
                    loss = scaled_losses_vec.sum()
                    if isinstance(ret, (np.ndarray, list)) and not isinstance(ret, tuple):
                        w = torch.as_tensor(ret, device=scaled_losses_vec.device, dtype=scaled_losses_vec.dtype)
                        if w.numel() == scaled_losses_vec.numel():
                            loss = (scaled_losses_vec * w).sum()
                    elif isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict) and "weights" in ret[1]:
                        w = torch.as_tensor(ret[1]["weights"], device=scaled_losses_vec.device,
                                            dtype=scaled_losses_vec.dtype)
                        if w.numel() == scaled_losses_vec.numel():
                            loss = (scaled_losses_vec * w).sum()

                optimizer.step()
                loss_scalar = float(loss.item())

        bs = batch["x"].shape[0]
        total_loss += loss_scalar * bs
        n_samples += bs

        if batch["has_emo_mosei"].any():
            pred = out["emotion_mosei_pred"][batch["has_emo_mosei"]]
            tgt = batch["emotion_mosei"][batch["has_emo_mosei"]]
            p, t = process_emotions(pred, tgt)
            emo_preds.extend(p)
            emo_tgts.extend(t)

        if batch["has_emo_resd"].any():
            pred = out["emotion_resd_logits"][batch["has_emo_resd"]].argmax(dim=1)
            tgt = batch["emotion_resd"][batch["has_emo_resd"]]
            resd_preds.extend(pred.cpu().numpy())
            resd_tgts.extend(tgt.cpu().numpy())

        if batch["has_pers"].any():
            pred = out["personality_preds"][batch["has_pers"]].detach().cpu().numpy()
            tgt = batch["personality"][batch["has_pers"]].cpu().numpy()
            pkl_preds.append(pred)
            pkl_tgts.append(tgt)

        if batch["has_ah"].any():
            pred = out["ah_logits"][batch["has_ah"]].argmax(dim=1)
            tgt = batch["ah"][batch["has_ah"]]
            ah_preds.extend(pred.cpu().numpy())
            ah_tgts.extend(tgt.cpu().numpy())

    results = {}

    if emo_tgts:
        emo_tgt = np.asarray(emo_tgts)
        emo_prd = np.asarray(emo_preds)
        results["mF1"] = mf1(emo_tgt, emo_prd)
        results["mUAR"] = uar(emo_tgt, emo_prd)

    if resd_tgts:
        results["mF1_RESD"] = mf1_ah(np.array(resd_tgts), np.array(resd_preds))
        results["mUAR_RESD"] = uar_ah(np.array(resd_tgts), np.array(resd_preds))

    if pkl_tgts:
        tgt = np.vstack(pkl_tgts)
        prd = np.vstack(pkl_preds)
        results["ACC"] = acc_func(tgt, prd)
        ccs = []
        for i in range(tgt.shape[1]):
            mask = ~np.isnan(tgt[:, i])
            if mask.sum() > 0:
                ccs.append(ccc(tgt[mask, i], prd[mask, i]))
        if ccs:
            results["CCC"] = float(np.mean(ccs))

    if ah_tgts:
        results["MF1_AH"] = mf1_ah(np.array(ah_tgts), np.array(ah_preds))
        results["UAR_AH"] = uar_ah(np.array(ah_tgts), np.array(ah_preds))

    train_loss = total_loss / max(1, n_samples)
    return train_loss, results

def train_model(
    cfg,
    train_loader: DataLoader,
    test_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
):
    device = torch.device(cfg.device)

    model_cfg = TransformerMTLConfig(
        in_dim=1024,
        encoder_type=getattr(cfg, "encoder_type", "transformer"),
        zeros_use_norm=getattr(cfg, "zeros_use_norm", True),
        d_model=getattr(cfg, "d_model", 256),
        n_heads=getattr(cfg, "n_heads", 4),
        num_layers=getattr(cfg, "num_layers", 4),
        dim_feedforward=getattr(cfg, "dim_feedforward", 1024),
        dropout=getattr(cfg, "dropout", 0.1),
        max_len=getattr(cfg, "max_len", 5000),
        mamba_d_state=getattr(cfg, "mamba_d_state", 16),
        mamba_d_conv=getattr(cfg, "mamba_d_conv", 3),
        mamba_expand=getattr(cfg, "mamba_expand", 2),
        mhla_chunk_size=getattr(cfg, "mhla_chunk_size", 64),
        mhla_qk_norm=getattr(cfg, "mhla_qk_norm", True),
        mhla_transform=getattr(cfg, "mhla_transform", "linear"),
        mhla_local_thres=getattr(cfg, "mhla_local_thres", 1.5),
        mhla_exp_sigma=getattr(cfg, "mhla_exp_sigma", 3.0),
        emo_mosei_out_dim=7,
        emo_resd_out_dim=7,
        pers_out_dim=5,
        ah_out_dim=2,
    )

    model = MultiTaskTransformerModel(model_cfg).to(device)

    criterion_full = MultiTaskLoss(
        label_smoothing=getattr(cfg, "label_smoothing", 0.0),
        regression_loss_type=getattr(cfg, "regression_loss_type", "mse"),
    )

    if not val_loaders:
        raise ValueError("Validation/control loaders are required for training.")

    selection_source = "validation"
    active_task_idx = _active_task_indices_from_loaders(val_loaders)
    active_datasets = [n for n in _task_order() if n in val_loaders]
    criterion = _CriterionView(criterion_full, active_task_idx)

    base_lr = float(getattr(cfg, "lr", 1e-4))
    enc_mult = float(getattr(cfg, "encoder_lr_mult", 1.0))
    head_mult = float(getattr(cfg, "heads_lr_mult", 3.0))
    base_encoder_lr = base_lr * enc_mult
    base_heads_lr = base_lr * head_mult

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": base_encoder_lr},
            {"params": model.emo_mosei_head.parameters(), "lr": base_heads_lr},
            {"params": model.emo_resd_head.parameters(), "lr": base_heads_lr},
            {"params": model.personality_head.parameters(), "lr": base_heads_lr},
            {"params": model.ah_head.parameters(), "lr": base_heads_lr},
        ]
    )

    weight_method = None
    group_method = None
    autolambda_method = None
    autolambda_train_loaders = None
    mt_method = getattr(cfg, "mt_weight_method", "none")

    if mt_method == "ntkmtl":
        weight_method = NTKMTL(
            n_tasks=len(active_task_idx),
            device=device,
            max_norm=getattr(cfg, "mt_max_norm", 1.0),
            ntk_exp=getattr(cfg, "ntk_exp", 0.5),
        )
    elif mt_method == "taskgroup":
        group_method = SelectiveTaskGroupUpdater(
            n_tasks=len(active_task_idx),
            beta=getattr(cfg, "group_decay", 1e-3),
            regroup_interval=getattr(cfg, "group_regroup_interval", 50),
            affinity_threshold=getattr(cfg, "group_affinity_threshold", 0.0),
            device=device,
        )
    elif mt_method == "fairgrad":
        weight_method = FairGrad(
            n_tasks=len(active_task_idx),
            device=device,
            alpha=getattr(cfg, "fairgrad_alpha", 1.0),
            max_norm=getattr(cfg, "mt_max_norm", 1.0),
        )
    elif mt_method == "stch":
        weight_method = STCH(
            n_tasks=len(active_task_idx),
            device=device,
            mu=getattr(cfg, "stch_mu", 1.0),
            warmup_epoch=getattr(cfg, "stch_warmup_epoch", 4),
        )
    elif mt_method in {"autolambda", "auto_lambda", "autol"}:
        autolambda_method = AutoLambda(
            model=model,
            n_tasks=len(active_task_idx),
            device=device,
            weight_init=getattr(cfg, "autolambda_init", 0.1),
            meta_lr=getattr(cfg, "autolambda_lr", 1e-4),
            virtual_lr=getattr(cfg, "autolambda_virtual_lr", 0.0),
            hessian_eps_scale=getattr(cfg, "autolambda_hessian_eps", 0.01),
        )
        autolambda_train_loaders = _build_autolambda_train_loaders(
            train_loader, active_datasets, cfg
        )

    phase_controller = PhaseController(cfg)

    best_score = -float("inf")
    best_selection = {}
    best_validation = None
    best_test = None
    best_checkpoint_path = None
    best_metrics_path = None

    run_dir = _get_run_dir(cfg, active_datasets)
    try:
        cfg_dict = {
            k: getattr(cfg, k)
            for k in dir(cfg)
            if not k.startswith("_") and not callable(getattr(cfg, k))
        }
        (run_dir / "config_snapshot.json").write_text(
            json.dumps(cfg_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    early_stop_cnt = 0

    for epoch in range(cfg.num_epochs):
        phase_state = phase_controller.get_phase_state(epoch, cfg.num_epochs, early_stop_cnt)
        _apply_optimizer_phase_lrs(optimizer, base_encoder_lr, base_heads_lr, phase_state)

        print(f"\n====== EPOCH {epoch + 1}/{cfg.num_epochs} ======")
        print(
            f"Phase: {phase_state['name']} | encoder_lr={optimizer.param_groups[0]['lr']:.6g} | "
            f"heads_lr={optimizer.param_groups[1]['lr']:.6g}"
        )
        if phase_controller.enabled:
            print(f"TRACE: {phase_controller.debug_string()}")

        if autolambda_method is not None:
            train_loss, train_metrics = train_one_epoch_autolambda(
                model=model,
                optimizer=optimizer,
                criterion_full=criterion_full,
                device=device,
                autolambda_method=autolambda_method,
                train_task_loaders=autolambda_train_loaders,
                val_loaders=val_loaders,
                task_names=active_datasets,
                steps_per_epoch=getattr(cfg, "autolambda_steps_per_epoch", 0),
            )
        else:
            train_loss, train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                weight_method=weight_method,
                group_method=group_method,
                active_task_idx=active_task_idx,
                phase_state=phase_state,
                epoch=epoch,
            )

        pretty_print_train(epoch, cfg.num_epochs, train_loss, train_metrics)
        if isinstance(weight_method, STCH):
            print(f"STCH: {weight_method.debug_string(active_datasets)}")
        if autolambda_method is not None:
            print(f"Auto-Lambda: {autolambda_method.debug_string(active_datasets)}")

        val_results = {
            ds: evaluate_dataset(model, loader, device)
            for ds, loader in val_loaders.items()
        }
        val_final = aggregate_multidataset(val_results)
        pretty_print_eval(val_final, "Validation/control")
        selection_final = val_final
        current_test = None

        phase_controller.update_after_epoch(selection_final, epoch, cfg.num_epochs)

        score = selection_final.get(cfg.selection_metric, -999)
        early_stop_cnt += 1

        if score > best_score:
            early_stop_cnt = 0
            best_score = score
            best_selection = selection_final
            best_validation = val_final
            best_test = current_test

            score_name = str(getattr(cfg, "selection_metric", "score"))
            safe_score = _fmt(score).replace(".", "p")
            ckpt_name = f"best_epoch{epoch + 1:03d}_{score_name}_{safe_score}.pt"
            ckpt_path = run_dir / ckpt_name
            metrics_path = run_dir / f"best_epoch{epoch + 1:03d}_metrics.json"

            payload = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
                "selection_source": selection_source,
                "metrics": selection_final,
                "selection_metrics": selection_final,
                "test_metrics": current_test,
                "validation_metrics": val_final,
                "autolambda_state": autolambda_method.state_dict() if autolambda_method is not None else None,
                "config": {
                    k: getattr(cfg, k)
                    for k in dir(cfg)
                    if not k.startswith("_") and not callable(getattr(cfg, k))
                },
            }
            torch.save(payload, ckpt_path)
            torch.save(payload, run_dir / "best_model.pt")

            metrics_payload = {
                "selection_source": selection_source,
                "metrics": selection_final,
                "selection_metrics": selection_final,
                "test_metrics": current_test,
                "validation_metrics": val_final,
            }
            metrics_path.write_text(
                json.dumps(metrics_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            best_checkpoint_path = ckpt_path
            best_metrics_path = metrics_path
            print(f"✔ Saved best model by {selection_source}:", ckpt_path)

        if early_stop_cnt >= int(getattr(cfg, "early_stop_patience", 5)):
            break

    if best_checkpoint_path is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    best_payload = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])

    test_results = {
        ds: evaluate_dataset(model, loader, device)
        for ds, loader in test_loaders.items()
    }
    best_test = aggregate_multidataset(test_results)
    pretty_print_eval(best_test, "Test at best validation checkpoint")

    best_payload["test_metrics"] = best_test
    torch.save(best_payload, run_dir / "best_model.pt")
    torch.save(best_payload, best_checkpoint_path)

    metrics_payload = {
        "selection_source": selection_source,
        "metrics": best_selection,
        "selection_metrics": best_selection,
        "test_metrics": best_test,
        "validation_metrics": best_validation,
    }
    best_metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = dict(best_selection)
    result["_selection_source"] = selection_source
    result["_test_metrics"] = best_test
    result["_validation_metrics"] = best_validation
    return result

