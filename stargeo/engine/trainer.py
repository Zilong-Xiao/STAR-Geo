from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from tqdm import tqdm

from .losses import SymmetricContrastiveLoss


def _unwrap(model: torch.nn.Module):
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_function: SymmetricContrastiveLoss,
    device: torch.device,
    mixed_precision: bool,
    scaler: Optional[torch.amp.GradScaler],
    scheduler=None,
    clip_grad_value: Optional[float] = 100.0,
    verbose: bool = True,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(dataloader, desc="train", leave=False) if verbose else dataloader
    for query, reference, query_tsf, reference_tsf, _ in iterator:
        query = query.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)
        query_tsf = query_tsf.to(device, non_blocking=True)
        reference_tsf = reference_tsf.to(device, non_blocking=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if mixed_precision and device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
            query_descriptor, reference_descriptor = model(
                query_images=query,
                reference_images=reference,
                query_tsf=query_tsf,
                reference_tsf=reference_tsf,
            )
            loss = loss_function(
                query_descriptor,
                reference_descriptor,
                _unwrap(model).clamped_logit_scale(),
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        batch_size = int(query.shape[0])
        running_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        if verbose:
            iterator.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return running_loss / max(sample_count, 1)
