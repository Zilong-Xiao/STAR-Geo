from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class VIGORMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_1_percent: float
    hit_rate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @property
    def r1(self) -> float:
        return self.recall_at_1


def extract_descriptors(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    mixed_precision: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    descriptors = []
    labels = []
    iterator = tqdm(dataloader, desc="extract", leave=False) if verbose else dataloader

    with torch.no_grad():
        for images, tsf_ids, batch_labels in iterator:
            images = images.to(device, non_blocking=True)
            tsf_ids = tsf_ids.to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                descriptor = model(
                    query_images=images,
                    query_tsf=tsf_ids,
                )
            descriptors.append(F.normalize(descriptor.float(), dim=-1).cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(descriptors, dim=0), torch.cat(labels, dim=0)


def _build_reference_positions(reference_labels: torch.Tensor) -> Dict[int, int]:
    if reference_labels.ndim != 1:
        raise ValueError(f"Reference labels must have shape [R], got {tuple(reference_labels.shape)}")
    mapping: Dict[int, int] = {}
    for position, reference_id in enumerate(reference_labels.tolist()):
        reference_id = int(reference_id)
        if reference_id in mapping:
            raise ValueError(f"Duplicate reference ID in gallery: {reference_id}")
        mapping[reference_id] = position
    return mapping


def compute_vigor_metrics(
    query_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    device: torch.device,
    step_size: int = 1024,
) -> VIGORMetrics:
    if query_labels.ndim != 2 or query_labels.shape[1] < 1:
        raise ValueError(f"Query labels must have shape [Q, >=1], got {tuple(query_labels.shape)}")

    reference_positions = _build_reference_positions(reference_labels)
    gallery_size = int(reference_descriptors.shape[0])
    recall_ks = (1, 5, 10, max(1, gallery_size // 100))
    correct = torch.zeros(len(recall_ks), dtype=torch.float64)
    hit_count = 0

    reference_descriptors = reference_descriptors.to(device, non_blocking=True)
    query_descriptors = query_descriptors.to(device, non_blocking=True)

    for start in tqdm(range(0, len(query_descriptors), step_size), desc="score", leave=False):
        end = min(start + step_size, len(query_descriptors))
        similarities = query_descriptors[start:end] @ reference_descriptors.t()
        chunk_labels = query_labels[start:end]

        strict_positions = torch.tensor(
            [reference_positions[int(row[0])] for row in chunk_labels.tolist()],
            dtype=torch.long,
            device=device,
        )
        row_indices = torch.arange(similarities.shape[0], device=device)
        strict_scores = similarities[row_indices, strict_positions]
        ranks = (similarities > strict_scores[:, None]).sum(dim=1)
        for metric_index, k in enumerate(recall_ks):
            correct[metric_index] += (ranks < k).sum().cpu()

        top1_positions = similarities.argmax(dim=1).cpu().tolist()
        for local_index, top1_position in enumerate(top1_positions):
            valid_reference_ids = [int(value) for value in chunk_labels[local_index].tolist()]
            valid_positions = {
                reference_positions[reference_id]
                for reference_id in valid_reference_ids
                if reference_id in reference_positions
            }
            if int(top1_position) in valid_positions:
                hit_count += 1

    query_count = max(1, int(query_descriptors.shape[0]))
    percentages = (correct / query_count * 100.0).tolist()
    return VIGORMetrics(
        recall_at_1=float(percentages[0]),
        recall_at_5=float(percentages[1]),
        recall_at_10=float(percentages[2]),
        recall_at_1_percent=float(percentages[3]),
        hit_rate=float(hit_count / query_count * 100.0),
    )


def evaluate_vigor(
    model: torch.nn.Module,
    query_dataloader,
    reference_dataloader,
    device: torch.device,
    mixed_precision: bool = True,
    step_size: int = 1024,
    verbose: bool = True,
) -> VIGORMetrics:
    reference_descriptors, reference_labels = extract_descriptors(
        model,
        reference_dataloader,
        device=device,
        mixed_precision=mixed_precision,
        verbose=verbose,
    )
    query_descriptors, query_labels = extract_descriptors(
        model,
        query_dataloader,
        device=device,
        mixed_precision=mixed_precision,
        verbose=verbose,
    )
    return compute_vigor_metrics(
        query_descriptors=query_descriptors,
        reference_descriptors=reference_descriptors,
        query_labels=query_labels,
        reference_labels=reference_labels,
        device=device,
        step_size=step_size,
    )
