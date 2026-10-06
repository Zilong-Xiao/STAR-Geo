from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from stargeo.datasets import (
    FUNCTION_LABELS,
    SCALE_LABELS,
    TOPOLOGY_LABELS,
    ImageTransform,
    TSFAnnotationStore,
    VIGORIndex,
    VIGOREvalDataset,
)
from stargeo.evaluation import evaluate_vigor
from stargeo.models import STARGeoConfig, build_star_geo
from stargeo.presets import get_preset
from stargeo.utils import load_checkpoint, seed_everything, worker_seed, write_json


def build_parser(default_backbone: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a STAR-Geo checkpoint on VIGOR.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tsf-annotations", required=True)
    parser.add_argument("--setting", choices=("same", "cross"), default=None)
    parser.add_argument("--backbone", choices=("dinov3", "convnext"), default=default_backbone)
    parser.add_argument("--backbone-name", default=None)
    parser.add_argument("--clip-backbone", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--ground-cutting", type=int, default=0)
    parser.add_argument("--score-step-size", type=int, default=1024)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON path for metrics.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(default_backbone: str = "dinov3", argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(default_backbone).parse_args(argv)
    seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if "star_geo_config" in checkpoint:
        model_config = STARGeoConfig(**checkpoint["star_geo_config"])
    else:
        model_config = get_preset(args.backbone).star_geo
    if args.backbone_name:
        model_config = replace(model_config, backbone_name=args.backbone_name)
    model_config = replace(model_config, pretrained_backbone=False)

    experiment_config = checkpoint.get("experiment_config", {})
    setting = args.setting or experiment_config.get("setting", "same")
    clip_backbone = args.clip_backbone or experiment_config.get("clip_backbone", "ViT-L/14")
    vocab = checkpoint.get("vocabularies", {})
    topology_labels = tuple(vocab.get("topology", TOPOLOGY_LABELS))
    scale_labels = tuple(vocab.get("scale", SCALE_LABELS))
    function_labels = tuple(vocab.get("function", FUNCTION_LABELS))

    print(f"Device: {device}")
    print(f"Backbone: {model_config.backbone_name}")
    print(f"Setting: VIGOR-{str(setting).capitalize()}")
    model = build_star_geo(
        config=model_config,
        clip_backbone=clip_backbone,
        device=device,
        topology_labels=topology_labels,
        scale_labels=scale_labels,
        function_labels=function_labels,
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=not args.non_strict)
    if args.non_strict:
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")
    model = model.to(device)

    data_config = model.get_data_config()
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    query_transform = ImageTransform(
        model_config.image_size,
        mean,
        std,
        ground_cutting=args.ground_cutting,
    )
    reference_transform = ImageTransform(model_config.image_size, mean, std)

    annotation_store = TSFAnnotationStore(args.tsf_annotations)
    test_index = VIGORIndex(args.data_root, setting, "test", annotation_store)
    query_dataset = VIGOREvalDataset(
        data_root=args.data_root,
        annotation_path=annotation_store,
        setting=setting,
        split="test",
        view="ground",
        transform=query_transform,
        index=test_index,
    )
    reference_dataset = VIGOREvalDataset(
        data_root=args.data_root,
        annotation_path=annotation_store,
        setting=setting,
        split="test",
        view="aerial",
        transform=reference_transform,
        index=test_index,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
    )

    metrics = evaluate_vigor(
        model=model,
        query_dataloader=query_loader,
        reference_dataloader=reference_loader,
        device=device,
        mixed_precision=args.mixed_precision,
        step_size=args.score_step_size,
        verbose=not args.quiet,
    )
    print(
        f"R@1={metrics.recall_at_1:.4f} R@5={metrics.recall_at_5:.4f} "
        f"R@10={metrics.recall_at_10:.4f} R@1%={metrics.recall_at_1_percent:.4f} "
        f"Hit={metrics.hit_rate:.4f}"
    )
    if args.output:
        write_json(
            args.output,
            {
                "checkpoint": str(Path(args.checkpoint)),
                "setting": setting,
                "metrics": metrics.to_dict(),
            },
        )


if __name__ == "__main__":
    main()
