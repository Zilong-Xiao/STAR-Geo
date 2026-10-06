from __future__ import annotations

import argparse
import json
import os
import time
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
    UniqueReferenceBatchSampler,
    VIGOREvalDataset,
    VIGORTrainDataset,
)
from stargeo.engine import SymmetricContrastiveLoss, train_one_epoch
from stargeo.evaluation import evaluate_vigor
from stargeo.models import build_star_geo
from stargeo.presets import get_preset
from stargeo.utils import (
    cosine_scheduler_with_warmup,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    unwrap_model,
    worker_seed,
    write_json,
)


def build_parser(default_backbone: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train STAR-Geo on VIGOR with independent ground and aerial TSF annotations."
    )
    parser.add_argument("--backbone", choices=("dinov3", "convnext"), default=default_backbone)
    parser.add_argument("--backbone-name", default=None, help="Optional timm model-name override.")
    parser.add_argument("--clip-backbone", default="ViT-L/14")
    parser.add_argument("--data-root", required=True, help="VIGOR root containing ground/, satellite/, and splits/.")
    parser.add_argument("--tsf-annotations", required=True, help="Unified ground+aerial TSF JSON/JSONL file.")
    parser.add_argument("--setting", choices=("same", "cross"), default="same")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad-value", type=float, default=100.0)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--module-lr", type=float, default=None)
    parser.add_argument("--prompt-lr", type=float, default=None)

    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--rotation-probability", type=float, default=0.75)
    parser.add_argument("--ground-cutting", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-ids", default=None, help="Visible CUDA device indices for DataParallel, e.g. 0,1,2,3.")
    parser.add_argument("--score-step-size", type=int, default=1024)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _gpu_ids(text: Optional[str]) -> Sequence[int]:
    if text is None or not text.strip():
        return tuple(range(torch.cuda.device_count()))
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _print_metrics(prefix: str, metrics) -> None:
    print(
        f"{prefix}: R@1={metrics.recall_at_1:.4f} "
        f"R@5={metrics.recall_at_5:.4f} R@10={metrics.recall_at_10:.4f} "
        f"R@1%={metrics.recall_at_1_percent:.4f} Hit={metrics.hit_rate:.4f}"
    )


def main(default_backbone: str = "dinov3", argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(default_backbone).parse_args(argv)
    seed_everything(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = get_preset(args.backbone)
    model_config = preset.star_geo
    if args.backbone_name:
        model_config = replace(model_config, backbone_name=args.backbone_name)
    model_config = replace(model_config, pretrained_backbone=args.pretrained_backbone)

    backbone_lr = preset.backbone_lr if args.backbone_lr is None else args.backbone_lr
    module_lr = preset.module_lr if args.module_lr is None else args.module_lr
    prompt_lr = preset.prompt_lr if args.prompt_lr is None else args.prompt_lr

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(
        args.output_dir
        or f"checkpoints/vigor_{args.setting}/{args.backbone}/{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "arguments": vars(args),
            "star_geo_config": model_config.to_dict(),
            "learning_rates": {
                "backbone": backbone_lr,
                "prompt": prompt_lr,
                "new_modules": module_lr,
            },
        },
    )

    print(f"Device: {device}")
    print(f"Backbone: {model_config.backbone_name}")
    print(f"Setting: VIGOR-{args.setting.capitalize()}")
    print("Loading frozen CLIP text encoder and STAR-Geo...")
    model = build_star_geo(
        config=model_config,
        clip_backbone=args.clip_backbone,
        device=device,
        topology_labels=TOPOLOGY_LABELS,
        scale_labels=SCALE_LABELS,
        function_labels=FUNCTION_LABELS,
    )
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)

    data_config = model.get_data_config()
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    query_train_transform = ImageTransform(
        model_config.image_size,
        mean,
        std,
        ground_cutting=args.ground_cutting,
    )
    reference_transform = ImageTransform(model_config.image_size, mean, std)
    query_eval_transform = ImageTransform(
        model_config.image_size,
        mean,
        std,
        ground_cutting=args.ground_cutting,
    )

    print("Loading VIGOR splits and validating independent ground/aerial TSF annotations...")
    annotation_store = TSFAnnotationStore(args.tsf_annotations)
    train_dataset = VIGORTrainDataset(
        data_root=args.data_root,
        annotation_path=annotation_store,
        setting=args.setting,
        query_transform=query_train_transform,
        reference_transform=reference_transform,
        flip_probability=args.flip_probability,
        rotation_probability=args.rotation_probability,
    )
    batch_sampler = UniqueReferenceBatchSampler(
        train_dataset.reference_ids,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        seed=args.seed,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )

    test_index = VIGORIndex(args.data_root, args.setting, "test", annotation_store)
    query_dataset = VIGOREvalDataset(
        data_root=args.data_root,
        annotation_path=annotation_store,
        setting=args.setting,
        split="test",
        view="ground",
        transform=query_eval_transform,
        index=test_index,
    )
    reference_dataset = VIGOREvalDataset(
        data_root=args.data_root,
        annotation_path=annotation_store,
        setting=args.setting,
        split="test",
        view="aerial",
        transform=reference_transform,
        index=test_index,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
    )
    print(
        f"Training pairs: {len(train_dataset):,}; test queries: {len(query_dataset):,}; "
        f"test gallery: {len(reference_dataset):,}"
    )

    checkpoint = None
    start_epoch = 1
    best_r1 = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            print(f"Checkpoint missing keys: {missing}")
        if unexpected:
            print(f"Checkpoint unexpected keys: {unexpected}")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_r1 = float(checkpoint.get("best_r1", 0.0))
        print(f"Resumed from epoch {start_epoch - 1}; best R@1={best_r1:.4f}")

    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            backbone_lr=backbone_lr,
            prompt_lr=prompt_lr,
            module_lr=module_lr,
            weight_decay=args.weight_decay,
        ),
        betas=(0.9, 0.999),
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps)
    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    model = model.to(device)
    gpu_ids = _gpu_ids(args.gpu_ids)
    if device.type == "cuda" and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(gpu_ids))
        print(f"DataParallel devices: {gpu_ids}")

    loss_function = SymmetricContrastiveLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.mixed_precision and device.type == "cuda",
    )

    history_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        batch_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            mixed_precision=args.mixed_precision,
            scaler=scaler,
            scheduler=scheduler,
            clip_grad_value=args.clip_grad_value,
            verbose=not args.quiet,
        )
        print(f"Epoch {epoch:03d}/{args.epochs}: loss={train_loss:.6f}")

        record = {"epoch": epoch, "train_loss": train_loss}
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            metrics = evaluate_vigor(
                model=model,
                query_dataloader=query_loader,
                reference_dataloader=reference_loader,
                device=device,
                mixed_precision=args.mixed_precision,
                step_size=args.score_step_size,
                verbose=not args.quiet,
            )
            _print_metrics("Test", metrics)
            record["test"] = metrics.to_dict()

            if metrics.r1 > best_r1:
                best_r1 = metrics.r1
                save_checkpoint(
                    output_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_r1=best_r1,
                    star_geo_config=model_config.to_dict(),
                    experiment_config=vars(args),
                    vocabularies={
                        "topology": TOPOLOGY_LABELS,
                        "scale": SCALE_LABELS,
                        "function": FUNCTION_LABELS,
                    },
                )
                print(f"Saved new best checkpoint: R@1={best_r1:.4f}")

        save_checkpoint(
            output_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_r1=best_r1,
            star_geo_config=model_config.to_dict(),
            experiment_config=vars(args),
            vocabularies={
                "topology": TOPOLOGY_LABELS,
                "scale": SCALE_LABELS,
                "function": FUNCTION_LABELS,
            },
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
