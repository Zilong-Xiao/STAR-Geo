from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - resolved by requirements.txt in release use
    raise ImportError("STAR-Geo requires timm. Install the dependencies from requirements.txt.") from exc

from clip import clip




def _create_timm_backbone(model_name: str, pretrained: bool, image_size: int) -> nn.Module:
    common = {"pretrained": pretrained, "num_classes": 0}
    try:
        return timm.create_model(model_name, img_size=image_size, **common)
    except TypeError as exc:
        # Some convolutional timm constructors use fully convolutional inputs and
        # do not expose an img_size argument. Retry without changing the model.
        if "img_size" not in str(exc):
            raise
        return timm.create_model(model_name, **common)


@dataclass
class STARGeoConfig:
    backbone_name: str
    backbone_type: str
    image_size: int = 384
    descriptor_dim: int = 768
    prompt_context_length: int = 16
    prompt_initialization: str = "The road type of the image is"
    mpg_hidden_dim: int = 768
    fusion_hidden_dim: int = 768
    gamma_init: float = 0.0
    temperature_init: float = 0.07
    pretrained_backbone: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text tower used with learnable prompt embeddings."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        seq_len = prompts.shape[1]
        positional = self.positional_embedding[:seq_len].to(device=prompts.device, dtype=prompts.dtype)
        x = prompts + positional
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(dtype=prompts.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
        x = x @ self.text_projection.to(device=x.device, dtype=x.dtype)
        return F.normalize(x.float(), dim=-1)


class FactorPromptLearner(nn.Module):
    """One CoOp-style prompt learner for one TSF factor."""

    def __init__(
        self,
        clip_model: nn.Module,
        class_names: Sequence[str],
        context_length: int,
        context_init: str,
    ):
        super().__init__()
        if not class_names:
            raise ValueError("class_names must not be empty")
        self.class_names = tuple(str(name) for name in class_names)
        self.context_length = int(context_length)
        dtype = clip_model.dtype
        context_dim = clip_model.ln_final.weight.shape[0]
        device = next(clip_model.parameters()).device

        context = torch.empty(self.context_length, context_dim, dtype=dtype, device=device)
        nn.init.normal_(context, std=0.02)

        if context_init.strip():
            initialized_tokens = clip.tokenize(context_init.strip()).to(device)
            with torch.no_grad():
                initialized_embeddings = clip_model.token_embedding(initialized_tokens).to(dtype=dtype)[0]
            eot_index = int(initialized_tokens[0].argmax().item())
            initialized_embeddings = initialized_embeddings[1:eot_index]
            count = min(self.context_length, initialized_embeddings.shape[0])
            context[:count].copy_(initialized_embeddings[:count])

        self.context = nn.Parameter(context)

        placeholder = " ".join(["X"] * self.context_length)
        prompt_strings = [f"{placeholder} {name}." for name in self.class_names]
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in prompt_strings]).to(device)
        with torch.no_grad():
            embeddings = clip_model.token_embedding(tokenized_prompts).to(dtype=dtype)

        self.register_buffer("token_prefix", embeddings[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix",
            embeddings[:, 1 + self.context_length :, :],
            persistent=False,
        )
        self.register_buffer("tokenized_prompts", tokenized_prompts, persistent=False)

    def forward(self, class_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if class_ids.ndim != 1:
            raise ValueError(f"Factor class IDs must have shape [B], got {tuple(class_ids.shape)}")
        if class_ids.dtype != torch.long:
            class_ids = class_ids.long()
        if torch.any(class_ids < 0) or torch.any(class_ids >= len(self.class_names)):
            raise IndexError(f"Class IDs are out of range for {len(self.class_names)} labels")

        prefix = self.token_prefix.index_select(0, class_ids)
        suffix = self.token_suffix.index_select(0, class_ids)
        context = self.context.unsqueeze(0).expand(class_ids.shape[0], -1, -1)
        prompts = torch.cat((prefix, context, suffix), dim=1)
        tokens = self.tokenized_prompts.index_select(0, class_ids)
        return prompts, tokens


class TSFPriorEncoder(nn.Module):
    """Factor-wise TSF encoding followed by Mixture-of-Priors Gating."""

    def __init__(
        self,
        clip_model: nn.Module,
        topology_labels: Sequence[str],
        scale_labels: Sequence[str],
        function_labels: Sequence[str],
        context_length: int,
        context_init: str,
        mpg_hidden_dim: int,
        descriptor_dim: int,
    ):
        super().__init__()
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)
        clip_model.eval()

        self.text_encoder = CLIPTextEncoder(clip_model)
        self.topology_prompt = FactorPromptLearner(
            clip_model, topology_labels, context_length, context_init
        )
        self.scale_prompt = FactorPromptLearner(
            clip_model, scale_labels, context_length, context_init
        )
        self.function_prompt = FactorPromptLearner(
            clip_model, function_labels, context_length, context_init
        )

        clip_dim = int(clip_model.text_projection.shape[1])
        self.mpg = nn.Sequential(
            nn.Linear(3 * clip_dim, mpg_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mpg_hidden_dim, 3),
        )
        nn.init.zeros_(self.mpg[-1].weight)
        nn.init.zeros_(self.mpg[-1].bias)

        self.prior_projection = nn.Linear(clip_dim, descriptor_dim)
        nn.init.normal_(self.prior_projection.weight, std=0.02)
        nn.init.zeros_(self.prior_projection.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # The CLIP text encoder is frozen even while prompt tokens are trained.
        self.text_encoder.eval()
        return self

    def _encode_factor(self, learner: FactorPromptLearner, ids: torch.Tensor) -> torch.Tensor:
        prompts, tokenized = learner(ids)
        return self.text_encoder(prompts, tokenized)

    def forward(
        self,
        tsf_ids: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if tsf_ids.ndim != 2 or tsf_ids.shape[1] != 3:
            raise ValueError(f"TSF IDs must have shape [B, 3], got {tuple(tsf_ids.shape)}")

        topology = self._encode_factor(self.topology_prompt, tsf_ids[:, 0])
        scale = self._encode_factor(self.scale_prompt, tsf_ids[:, 1])
        function = self._encode_factor(self.function_prompt, tsf_ids[:, 2])

        weights = torch.softmax(self.mpg(torch.cat((topology, scale, function), dim=-1)), dim=-1)
        mixture = (
            weights[:, 0:1] * topology
            + weights[:, 1:2] * scale
            + weights[:, 2:3] * function
        )
        prior = F.normalize(self.prior_projection(mixture), dim=-1)
        if return_weights:
            return prior, weights
        return prior


class _LateStageVisualEncoder(nn.Module):
    output_dim: int

    def get_data_config(self) -> Mapping[str, object]:
        return timm.data.resolve_model_data_config(self.backbone)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enabled)


class DINOv3LateStageEncoder(_LateStageVisualEncoder):
    """Injects the road prior into the CLS token before the final ViT block."""

    def __init__(
        self,
        model_name: str,
        image_size: int,
        prior_dim: int,
        pretrained: bool,
        gamma_init: float,
    ):
        super().__init__()
        self.backbone = _create_timm_backbone(model_name, pretrained, image_size)
        required = ("patch_embed", "blocks", "norm")
        missing = [name for name in required if not hasattr(self.backbone, name)]
        if missing:
            raise TypeError(f"Backbone {model_name!r} is not a supported ViT; missing {missing}")
        self.output_dim = int(getattr(self.backbone, "num_features", self.backbone.embed_dim))
        self.prior_injection = nn.Linear(prior_dim, self.output_dim)
        nn.init.normal_(self.prior_injection.weight, std=0.02)
        nn.init.zeros_(self.prior_injection.bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _rope_embedding(self, patch_tokens: torch.Tensor):
        rope = getattr(self.backbone, "rope", None)
        if rope is None or not hasattr(rope, "get_embed"):
            return None
        if patch_tokens.ndim == 4:
            if patch_tokens.shape[-1] == self.output_dim:
                height, width = int(patch_tokens.shape[1]), int(patch_tokens.shape[2])
            else:
                height, width = int(patch_tokens.shape[2]), int(patch_tokens.shape[3])
        else:
            grid = getattr(self.backbone.patch_embed, "grid_size", None)
            if grid is not None:
                height, width = int(grid[0]), int(grid[1])
            else:
                side = int(math.sqrt(patch_tokens.shape[1]))
                height, width = side, max(1, patch_tokens.shape[1] // max(side, 1))
        return rope.get_embed([height, width])

    @staticmethod
    def _run_block(block: nn.Module, tokens: torch.Tensor, rope_embedding):
        if rope_embedding is None:
            return block(tokens)
        for keyword in ("rot_pos_embed", "rope", "freqs_cis", "rotary_emb"):
            try:
                return block(tokens, **{keyword: rope_embedding})
            except TypeError:
                continue
        return block(tokens)

    def _prepare_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, object]:
        patch_tokens = self.backbone.patch_embed(images)
        rope_embedding = self._rope_embedding(patch_tokens)

        used_native_pos_embed = hasattr(self.backbone, "_pos_embed")
        if used_native_pos_embed:
            tokens = self.backbone._pos_embed(patch_tokens)
        else:
            tokens = patch_tokens
            if tokens.ndim == 4:
                if tokens.shape[-1] == self.output_dim:
                    tokens = tokens.flatten(1, 2)
                else:
                    tokens = tokens.flatten(2).transpose(1, 2)
            cls_token = getattr(self.backbone, "cls_token", None)
            if cls_token is not None:
                tokens = torch.cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
            position = getattr(self.backbone, "pos_embed", None)
            if position is not None:
                tokens = tokens + position[:, : tokens.shape[1]]

        patch_drop = getattr(self.backbone, "patch_drop", None)
        if patch_drop is not None:
            tokens = patch_drop(tokens)
        pos_drop = getattr(self.backbone, "pos_drop", None)
        if pos_drop is not None and not used_native_pos_embed:
            tokens = pos_drop(tokens)
        norm_pre = getattr(self.backbone, "norm_pre", None)
        if norm_pre is not None:
            tokens = norm_pre(tokens)
        return tokens, rope_embedding

    def forward(self, images: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        tokens, rope_embedding = self._prepare_tokens(images)
        blocks = self.backbone.blocks
        if len(blocks) == 0:
            raise RuntimeError("DINOv3 backbone contains no transformer blocks")
        for block in blocks[:-1]:
            tokens = self._run_block(block, tokens, rope_embedding)

        injected = self.prior_injection(prior).to(dtype=tokens.dtype)
        tokens = tokens.clone()
        tokens[:, 0] = tokens[:, 0] + torch.tanh(self.gamma) * injected
        tokens = self._run_block(blocks[-1], tokens, rope_embedding)
        tokens = self.backbone.norm(tokens)

        if hasattr(self.backbone, "forward_head"):
            descriptor = self.backbone.forward_head(tokens, pre_logits=True)
        else:
            descriptor = tokens[:, 0]
        if descriptor.ndim > 2:
            descriptor = descriptor.flatten(1)
        return descriptor


class ConvNeXtLateStageEncoder(_LateStageVisualEncoder):
    """Broadcasts the road prior over the spatial map before the final ConvNeXt stage."""

    def __init__(
        self,
        model_name: str,
        image_size: int,
        prior_dim: int,
        pretrained: bool,
        gamma_init: float,
    ):
        super().__init__()
        self.backbone = _create_timm_backbone(model_name, pretrained, image_size)
        if not hasattr(self.backbone, "stem") or not hasattr(self.backbone, "stages"):
            raise TypeError(f"Backbone {model_name!r} is not a supported timm ConvNeXt model")
        stages = list(self.backbone.stages)
        if len(stages) < 2:
            raise TypeError(f"ConvNeXt backbone {model_name!r} must expose multiple stages")
        self.output_dim = int(self.backbone.num_features)
        final_input_dim = self._infer_final_stage_input_dim(stages[-1])
        self.prior_injection = nn.Linear(prior_dim, final_input_dim)
        nn.init.normal_(self.prior_injection.weight, std=0.02)
        nn.init.zeros_(self.prior_injection.bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _infer_final_stage_input_dim(stage: nn.Module) -> int:
        downsample = getattr(stage, "downsample", None)
        if downsample is not None:
            for module in downsample.modules():
                if isinstance(module, nn.Conv2d):
                    return int(module.in_channels)
        for module in stage.modules():
            if isinstance(module, nn.Conv2d):
                return int(module.in_channels)
        raise TypeError("Unable to infer the input channels of the final ConvNeXt stage")

    def forward(self, images: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(images)
        stages = list(self.backbone.stages)
        for stage in stages[:-1]:
            features = stage(features)

        injected = self.prior_injection(prior).to(dtype=features.dtype)
        features = features + torch.tanh(self.gamma) * injected[:, :, None, None]
        features = stages[-1](features)

        norm_pre = getattr(self.backbone, "norm_pre", None)
        if norm_pre is not None:
            features = norm_pre(features)
        if hasattr(self.backbone, "forward_head"):
            descriptor = self.backbone.forward_head(features, pre_logits=True)
        else:
            descriptor = features.mean(dim=(-2, -1))
        if descriptor.ndim > 2:
            descriptor = descriptor.flatten(1)
        return descriptor


class STARGeo(nn.Module):
    """STAR-Geo with factor-wise TSF encoding, MPG, and dual-stage prior fusion."""

    def __init__(
        self,
        config: STARGeoConfig,
        clip_model: nn.Module,
        topology_labels: Sequence[str],
        scale_labels: Sequence[str],
        function_labels: Sequence[str],
    ):
        super().__init__()
        self.config = config
        self.prior_encoder = TSFPriorEncoder(
            clip_model=clip_model,
            topology_labels=topology_labels,
            scale_labels=scale_labels,
            function_labels=function_labels,
            context_length=config.prompt_context_length,
            context_init=config.prompt_initialization,
            mpg_hidden_dim=config.mpg_hidden_dim,
            descriptor_dim=config.descriptor_dim,
        )

        backbone_type = config.backbone_type.lower()
        if backbone_type == "dinov3":
            self.visual_encoder = DINOv3LateStageEncoder(
                model_name=config.backbone_name,
                image_size=config.image_size,
                prior_dim=config.descriptor_dim,
                pretrained=config.pretrained_backbone,
                gamma_init=config.gamma_init,
            )
        elif backbone_type == "convnext":
            self.visual_encoder = ConvNeXtLateStageEncoder(
                model_name=config.backbone_name,
                image_size=config.image_size,
                prior_dim=config.descriptor_dim,
                pretrained=config.pretrained_backbone,
                gamma_init=config.gamma_init,
            )
        else:
            raise ValueError("backbone_type must be 'dinov3' or 'convnext'")

        self.visual_projection = nn.Linear(self.visual_encoder.output_dim, config.descriptor_dim)
        nn.init.normal_(self.visual_projection.weight, std=0.02)
        nn.init.zeros_(self.visual_projection.bias)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * config.descriptor_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.fusion_hidden_dim, 1),
        )
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.temperature_init), dtype=torch.float32)
        )

    def get_data_config(self) -> Mapping[str, object]:
        return self.visual_encoder.get_data_config()

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self.visual_encoder.set_grad_checkpointing(enabled)

    def encode(self, images: torch.Tensor, tsf_ids: torch.Tensor) -> torch.Tensor:
        prior = self.prior_encoder(tsf_ids)
        visual = self.visual_encoder(images, prior)
        visual = F.normalize(self.visual_projection(visual), dim=-1)
        beta = torch.sigmoid(self.fusion_mlp(torch.cat((visual, prior), dim=-1)))
        return F.normalize((1.0 - beta) * visual + beta * prior, dim=-1)

    def forward(
        self,
        query_images: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None,
        query_tsf: Optional[torch.Tensor] = None,
        reference_tsf: Optional[torch.Tensor] = None,
    ):
        if query_tsf is None:
            raise ValueError("query_tsf is required")
        query_descriptor = self.encode(query_images, query_tsf)
        if reference_images is None:
            return query_descriptor
        if reference_tsf is None:
            raise ValueError("reference_tsf is required for paired training")
        reference_descriptor = self.encode(reference_images, reference_tsf)
        return query_descriptor, reference_descriptor

    def clamped_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def optimizer_parameter_groups(
        self,
        backbone_lr: float,
        prompt_lr: float,
        module_lr: float,
        weight_decay: float,
    ) -> Iterable[Dict[str, object]]:
        prompt_ids = {
            id(parameter)
            for module in (
                self.prior_encoder.topology_prompt,
                self.prior_encoder.scale_prompt,
                self.prior_encoder.function_prompt,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        backbone_ids = {
            id(parameter)
            for parameter in self.visual_encoder.backbone.parameters()
            if parameter.requires_grad
        }

        groups = {"backbone": [], "prompt": [], "module": []}
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in prompt_ids:
                groups["prompt"].append(parameter)
            elif id(parameter) in backbone_ids:
                groups["backbone"].append(parameter)
            else:
                groups["module"].append(parameter)

        return [
            {
                "name": "backbone",
                "params": groups["backbone"],
                "lr": float(backbone_lr),
                "weight_decay": float(weight_decay),
            },
            {
                "name": "prompts",
                "params": groups["prompt"],
                "lr": float(prompt_lr),
                "weight_decay": float(weight_decay),
            },
            {
                "name": "star_geo_modules",
                "params": groups["module"],
                "lr": float(module_lr),
                "weight_decay": float(weight_decay),
            },
        ]


def build_star_geo(
    config: STARGeoConfig,
    clip_backbone: str,
    device: torch.device | str,
    topology_labels: Sequence[str],
    scale_labels: Sequence[str],
    function_labels: Sequence[str],
) -> STARGeo:
    clip_model, _ = clip.load(clip_backbone, device=device, jit=False)
    clip_model = clip_model.float()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    return STARGeo(
        config=config,
        clip_model=clip_model,
        topology_labels=topology_labels,
        scale_labels=scale_labels,
        function_labels=function_labels,
    )
