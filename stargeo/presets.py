from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .models import STARGeoConfig


@dataclass(frozen=True)
class BackbonePreset:
    star_geo: STARGeoConfig
    backbone_lr: float
    module_lr: float
    prompt_lr: float = 1e-5


PRESETS: Dict[str, BackbonePreset] = {
    "dinov3": BackbonePreset(
        star_geo=STARGeoConfig(
            backbone_name="vit_base_patch16_dinov3.lvd1689m",
            backbone_type="dinov3",
            image_size=384,
            descriptor_dim=768,
            prompt_context_length=16,
            prompt_initialization="The road type of the image is",
            mpg_hidden_dim=768,
            fusion_hidden_dim=768,
            gamma_init=0.0,
            temperature_init=0.07,
        ),
        backbone_lr=1e-4,
        module_lr=1e-4,
        prompt_lr=1e-5,
    ),
    "convnext": BackbonePreset(
        star_geo=STARGeoConfig(
            backbone_name="convnext_base.fb_in22k_ft_in1k_384",
            backbone_type="convnext",
            image_size=384,
            descriptor_dim=768,
            prompt_context_length=16,
            prompt_initialization="The road type of the image is",
            mpg_hidden_dim=768,
            fusion_hidden_dim=768,
            gamma_init=0.0,
            temperature_init=0.07,
        ),
        backbone_lr=1e-3,
        module_lr=1e-3,
        prompt_lr=1e-5,
    ),
}


def get_preset(name: str) -> BackbonePreset:
    key = name.lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown backbone preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[key]
