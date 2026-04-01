"""Conditioner factory and checkpoint helpers."""

from __future__ import annotations

import torch

from ecgtwin.models.foundation_conditioner import FoundationConditioner
from ecgtwin.models.ib_extractor import IBExtractor


def conditioner_hparams(cfg) -> dict[str, int | float]:
    """Return the active conditioner hyperparameters from config."""
    conditioner_type = cfg.MODEL.CONDITIONER.TYPE.lower()
    if conditioner_type == "foundation_jepa":
        return {
            "embed_dim": cfg.MODEL.FOUNDATION.EMBED_DIM,
            "num_heads": cfg.MODEL.FOUNDATION.NUM_HEADS,
            "ff_hidden_size": cfg.MODEL.FOUNDATION.FF_HIDDEN_SIZE,
            "num_layers": cfg.MODEL.FOUNDATION.NUM_LAYERS,
            "dropout": cfg.MODEL.FOUNDATION.DROPOUT,
            "text_embed_dim": cfg.MODEL.FOUNDATION.TEXT_EMBED_DIM,
            "patient_info_size": cfg.MODEL.FOUNDATION.PATIENT_INFO_SIZE,
        }
    if conditioner_type == "ibe":
        return {
            "embed_dim": cfg.MODEL.IBE.EMBED_DIM,
            "num_heads": cfg.MODEL.IBE.NUM_HEADS,
            "ff_hidden_size": cfg.MODEL.IBE.FF_HIDDEN_SIZE,
            "num_layers": cfg.MODEL.IBE.NUM_LAYERS,
            "dropout": 0.0,
            "text_embed_dim": cfg.MODEL.IBE.TEXT_EMBED_DIM,
            "patient_info_size": cfg.MODEL.IBE.PATIENT_INFO_SIZE,
        }
    raise NotImplementedError(f"Unknown conditioner type: {cfg.MODEL.CONDITIONER.TYPE}")


def conditioner_embed_dim(cfg) -> int:
    """Return the pooled base-vector width produced by the active conditioner."""
    return int(conditioner_hparams(cfg)["embed_dim"])


def build_conditioner(cfg):
    """Instantiate the configured runtime conditioner."""
    params = conditioner_hparams(cfg)
    conditioner_type = cfg.MODEL.CONDITIONER.TYPE.lower()
    common_kwargs = {
        "embed_dim": params["embed_dim"],
        "num_heads": params["num_heads"],
        "ff_hidden_size": params["ff_hidden_size"],
        "num_layers": params["num_layers"],
        "text_embed_dim": params["text_embed_dim"],
        "patient_info_size": params["patient_info_size"],
        "base_vector_mode": cfg.MODEL.BASE_VECTOR.MODE,
        "base_vector_bottleneck_dim": cfg.MODEL.BASE_VECTOR.BOTTLENECK_DIM,
    }
    if conditioner_type == "foundation_jepa":
        return FoundationConditioner(dropout=params["dropout"], **common_kwargs)
    if conditioner_type == "ibe":
        return IBExtractor(**common_kwargs)
    raise NotImplementedError(f"Unknown conditioner type: {cfg.MODEL.CONDITIONER.TYPE}")


def resolve_conditioner_checkpoint(cfg) -> str:
    """Return the checkpoint path that should be used for runtime conditioning."""
    if getattr(cfg.CHECKPOINTS, "CONDITIONER_PATH", ""):
        return cfg.CHECKPOINTS.CONDITIONER_PATH
    return cfg.CHECKPOINTS.IBE_PATH


def load_conditioner(cfg, checkpoint_path: str | None = None, map_location: str | torch.device = "cpu"):
    """Instantiate and load the configured conditioner weights."""
    conditioner = build_conditioner(cfg)
    state_dict = torch.load(checkpoint_path or resolve_conditioner_checkpoint(cfg), map_location=map_location)
    conditioner.load_state_dict(state_dict)
    return conditioner
