"""Sparse autoencoders for concept discovery and control in diffusion models."""

from .model import SAEConfig, SAEOutput, SparseAutoencoder
from .data import ActivationStore, ShardMeta, ShardWriter
from .train import TrainConfig, train_sae, feature_density
from .loader import load_pipeline, inspect as inspect_checkpoint, find_snapshot
from .hooks import (
    ActivationCapture,
    CaptureConfig,
    SteeringHook,
    SteerSpec,
    SD15_SITES,
    DEFAULT_SITE,
    RESIDUAL_SITE,
    bucket_of,
    get_module,
)

__all__ = [
    "SAEConfig", "SAEOutput", "SparseAutoencoder",
    "ActivationStore", "ShardMeta", "ShardWriter",
    "TrainConfig", "train_sae", "feature_density",
    "ActivationCapture", "CaptureConfig", "SteeringHook", "SteerSpec",
    "SD15_SITES", "DEFAULT_SITE", "RESIDUAL_SITE", "bucket_of", "get_module",
    "load_pipeline", "inspect_checkpoint", "find_snapshot",
]
