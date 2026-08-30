#!/usr/bin/env python3

import logging
from typing import Callable, Optional

from .._warnings import configure_runtime_warnings

configure_runtime_warnings()

from torchbiggraph.config import ConfigSchema
from .train_cpu_ext import TrainingCoordinatorExt
import torch


logger = logging.getLogger("unig.tbg_ext")


def train(
    config: ConfigSchema,
    rank: int = 0,
    subprocess_init: Optional[Callable[[], None]] = None,
    *,
    alpha_ot_rna: float = 5e4,
    alpha_ot_atac: float = 5e4,
    ot_swd_num_projections: int = 50,
    z_softmax_temp: float = 1.0,
    s: float = 0.95,
    rna_noise_level: float = 0.05,
    alpha_triplet: float = 1000,
    begin_triplet_epoch: int = 10,
    triplet_update_freq: int = 2,
    triplet_num_samples: int = 512,
    seed: int = 42,
    alpha_ot_gp: float = 0.0,
    rna_aggregation: bool = True,
    atac_aggregation: bool = True,
) -> None:
    """Train a model with joint loss terms.

    Args:
        config: The parsed config.
        rank: The rank of the current process.
        subprocess_init: A function to call at the beginning of each subprocess.
        alpha_ot_rna: Weight for RNA OT reconstruction loss.
        alpha_ot_atac: Weight for ATAC OT reconstruction loss.
        ot_swd_num_projections: Number of projections for Sliced Wasserstein Distance.
        z_softmax_temp: Softmax temperature for dynamic target enhancement.
        s: Binomial probability for ATAC regularization.
        rna_noise_level: Noise level for RNA regularization.
        alpha_triplet: Weight for triplet loss on embeddings.
        begin_triplet_epoch: Number of epochs for pretraining.
        triplet_update_freq: Frequency of triplet updates.
        triplet_num_samples: Number of samples for triplet loss.
        seed: Random seed for reproducibility.
        alpha_ot_gp: Weight for Gene-Peak OT loss.
        rna_aggregation: Whether to aggregate RNA data.
        atac_aggregation: Whether to aggregate ATAC data.
    """
    configure_runtime_warnings()

    if config.num_gpus > 0:
        # Custom GPU trainer not implemented for joint loss yet
        raise NotImplementedError("Joint loss training on GPU is not yet supported.")

    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Extended trainer: seed={seed}, epochs={getattr(config, 'num_epochs', 'unknown')}")

    coordinator = TrainingCoordinatorExt(
        config,
        subprocess_init=subprocess_init,
        rank=rank,
        # Joint loss parameters
        alpha_ot_rna=alpha_ot_rna,
        alpha_ot_atac=alpha_ot_atac,
        ot_swd_num_projections=ot_swd_num_projections,
        z_softmax_temp=z_softmax_temp,
        s=s,
        rna_noise_level=rna_noise_level,
        alpha_triplet=alpha_triplet,
        begin_triplet_epoch=begin_triplet_epoch,
        triplet_update_freq=triplet_update_freq,
        triplet_num_samples=triplet_num_samples,
        alpha_ot_gp=alpha_ot_gp,
        rna_aggregation=rna_aggregation,
        atac_aggregation=atac_aggregation,
    )
    
    coordinator.train()
    coordinator.close()
