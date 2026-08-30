#!/usr/bin/env python3

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from .._warnings import configure_runtime_warnings

configure_runtime_warnings()

import torch
from torch.optim import Optimizer
import torch.nn.functional as F
import math

from torchbiggraph.batching import AbstractBatchProcessor
from torchbiggraph.config import ConfigSchema
from torchbiggraph.losses import LOSS_FUNCTIONS, AbstractLossFunction
from torchbiggraph.model import MultiRelationEmbedder
from torchbiggraph.stats import Stats, StatsHandler
from torchbiggraph.train_cpu import TrainingCoordinator as TrainingCoordinatorBase, Trainer as TrainerBase
from torchbiggraph.types import SINGLE_TRAINER, Rank

from ..utils import (
    ot_loss_swd,
    find_mnn_triplets,
)
import numpy as np
import pandas as pd
from pathlib import Path
import ot

logger = logging.getLogger("unig.tbg_ext.cpu")


class JointTrainer(TrainerBase):

    def __init__(
        self,
        config: ConfigSchema,
        model_optimizer: Optimizer,
        loss_fn: AbstractLossFunction,
        relation_weights: List[float],
        *,
        alpha_ot_rna: float = 5e4,
        alpha_ot_atac: float = 5e4,
        ot_swd_num_projections: int = 50, 
        alpha_ot_gp: float = 0.0,
        z_softmax_temp: float = 1.0,
        s: float = 0.95,  # Binomial dropout prob for ATAC
        rna_noise_level: float = 0.05,  # Noise level for RNA
        alpha_triplet: float = 1.0,
        begin_triplet_epoch: int = 10,
        triplet_update_freq: int = 2,
        triplet_num_samples: int = 512,
        rna_aggregation: bool = True,
        atac_aggregation: bool = True,
    ) -> None:
        super().__init__(model_optimizer, loss_fn, relation_weights)
        
        self.config = config
        self.alpha_ot_rna = alpha_ot_rna
        self.alpha_ot_atac = alpha_ot_atac
        self.ot_swd_num_projections = ot_swd_num_projections
        self.z_softmax_temp = z_softmax_temp
        self.s = s
        self.rna_noise_level = rna_noise_level
        self.alpha_triplet = alpha_triplet
        self.begin_triplet_epoch = begin_triplet_epoch
        self.triplet_update_freq = triplet_update_freq
        self.triplet_num_samples = triplet_num_samples
        self.rna_aggregation = rna_aggregation
        self.atac_aggregation = atac_aggregation
        self.alpha_ot_gp = alpha_ot_gp
        
        # Initialize Triplet Loss function
        if alpha_triplet > 0:
            self.triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')

        # State for triplet mining
        self.last_triplet_update_epoch = -1
        self.mnn_pairs_dict = {}
        
    def _update_triplets(self, model: MultiRelationEmbedder) -> None:
        """Dynamically finds MNNs and updates them for online triplet mining."""
        current_epoch = getattr(self, 'current_epoch', 0)
        print(f"--- Updating MNN pairs for Epoch {current_epoch} ---")
        self.last_triplet_update_epoch = current_epoch

        try:
            # Find all entity types starting with 'C'
            cell_entity_types = [name for name in self.config.entities if name.startswith('C')]
            
            if len(cell_entity_types) < 2:
                print(f"Warning: Less than 2 cell entity types found. Skipping MNN update.")
                return

            with torch.no_grad():
                # Get embeddings for all cell types
                cell_embs = {}
                for c_type in cell_entity_types:
                    emb_key = model.EMB_PREFIX + c_type
                    if emb_key in model.lhs_embs:
                        cell_embs[c_type] = model.lhs_embs[emb_key].weight.cpu().numpy()
                    else:
                        print(f"Warning: Embedding module for {c_type} not found.")

            # Compute MNNs for all pairs
            self.mnn_pairs_dict = {} # Store pairs per type combination
            
            for i in range(len(cell_entity_types)):
                for j in range(i + 1, len(cell_entity_types)):
                    c1_type = cell_entity_types[i]
                    c2_type = cell_entity_types[j]
                    
                    if c1_type not in cell_embs or c2_type not in cell_embs:
                        continue
                        
                    E_c1_np = cell_embs[c1_type]
                    E_c2_np = cell_embs[c2_type]
                    
                    print(
                        f"Finding MNNs between {E_c1_np.shape[0]} {c1_type} cells "
                        f"and {E_c2_np.shape[0]} {c2_type} cells..."
                    )
                    mnn_pairs = find_mnn_triplets(E_c1_np, E_c2_np)
                    
                    if mnn_pairs:
                        print(
                            f"Found {len(mnn_pairs)} unique MNN pairs between "
                            f"{c1_type} and {c2_type}."
                        )
                        self.mnn_pairs_dict[(c1_type, c2_type)] = {
                            'anchors': np.array([p[0] for p in mnn_pairs]),
                            'positives': np.array([p[1] for p in mnn_pairs])
                        }
                    else:
                        print(f"Warning: No MNN pairs found between {c1_type} and {c2_type}.")
        
        except Exception as e:
            print(f"Error during MNN pair update: {e}")
            import traceback
            traceback.print_exc()
            self.mnn_pairs_dict = {}


    def _process_one_batch(self, model: MultiRelationEmbedder, batch_edges) -> Stats:
        current_epoch = getattr(self, 'current_epoch', None)
        # Run base pipeline to compute TBG losses and step optimizer
        for p in model.parameters():
            p.grad = None
        scores, reg = model(batch_edges)
        # Compute ranking loss ignoring per-edge values so that relation weights drive ranking,
        # while keeping per-edge values available for reconstruction losses.
        class _BatchEdgesNoWeight:
            def __init__(self, inner):
                self._inner = inner
            def __getattr__(self, name):
                return getattr(self._inner, name)
            def has_weight(self):
                return False
            def get_weight(self):
                return None
            def __len__(self):
                return len(self._inner)

        loss = self.calc_loss(scores, _BatchEdgesNoWeight(batch_edges))
        # Scalars for logging/stats must be detached from autograd
        reg_scalar = float(reg.detach()) if reg is not None else 0.0
        loss_scalar = float(loss.detach())
        
        # Track regularization components
        wd_loss_scalar = 0.0
        if model.wd > 0 and torch.rand(()) < 1.0 / model.wd_interval:
            wd_loss_val = model.wd * model.wd_interval * model.l2_norm()
            wd_loss_scalar = float(wd_loss_val.detach())
            loss = loss + wd_loss_val
        
        stats = Stats(
            loss=loss_scalar,
            reg=reg_scalar,
            violators_lhs=int((scores.lhs_neg > scores.lhs_pos.unsqueeze(1)).sum()),
            violators_rhs=int((scores.rhs_neg > scores.rhs_pos.unsqueeze(1)).sum()),
            count=len(batch_edges),
        )
        if reg is not None:
            loss = loss + reg
            
        # Store regularization losses for epoch-level aggregation
        if not hasattr(stats, 'metrics') or stats.metrics is None:
            stats.metrics = {}
        stats.metrics['wd_loss'] = wd_loss_scalar
        setattr(self, "wd_loss_epoch_sum", float(getattr(self, "wd_loss_epoch_sum", 0.0)) + stats.metrics['wd_loss'])
        setattr(self, "wd_loss_epoch_count", int(getattr(self, "wd_loss_epoch_count", 0)) + 1)

        if self.alpha_ot_rna > 0 or self.alpha_ot_atac > 0 or self.alpha_ot_gp > 0:
            # Determine relation index and schema
            if batch_edges.has_scalar_relation_type():
                rel_idx = batch_edges.get_relation_type_as_scalar()
                rel_type = batch_edges.get_relation_type()
            else:
                # Fallback: use first relation
                rel_idx = 0
                rel_type = 0
            relation = model.relations[rel_idx]

            lhs_module = model.lhs_embs[model.EMB_PREFIX + relation.lhs]
            rhs_module = model.rhs_embs[model.EMB_PREFIX + relation.rhs]

            lhs_pos_raw = lhs_module(batch_edges.lhs)
            rhs_pos_raw = rhs_module(batch_edges.rhs)

            lhs_adj = model.adjust_embs(lhs_pos_raw, rel_type, relation.lhs, None)
            rhs_adj = model.adjust_embs(
                rhs_pos_raw,
                rel_type,
                relation.rhs,
                model.rhs_operators[rel_idx] if model.num_dynamic_rels == 0 else None,
            )

            # Unique entities on lhs and rhs to form compact matrices
            lhs_ids = batch_edges.lhs if hasattr(batch_edges.lhs, 'numel') else batch_edges.lhs.to_tensor()
            rhs_ids = batch_edges.rhs if hasattr(batch_edges.rhs, 'numel') else batch_edges.rhs.to_tensor()
            lhs_unique, lhs_inv = torch.unique(lhs_ids, sorted=True, return_inverse=True)
            rhs_unique, rhs_inv = torch.unique(rhs_ids, sorted=True, return_inverse=True)

            # Aggregate embeddings for unique ids by first occurrence (build both sides)
            E_lhs = lhs_adj.new_zeros((lhs_unique.numel(), lhs_adj.size(-1)))
            E_lhs.index_copy_(0, lhs_inv, lhs_adj)
            E_rhs = rhs_adj.new_zeros((rhs_unique.numel(), rhs_adj.size(-1)))
            E_rhs.index_copy_(0, rhs_inv, rhs_adj)

            # Decide which side is cells and which is features
            lhs_is_cell = relation.lhs.startswith('C')
            rhs_is_cell = relation.rhs.startswith('C')
            
            is_rna_rel = (lhs_is_cell and 'G' in relation.rhs) or (rhs_is_cell and 'G' in relation.lhs)
            is_atac_rel = (lhs_is_cell and 'P' in relation.rhs) or (rhs_is_cell and 'P' in relation.lhs)
            is_gp_rel = ('G' in relation.lhs and 'P' in relation.rhs) or ('P' in relation.lhs and 'G' in relation.rhs)

            is_cc_relation = lhs_is_cell and rhs_is_cell
            if lhs_is_cell and not rhs_is_cell:
                E_cells = E_lhs
                E_feats = E_rhs
                cell_inv = lhs_inv
                feat_inv = rhs_inv
            elif rhs_is_cell and not lhs_is_cell:
                E_cells = E_rhs
                E_feats = E_lhs
                cell_inv = rhs_inv
                feat_inv = lhs_inv
            else:
                # Either C-C (both sides cells) or no cells; skip reconstruction based on cells
                E_cells = None
                E_feats = None

            # --- 1. Gene-Peak Sinkhorn OT Loss ---
            if self.alpha_ot_gp > 0 and is_gp_rel and current_epoch > self.begin_triplet_epoch:
                try:
                    # Determine which side is genes and which is peaks
                    if 'G' in relation.lhs and 'P' in relation.rhs:
                        # G-P relationship: E_lhs = genes, E_rhs = peaks
                        E_genes = E_lhs
                        E_peaks = E_rhs
                    else:
                        # P-G relationship: E_lhs = peaks, E_rhs = genes, need to swap
                        E_genes = E_rhs
                        E_peaks = E_lhs
                
                    eps = 1e-8
                    
                    sim = torch.mm(E_genes, E_peaks.t())           # [Ng, Np]
                    sim_min, sim_max = sim.min(), sim.max()
                    sim_norm = (sim - sim_min) / (sim_max - sim_min + eps)
                    cost_matrix = 1.0 - sim_norm
                    
                    # Create uniform marginal distributions for genes and peaks as PyTorch tensors.
                    n_genes = E_genes.shape[0]
                    n_peaks = E_peaks.shape[0]
                    device = E_genes.device
                    a = torch.ones(n_genes, device=device) / n_genes
                    b = torch.ones(n_peaks, device=device) / n_peaks

                    # Compute the OT loss using the POT library's PyTorch backend.
                    # ot.sinkhorn2 can directly handle torch tensors and returns a differentiable tensor.
                    reg = 0.05
                    gp_ot_val = ot.sinkhorn2(a, b, cost_matrix, reg, numItermax=500, stopThr=1e-8)

                    gp_ot_scalar = float(gp_ot_val.detach().cpu())
                    loss = loss + self.alpha_ot_gp * gp_ot_val
                    stats.metrics = stats.metrics or {}
                    stats.metrics['gp_ot'] = gp_ot_scalar
                    sum_key, count_key = "gp_ot_epoch_sum", "gp_ot_epoch_count"
                    setattr(self, sum_key, float(getattr(self, sum_key, 0.0)) + gp_ot_scalar)
                    setattr(self, count_key, int(getattr(self, count_key, 0)) + 1)
                except Exception as e:
                    print(f"  [Warning] Failed to compute Gene-Peak OT loss: {e}")

            # Reconstructed data for this batch's submatrix (cells x features)
            # Skip for C-C relations
            if E_cells is not None and E_feats is not None and not is_cc_relation and current_epoch > self.begin_triplet_epoch:
                X_pred = E_cells @ E_feats.t()

                # 3. Build the dense target matrix (X_target) for the batch
                try:
                    target_weights = torch.as_tensor(batch_edges.get_weight(), device=X_pred.device, dtype=X_pred.dtype)
                    # Create a sparse tensor and then convert to dense to build X_target efficiently
                    indices = torch.stack([cell_inv, feat_inv])
                    X_target = torch.sparse_coo_tensor(indices, target_weights, X_pred.shape).to_dense()

                except Exception:
                    X_target = None
                
                if X_target is not None and (self.alpha_ot_rna > 0 or self.alpha_ot_atac > 0):
                    # --- Dynamic Target Enhancement (Modality-Specific) ---
                    # 1. Compute dynamic cell-similarity (Z) from embeddings (H)
                    sim_matrix = E_cells @ E_cells.T
                    Z_batch = torch.nn.functional.softmax(sim_matrix / self.z_softmax_temp, dim=-1)

                    # 2. Enhance the target matrix based on modality
                    X_enhanced_target = None
                    if is_rna_rel:
                        if self.rna_aggregation:
                            # For RNA, add small Gaussian noise directly to the target data before enhancement
                            noise = torch.randn_like(X_target) * self.rna_noise_level
                            X_target_noisy = X_target + noise
                            X_enhanced_target = Z_batch @ X_target_noisy
                        else:
                            # For SC mode, skip aggregation
                            X_enhanced_target = X_target
                    elif is_atac_rel:
                        if self.atac_aggregation:
                            # For ATAC, use the binomial dropout mask (R_batch) on the similarity matrix
                            R_batch = torch.bernoulli(torch.full_like(Z_batch, self.s))
                            Z_masked_batch = Z_batch * R_batch
                            X_enhanced_target = Z_masked_batch @ X_target
                        else:
                            X_enhanced_target = X_target
                 
                    # --- Loss Calculation with Enhanced Target ---
                    try:
                        ot_val = ot_loss_swd(
                            mat_pred=X_pred,
                            mat_target=X_enhanced_target,
                            num_projections=self.ot_swd_num_projections
                        )
                        ot_scalar = float(ot_val.detach())

                        # Apply modality-specific weighting
                        applied_alpha = 0.0
                        loss_key = None
                        if is_rna_rel and self.alpha_ot_rna > 0:
                            applied_alpha = self.alpha_ot_rna
                            loss_key = 'ot_rna'
                        elif is_atac_rel and self.alpha_ot_atac > 0:
                            applied_alpha = self.alpha_ot_atac
                            loss_key = 'ot_atac'

                        if applied_alpha > 0 and loss_key:
                            loss = loss + applied_alpha * ot_val
                            
                            # Track and expose to parent via Stats.metrics
                            if not hasattr(stats, 'metrics') or stats.metrics is None:
                                stats.metrics = {}
                            stats.metrics[loss_key] = ot_scalar

                            # Epoch accumulators
                            sum_key = f"{loss_key}_epoch_sum"
                            count_key = f"{loss_key}_epoch_count"
                            setattr(self, sum_key, float(getattr(self, sum_key, 0.0)) + ot_scalar)
                            setattr(self, count_key, int(getattr(self, count_key, 0)) + 1)
                            
                    except (ValueError, RuntimeError) as e:
                        # Catch potential dimension mismatches or other errors in SWD
                        print(f"Warning: SWD calculation failed. Skipping OT loss for this batch. Error: {e}")

                
            else:
                # No valid cell side; skip OT based on cells
                pass
            
        if self.alpha_triplet > 0 and current_epoch > self.begin_triplet_epoch:
            if hasattr(self, 'mnn_pairs_dict') and self.mnn_pairs_dict:
                try:
                    # 1. Sample a pair of cell types randomly
                    available_pairs = list(self.mnn_pairs_dict.keys())
                    c1_type, c2_type = available_pairs[np.random.randint(len(available_pairs))]
                    
                    pair_data = self.mnn_pairs_dict[(c1_type, c2_type)]
                    anchors = pair_data['anchors']
                    positives = pair_data['positives']
                    
                    # 2. Sample a mini-batch of MNN pairs from the selected cell types
                    num_available_pairs = len(anchors)
                    if num_available_pairs > 0:
                        sample_size = min(self.triplet_num_samples, num_available_pairs)
                        sampled_indices = np.random.choice(num_available_pairs, sample_size, replace=False)

                        # 3. Get local indices for the sampled pairs and move to device
                        device = model.lhs_embs[model.EMB_PREFIX + c1_type].weight.device
                        anchor_local_indices = torch.from_numpy(anchors[sampled_indices]).long().to(device)
                        positive_local_indices = torch.from_numpy(positives[sampled_indices]).long().to(device)
                        
                        # 4. Get the embedding modules
                        c1_emb_module = model.lhs_embs[model.EMB_PREFIX + c1_type]
                        c2_emb_module = model.lhs_embs[model.EMB_PREFIX + c2_type]

                        n_c1 = c1_emb_module.weight.size(0)
                        n_c2 = c2_emb_module.weight.size(0)

                        # # Safety check: clamp to max valid index to prevent crash
                        # anchor_local_indices = torch.clamp(anchor_local_indices, 0, n_c1 - 1)
                        # positive_local_indices = torch.clamp(positive_local_indices, 0, n_c2 - 1)

                        # 5. Fetch embeddings for anchors and positives
                        anchor_embs = c1_emb_module.weight.index_select(0, anchor_local_indices)
                        positive_embs = c2_emb_module.weight.index_select(0, positive_local_indices)

                        # 6. Online negative sampling: sample random negatives from C1 for each anchor
                        negative_local_indices = torch.randint(0, n_c1, (sample_size,), device=device)
                        
                        # Optional: Ensure negative is not the same as anchor.
                        for i in range(sample_size):
                            while negative_local_indices[i] == anchor_local_indices[i]:
                                negative_local_indices[i] = torch.randint(0, n_c1, (1,), device=device)[0]

                        negative_embs = c1_emb_module.weight.index_select(0, negative_local_indices)

                        # 7. Calculate triplet loss on embeddings
                        triplet_val = self.triplet_loss_fn(anchor_embs, positive_embs, negative_embs)
                        triplet_scalar = float(triplet_val.detach())
                        
                        loss = loss + self.alpha_triplet * triplet_val

                        # Track and expose to parent via Stats.metrics
                        if not hasattr(stats, 'metrics') or stats.metrics is None:
                            stats.metrics = {}
                        stats.metrics['triplet_align'] = triplet_scalar
                        
                        # Epoch accumulators for alignment loss
                        sum_key = "triplet_align_epoch_sum"
                        count_key = "triplet_align_epoch_count"
                        setattr(self, sum_key, float(getattr(self, sum_key, 0.0)) + triplet_scalar)
                        setattr(self, count_key, int(getattr(self, count_key, 0)) + 1)

                except (ValueError, RuntimeError, IndexError) as e:
                    print(f"Warning: Triplet loss calculation failed. Skipping for this batch. Error: {e}")
        
    

        loss.backward()
        self.model_optimizer.step(closure=None)
        for optimizer in self.unpartitioned_optimizers.values():
            optimizer.step(closure=None)
        for optimizer in self.partitioned_optimizers.values():
            optimizer.step(closure=None)
        return stats



class TrainingCoordinatorExt(TrainingCoordinatorBase):
    def __init__(
        self,
        config: ConfigSchema,
        model: Optional[MultiRelationEmbedder] = None,
        trainer: Optional[AbstractBatchProcessor] = None,
        evaluator: Optional[AbstractBatchProcessor] = None,
        rank: Rank = SINGLE_TRAINER,
        subprocess_init: Optional[Callable[[], None]] = None,
        *,
        alpha_ot_rna: float = 5e4,
        alpha_ot_atac: float = 5e4,
        ot_swd_num_projections: int = 50,
        z_softmax_temp: float = 1.0,
        s: float = 0.95,  # Binomial dropout prob for ATAC
        rna_noise_level: float = 0.05,  # Noise level for RNA
        alpha_triplet: float = 1000,
        begin_triplet_epoch: int = 10,
        triplet_update_freq: int = 2,
        triplet_num_samples: int = 512,
        alpha_ot_gp: float = 0.0,
        rna_aggregation: bool = True,
        atac_aggregation: bool = True,
        stats_handler: StatsHandler = StatsHandler(),
    ):
        self.extra_cfg = dict(
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
        super().__init__(config, model, trainer, evaluator, rank, subprocess_init, stats_handler=stats_handler)
        
        # Replace default trainer with our joint trainer
        try:
            # Get the optimizer from the base trainer that was just created
            base_trainer = self.trainer
            
            # 1. Rebuild loss_fn and relation_weights from config
            loss_fn = LOSS_FUNCTIONS.get_class(self.config.loss_fn)(margin=self.config.margin)
            relation_weights = [relation.weight for relation in self.config.relations]

            # 2. Get the optimizer from base trainer
            model_optimizer = base_trainer.model_optimizer

            # 3. Create the JointTrainer instance
            joint = JointTrainer(
                config=self.config,
                model_optimizer=model_optimizer,
                loss_fn=loss_fn,
                relation_weights=relation_weights,
                alpha_ot_rna=self.extra_cfg['alpha_ot_rna'],
                alpha_ot_atac=self.extra_cfg['alpha_ot_atac'],
                ot_swd_num_projections=self.extra_cfg['ot_swd_num_projections'],
                z_softmax_temp=self.extra_cfg['z_softmax_temp'],
                s=self.extra_cfg['s'],
                rna_noise_level=self.extra_cfg['rna_noise_level'],
                alpha_triplet=self.extra_cfg['alpha_triplet'],
                begin_triplet_epoch=self.extra_cfg['begin_triplet_epoch'],
                triplet_update_freq=self.extra_cfg['triplet_update_freq'],
                triplet_num_samples=self.extra_cfg['triplet_num_samples'],
                alpha_ot_gp=self.extra_cfg['alpha_ot_gp'],
                rna_aggregation=self.extra_cfg['rna_aggregation'],
                atac_aggregation=self.extra_cfg['atac_aggregation'],
            )

            # 4. Carry over optimizers for unpartitioned/partitioned entities
            #    These are created in the super().__init__ call.
            joint.unpartitioned_optimizers = base_trainer.unpartitioned_optimizers
            joint.partitioned_optimizers = base_trainer.partitioned_optimizers
            
            # 5. Assign the new trainer
            self.trainer = joint
            print("Joint trainer ready.")
            logger.info("Replaced default Trainer with JointTrainer (extra losses enabled)")
        except Exception as e:
            print(f"Failed to setup JointTrainer: {e}")
            logger.warning(f"Failed to replace trainer with JointTrainer: {e}")

        # Epoch tracking for progress prints
        self._last_epoch_idx: Optional[int] = None
        self._num_epochs = getattr(self.config, 'num_epochs', None)

    # Inject current epoch into trainer and print epoch start
    def _coordinate_train(self, edges, eval_edge_idxs, epoch_idx) -> Stats:
        # Set current epoch (1-based) for batch prints
        current_epoch = int(epoch_idx) + 1
        try:
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.current_epoch = current_epoch
        except Exception:
            pass

        # Check if it's time to update triplets
        if self.trainer.alpha_triplet > 0 and current_epoch > self.trainer.begin_triplet_epoch:
            # Trigger update on the first alignment epoch, and then based on frequency
            is_first_align_epoch = (self.trainer.last_triplet_update_epoch < self.trainer.begin_triplet_epoch)
            is_update_epoch = (current_epoch - (self.trainer.begin_triplet_epoch + 1)) % self.trainer.triplet_update_freq == 0
            
            if self.trainer.last_triplet_update_epoch != current_epoch and (is_first_align_epoch or is_update_epoch):
                self.trainer._update_triplets(self.model)

        # Print epoch start once and set loss history baselines
        if self._last_epoch_idx != epoch_idx:
            total = self._num_epochs if self._num_epochs is not None else '?'
            print(f"===== Epoch {epoch_idx + 1} / {total} START =====")
            self._last_epoch_idx = epoch_idx
            # Reset per-epoch accumulators on trainer
            try:
                self.trainer.ot_rna_epoch_sum = 0.0
                self.trainer.ot_rna_epoch_count = 0
                self.trainer.ot_atac_epoch_sum = 0.0
                self.trainer.ot_atac_epoch_count = 0
                self.trainer.triplet_align_epoch_sum = 0.0
                self.trainer.triplet_align_epoch_count = 0
                self.trainer.gp_ot_epoch_sum = 0.0
                self.trainer.gp_ot_epoch_count = 0
                self.trainer.wd_loss_epoch_sum = 0.0
                self.trainer.wd_loss_epoch_count = 0
            except Exception:
                pass
            # Delegate to base implementation
        return super()._coordinate_train(edges, eval_edge_idxs, epoch_idx)


    # Print epoch end when checkpoint is written (end of epoch)
    def _maybe_write_checkpoint(
        self,
        epoch_idx: int,
        edge_path_idx: int,
        edge_chunk_idx: int,
        current_index: int,
    ) -> List[Dict[str, Any]]:
        # Aggregate PBG (embedding) loss for this pass (epoch)
        try:
            train_stats_list = [s.train for s in self.bucket_scheduler.get_stats_for_pass() if s.train is not None]
            if len(train_stats_list) > 0:
                train_avg = Stats.average_list(train_stats_list)
                train_avg_dict = train_avg.to_dict()
                pbg_loss_epoch = float(train_avg_dict.get('metrics', {}).get('loss', float('nan')))
            else:
                pbg_loss_epoch = float('nan')
        except Exception:
            pbg_loss_epoch = float('nan')
        
        out = super()._maybe_write_checkpoint(epoch_idx, edge_path_idx, edge_chunk_idx, current_index)
        # --- Compute per-epoch OT means for RNA and ATAC ---
        ot_avg_rna, ot_avg_atac = float('nan'), float('nan')

        def get_avg_ot_loss(loss_key: str):
            avg_loss = float('nan')
            try:
                # Prefer parent-aggregated metrics if available
                metrics_epoch = train_avg_dict.get('metrics', {})
                if isinstance(metrics_epoch, dict) and loss_key in metrics_epoch:
                    avg_loss = float(metrics_epoch[loss_key])
                # Fallback to per-epoch accumulators
                if (avg_loss != avg_loss): # NaN check
                    count = getattr(self.trainer, f"{loss_key}_epoch_count", 0)
                    if count > 0:
                        total = getattr(self.trainer, f"{loss_key}_epoch_sum", 0.0)
                        avg_loss = float(total) / float(count)
            except Exception:
                pass
            return avg_loss

        ot_avg_rna = get_avg_ot_loss('ot_rna')
        ot_avg_atac = get_avg_ot_loss('ot_atac')
        triplet_avg_align = get_avg_ot_loss('triplet_align')
        gp_ot_avg = get_avg_ot_loss('gp_ot')
        wd_avg_loss = get_avg_ot_loss('wd_loss')

        # Weighted means (as requested)
        alpha_ot_rna = float(self.extra_cfg.get('alpha_ot_rna', 0.0))
        alpha_ot_atac = float(self.extra_cfg.get('alpha_ot_atac', 0.0))
        alpha_triplet = float(self.extra_cfg.get('alpha_triplet', 0.0))
        alpha_ot_gp = float(self.extra_cfg.get('alpha_ot_gp', 0.0))
        weighted_ot_avg_rna = alpha_ot_rna * ot_avg_rna
        weighted_ot_avg_atac = alpha_ot_atac * ot_avg_atac
        weighted_triplet_avg_align = alpha_triplet * triplet_avg_align
        weighted_gp_ot_avg = alpha_ot_gp * gp_ot_avg
        
        # Calculate total loss (sum of all weighted components including regularization)
        # Handle NaN values gracefully - if a component is NaN, treat it as 0 for total calculation
        def safe_add(*values):
            result = 0.0
            for val in values:
                if val == val:  # NaN check (NaN != NaN is True)
                    result += val
            return result
        
        total_loss = safe_add(pbg_loss_epoch, weighted_ot_avg_rna, weighted_ot_avg_atac, weighted_triplet_avg_align, weighted_gp_ot_avg, wd_avg_loss)
        
        # Write to joint_losses_per_epoch.tsv under checkpoint_path
        try:
            ckpt_dir = getattr(self.config, 'checkpoint_path', None)
            if ckpt_dir is not None:
                os.makedirs(ckpt_dir, exist_ok=True)
                out_path = os.path.join(ckpt_dir, 'joint_losses_per_epoch.tsv')
                header_needed = not os.path.exists(out_path)

                # Dynamically build header and data rows based on active losses
                header = ['epoch', 'loss_total', 'loss_pbg']
                data_row = [total_loss, pbg_loss_epoch] # epoch will be prepended later

                if self.extra_cfg.get('alpha_ot_rna', 0.0) > 0:
                    header.append('loss_ot_rna_weighted')
                    data_row.append(weighted_ot_avg_rna)
                
                if self.extra_cfg.get('alpha_ot_atac', 0.0) > 0:
                    header.append('loss_ot_atac_weighted')
                    data_row.append(weighted_ot_avg_atac)

                if self.extra_cfg.get('alpha_triplet', 0.0) > 0:
                    header.append('loss_triplet_align_weighted')
                    data_row.append(weighted_triplet_avg_align)
                
                if self.extra_cfg.get('alpha_ot_gp', 0.0) > 0:
                    header.append('loss_gp_ot_weighted')
                    data_row.append(weighted_gp_ot_avg)
                
                header.append('loss_wd')
                data_row.append(wd_avg_loss)
                
                with open(out_path, 'a') as f:
                    if header_needed:
                        f.write('\t'.join(header) + '\n')
                    def fmt(x):
                        try:
                            if math.isnan(x) or math.isinf(x):
                                return "N/A"
                            return f"{x:.6f}"
                        except (TypeError, ValueError):
                            return "N/A"
                    
                    # Format epoch as integer, other values as floats
                    epoch_str = str(int(epoch_idx) + 1)
                    loss_strs = [fmt(v) for v in data_row]
                    
                    f.write('\t'.join([epoch_str] + loss_strs) + '\n')

        except Exception as e:
            print(f"  [Warning] Failed to write joint loss summary: {e}")
        
        return out
