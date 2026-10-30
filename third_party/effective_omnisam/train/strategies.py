from typing import Protocol, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import sys
import os

# Add utils to path for pseudo-label generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from generate_density_pseudo_labels import generate_density_pseudo_label_batch

class BaseMemStrategy(Protocol):
    def forward(self, model, batch, criterion, device) -> Tuple[torch.Tensor,...]:
        ...
    def forward_eval(self, model, images, frame_idx): ...
    def reset_memory(self, model): ...

class WithMemStrategy:
    def forward(self,
                global_step,
                model,
                batch,
                criterion,
                device,
                optimizer,
                callbacks,
                rank
                ):
        images, labels, *_ = batch
        images, labels = images.to(device), labels.to(device)

        frame_cnt = images.shape[1]
        loss_seq = 0
        random_enable_mem = np.random.choice([1, 0], p=[0.8, 0.2])

        # Check if model has SG-DRF enabled
        enable_sgdrf = hasattr(model.module, 'enable_sgdrf') and model.module.enable_sgdrf

        for frame_idx in range(frame_cnt):
            frame_img = images[:, frame_idx, :, :, :]  # [B, C, H, W]
            frame_lbl = labels[:, frame_idx, :, :]      # [B, H, W]

            if enable_sgdrf:
                # SG-DRF enabled: request density map and stats
                if random_enable_mem:
                    pred, _, _, density_map, sgdrf_stats = model(
                        frame_idx, frame_img,
                        return_sgdrf_aux=True
                    )
                else:
                    pred, _, _, density_map, sgdrf_stats = model(
                        0, frame_img,
                        return_sgdrf_aux=True
                    )

                # Compute segmentation loss
                seg_loss = criterion(pred, frame_lbl)
                total_loss = seg_loss

                # Compute additional SG-DRF losses if in joint training
                if model.module.sgdrf_training_stage == 'joint':
                    # 1. Density loss (MSE against pseudo-label)
                    with torch.no_grad():
                        density_pseudo_label = generate_density_pseudo_label_batch(frame_lbl)  # (B, 1, H, W)
                        # Downsample to match density_map resolution if needed
                        if density_pseudo_label.shape != density_map.shape:
                            density_pseudo_label = F.interpolate(
                                density_pseudo_label,
                                size=density_map.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )

                    density_loss = F.mse_loss(density_map, density_pseudo_label)

                    # 2. Consistency loss (density vs segmentation uncertainty)
                    consistency_loss = model.module.compute_sgdrf_consistency_loss(density_map, pred)

                    # Get loss weights from config (default values)
                    density_weight = getattr(model.module, 'sgdrf_density_loss_weight', 0.1)
                    consistency_weight = getattr(model.module, 'sgdrf_consistency_loss_weight', 0.05)

                    # Total loss
                    total_loss = seg_loss + density_weight * density_loss + consistency_weight * consistency_loss

                    # Logging (optional)
                    if global_step % 100 == 0 and rank == 0:
                        print(f"[SG-DRF] Step {global_step}: seg_loss={seg_loss.item():.4f}, "
                              f"density_loss={density_loss.item():.4f}, "
                              f"consistency_loss={consistency_loss.item():.4f}")
                        if sgdrf_stats:
                            print(f"[SG-DRF Routing] {sgdrf_stats}")

                elif model.module.sgdrf_training_stage == 'cold_start':
                    # Cold-start: only train density estimator with pseudo-label
                    with torch.no_grad():
                        density_pseudo_label = generate_density_pseudo_label_batch(frame_lbl)
                        if density_pseudo_label.shape != density_map.shape:
                            density_pseudo_label = F.interpolate(
                                density_pseudo_label,
                                size=density_map.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )

                    density_loss = F.mse_loss(density_map, density_pseudo_label)
                    # Keep segmentation as the primary cold-start objective.
                    density_weight = getattr(model.module, 'sgdrf_density_loss_weight', 0.1)
                    total_loss = seg_loss + density_weight * density_loss

                    # Logging
                    if global_step % 100 == 0 and rank == 0:
                        print(f"[SG-DRF Cold-start] Step {global_step}: seg_loss={seg_loss.item():.4f}, density_loss={density_loss.item():.4f}, density_weight={density_weight:.4f}")
                        if sgdrf_stats:
                            print(f"[SG-DRF Routing] {sgdrf_stats}")

                loss = total_loss

            else:
                # Original OmniSAM behavior (without SG-DRF)
                if random_enable_mem:
                    pred = model(frame_idx, frame_img)
                else:
                    pred = model(0, frame_img)

                loss = criterion(pred, frame_lbl)

            optimizer.zero_grad()
            loss.backward()
            clip_norm = float(getattr(model.module, "sgdrf_grad_clip_norm", 1.0)) if enable_sgdrf else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            loss_seq += loss.item()

        loss_avg = loss_seq / frame_cnt

        for cb in callbacks:
            if hasattr(cb, 'on_iter_end'):
                cb.on_iter_end(global_step=global_step,
                            loss=loss_avg,
                            lr=optimizer.param_groups[0]['lr'],
                            rank=rank)

        self.reset_memory(model)


    def forward_eval(self, model, images, frame_idx):
        # During evaluation, we don't need SG-DRF auxiliary outputs
        # Just get the prediction
        enable_sgdrf = hasattr(model.module, 'enable_sgdrf') and model.module.enable_sgdrf

        if enable_sgdrf:
            # SG-DRF enabled but don't request aux outputs
            pred, _, _ = model(frame_idx, images[:, frame_idx], mem_type='source', return_sgdrf_aux=False)
            return pred
        else:
            # Original OmniSAM behavior
            return model(frame_idx, images[:, frame_idx], mem_type='source')

    def reset_memory(self, model):
        # Reset memory bank
        if hasattr(model.module, 'memory_bank'):
            model.module.memory_bank._init_output_dict_source()
            model.module.memory_bank._init_output_dict_target()


class NoMemStrategy(WithMemStrategy):
    def forward(self,
                global_step,
                model,
                batch,
                criterion,
                device,
                optimizer,
                callbacks,
                rank
                ):
        images, labels, *_ = batch
        images, labels = images.to(device), labels.to(device)

        # Check if SG-DRF is enabled
        enable_sgdrf = hasattr(model.module, 'enable_sgdrf') and model.module.enable_sgdrf

        if enable_sgdrf:
            # SG-DRF enabled: request density map and stats
            pred, _, _, density_map, sgdrf_stats = model(
                0, images,
                return_sgdrf_aux=True
            )

            # Compute segmentation loss
            seg_loss = criterion(pred, labels)
            total_loss = seg_loss

            # Compute additional SG-DRF losses if in joint training
            if model.module.sgdrf_training_stage == 'joint':
                # 1. Density loss
                with torch.no_grad():
                    density_pseudo_label = generate_density_pseudo_label_batch(labels)
                    if density_pseudo_label.shape != density_map.shape:
                        density_pseudo_label = F.interpolate(
                            density_pseudo_label,
                            size=density_map.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )

                density_loss = F.mse_loss(density_map, density_pseudo_label)

                # 2. Consistency loss
                consistency_loss = model.module.compute_sgdrf_consistency_loss(density_map, pred)

                # Get loss weights
                density_weight = getattr(model.module, 'sgdrf_density_loss_weight', 0.1)
                consistency_weight = getattr(model.module, 'sgdrf_consistency_loss_weight', 0.05)

                # Total loss
                total_loss = seg_loss + density_weight * density_loss + consistency_weight * consistency_loss

                if global_step % 100 == 0 and rank == 0:
                    print(f"[SG-DRF] Step {global_step}: seg_loss={seg_loss.item():.4f}, "
                          f"density_loss={density_loss.item():.4f}, "
                          f"consistency_loss={consistency_loss.item():.4f}")
                    if sgdrf_stats:
                        print(f"[SG-DRF Routing] {sgdrf_stats}")

            elif model.module.sgdrf_training_stage == 'cold_start':
                # Cold-start keeps segmentation active; density is auxiliary
                with torch.no_grad():
                    density_pseudo_label = generate_density_pseudo_label_batch(labels)
                    if density_pseudo_label.shape != density_map.shape:
                        density_pseudo_label = F.interpolate(
                            density_pseudo_label,
                            size=density_map.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )

                density_loss = F.mse_loss(density_map, density_pseudo_label)
                density_weight = getattr(model.module, 'sgdrf_density_loss_weight', 0.1)
                total_loss = seg_loss + density_weight * density_loss

                if global_step % 100 == 0 and rank == 0:
                    print(f"[SG-DRF Cold-start] Step {global_step}: seg_loss={seg_loss.item():.4f}, density_loss={density_loss.item():.4f}, density_weight={density_weight:.4f}")
                    if sgdrf_stats:
                        print(f"[SG-DRF Routing] {sgdrf_stats}")

            loss = total_loss

        else:
            # Original OmniSAM behavior
            pred = model(0, images)
            loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_norm = float(getattr(model.module, "sgdrf_grad_clip_norm", 1.0)) if enable_sgdrf else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        for cb in callbacks:
            if hasattr(cb, 'on_iter_end'):
                cb.on_iter_end(global_step=global_step,
                            loss=loss.item(),
                            lr=optimizer.param_groups[0]['lr'],
                            rank=rank)

        self.reset_memory(model)

    def forward_eval(self, model, images, frame_idx):
        enable_sgdrf = hasattr(model.module, 'enable_sgdrf') and model.module.enable_sgdrf

        if enable_sgdrf:
            pred, _, _ = model(0, images[:, frame_idx], mem_type='source', return_sgdrf_aux=False)
            return pred
        else:
            return model(0, images[:, frame_idx], mem_type='source')
