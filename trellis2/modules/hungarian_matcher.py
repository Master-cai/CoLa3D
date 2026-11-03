


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


__all__ = ["HungarianMatcher"]


class HungarianMatcher(nn.Module):


    def __init__(
        self,
        cost_bce: float = 1.0,
        cost_dice: float = 1.0,
    ):
        super().__init__()
        self.cost_bce = cost_bce
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        mask_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        num_objects: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        valid_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        batch_size, num_points, num_queries = mask_logits.shape
        max_objects = gt_masks.shape[-1]
        device = mask_logits.device


        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, num_points, dtype=torch.bool, device=device
            )
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            if valid_mask.shape != (batch_size, num_points):
                raise ValueError(
                    "valid_mask must have shape (B, N), "
                    f"got {tuple(valid_mask.shape)} vs {(batch_size, num_points)}"
                )
        if valid_queries is None:
            valid_queries = torch.ones(
                batch_size, num_queries, dtype=torch.bool, device=device
            )
        else:
            valid_queries = valid_queries.to(device=device, dtype=torch.bool)
            if valid_queries.shape != (batch_size, num_queries):
                raise ValueError(
                    "valid_queries must have shape (B, Q), "
                    f"got {tuple(valid_queries.shape)} vs {(batch_size, num_queries)}"
                )


        pred_probs = torch.sigmoid(mask_logits)

        results = []
        for b in range(batch_size):
            n_valid_obj = int(num_objects[b].item())
            query_mask = valid_queries[b].bool()
            query_indices = query_mask.nonzero(as_tuple=False).squeeze(1)

            if n_valid_obj == 0 or int(query_mask.sum().item()) == 0:

                results.append(
                    torch.full((num_queries,), -1, dtype=torch.long, device=device)
                )
                continue


            batch_valid = valid_mask[b]
            pred_batch = pred_probs[b][batch_valid][:, query_mask]

            gt_batch = gt_masks[b][
                batch_valid, :n_valid_obj
            ].float()


            cost_matrix = self._compute_cost_matrix(pred_batch, gt_batch)


            cost_np = cost_matrix.cpu().numpy()

            cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=-1e6)
            row_indices, col_indices = linear_sum_assignment(cost_np)


            batch_result = torch.full(
                (num_queries,), -1, dtype=torch.long, device=device
            )
            batch_result[query_indices[col_indices]] = torch.tensor(
                row_indices, dtype=torch.long, device=device
            )
            results.append(batch_result)

        return torch.stack(results)

    def _compute_cost_matrix(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:


        N, Q = pred.shape
        O = gt.shape[1]


        eps = 1e-5
        pred_clamped = pred.clamp(eps, 1 - eps)


        bce = -(
            gt.unsqueeze(2) * torch.log(pred_clamped.unsqueeze(1))
            + (1 - gt.unsqueeze(2)) * torch.log(1 - pred_clamped.unsqueeze(1))
        )
        bce_cost = bce.mean(dim=0)


        intersection = (pred.unsqueeze(1) * gt.unsqueeze(2)).sum(dim=0)
        pred_sum = pred.sum(dim=0).unsqueeze(0)
        gt_sum = gt.sum(dim=0).unsqueeze(1)

        dice = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
        dice_cost = 1 - dice


        cost = self.cost_bce * bce_cost + self.cost_dice * dice_cost

        return cost

    @torch.no_grad()
    def get_reordered_gt(
        self,
        matching: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:


        batch_size, num_points, max_objects = gt_masks.shape
        num_queries = matching.shape[1]
        device = gt_masks.device

        gt_reordered = torch.zeros(
            batch_size, num_points, num_queries, dtype=gt_masks.dtype, device=device
        )

        for b in range(batch_size):
            for q in range(num_queries):
                matched_obj = matching[b, q].item()
                if matched_obj >= 0:

                    gt_reordered[b, :, q] = gt_masks[b, :, matched_obj]

        return gt_reordered


def dice_loss_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-4,
) -> torch.Tensor:


    intersection = (pred * target).sum(dim=0)
    pred_sum = pred.sum(dim=0)
    target_sum = target.sum(dim=0)


    dice_per_class = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)


    valid_classes = target_sum > 0
    if valid_classes.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    mean_dice = dice_per_class[valid_classes].mean()
    return 1 - mean_dice
