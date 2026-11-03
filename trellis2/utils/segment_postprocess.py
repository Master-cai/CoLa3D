


from typing import Tuple, List, Dict
from collections import deque, Counter

import torch


_OFFSETS_6 = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]


def multihot_assign(
    probs: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:


    multihot = probs > threshold

    no_assign = ~multihot.any(dim=1)
    if no_assign.any():
        fallback_idx = probs[no_assign].argmax(dim=1)
        multihot[no_assign, fallback_idx] = True
    return multihot


def query_nms(
    multihot: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:


    Q = multihot.shape[1]
    order = scores.argsort(descending=True).tolist()

    keep: List[int] = []

    for q in order:
        mask_q = multihot[:, q]
        if not mask_q.any():
            continue

        merged = False
        for k in keep:
            mask_k = multihot[:, k]
            intersection = (mask_q & mask_k).sum().item()
            union = (mask_q | mask_k).sum().item()
            if union == 0:
                continue
            iou = intersection / union
            if iou > iou_threshold:

                multihot[:, k] = mask_q | mask_k
                multihot[:, q] = False
                merged = True
                break

        if not merged:
            keep.append(q)

    return multihot


def connected_components_3d(
    coords: torch.Tensor,
) -> List[List[int]]:


    M = coords.shape[0]
    if M == 0:
        return []


    coords_cpu = coords.cpu().tolist()
    coord_to_idx: Dict[Tuple[int, int, int], int] = {}
    for i, (x, y, z) in enumerate(coords_cpu):
        coord_to_idx[(x, y, z)] = i

    visited = [False] * M
    components: List[List[int]] = []

    for start in range(M):
        if visited[start]:
            continue
        component: List[int] = []
        queue = deque([start])
        visited[start] = True
        while queue:
            idx = queue.popleft()
            component.append(idx)
            cx, cy, cz = coords_cpu[idx]
            for dx, dy, dz in _OFFSETS_6:
                nb = (cx + dx, cy + dy, cz + dz)
                nb_idx = coord_to_idx.get(nb)
                if nb_idx is not None and not visited[nb_idx]:
                    visited[nb_idx] = True
                    queue.append(nb_idx)
        components.append(component)

    return components


def suppress_fragments(
    multihot: torch.Tensor,
    primary_labels: torch.Tensor,
    coords: torch.Tensor,
    probs: torch.Tensor,
    fragment_min_voxels: int = 5,
    fragment_ratio: float = 0.05,
    max_refine_iters: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:


    N = coords.shape[0]
    coords_cpu = coords.cpu()

    multihot = multihot.cpu()
    primary_labels = primary_labels.cpu()
    probs = probs.cpu()


    coords_list = coords_cpu.tolist()
    coord_map: Dict[Tuple[int, int, int], int] = {
        (int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(coords_list)
    }

    for _iter in range(max_refine_iters):
        changed = False
        active_labels = primary_labels.unique().tolist()

        for label in active_labels:
            label_mask = primary_labels == label
            label_indices = label_mask.nonzero(as_tuple=True)[0]
            total = label_indices.numel()
            if total == 0:
                continue


            label_coords = coords_cpu[label_indices]
            components = connected_components_3d(label_coords)


            if len(components) <= 1:
                continue


            largest_idx = max(range(len(components)), key=lambda i: len(components[i]))

            for comp_idx, comp in enumerate(components):
                comp_size = len(comp)

                if comp_idx == largest_idx:
                    continue


                is_fragment = (
                    comp_size < fragment_min_voxels
                    or comp_size < fragment_ratio * total
                )
                if not is_fragment:
                    continue


                for local_idx in comp:
                    global_idx = label_indices[local_idx].item()
                    cx, cy, cz = coords_cpu[global_idx].tolist()
                    cx, cy, cz = int(cx), int(cy), int(cz)

                    neighbor_labels: List[int] = []
                    for dx, dy, dz in _OFFSETS_6:
                        nb = (cx + dx, cy + dy, cz + dz)
                        nb_global = coord_map.get(nb)
                        if nb_global is not None:
                            nb_label = primary_labels[nb_global].item()
                            if nb_label != label:
                                neighbor_labels.append(nb_label)

                    if not neighbor_labels:
                        continue


                    vote = Counter(neighbor_labels).most_common(1)[0][0]


                    primary_labels[global_idx] = vote

                    multihot[global_idx, label] = False
                    multihot[global_idx, vote] = True
                    changed = True

        if not changed:
            break

    return multihot, primary_labels


def suppress_fragments_aggressive(
    multihot: torch.Tensor,
    primary_labels: torch.Tensor,
    coords: torch.Tensor,
    probs: torch.Tensor,
    multilabel_mask: torch.Tensor,
    max_refine_iters: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:


    coords_cpu = coords.cpu()
    multihot = multihot.cpu()
    primary_labels = primary_labels.cpu()
    probs = probs.cpu()
    multilabel_mask = multilabel_mask.cpu()


    coords_list = coords_cpu.tolist()
    coord_map: Dict[Tuple[int, int, int], int] = {
        (int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(coords_list)
    }

    for _iter in range(max_refine_iters):
        changed = False
        active_labels = primary_labels.unique().tolist()

        for label in active_labels:
            label_mask = primary_labels == label
            label_indices = label_mask.nonzero(as_tuple=True)[0]
            if label_indices.numel() == 0:
                continue


            label_coords = coords_cpu[label_indices]
            components = connected_components_3d(label_coords)

            if len(components) <= 1:
                continue


            largest_idx = max(range(len(components)), key=lambda i: len(components[i]))

            for comp_idx, comp in enumerate(components):
                if comp_idx == largest_idx:
                    continue


                global_indices = [label_indices[li].item() for li in comp]
                if any(multilabel_mask[gi].item() for gi in global_indices):
                    continue


                for local_idx in comp:
                    global_idx = label_indices[local_idx].item()
                    cx, cy, cz = coords_cpu[global_idx].tolist()
                    cx, cy, cz = int(cx), int(cy), int(cz)

                    neighbor_labels: List[int] = []
                    for dx, dy, dz in _OFFSETS_6:
                        nb = (cx + dx, cy + dy, cz + dz)
                        nb_global = coord_map.get(nb)
                        if nb_global is not None:
                            nb_label = primary_labels[nb_global].item()
                            if nb_label != label:
                                neighbor_labels.append(nb_label)

                    if neighbor_labels:
                        vote = Counter(neighbor_labels).most_common(1)[0][0]
                        primary_labels[global_idx] = vote
                        multihot[global_idx, label] = False
                        multihot[global_idx, vote] = True
                    else:

                        multihot[global_idx, label] = False
                    changed = True

        if not changed:
            break


    orphans = ~multihot.any(dim=1)
    if orphans.any():
        best = probs[orphans].argmax(dim=1)
        multihot[orphans, best] = True

        orphan_masked = probs[orphans].clone()
        orphan_masked[~multihot[orphans]] = -1.0
        primary_labels[orphans] = orphan_masked.argmax(dim=1)

    return multihot, primary_labels


def compact_queries(
    multihot: torch.Tensor,
    primary_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:


    Q = multihot.shape[1]
    active_mask = multihot.any(dim=0)
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    K = active_indices.numel()

    if K == 0:

        N = multihot.shape[0]
        return (
            torch.ones(N, 1, dtype=torch.bool, device=multihot.device),
            torch.zeros(N, dtype=torch.long, device=primary_labels.device),
            1,
            torch.full((Q,), -1, dtype=torch.long, device=multihot.device),
        )


    query_to_part = torch.full((Q,), -1, dtype=torch.long, device=multihot.device)
    query_to_part[active_indices] = torch.arange(
        K, dtype=torch.long, device=multihot.device
    )


    multihot_compact = multihot[:, active_indices]


    primary_labels_compact = query_to_part[primary_labels]

    primary_labels_compact = primary_labels_compact.clamp(min=0)

    return multihot_compact, primary_labels_compact, K, query_to_part


def postprocess_segmentation(
    mask_logits: torch.Tensor,
    coords: torch.Tensor,
    assign_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    fragment_min_voxels: int = 5,
    fragment_ratio: float = 0.05,
    max_refine_iters: int = 2,
    min_object_voxels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, int]:


    N, Q = mask_logits.shape


    probs = torch.sigmoid(mask_logits)
    multihot = multihot_assign(probs, threshold=assign_threshold)


    scores = (probs * multihot.float()).sum(dim=0)
    multihot = query_nms(multihot, scores, iou_threshold=nms_iou_threshold)


    query_sizes = multihot.float().sum(dim=0)
    tiny_queries = query_sizes < min_object_voxels
    if tiny_queries.any():

        for q in tiny_queries.nonzero(as_tuple=True)[0].tolist():
            voxel_mask = multihot[:, q]
            multihot[:, q] = False


            orphans = voxel_mask & ~multihot.any(dim=1)
            if orphans.any():

                remaining_probs = probs[orphans].clone()
                remaining_probs[:, tiny_queries] = -1.0
                best = remaining_probs.argmax(dim=1)
                multihot[orphans, best] = True


    no_assign = ~multihot.any(dim=1)
    if no_assign.any():
        fallback_idx = probs[no_assign].argmax(dim=1)
        multihot[no_assign, fallback_idx] = True


    masked_probs = probs.clone()
    masked_probs[~multihot] = -1.0
    primary_labels = masked_probs.argmax(dim=1)


    multihot, primary_labels = suppress_fragments(
        multihot=multihot,
        primary_labels=primary_labels,
        coords=coords,
        probs=probs,
        fragment_min_voxels=fragment_min_voxels,
        fragment_ratio=fragment_ratio,
        max_refine_iters=max_refine_iters,
    )


    multihot_final, primary_labels, num_parts, _ = compact_queries(
        multihot,
        primary_labels,
    )

    return multihot_final, primary_labels, num_parts


def postprocess_segmentation_mask_nms(
    mask_logits: torch.Tensor,
    coords: torch.Tensor,
    assign_threshold: float = 0.5,
    min_object_voxels: int = 20,
    containment_threshold: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, int]:


    N, Q = mask_logits.shape
    device = mask_logits.device


    probs = torch.sigmoid(mask_logits)
    multihot_init = multihot_assign(probs, threshold=assign_threshold)


    coords_cpu = coords.cpu()
    probs_cpu = probs.cpu()
    multihot_cpu = multihot_init.cpu()

    proposals: List[Dict] = []
    for q in range(Q):
        q_mask = multihot_cpu[:, q]
        if int(q_mask.sum()) < min_object_voxels:
            continue

        q_indices = q_mask.nonzero(as_tuple=True)[0]
        q_coords = coords_cpu[q_indices]
        ccs = connected_components_3d(q_coords)

        for cc in ccs:
            if len(cc) < min_object_voxels:
                continue
            global_idx = q_indices[torch.tensor(cc, dtype=torch.long)]
            mask = torch.zeros(N, dtype=torch.bool)
            mask[global_idx] = True
            score = float(probs_cpu[global_idx, q].mean().item())
            proposals.append({"mask": mask, "score": score, "q": q})


    proposals.sort(key=lambda p: -p["score"])
    kept: List[Dict] = []
    for prop in proposals:
        mask = prop["mask"]
        size = int(mask.sum())
        if size == 0:
            continue

        absorb = False
        for k in kept:
            k_mask = k["mask"]
            k_size = int(k_mask.sum())
            inter = int((mask & k_mask).sum())
            if inter == 0:
                continue


            min_size = min(size, k_size)
            if inter / min_size > containment_threshold:
                absorb = True
                break
        if not absorb:
            kept.append(prop)


    K = len(kept)
    if K == 0:


        multihot_final = torch.ones(N, 1, dtype=torch.bool, device=device)
        primary_labels = torch.zeros(N, dtype=torch.long, device=device)
        return multihot_final, primary_labels, 1

    multihot_final = torch.zeros(N, K, dtype=torch.bool)
    proposal_qs = torch.empty(K, dtype=torch.long)
    for i, prop in enumerate(kept):
        multihot_final[:, i] = prop["mask"]
        proposal_qs[i] = prop["q"]


    assigned = multihot_final.any(dim=1)
    if (~assigned).any():
        orphan_idx = (~assigned).nonzero(as_tuple=True)[0]


        orphan_probs = probs_cpu[orphan_idx][:, proposal_qs]
        best = orphan_probs.argmax(dim=1)
        multihot_final[orphan_idx, best] = True


    masked_probs = probs_cpu[:, proposal_qs].clone()
    masked_probs[~multihot_final] = -1.0
    primary_labels = masked_probs.argmax(dim=1)

    return multihot_final.to(device), primary_labels.to(device), K


def postprocess_segmentation_aggressive(
    mask_logits: torch.Tensor,
    coords: torch.Tensor,
    assign_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_refine_iters: int = 2,
    min_object_voxels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, int]:


    N, Q = mask_logits.shape


    probs = torch.sigmoid(mask_logits)
    multihot = multihot_assign(probs, threshold=assign_threshold)


    multilabel_mask = multihot.sum(dim=1) >= 2


    scores = (probs * multihot.float()).sum(dim=0)
    multihot = query_nms(multihot, scores, iou_threshold=nms_iou_threshold)


    query_sizes = multihot.float().sum(dim=0)
    tiny_queries = query_sizes < min_object_voxels
    if tiny_queries.any():
        for q in tiny_queries.nonzero(as_tuple=True)[0].tolist():
            voxel_mask = multihot[:, q]
            multihot[:, q] = False
            orphans = voxel_mask & ~multihot.any(dim=1)
            if orphans.any():
                remaining_probs = probs[orphans].clone()
                remaining_probs[:, tiny_queries] = -1.0
                best = remaining_probs.argmax(dim=1)
                multihot[orphans, best] = True


    no_assign = ~multihot.any(dim=1)
    if no_assign.any():
        fallback_idx = probs[no_assign].argmax(dim=1)
        multihot[no_assign, fallback_idx] = True


    masked_probs = probs.clone()
    masked_probs[~multihot] = -1.0
    primary_labels = masked_probs.argmax(dim=1)


    multihot, primary_labels = suppress_fragments_aggressive(
        multihot=multihot,
        primary_labels=primary_labels,
        coords=coords,
        probs=probs,
        multilabel_mask=multilabel_mask,
        max_refine_iters=max_refine_iters,
    )


    multihot_final, primary_labels, num_parts, _ = compact_queries(
        multihot,
        primary_labels,
    )

    return multihot_final, primary_labels, num_parts
