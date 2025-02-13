import math
from typing import Dict, Iterator, List, Tuple, Union
import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


def clip_grad_norms(
    param_groups: List[dict], max_norm: float = math.inf
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [
            min(g_norm, torch.tensor(max_norm), key=lambda x: x.item())
            for g_norm in grad_norms
        ]
        if max_norm > 0
        else grad_norms
    )
    return grad_norms, grad_norms_clipped


def get_parameter_number(net: nn.Module) -> Dict[str, int]:
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def batch_slicer(total: int, parallel: int) -> Iterator[Tuple[int, int]]:
    assert total >= 0 and parallel >= 1

    start = 0
    remain = total
    while (remain := remain - parallel) > (-parallel):
        if remain >= 0:
            pick_count = parallel
        else:
            pick_count = remain + parallel
        end = start + pick_count
        yield start, end
        start = end


def get_inner_model(model: Union[nn.Module, DDP, DataParallel]) -> nn.Module:
    return model.module if isinstance(model, (DDP, DataParallel)) else model
