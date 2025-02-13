from typing import Optional, Tuple
import torch


def zoom(
    problems: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    problem: [batch_size, problem_size, (x, y, ...)]

    mask: (batch_size, problem_size), mask[:, 0] must all be False.

    return: [problem_zoom, gap(batch_size,)]
    '''
    if mask is not None:
        depot = problems[:, 0:1, :].repeat(1, problems.size(1), 1)
        problems = torch.where(
            mask.unsqueeze(2).repeat(1, 1, problems.size(2)), depot, problems
        )

    problems_xy = problems[:, :, :2]
    max_coord = problems_xy.max(1)[0]  # (batch_size, 2)
    min_coord = problems_xy.min(1)[0]  # (batch_size, 2)
    x_gap = max_coord[:, 0] - min_coord[:, 0]  # (batch_size,)
    y_gap = max_coord[:, 1] - min_coord[:, 1]  # (batch_size,)
    xy_gap = torch.cat([x_gap[None, :], y_gap[None, :]])  # (2, batch_size)
    gap = xy_gap.max(0)[0]  # (batch_size,)
    problem_zoom = (problems_xy - min_coord[:, None, :]) / gap[:, None, None]

    if problems.size(2) > 2:
        problem_zoom = torch.cat((problem_zoom, problems[:, :, 2:]), dim=2)

    return problem_zoom, gap


def cal_distance(
    problems: torch.Tensor, routes: torch.Tensor, cycle: bool = False
) -> torch.Tensor:
    '''
    problems: [batch_size, problem_size, (x, y, ...)]

    routes: (batch_size, node_indexes)

    return: (batch_size,)
    '''
    routes = routes.unsqueeze(-1).repeat(1, 1, 2)
    seq_nodes = torch.gather(
        problems[:, :, :2], 1, routes
    )  # (batch_size, node_indexes, 2)

    if not cycle:
        next_nodes = seq_nodes[:, 1:, :]
        return torch.linalg.vector_norm(next_nodes - seq_nodes[:, :-1, :], dim=-1).sum(
            -1
        )
    else:
        next_nodes = torch.cat((seq_nodes[:, 1:, :], seq_nodes[:, 0:1, :]), 1)
        return torch.linalg.vector_norm(next_nodes - seq_nodes, dim=-1).sum(-1)
