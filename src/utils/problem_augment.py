import math
import random
import torch


def _augment_xy_data_by_8_fold(problems: torch.Tensor) -> torch.Tensor:
    '''
    problems: (batch, problem, 2)
    '''

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def _SR_transform(x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if idx < 0.5:
        phi = idx * 4 * math.pi
    else:
        phi = (idx - 0.5) * 4 * math.pi

    x = x - 1 / 2
    y = y - 1 / 2

    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y

    if idx < 0.5:
        dat = torch.cat((x_prime + 1 / 2, y_prime + 1 / 2), dim=2)
    else:
        dat = torch.cat((y_prime + 1 / 2, x_prime + 1 / 2), dim=2)
    return dat


def _augment_xy_data_by_N_fold(problems: torch.Tensor, N: int) -> torch.Tensor:
    '''
    problems: (batch, problem, 2)
    '''

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    idx = torch.rand(N - 1)

    for i in range(N - 1):
        problems = torch.cat((problems, _SR_transform(x, y, idx[i])), dim=0)
    # (N*batch, problem, 2)
    return problems


def augment_1(
    problems: torch.Tensor, N_aug: int = 8, force_SR: bool = False
) -> torch.Tensor:
    '''
    input: [batch_size, problem_size, (x, y, ...)]

    return: [N_aug*batch, problem, (x, y, ...)]
    '''
    if N_aug == 8 and not force_SR:
        aug_problem = _augment_xy_data_by_8_fold(problems[:, :, :2])
    else:
        aug_problem = _augment_xy_data_by_N_fold(problems[:, :, :2], N_aug)
    if problems.size(2) > 2:
        other_attr = problems[:, :, 2:].repeat(N_aug, 1, 1)
        aug_problem = torch.cat((aug_problem, other_attr), dim=2)
    return aug_problem


def _get_rotate_mat(theta_f: float) -> torch.Tensor:
    theta = torch.tensor(theta_f)
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


def _rotate_tensor(x: torch.Tensor, d: float) -> torch.Tensor:
    rot_mat = _get_rotate_mat(d / 360 * 2 * math.pi).to(x.device)
    return torch.matmul(x - 0.5, rot_mat) + 0.5


def augment_2(
    problems: torch.Tensor,
    N_aug: int,
    merge_to_batch: bool = True,
    one_is_keep: bool = True,
) -> torch.Tensor:
    '''
    problems: [batch_size, graph_size, (x, y, ...)]

    return:
    if merge_to_batch (N_aug*batch_size, graph_size, node_dim), else (batch_size, N_aug, graph_size, node_dim)
    '''
    _, graph_size, node_dim = problems.size()
    problems_xy = problems[:, :, :2].unsqueeze(1).repeat(1, N_aug, 1, 1)

    augments = ['Rotate', 'Flip_x-y', 'Flip_x_cor', 'Flip_y_cor']

    for i in range(N_aug):
        if one_is_keep and i == 0:
            continue
        random.shuffle(augments)
        id_ = torch.rand(4)
        for aug in augments:
            if aug == 'Rotate':
                problems_xy[:, i] = _rotate_tensor(
                    problems_xy[:, i], int(id_[0] * 4 + 1) * 90
                )
            elif aug == 'Flip_x-y':
                if int(id_[1] * 2 + 1) == 1:
                    data = problems_xy[:, i].clone()
                    problems_xy[:, i, :, 0] = data[:, :, 1]
                    problems_xy[:, i, :, 1] = data[:, :, 0]
            elif aug == 'Flip_x_cor':
                if int(id_[2] * 2 + 1) == 1:
                    problems_xy[:, i, :, 0] = 1 - problems_xy[:, i, :, 0]
            elif aug == 'Flip_y_cor':
                if int(id_[3] * 2 + 1) == 1:
                    problems_xy[:, i, :, 1] = 1 - problems_xy[:, i, :, 1]

    # problems_xy: (batch_size, N_aug, graph_size, 2)

    if problems.size(2) > 2:
        other_attr = (
            problems[:, :, 2:].unsqueeze(1).repeat(1, N_aug, 1, 1)
        )  # (batch_size, N_aug, graph_size, 1+)
        aug_problem = torch.cat(
            (problems_xy, other_attr), dim=-1
        )  # (batch_size, N_aug, graph_size, node_dim)
    else:
        aug_problem = problems_xy

    if merge_to_batch:
        aug_problem = aug_problem.permute(1, 0, 2, 3).reshape(
            -1, graph_size, node_dim
        )  # (N_aug*batch_size, graph_size, node_dim)

    return aug_problem


augment = augment_1
