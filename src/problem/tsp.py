from typing import Any, Optional, Mapping
import torch

from route_solver import TspSolver, cal_distance

from .cvrp import cal_reward, get_duplicate_mask
from .cvrp_balance import cal_reward_MO


def random_generate(batch_size: int, customer_number: int) -> torch.Tensor:
    '''
    return: (batch_size, customer_number, 2)
    '''
    return torch.rand((batch_size, customer_number, 2))


def generate_datasets(
    batch_size: int, customer_size: int, save_path: Optional[str] = None, **kwargs: Any
) -> torch.Tensor:
    data = random_generate(batch_size, customer_size)

    if save_path is not None:
        torch.save(data, save_path)

    return data


def cal_reward_MO_refine(
    problems: torch.Tensor,
    actions: torch.Tensor,
    tsp_solvers: Mapping[int, TspSolver],
) -> torch.Tensor:
    '''
    problems: [batch_size, problem_size, 2(x,y)]

    actions: (batch_size, vehicle_num, node_indexes)

    return: [batch_size, 2(objective_sum, objective_max)]
    '''
    batch_size, vehicle_num, _ = actions.size()

    problems_aggregate = problems.repeat_interleave(
        vehicle_num, 0
    )  # (batch_size*vehicle_num, problem_size, 2)

    actions_aggregate = actions.view(
        batch_size * vehicle_num, -1
    )  # (batch_size*vehicle_num, node_indexes)

    # tsp_refine
    problems_of_tsp = problems_aggregate.gather(
        1, actions_aggregate.unsqueeze(2).repeat(1, 1, 2)
    )

    mask = actions_aggregate == 0
    mask[:, 0] = False

    duplicate = get_duplicate_mask(actions_aggregate)
    mask[duplicate] = True

    tsp_length_list = []
    for tsp_solver in tsp_solvers.values():
        length, _ = tsp_solver.solve(problems_of_tsp, mask, 'argmax')
        tsp_length_list.append(length)
    tsp_length = torch.stack(tsp_length_list, dim=1).min(dim=1)[
        0
    ]  # (batch_size*vehicle_num,)

    # compare with original
    direct_length = cal_distance(
        problems_aggregate, actions_aggregate
    )  # (batch_size*vehicle_num,)

    route_length = torch.stack((direct_length, tsp_length), dim=1).min(1)[0]

    car_route_length = route_length.view(
        batch_size, vehicle_num
    )  # (batch_size, car_number)

    objective_sum = car_route_length.sum(1)  # (batch_size,)
    objective_max = car_route_length.max(1)[0]  # (batch_size,)

    return -torch.stack((objective_sum, objective_max), dim=1)
