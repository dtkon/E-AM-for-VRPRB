from typing import List, Optional, Tuple
import torch

from route_solver import cal_distance


# model reference: Multi-Period Workload Balancing in Last-Mile Urban Delivery


def cal_reward(
    problems: torch.Tensor,
    actions: List[torch.Tensor],
    preference: Optional[torch.Tensor] = None,
    remain_stage: Optional[int] = None,
    cumulative_demand: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: List[(batch_size, node_indexes)], len(actions)==car_number
    action[i]: 0-2-5-0-1-4-7-0-0-0, represents permutation.

    preference: (batch_size, 3)

    remain_stage: 0 means only 1 stage left (or single period).

    cumulative_demand: (batch_size, car_number), cumulative served demand for one car.

    return: [preferenced objective:(batch_size,), updated cumulative_demand:(batch_size, car_number)]
    '''
    batch_size = problems.size(0)
    car_number = len(actions)

    if preference is None:
        preference = torch.ones((batch_size, 3), device=problems.device)

    if cumulative_demand is None:
        cumulative_demand = torch.zeros(
            (batch_size, car_number), device=problems.device
        )

    car_route_length_list = []
    car_route_demand_list = []

    for one_car_actions in actions:
        # one_car_actions: (batch_size, node_indexes)

        one_car_route_length = cal_distance(problems, one_car_actions)  # (batch_size,)
        car_route_length_list.append(one_car_route_length)

        one_car_route_demand = torch.gather(problems[:, :, 2], 1, one_car_actions).sum(
            -1
        )  # (batch_size,)
        car_route_demand_list.append(one_car_route_demand)

    car_route_length = torch.stack(
        car_route_length_list, dim=1
    )  # (batch_size, car_number)
    car_route_demand = torch.stack(
        car_route_demand_list, dim=1
    )  # (batch_size, car_number)

    updated_cumulative_demand = (
        cumulative_demand + car_route_demand
    )  # (batch_size, car_number)

    objective_0 = car_route_length.sum(1)  # (batch_size,)
    objective_1 = car_route_length.max(1)[0]  # (batch_size,)

    if remain_stage is not None and remain_stage == 0:
        objective_2 = (
            updated_cumulative_demand.max(1)[0] - updated_cumulative_demand.min(1)[0]
        )  # (batch_size,)
        obj = (
            objective_0 * preference[:, 0]
            + objective_1 * preference[:, 1]
            + objective_2 * preference[:, 2]
        )
    else:
        obj = objective_0 * preference[:, 0] + objective_1 * preference[:, 1]

    return obj, updated_cumulative_demand
