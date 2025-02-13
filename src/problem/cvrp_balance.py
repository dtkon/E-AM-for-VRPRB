from typing import Any, List, Mapping, Optional
from collections import defaultdict
import math
import torch
import numpy as np

from route_solver import TspSolver, CvrpSolver, cal_distance

from .cvrp import (
    split_action_with_wait,
    align_actions_to_wait,
    get_duplicate_mask,
    routes_number,
    CAPACITIES,
)


def random_generate(
    batch_size: int,
    customer_number: int,
    max_demand: int,
    capacity: int,
    vehicle_num: int,
    generate_size: int = 0,
) -> torch.Tensor:
    '''
    return: (batch_size, customer_number+1, 3)

    demand range: 1 ~ max_demand, then divided by capacity

    depot demand = 0
    '''
    if vehicle_num < math.ceil(customer_number / capacity) + 1:
        raise RuntimeError('fleet capacity too small')

    if generate_size == 0:
        generate_size = batch_size * 2

    problems = torch.rand((batch_size, customer_number + 1, 2))

    all_good_demand = torch.tensor([])

    while all_good_demand.size(0) < batch_size:
        demand = torch.randint(1, max_demand + 1, (generate_size, customer_number + 1))
        demand[:, 0] = 0
        if_good_demand = (
            vehicle_num >= torch.ceil(demand.sum(1) / capacity) + 1
        )  # (generate_size,)
        good_demand = demand[if_good_demand]
        all_good_demand = torch.cat((all_good_demand, good_demand))

    sel_good_demand = all_good_demand[:batch_size] / capacity  # (batch_size, cus_num+1)

    return torch.cat((problems, sel_good_demand.unsqueeze(2)), dim=2)


def generate_datasets(
    batch_size: int,
    customer_size: int,
    vehicle_num: int,
    vehicle_capacity: int = -1,
    save_npy_path: Optional[str] = None,
    **kwargs: Any
) -> torch.Tensor:
    if vehicle_capacity == -1:
        capacity = CAPACITIES[customer_size]
    else:
        capacity = vehicle_capacity

    data = random_generate(batch_size, customer_size, 9, capacity, vehicle_num)

    if save_npy_path is not None:
        np.save(save_npy_path, data.cpu().numpy())

    return data


def cal_reward_MO(
    problems: torch.Tensor, actions: torch.Tensor, only_max: bool = False
) -> torch.Tensor:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, vehicle_num, node_indexes)

    return: [batch_size, 2(objective_sum, objective_max)]
    '''
    batch_size, vehicle_num, _ = actions.size()
    problems = problems.repeat_interleave(vehicle_num, 0)

    route_length = cal_distance(
        problems, actions.view(batch_size * vehicle_num, -1)
    ).view(batch_size, vehicle_num)

    objective_max = route_length.max(1)[0]  # (batch_size,)

    if not only_max:
        objective_sum = route_length.sum(1)  # (batch_size,)
        return -torch.stack((objective_sum, objective_max), dim=1)
    else:
        return -objective_max


def cal_reward_MO_refine_MultiTrip(
    problems: torch.Tensor,
    actions: torch.Tensor,
    tsp_solvers: Optional[Mapping[int, TspSolver]] = None,
    cvrp_solvers: Optional[Mapping[int, CvrpSolver]] = None,
) -> torch.Tensor:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, vehicle_num, node_indexes)

    return: [batch_size, 2(objective_sum, objective_max)]
    '''
    batch_size, vehicle_num, _ = actions.size()

    problems_aggregate = problems.repeat_interleave(
        vehicle_num, 0
    )  # (batch_size*vehicle_num, problem_size, 3)

    actions_aggregate = actions.view(
        batch_size * vehicle_num, -1
    )  # (batch_size*vehicle_num, node_indexes)

    # vrp_refine
    if cvrp_solvers is not None:
        problems_of_cvrp = problems_aggregate.gather(
            1, actions_aggregate.unsqueeze(2).repeat(1, 1, 3)
        )  # (batch_size*vehicle_num, node_indexes, 3)
        mask = actions_aggregate == 0
        mask[:, 0] = False

        duplicate = get_duplicate_mask(actions_aggregate)
        mask[duplicate] = True

        cvrp_length_list = []
        for cvrp_solver in cvrp_solvers.values():
            length, _ = cvrp_solver.solve(problems_of_cvrp, mask, 'argmax')
            cvrp_length_list.append(length)
        cvrp_length = torch.stack(cvrp_length_list, dim=1).min(dim=1)[
            0
        ]  # (batch_size*vehicle_num,)
    else:
        cvrp_length = torch.empty(batch_size * vehicle_num, device=problems.device)
        cvrp_length[:] = torch.inf

    # tsp_refine
    tsp_sum_length = torch.zeros(batch_size * vehicle_num, device=problems.device)

    if tsp_solvers is not None:
        actions_aggregate_align = align_actions_to_wait(actions_aggregate)
        for (
            no_need_calculate,
            tsp_in_vrp,
            problems_of_tsp,
            mask,
        ) in split_action_with_wait(problems_aggregate, actions_aggregate_align):
            if not no_need_calculate:
                tsp_length_list = []
                for tsp_solver in tsp_solvers.values():
                    length, _ = tsp_solver.solve(problems_of_tsp, mask, 'argmax')
                    tsp_length_list.append(length)
                tsp_length = torch.stack(tsp_length_list, dim=1).min(dim=1)[
                    0
                ]  # (batch_size*vehicle_num,)

                tsp_sum_length += tsp_length
    else:
        tsp_sum_length[:] = torch.inf

    # compare with original
    direct_length = cal_distance(
        problems_aggregate, actions_aggregate
    )  # (batch_size*vehicle_num,)

    route_length = torch.stack((direct_length, tsp_sum_length, cvrp_length), dim=1).min(
        1
    )[0]

    car_route_length = route_length.view(
        batch_size, vehicle_num
    )  # (batch_size, car_number)

    objective_sum = car_route_length.sum(1)  # (batch_size,)
    objective_max = car_route_length.max(1)[0]  # (batch_size,)

    return -torch.stack((objective_sum, objective_max), dim=1)


def check_action_legal(problems: torch.Tensor, actions: torch.Tensor) -> bool:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, vehicle_num, node_indexes)
    '''
    batch_size, vehicle_num, _ = actions.size()

    problems_aggregate = problems.repeat_interleave(
        vehicle_num, 0
    )  # (batch_size*vehicle_num, problem_size, 3)

    actions_aggregate = actions.view(
        batch_size * vehicle_num, -1
    )  # (batch_size*vehicle_num, node_indexes)

    actions_aggregate_align = align_actions_to_wait(actions_aggregate)

    for (
        no_need_calculate,
        tsp_in_vrp,
        problems_of_tsp,
        mask,
    ) in split_action_with_wait(problems_aggregate, actions_aggregate_align):
        if not no_need_calculate:
            demands_of_tsp = problems_of_tsp[:, :, 2]
            demands_of_tsp[mask] = 0
            if torch.any(demands_of_tsp.sum(1) > 1 + 1e-6):
                return False

    return True


def convert_cvrp_actions_to_max_vehicle_num(
    problem: torch.Tensor, actions: torch.Tensor, vehicle_num: int
) -> List[torch.Tensor]:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, node_indexes)

    return: merged_actions: List[(batch_size, node_indexes)], len(merged_actions)==car_number
    '''
    vehicle_batch_action_list: List[List[torch.Tensor]] = [
        [] for _ in range(vehicle_num)
    ]

    route_nums = routes_number(actions)
    for batch_i, batch_action in enumerate(actions):
        in_depot = torch.nonzero(batch_action == 0)
        length_route_dict = defaultdict(list)

        route_num = 0
        for i in range(in_depot.size(0) - 1):
            one_route = batch_action[in_depot[i] : in_depot[i + 1]]
            one_route_length = cal_distance(
                problem[batch_i : batch_i + 1], one_route.unsqueeze(0), cycle=True
            ).item()
            if one_route_length != 0:
                length_route_dict[one_route_length].append(one_route)
                route_num += 1
        assert route_nums[batch_i] == route_num

        sorted_length = sorted(length_route_dict)
        while route_num > vehicle_num:
            if len(length_route_dict[sorted_length[0]]) > 1:
                length_route_dict[sorted_length[0]][-2] = torch.cat(
                    (
                        length_route_dict[sorted_length[0]][-2],
                        length_route_dict[sorted_length[0]][-1],
                    )
                )
                del length_route_dict[sorted_length[0]][-1]
            else:
                length_route_dict[sorted_length[0] + sorted_length[1]].append(
                    torch.cat(
                        (
                            length_route_dict[sorted_length[0]][-1],
                            length_route_dict[sorted_length[1]][-1],
                        )
                    )
                )
                del length_route_dict[sorted_length[0]]

                if len(length_route_dict[sorted_length[1]]) > 1:
                    del length_route_dict[sorted_length[1]][-1]
                else:
                    del length_route_dict[sorted_length[1]]

                sorted_length = sorted(length_route_dict)

            route_num -= 1

        i = 0
        for this_length_routes in length_route_dict.values():
            for route in this_length_routes:
                vehicle_batch_action_list[i].append(route)
                i += 1

    vehicle_action: List[torch.Tensor] = []
    for i in range(vehicle_num):
        vehicle_action.append(
            torch.cat(
                (
                    torch.nn.utils.rnn.pad_sequence(
                        vehicle_batch_action_list[i], batch_first=True
                    ),
                    torch.zeros(
                        (actions.size(0), 1), device=actions.device, dtype=torch.int64
                    ),
                ),
                dim=1,
            )
        )

    return vehicle_action


def cal_reward_MO_1D(problems: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, node_indexes)

    return: [batch_size, 2(objective_sum, objective_max)]
    '''
    length_per_vehicle = []
    actions_align = align_actions_to_wait(actions)
    for no_need_calculate, tsp_in_vrp, _, _ in split_action_with_wait(
        problems, actions_align
    ):
        if not no_need_calculate:
            tsp_length = cal_distance(problems, tsp_in_vrp, cycle=True)  # (batch_size,)
            length_per_vehicle.append(tsp_length)
    length = torch.stack(length_per_vehicle, 1)  # (batch_size, route_num)
    sum_and_max = torch.stack((length.sum(1), length.max(1)[0]), dim=1)
    return -sum_and_max
