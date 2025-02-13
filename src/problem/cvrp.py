from typing import Any, Optional, Tuple, Iterator, overload
import torch

from route_solver import cal_distance

# https://github.com/ai4co/rl4co/blob/d557327b8e4c0cfa95286c06428cade849e55d3f/rl4co/envs/routing/cvrp/generator.py
CAPACITIES = {
    10: 20,
    15: 25,
    20: 30,
    30: 33,
    40: 37,
    50: 40,
    60: 43,
    75: 45,
    100: 50,
    125: 55,
    150: 60,
    200: 70,
    500: 100,
    1000: 150,
}


def random_generate(
    batch_size: int, customer_number: int, max_demand: int, capacity: int
) -> torch.Tensor:
    '''
    return: (batch_size, customer_number+1, 3)

    demand range: 1 ~ max_demand, then divided by capacity

    depot demand = 0
    '''
    problems = torch.rand((batch_size, customer_number + 1, 2))
    demand = (
        torch.randint(1, max_demand + 1, (batch_size, customer_number + 1, 1))
        / capacity
    )
    demand[:, 0, 0] = 0
    return torch.cat((problems, demand), dim=2)


def generate_datasets(
    batch_size: int,
    customer_size: int,
    vehicle_capacity: int = -1,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> torch.Tensor:
    if vehicle_capacity == -1:
        capacity = CAPACITIES[customer_size]
    else:
        capacity = vehicle_capacity

    data = random_generate(batch_size, customer_size, 9, capacity)

    if save_path is not None:
        torch.save(data, save_path)

    return data


def cal_reward(problems: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, node_indexes), a row: 0-2-5-0-1-4-7-0-0-0, represents permutation

    return: (batch_size,)
    '''
    return -cal_distance(problems, actions)


def check_action_legal(problems: torch.Tensor, actions: torch.Tensor) -> bool:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    actions: (batch_size, node_indexes)
    '''
    actions_align = align_actions_to_wait(actions)

    for (
        no_need_calculate,
        tsp_in_vrp,
        problems_of_tsp,
        mask,
    ) in split_action_with_wait(problems, actions_align):
        if not no_need_calculate:
            demands_of_tsp = problems_of_tsp[:, :, 2]
            demands_of_tsp[mask] = 0
            if torch.any(demands_of_tsp.sum(1) > 1 + 1e-6):
                return False

    return True


def routes_number(actions: torch.Tensor) -> torch.Tensor:
    '''
    actions: (batch_size, node_indexes), a row: 0-2-5-0-0-1-4-7-0-0-0, represents permutation

    return: (batch_size,), for above action, routes number is 2
    '''
    is_depot = actions == 0
    duplicate = get_duplicate_mask(is_depot)
    is_depot[duplicate] = False
    return is_depot.sum(1) - 1


def get_duplicate_mask(actions: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros(
                actions.size(0),
                1,
                device=actions.device,
                dtype=torch.bool,
            ),
            actions[:, :-1] == actions[:, 1:],
        ),
        dim=1,
    )


@overload
def _split_action_no_wait(
    problems: None, actions: torch.Tensor, have_duplicate: bool = True
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]]: ...


@overload
def _split_action_no_wait(
    problems: torch.Tensor, actions: torch.Tensor, have_duplicate: bool = True
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: ...


def _split_action_no_wait(
    problems: Optional[torch.Tensor], actions: torch.Tensor, have_duplicate: bool = True
):
    '''
    WARNING: this function is INEFFICIENT!
    SUGGESTION: first use align_actions_to_wait, then use split_action_with_wait.

    actions:
        [0-2-5-0-0-1-4-7-0-0-0,
         0-2-0-1-3-4-0-0-0-0-0].
    no wait.
    represents permutation.

    yield: if is no need to calculate, modified actions, sub problem, mask
    '''
    actions = actions.clone()
    any_batch_in_depot = torch.nonzero(torch.any(actions == 0, dim=0))

    for i in range(any_batch_in_depot.size(0) - 1):
        tsp_in_vrp = actions[:, : any_batch_in_depot[i + 1]]
        problems_of_tsp = (
            problems.gather(1, tsp_in_vrp.unsqueeze(2).repeat(1, 1, 3))
            if problems is not None
            else None
        )

        mask = tsp_in_vrp == 0
        mask[:, 0] = False

        if have_duplicate:
            duplicate = get_duplicate_mask(tsp_in_vrp)
            mask[duplicate] = True

        not_finish_tsp = (actions[:, any_batch_in_depot[i + 1]] != 0).view(-1)
        mask[:, 1:][not_finish_tsp] = True

        no_need_calculate = torch.all(
            mask[:, 1:] == True
        )  # if mask.size(1) == 1, will be True

        yield no_need_calculate, tsp_in_vrp, problems_of_tsp, mask

        tsp_in_vrp[~not_finish_tsp] = 0


@overload
def split_action_with_wait(
    problems: None, actions: torch.Tensor, have_duplicate: bool = True
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]]: ...


@overload
def split_action_with_wait(
    problems: torch.Tensor, actions: torch.Tensor, have_duplicate: bool = True
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: ...


def split_action_with_wait(
    problems: Optional[torch.Tensor], actions: torch.Tensor, have_duplicate: bool = True
):
    '''
    actions:
        [0-2-5-0-3-1-4-7-0-8-0,
         0-2-0-0-3-4-0-0-0-1-0].
    with wait.
    represents permutation.

    yield: if is no need to calculate, actions, sub problems, mask
    '''
    all_batch_in_depot = torch.nonzero(actions.sum(0) == 0)

    for i in range(all_batch_in_depot.size(0) - 1):
        tsp_in_vrp = actions[:, all_batch_in_depot[i] : all_batch_in_depot[i + 1]]
        problems_of_tsp = (
            problems.gather(1, tsp_in_vrp.unsqueeze(2).repeat(1, 1, 3))
            if problems is not None
            else None
        )

        mask = tsp_in_vrp == 0
        mask[:, 0] = False

        if have_duplicate:
            duplicate = get_duplicate_mask(tsp_in_vrp)
            mask[duplicate] = True

        yield torch.all(tsp_in_vrp == 0), tsp_in_vrp, problems_of_tsp, mask


def align_actions_to_wait(actions: torch.Tensor) -> torch.Tensor:
    '''
    actions:
        [0-2-5-0-0-1-4-7-0-0-0,
         0-2-0-1-3-4-0-0-0-0-0].
    no wait.
    represents permutation.
    '''
    actions = _erase_duplicate_zero(actions)

    batch_size = actions.size(0)

    i = 0
    while i < actions.size(1) - 1:
        if not (torch.all(actions[:, i] == 0) or torch.all(actions[:, i] != 0)):
            need_push_back = (actions[:, i] == 0) & (actions[:, i + 1] != 0)
            if torch.any(need_push_back):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            batch_size, 1, device=actions.device, dtype=torch.long
                        ),
                    ),
                    1,
                )
                need_push_back_square = need_push_back.unsqueeze(1).repeat(
                    1, actions.size(1) - (i + 2)
                )
                actions[:, i + 2 :] = torch.where(
                    need_push_back_square, actions[:, i + 1 : -1], actions[:, i + 2 :]
                )
                actions[:, i + 1][need_push_back] = 0
        i += 1

    return actions


def _erase_duplicate_zero(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()

    for i in range(actions.size(1) - 2, 0, -1):
        copy_target = actions[:, i + 1]
        keep_target = actions[:, i]

        need_copy = (actions[:, i] == 0) & (actions[:, i - 1] == 0) & (copy_target != 0)
        if torch.any(need_copy):
            actions[:, i] = torch.where(need_copy, copy_target, keep_target)

    return actions
