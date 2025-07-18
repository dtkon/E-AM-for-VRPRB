from typing import List, Tuple
import numpy as np
import random


def get_demand(problem: np.ndarray, part_tour: List[int]) -> float:
    '''
    problem: [problem_size, (x, y, d, ...)]
    '''
    seq_demand = problem[:, 2][part_tour]
    return seq_demand.sum().item()


def get_distance(
    problem: np.ndarray, part_tour: List[int], cycle: bool = False
) -> float:
    '''
    problem: [problem_size, (x, y, ...)]
    '''
    seq_cord = problem[:, :2][part_tour]
    if not cycle:
        next_nodes = seq_cord[1:, :]
        return np.linalg.norm(next_nodes - seq_cord[:-1, :], axis=-1).sum().item()
    else:
        next_nodes = np.concatenate((seq_cord[1:, :], seq_cord[0:1, :]), 0)
        return np.linalg.norm(next_nodes - seq_cord, axis=-1).sum().item()


def get_cap(j: int, i: int, problem: np.ndarray, giant_tour: List[int]) -> float:
    if j > i:
        return np.inf
    else:
        return get_demand(problem, giant_tour[j : i + 1])


def get_cost(j: int, i: int, problem: np.ndarray, giant_tour: List[int]) -> float:
    if j > i:
        return np.inf
    else:
        return get_distance(problem, [0] + giant_tour[j : i + 1] + [0])


def dp_split(
    problem: np.ndarray, giant_tour: List[int], vehicle_num: int, capacity: float = 1.0
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_sum = np.zeros((vehicle_num + 1, len(giant_tour)))
    state_sum[:, 1:] = np.nan
    state_max = state_sum.copy()
    action_sum = state_sum.copy()  # .astype(np.int32)
    action_max = state_sum.copy()
    for k in range(1, vehicle_num + 1):
        for i in range(1, len(giant_tour)):
            min_val_sum = np.inf
            min_act_sum = -1
            min_val_max = np.inf
            min_act_max = -1
            for j in range(0, i + 1):
                if get_cap(j + 1, i, problem, giant_tour) <= capacity:
                    val_sum = state_sum[k - 1][j] + get_cost(
                        j + 1, i, problem, giant_tour
                    )
                    val_max = max(
                        state_max[k - 1][j], get_cost(j + 1, i, problem, giant_tour)
                    )
                    if val_sum < min_val_sum:
                        min_val_sum = val_sum
                        min_act_sum = j + 1
                    if val_max < min_val_max:
                        min_val_max = val_max
                        min_act_max = j + 1
            if not np.isinf(min_val_sum):
                state_sum[k][i] = min_val_sum
                action_sum[k][i] = min_act_sum
            if not np.isinf(min_val_max):
                state_max[k][i] = min_val_max
                action_max[k][i] = min_act_max
    return (
        state_sum[-1, -1],
        state_max[-1, -1],
        action_sum,
        action_max,
        state_sum,
        state_max,
    )


def get_max_distance(problem: np.ndarray, route: List[int]) -> float:
    '''
    problem: [problem_size, (x, y, ...)]

    route: must start and end at 0
    '''
    tours = []
    tour = [0]
    for i in range(1, len(route)):
        if route[i] != 0:
            tour.append(route[i])
        else:
            tour.append(route[i])
            tours.append(tour)
            tour = [0]

    max_distance = -1.0
    for t in tours:
        distance = get_distance(problem, t)
        if distance > max_distance:
            max_distance = distance
    return max_distance


def demand_is_legal(
    problem: np.ndarray, route: List[int], capacity: float = 1.0
) -> bool:
    '''
    problem: [problem_size, (x, y, d, ...)]
    '''
    tours = []
    tour = [0]
    for i in range(1, len(route)):
        if route[i] != 0:
            tour.append(route[i])
        else:
            tour.append(route[i])
            tours.append(tour)
            tour = [0]

    for t in tours:
        demand = get_demand(problem, t)
        if demand > capacity:
            return False
    return True


def get_route_from_action(giant_tour: List[int], action: np.ndarray) -> List[int]:
    assert len(giant_tour) == action.shape[1]

    if np.isnan(action[-1, -1]):
        return []

    tours: List[Tuple[int, int]] = []
    current_end = action.shape[1] - 1
    for i in range(action.shape[0] - 1, 0, -1):
        start = int(action[i, current_end])
        tours.insert(0, (start, current_end))
        current_end = start - 1
        if current_end < 0:
            break
    route = [0]
    for pair in tours:
        route.extend(giant_tour[pair[0] : pair[1] + 1])
        route.append(0)
    return route


if __name__ == '__main__':
    random.seed(2)
    problems = np.load('datasets/cvrp20-v4.npy')
    tour = list(range(21))
    random.shuffle(tour)
    val_sum, val_min, action_sum, action_min, *_ = dp_split(problems[0], tour, 4)
    print(val_sum, val_min)
    route_sum = get_route_from_action(tour, action_sum)
    route_min = get_route_from_action(tour, action_min)
    print(route_sum, route_min)
    print(get_distance(problems[0], route_sum))
    print(get_max_distance(problems[0], route_min))
