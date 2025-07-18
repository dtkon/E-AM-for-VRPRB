from typing import List, Optional, Tuple, cast, Generator
import random, os
import numpy as np
import geatpy

import split_tour


def gen_gaint_tour(customer_size: int) -> List[int]:
    tour = list(range(1, customer_size + 1))
    random.shuffle(tour)
    return [0] + tour


def gen_initial_population(
    polulation_size: int,
    problem: np.ndarray,
    vehicle_num: int,
    capacity: float = 1.0,
) -> List[List[int]]:
    '''
    return: route
    '''
    P_route = []
    P_giant_tour = []
    customer_size = problem.shape[0] - 1
    for _ in range(polulation_size // 2):
        P_giant_tour.append(gen_gaint_tour(customer_size))
    for giant_tour in P_giant_tour:
        _, _, action_sum, action_max, *_ = split_tour.dp_split(
            problem, giant_tour, vehicle_num, capacity
        )
        route_sum = split_tour.get_route_from_action(giant_tour, action_sum)
        route_max = split_tour.get_route_from_action(giant_tour, action_max)
        assert len(route_sum) != 0 and len(route_max) != 0
        P_route.append(route_sum)
        P_route.append(route_max)
    return P_route


def convert_route_to_gaint_tour(route: List[int]) -> List[int]:
    tour = [0]
    for i in route:
        if i != 0:
            tour.append(i)
    return tour


def order_crossover(giant_tour_1: List[int], giant_tour_2: List[int]) -> List[int]:
    assert len(giant_tour_1) == len(giant_tour_2)

    splits = random.sample(range(1, len(giant_tour_1)), 2)
    split_start = min(splits)
    split_end = max(splits)

    child = [0] + [-1] * (len(giant_tour_1) - 1)
    child[split_start:split_end] = giant_tour_1[split_start:split_end]

    re_parent_1 = giant_tour_2[split_end:] + giant_tour_2[1:split_end]
    for i in child[split_start:split_end]:
        re_parent_1.remove(i)
    target_point = split_end % len(giant_tour_1)
    for i in re_parent_1:
        child[target_point] = i
        target_point = (target_point + 1) % len(giant_tour_1)
        if target_point == 0:
            target_point += 1
    return child


def recombination(
    population: List[List[int]],
    problem: np.ndarray,
    vehicle_num: int,
    capacity: float = 1.0,
    save_path: Optional[str] = None,
) -> List[List[int]]:
    P_giant_tour = [convert_route_to_gaint_tour(r) for r in population]
    random.shuffle(P_giant_tour)
    offspring_giant_tour = []
    for i in range(0, len(P_giant_tour), 2):
        offspring_giant_tour.append(
            order_crossover(P_giant_tour[i], P_giant_tour[i + 1])
        )

    P_route = []
    for giant_tour in offspring_giant_tour:
        _, _, action_sum, action_max, *_ = split_tour.dp_split(
            problem, giant_tour, vehicle_num, capacity
        )
        route_sum = split_tour.get_route_from_action(giant_tour, action_sum)
        route_max = split_tour.get_route_from_action(giant_tour, action_max)

        if len(route_sum) != 0:
            P_route.append(route_sum)
        else:
            P_route.append(random.choice(population))
            if save_path is not None:
                with open(os.path.join(save_path, 'log_bad_split.txt'), 'a') as f:
                    f.write(f'bad sum split:\n{action_sum}\n\n')

        if len(route_sum) != 0:
            P_route.append(route_max)
        else:
            P_route.append(random.choice(population))
            if save_path is not None:
                with open(os.path.join(save_path, 'log_bad_split.txt'), 'a') as f:
                    f.write(f'bad max split:\n{action_max}\n\n')
    return P_route


def exchange(route: List[int], customer_1: int, customer_2: int) -> List[int]:
    assert customer_1 != customer_2 and customer_1 != 0 and customer_2 != 0
    route = route.copy()
    cus1_i = route.index(customer_1)
    cus2_i = route.index(customer_2)
    route[cus1_i] = customer_2
    route[cus2_i] = customer_1
    return route


def relocate(route: List[int], customer_1: int, customer_2: int) -> List[int]:
    assert customer_1 != customer_2 and customer_1 != 0 and customer_2 != 0
    route = route.copy()
    cus1_i = route.index(customer_1)
    cus2_i = route.index(customer_2)
    route.insert(cus2_i, customer_1)
    if cus1_i < cus2_i:
        cus1 = route.pop(cus1_i)
    elif cus1_i > cus2_i:
        cus1 = route.pop(cus1_i + 1)
    else:
        raise RuntimeError('impossible')
    assert cus1 == customer_1
    return route


def two_opt(route: List[int], customer_1: int, customer_2: int) -> List[int]:
    assert customer_1 != customer_2 and customer_1 != 0 and customer_2 != 0
    route = route.copy()
    cus1_i = route.index(customer_1)
    cus2_i = route.index(customer_2)

    small_i = min(cus1_i, cus2_i)
    big_i = max(cus1_i, cus2_i)

    small_i_end = route.index(0, small_i)
    big_i_end = route.index(0, big_i)

    if small_i_end != big_i_end:  # in two routes
        part1 = route[small_i:small_i_end]
        part2 = route[big_i:big_i_end]
        route[big_i:big_i_end] = part1
        route[small_i:small_i_end] = part2
    else:  # in same route, reverse
        small_i = min(cus1_i, cus2_i)
        big_i = max(cus1_i, cus2_i)
        route[small_i : big_i + 1] = reversed(route[small_i : big_i + 1])
    return route


_local_search_k_opts = (exchange, relocate, two_opt)


def local_search_k(
    k: int, route: List[int], weight: float, problem: np.ndarray, capacity: float = 1
) -> List[int]:
    opt = _local_search_k_opts[k - 1]

    retry_time = 0
    while retry_time < 10:
        cus1, cus2 = random.sample(range(1, problem.shape[0]), 2)
        s = opt(route, cus1, cus2)
        if not split_tour.demand_is_legal(problem, s, capacity):
            retry_time += 1
            continue

        fitness_old = split_tour.get_distance(
            problem, route
        ) * weight + split_tour.get_max_distance(problem, route) * (1 - weight)
        fitness_new = split_tour.get_distance(
            problem, s
        ) * weight + split_tour.get_max_distance(problem, s) * (1 - weight)
        if fitness_new < fitness_old:
            ret = s
        else:
            ret = route.copy()
        return ret
    return route.copy()


def tournament_select(
    population: List[List[int]],
    tournament_size: int,
    weight: float,
    problem: np.ndarray,
) -> int:
    choose_index = list(range(len(population)))
    random.shuffle(choose_index)
    choose_individual = []
    for i in choose_index[:tournament_size]:
        choose_individual.append(population[i])

    choose_fitness = []
    for s in choose_individual:
        fitness = split_tour.get_distance(
            problem, s
        ) * weight + split_tour.get_max_distance(problem, s) * (1 - weight)
        choose_fitness.append(fitness)

    best_index = choose_fitness.index(min(choose_fitness))
    return best_index


def pareto_sort(
    population: List[List[int]],
    problem: np.ndarray,
    needNum: Optional[int] = None,
    needLevel: Optional[int] = None,
) -> List[List[int]]:
    objv_list = []
    for s in population:
        objv_list.append(
            [
                split_tour.get_distance(problem, s),
                split_tour.get_max_distance(problem, s),
            ]
        )
    objv = np.array(objv_list)

    # nd_sort
    levels, criLevel = cast(
        Tuple[np.ndarray, int], geatpy.ndsortESS(objv, needNum, needLevel)
    )
    dis = cast(np.ndarray, geatpy.crowdis(objv, levels))
    ###

    if needNum is None:
        needNum = len(P)

    sortP = []
    for lv in range(1, criLevel):
        indexs = np.argwhere(levels == lv)
        indexs_sorted = sorted(indexs, key=lambda x: dis[x[0]], reverse=True)
        for i in indexs_sorted:
            sortP.append(population[i[0]])
    indexs = np.argwhere(levels == criLevel)
    indexs_sorted = sorted(indexs, key=lambda x: dis[x[0]], reverse=True)
    for i in indexs_sorted:
        if len(sortP) < needNum:
            sortP.append(population[i[0]])
    return sortP


def CMMA(
    problem: np.ndarray,
    vehicle_num: int,
    capacity: float = 1.0,
    population_size: int = 200,
    N_gen: int = 12000,
    Iter: int = 80,
    LS_iter: int = 3000,
    tournament_tau: float = 0.5,
    initial_P: Optional[List[List[int]]] = None,
    save_path: Optional[str] = None,
) -> Generator[List[List[int]], None, None]:
    if initial_P is None:
        P = gen_initial_population(population_size, problem, vehicle_num, capacity)
    else:
        P = initial_P

    for i_0 in range(N_gen):
        P_2 = recombination(P, problem, vehicle_num, capacity, save_path)
        new_P = P + P_2

        for i_1 in range(Iter):
            weight = random.random()
            s_i = tournament_select(
                new_P, int(tournament_tau * len(new_P)), weight, problem
            )
            for i_2 in range(LS_iter):
                k = random.randint(1, 3)
                s = local_search_k(k, new_P[s_i], weight, problem, capacity)
                new_P[s_i] = s
        P = pareto_sort(new_P, problem, population_size)
        yield P


if __name__ == '__main__':
    random.seed(2)
    problems = np.load('datasets/cvrp20-v4.npy')

    P_gen = CMMA(
        problems[0],
        4,
        population_size=50,
        N_gen=2,
        Iter=40,
        LS_iter=300,
        tournament_tau=0.25,
    )

    for P in P_gen:
        print(P[0])
    print(P[0])
