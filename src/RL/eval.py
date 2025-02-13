import random
import time
import platform
import math
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, cast
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import geatpy
import numpy as np


from problem.cvrp import generate_datasets as generate_datasets_cvrp
from problem.cvrp_balance import generate_datasets as generate_datasets_cvrp_balance
from problem.tsp import generate_datasets as generate_datasets_tsp
from route_solver import TspSolver, pretrained_model, zoom
from utils import augment, batch_slicer

if TYPE_CHECKING:
    from .agent import Agent
    from NN.actor import AM_Actor, AM_preActor

PREF_11 = torch.tensor([[x, 1 - x] for x in torch.arange(0.0, 1.1, 0.1)])
PREF_101 = torch.tensor([[x, 1 - x] for x in torch.arange(0.0, 1.01, 0.01)])

PENALTY_VALUE = 10000


def eval(
    rank: int,
    agent: 'Agent',
    mp_ret: Optional[mp.Queue] = None,
    init_dist: bool = True,
    other_for_env: Optional[dict] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    float,
]:
    '''
    return: rewards, infeasible, action, addition_rewards, time_used
    '''

    if rank == 0:
        print('\nEvaluating...', flush=True)

    option = agent.option

    # agent.eval()

    random_state_backup = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        random.getstate(),
    )

    torch.manual_seed(option.seed)
    random.seed(option.seed)

    if option.val_dataset is None:
        if 'cvrp' in option.problem:
            if option.problem == 'cvrp':
                generate_datasets: Callable[..., torch.Tensor] = generate_datasets_cvrp
            else:
                generate_datasets = generate_datasets_cvrp_balance
        elif 'tsp' in option.problem:
            generate_datasets = generate_datasets_tsp
        else:
            raise NotImplementedError
        val_problem = generate_datasets(
            option.val_range[1] - option.val_range[0],
            option.val_customer_size,
            vehicle_num=option.val_vehicle_num,
            vehicle_capacity=option.vehicle_capacity,
        )
    else:
        val_problem = torch.tensor(
            np.load(option.val_dataset)[option.val_range[0] : option.val_range[1]]
        )
        if 'cvrp' in option.problem:
            assert val_problem.size(1) == option.val_customer_size + 1
        elif 'tsp' in option.problem:
            assert val_problem.size(1) == option.val_customer_size
        else:
            raise NotImplementedError

        if option.zoom_on:
            val_problem, zoom_ratio = zoom(val_problem)

            if math.isnan(option.first_zoom_ratio):
                option.first_zoom_ratio = float(zoom_ratio[0].item())

    if option.distributed and init_dist:
        device = torch.device('cuda', rank)
        dist.init_process_group(
            backend='gloo' if platform.system() == 'Windows' else 'nccl',
            world_size=option.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.pre_actor.to(device)

        if option.normalization == 'batch':
            agent.pre_actor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                agent.pre_actor
            ).to(device)

        agent.actor = cast(
            'AM_Actor',
            torch.nn.parallel.DistributedDataParallel(
                agent.actor, device_ids=[rank], find_unused_parameters=False
            ),
        )
        agent.pre_actor = cast(
            'AM_preActor',
            torch.nn.parallel.DistributedDataParallel(
                agent.pre_actor, device_ids=[rank], find_unused_parameters=False
            ),
        )
    else:
        if option.use_cuda:
            device = torch.device('cuda', rank)
        else:
            device = option.device

    if option.distributed:
        val_sampler: torch.utils.data.distributed.DistributedSampler = (
            torch.utils.data.distributed.DistributedSampler(
                cast(Dataset, val_problem), shuffle=False
            )
        )
        val_dataloader: DataLoader = DataLoader(
            cast(Dataset, val_problem),
            batch_size=option.val_batch_size // option.world_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=val_sampler,
        )
    else:
        val_dataloader = DataLoader(
            cast(Dataset, val_problem),
            batch_size=option.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    other_for_actor = {
        'vehicle_num': option.val_vehicle_num,
        'will_re_tsp': '_re_' in option.problem,
    }

    if other_for_env is None:
        if '_re_' in option.problem:
            tsp_solvers = {
                20: TspSolver(*pretrained_model['tsp20']).to_device(device),
                # 100: TspSolver(*pretrained_model['tsp100']).to_device(device),
            }
            other_for_env = {'tsp_solvers': tsp_solvers}
        else:
            other_for_env = {}

    start_time = time.time()

    if option.is_mo_problem:
        rewards, infeasible, action, addtion_rewards = _evaluate_MO(
            rank,
            agent,
            val_dataloader,
            device,
            other_for_actor,
            other_for_env,
            save_action=(option.problem != 'mo_cvrp_baseline'),
        )
    else:
        rewards, infeasible, action, addtion_rewards = _evaluate(
            rank, agent, val_dataloader, device, other_for_actor, other_for_env
        )

    if option.zoom_on:
        zoom_ratio = zoom_ratio.to(rewards.device)
        rewards *= zoom_ratio

        if addtion_rewards is not None:
            addtion_rewards *= zoom_ratio

    time_used = time.time() - start_time

    if option.distributed:
        dist.barrier()

        rewards = gather_tensor_and_concat(rewards.contiguous())
        if infeasible is not None:
            infeasible = gather_tensor_and_concat(infeasible.contiguous())

        dist.barrier()

        if mp_ret is not None and rank == 0:
            mp_ret.put(
                (
                    rewards.cpu(),
                    infeasible.cpu() if infeasible is not None else None,
                    action.cpu() if action is not None else None,
                    addtion_rewards.cpu() if addtion_rewards is not None else None,
                    time_used,
                )
            )

    torch.set_rng_state(random_state_backup[0])
    torch.cuda.set_rng_state(random_state_backup[1])
    random.setstate(random_state_backup[2])

    return rewards, infeasible, action, addtion_rewards, time_used


def _evaluate(
    rank: int,
    agent: 'Agent',
    val_dataloader: DataLoader,
    device: torch.device,
    other_for_actor: dict,
    other_for_env: dict,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, None]:
    '''
    return: rewards, infeasible, action
    '''

    # calculate tqdm total
    val_size = (
        agent.option.val_range[1] - agent.option.val_range[0]
        if not agent.option.distributed
        else (agent.option.val_range[1] - agent.option.val_range[0])
        // agent.option.world_size
    )
    val_batch_size = (
        agent.option.val_batch_size
        if not agent.option.distributed
        else agent.option.val_batch_size // agent.option.world_size
    )
    batch_num = val_size // val_batch_size
    if agent.option.eval_type == 'greedy':
        total = batch_num
    elif agent.option.eval_type == 'sample':
        split_batch_num = (
            val_batch_size
            * agent.option.val_N_aug
            * agent.option.sample_times
            // agent.option.max_parallel
        )
        if split_batch_num == 0:
            split_batch_num = 1
        total = batch_num * split_batch_num
    elif agent.option.eval_type == 'greedy_aug':
        split_batch_num = (
            val_batch_size * agent.option.val_N_aug // agent.option.max_parallel
        )
        if split_batch_num == 0:
            split_batch_num = 1
        total = batch_num * split_batch_num
    else:
        raise NotImplementedError

    eval_bar = tqdm(
        total=total,
        disable=agent.option.no_progress_bar or rank != 0,
        desc='evaluating',
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    )

    action_list: List[torch.Tensor] = []
    reward_list = []
    infeasible_list = []

    max_action_length = 0

    with torch.no_grad():
        for batch_problem in val_dataloader:
            batch_problem = batch_problem.to(device)

            if agent.option.eval_type == 'greedy':
                reward, infeasible, action = solve_so(
                    batch_problem,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    1,
                    0,
                    eval_bar,
                    agent.option.max_parallel,
                )

                eval_bar.update()
            elif agent.option.eval_type == 'sample':
                reward, infeasible, action = solve_so(
                    batch_problem,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    agent.option.val_N_aug,
                    agent.option.sample_times,
                    eval_bar,
                    agent.option.max_parallel,
                )
            elif agent.option.eval_type == 'greedy_aug':
                reward, infeasible, action = solve_so(
                    batch_problem,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    agent.option.val_N_aug,
                    0,
                    eval_bar,
                    agent.option.max_parallel,
                )
            else:
                raise NotImplementedError

            action_list.append(action)
            reward_list.append(reward)
            infeasible_list.append(infeasible)

            if action.size(-1) > max_action_length:
                max_action_length = action.size(-1)

    eval_bar.close()

    for i, t in enumerate(action_list):
        action_list[i] = F.pad(t, (0, max_action_length - t.size(-1)), 'constant', 0)

    return (
        torch.cat(reward_list),  # (val_size,)
        (
            torch.cat(cast(List[torch.Tensor], infeasible_list))
            if infeasible is not None
            else None
        ),  # (val_size, N_aug, sample_times)
        torch.cat(action_list),  # (val_size, action_length)
        None,
    )


def _evaluate_MO(
    rank: int,
    agent: 'Agent',
    val_dataloader: DataLoader,
    device: torch.device,
    other_for_actor: dict,
    other_for_env: dict,
    save_action: bool = True,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    '''
    return: rewards, infeasible, action, addtion_rewards
    '''

    if agent.option.quick_eval:
        prefs = PREF_11.to(device)
    else:
        prefs = PREF_101.to(device)

    if agent.option.so_mode == 0:
        prefs = prefs[-1:]
    elif agent.option.so_mode == 1:
        prefs = prefs[:1]

    # calculate tqdm total
    val_size = (
        agent.option.val_range[1] - agent.option.val_range[0]
        if not agent.option.distributed
        else (agent.option.val_range[1] - agent.option.val_range[0])
        // agent.option.world_size
    )
    val_batch_size = (
        agent.option.val_batch_size
        if not agent.option.distributed
        else agent.option.val_batch_size // agent.option.world_size
    )
    batch_num = val_size // val_batch_size
    if agent.option.eval_type == 'greedy':
        total = batch_num * prefs.size(0)
    elif (agent.option.eval_type == 'sample') or ('pareto' in agent.option.eval_type):

        sample_times = agent.option.sample_times
        if sample_times == 0:
            sample_times = 1
        split_batch_num = math.ceil(
            val_batch_size
            * agent.option.val_N_aug
            * sample_times
            / agent.option.max_parallel
        )
        if split_batch_num == 0:
            split_batch_num = 1
        total = batch_num * split_batch_num * prefs.size(0)
    elif agent.option.eval_type == 'greedy_aug':
        split_batch_num = math.ceil(
            val_batch_size * agent.option.val_N_aug / agent.option.max_parallel
        )
        if split_batch_num == 0:
            split_batch_num = 1
        total = batch_num * split_batch_num * prefs.size(0)
    else:
        raise NotImplementedError

    eval_bar = tqdm(
        total=total,
        disable=agent.option.no_progress_bar or rank != 0,
        desc='evaluating',
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    )

    action_list: List[Optional[torch.Tensor]] = []
    reward_list = []
    infeasible_list = []
    addition_reward_list = []

    max_action_length = 0

    with torch.no_grad():
        for batch_problem in val_dataloader:
            batch_problem = batch_problem.to(device)

            if agent.option.eval_type == 'greedy':
                reward, infeasible, action, addition_reward = solve_mo(
                    batch_problem,
                    prefs,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    1,
                    0,
                    eval_bar,
                    select_mode='ws',
                    max_parallel=agent.option.max_parallel,
                    save_action=save_action,
                )
            elif agent.option.eval_type == 'sample':
                reward, infeasible, action, addition_reward = solve_mo(
                    batch_problem,
                    prefs,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    agent.option.val_N_aug,
                    agent.option.sample_times,
                    eval_bar,
                    select_mode='ws',
                    max_parallel=agent.option.max_parallel,
                    save_action=save_action,
                )
            elif agent.option.eval_type == 'greedy_aug':
                reward, infeasible, action, addition_reward = solve_mo(
                    batch_problem,
                    prefs,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    agent.option.val_N_aug,
                    0,
                    eval_bar,
                    select_mode='ws',
                    max_parallel=agent.option.max_parallel,
                    save_action=save_action,
                )
            elif 'pareto' in agent.option.eval_type:
                reward, infeasible, action, addition_reward = solve_mo(
                    batch_problem,
                    prefs,
                    agent.actor,
                    agent.pre_actor,
                    agent.env,
                    other_for_actor,
                    other_for_env,
                    agent.option.val_N_aug,
                    agent.option.sample_times,
                    eval_bar,
                    select_mode='pareto',
                    max_parallel=agent.option.max_parallel,
                    with_sample=('sample' in agent.option.eval_type),
                    save_action=save_action,
                )
            else:
                raise NotImplementedError

            action_list.append(action)
            reward_list.append(reward)
            infeasible_list.append(infeasible)
            addition_reward_list.append(addition_reward)

            if action is not None and action.size(-1) > max_action_length:
                max_action_length = action.size(-1)

    if action is not None:
        for i, t in enumerate(cast(List[torch.Tensor], action_list)):
            action_list[i] = F.pad(
                t, (0, max_action_length - t.size(-1)), 'constant', 0
            )

    eval_bar.close()

    return (
        torch.cat(reward_list, dim=0),  # [val_size, need_num(pref_num), reward_dim]
        (
            torch.cat(cast(List[torch.Tensor], infeasible_list))
            if infeasible is not None
            else None
        ),  # (val_size, pref_num, N_aug, sample_times)
        (
            torch.cat(cast(List[torch.Tensor], action_list))
            if action is not None
            else None
        ),  # (val_size, pref_num, vehicle_num, action_length)
        (
            torch.cat(cast(List[torch.Tensor], addition_reward_list))
            if addition_reward is not None
            else None
        ),  # [val_size, need_num(pref_num), reward_dim]
    )


def solve_so(
    problem: torch.Tensor,
    actor: nn.Module,
    pre_actor: nn.Module,
    env: Callable[..., torch.Tensor],
    other_for_actor: dict,
    other_for_env: dict,
    N_aug: int,
    sample_times: int,
    progress_bar: Optional[tqdm],
    max_parallel: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    '''
    sample_times: 0 for greedy

    return: rewards, infeasible, action
    '''
    batch_size = problem.size(0)

    problem = augment(problem, N_aug)  # [N_aug*batch_size, problem_size, 3]
    if sample_times > 1:
        problem = problem.repeat(
            sample_times, 1, 1
        )  # [sample_times*N_aug*batch_size, problem_size, 3]

    if max_parallel is None:
        max_parallel = problem.size(0)

    actions_list = []
    infeasible_list = []
    split_reward_list = []
    for split_start, split_end in batch_slicer(problem.size(0), max_parallel):
        split_problem = problem[split_start:split_end]

        enc_problems = pre_actor(split_problem, **other_for_actor)

        actions, log_prob, other = actor(
            split_problem,
            *enc_problems,
            decoder_type='sample' if sample_times > 0 else 'greedy',
            **other_for_actor,
        )
        reward = env(split_problem, actions, **other_for_env)  # (max_paral,)

        if other is not None:
            reward[other] = reward[other] - PENALTY_VALUE

        if progress_bar is not None:
            progress_bar.update()

        actions_list.append(actions)
        infeasible_list.append(other)
        split_reward_list.append(reward)

    if other is not None:
        all_infeasible = torch.cat(infeasible_list)  # (sample_times*N_aug*batch_size,)
        all_infeasible = all_infeasible.reshape(
            sample_times if sample_times > 0 else 1, N_aug, batch_size
        )
        all_infeasible = all_infeasible.permute(
            2, 1, 0
        )  # (batch_size, N_aug, sample_times)
    else:
        all_infeasible = None

    all_action = torch.cat(
        actions_list
    )  # (sample_times*N_aug*batch_size, action_length)

    all_reward = torch.cat(split_reward_list)  # (sample_times*N_aug*batch_size,)

    all_reward_best_index = all_reward.view(-1, batch_size).max(dim=0)[
        1
    ]  # (batch_size,)

    best_reward = all_reward.view(-1, batch_size)[
        all_reward_best_index, torch.arange(batch_size)
    ]

    best_action = all_action.view(-1, batch_size, all_action.size(-1))[
        all_reward_best_index, torch.arange(batch_size)
    ]

    return (
        best_reward,  # (batch_size,)
        all_infeasible,  # (batch_size, N_aug, sample_times)
        best_action,  # (batch_size, action_length)
    )


def solve_mo(
    problem: torch.Tensor,
    prefs: torch.Tensor,
    actor: nn.Module,
    pre_actor: nn.Module,
    env: Callable[..., torch.Tensor],
    other_for_actor: dict,
    other_for_env: dict,
    N_aug: int,
    sample_times: int,
    progress_bar: Optional[tqdm],
    select_mode: str = 'ws',
    max_parallel: Optional[int] = None,
    needNum: Optional[int] = None,
    needLevel: Optional[int] = None,
    with_sample: bool = False,
    save_action: bool = True,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    '''
    sample_times: 0 for greedy

    select_mode: ws or pareto

    return: rewards (batch_size, needNum, reward_dim), infeasible, action, sample_rewards
    '''
    batch_size = problem.size(0)
    pref_num, reward_dim = prefs.size()
    vehicle_num = cast(int, other_for_actor['vehicle_num'])

    if needNum is None:
        needNum = pref_num

    aug_problem = augment(problem, N_aug)  # [N_aug*batch_size, problem_size, 3]
    if sample_times > 1:
        aug_problem = aug_problem.repeat(
            sample_times, 1, 1
        )  # [sample_times*N_aug*batch_size, problem_size, 3]

    if max_parallel is None:
        max_parallel = aug_problem.size(0)

    max_action_length = 0

    split_pref_action_list: List[torch.Tensor] = []
    split_pref_infeasible_list = []
    split_pref_reward_list = []
    for split_start, split_end in batch_slicer(aug_problem.size(0), max_parallel):
        split_problem = aug_problem[split_start:split_end]

        enc_problems = pre_actor(split_problem, **other_for_actor)

        action_list = []
        infeasible_list = []
        pref_reward_list = []

        actions: torch.Tensor
        log_prob: torch.Tensor
        other: Optional[torch.Tensor]
        for pref in prefs:
            actions, log_prob, other = actor(
                pref,
                split_problem,
                *enc_problems,
                decoder_type='sample' if sample_times > 0 else 'greedy',
                **other_for_actor,
            )
            reward = env(split_problem, actions, **other_for_env).reshape(
                log_prob.size(0), -1
            )  # (max_paral, reward_dim)

            if other is not None:
                reward[other] = reward[other] - PENALTY_VALUE

            infeasible_list.append(other)
            pref_reward_list.append(reward)

            if save_action:
                action_list.append(
                    actions.permute(2, 0, 1)
                )  # (action_length, max_paral, vehicle_num)

            if progress_bar is not None:
                progress_bar.update()

        pref_reward = torch.stack(
            pref_reward_list, dim=1
        )  # (max_paral, pref_num, reward_dim)
        split_pref_reward_list.append(pref_reward)

        if save_action:
            pref_action = pad_sequence(
                action_list, padding_value=0
            )  # (action_length, pref_num, max_paral, vehicle_num)
            split_pref_action_list.append(
                pref_action.permute(2, 1, 3, 0)
            )  # (max_paral, pref_num, vehicle_num, action_length)

            if pref_action.size(0) > max_action_length:
                max_action_length = pref_action.size(0)

        if other is not None:
            pref_infeasible = torch.stack(
                infeasible_list, dim=1  # type: ignore
            )  # (max_paral, pref_num)
            split_pref_infeasible_list.append(pref_infeasible)

    all_pref_reward = torch.cat(
        split_pref_reward_list
    )  # [sample_times*N_aug*batch_size, pref_num, reward_dim]

    if save_action:
        for i, t in enumerate(split_pref_action_list):
            split_pref_action_list[i] = F.pad(
                t, (0, max_action_length - t.size(-1)), 'constant', 0
            )

        all_pref_action = torch.cat(
            split_pref_action_list
        )  # [sample_times*N_aug*batch_size, pref_num, vehicle_num, action_length]

    if other is not None:
        all_pref_infeasible = torch.cat(
            split_pref_infeasible_list
        )  # (sample_times*N_aug*batch_size, pref_num)
        all_pref_infeasible = all_pref_infeasible.reshape(
            sample_times if sample_times > 0 else 1, N_aug, batch_size, pref_num
        )
        all_pref_infeasible = all_pref_infeasible.permute(
            2, 3, 1, 0
        )  # (batch_size, pref_num, N_aug, sample_times)
    else:
        all_pref_infeasible = None

    if with_sample or select_mode == 'ws':
        all_pref_reward_ws = (
            (all_pref_reward * prefs.unsqueeze(0))
            .sum(2)
            .view(-1, batch_size * pref_num)
        )  # [sample_times*N_aug, batch_size*pref_num]

        all_pref_reward_best_ws_index = all_pref_reward_ws.max(dim=0)[
            1
        ]  # (batch_size*pref_num,)

        sample_best_reward = all_pref_reward.view(
            -1, batch_size * pref_num, reward_dim
        )[all_pref_reward_best_ws_index, torch.arange(batch_size * pref_num)].view(
            batch_size, pref_num, reward_dim
        )

        if select_mode == 'ws':
            ret_action = None
            if save_action:
                sample_best_action = all_pref_action.view(
                    -1, batch_size * pref_num, vehicle_num, all_pref_action.size(-1)
                )[
                    all_pref_reward_best_ws_index, torch.arange(batch_size * pref_num)
                ].view(
                    batch_size, pref_num, vehicle_num, all_pref_action.size(-1)
                )
                ret_action = sample_best_action

            return (
                sample_best_reward,  # (batch_size, pref_num, reward_dim)
                all_pref_infeasible,  # (batch_size, pref_num, N_aug, sample_times)
                ret_action,  # (batch_size, pref_num, vehicle_num, action_length)
                None,
            )

    all_pref_reward = all_pref_reward.reshape(
        -1, batch_size, pref_num, reward_dim
    ).permute(
        1, 0, 2, 3
    )  # [batch_size, sample_times*N_aug, pref_num, reward_dim]

    all_reward = all_pref_reward.reshape(
        batch_size, -1, reward_dim
    )  # [batch_size, sample_times*N_aug*pref_num, reward_dim]

    if save_action:
        all_pref_action = all_pref_action.reshape(
            -1, batch_size, pref_num, vehicle_num, all_pref_action.size(-1)
        ).permute(
            1, 0, 2, 3, 4
        )  # [batch_size, sample_times*N_aug, pref_num, vehicle_num, action_length]

        all_action = all_pref_action.reshape(
            batch_size, -1, vehicle_num, all_pref_action.size(-1)
        )  # [batch_size, sample_times*N_aug*pref_num, vehicle_num, action_length]

    if other is not None:
        all_pref_infeasible = all_pref_infeasible.reshape(batch_size, -1)

    all_objv: np.ndarray = -all_reward.cpu().numpy()

    if progress_bar is not None:
        sub_eval_bar = tqdm(
            total=batch_size,
            disable=progress_bar.disable,
            desc='pareto sorting',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        )

    select_infeasible_list = []
    selected_action_list = []
    selected_reward_list = []
    for p_i, objv in enumerate(all_objv):
        levels, criLevel = cast(
            Tuple[np.ndarray, np.ndarray], geatpy.ndsortESS(objv, needNum, needLevel)
        )
        dis = cast(np.ndarray, geatpy.crowdis(objv, levels))

        sortI = []
        sortA = []
        sortR = []
        for lv in range(1, criLevel + 1):
            indexs = np.argwhere(levels == lv)  # (n, 1)
            indexs_sorted: List[np.ndarray] = sorted(
                indexs, key=lambda x: dis[x.item()], reverse=True
            )
            for nd_i in indexs_sorted:
                sortR.append(all_reward[p_i, nd_i.item()])  # (reward_dim,)
                if save_action:
                    sortA.append(
                        all_action[p_i, nd_i.item()]
                    )  # (vehicle_num, action_length)
                sortI.append(all_pref_infeasible[p_i, nd_i.item()])

        single_reward = torch.stack(sortR[:needNum])  # (needNum, reward_dim)
        if save_action:
            single_action = torch.stack(
                sortA[:needNum]
            )  # (needNum, vehicle_num, action_length)
        single_infeasible = torch.stack(sortI[:needNum])  # (needNum,)

        selected_reward_list.append(single_reward)
        if save_action:
            selected_action_list.append(single_action)
        select_infeasible_list.append(single_infeasible)

        if progress_bar is not None:
            sub_eval_bar.update()

    select_reward = torch.stack(
        selected_reward_list
    )  # (batch_size, needNum, reward_dim)
    ret_action = None
    if save_action:
        select_action = torch.stack(
            selected_action_list
        )  # (batch_size, needNum, vehicle_num, action_length)
        ret_action = select_action
    select_infeasible = torch.stack(select_infeasible_list)  # (batch_size, needNum)

    if progress_bar is not None:
        sub_eval_bar.close()

    return (
        select_reward,
        select_infeasible,
        ret_action,
        sample_best_reward if with_sample else None,
    )


def gather_tensor_and_concat(tensor: torch.Tensor) -> torch.Tensor:
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)
