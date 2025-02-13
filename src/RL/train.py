import random
import platform
from typing import TYPE_CHECKING, Callable, cast
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm
import numpy as np

from problem.cvrp import generate_datasets as generate_datasets_cvrp
from problem.cvrp_balance import generate_datasets as generate_datasets_cvrp_balance
from problem.tsp import generate_datasets as generate_datasets_tsp
from route_solver import TspSolver, pretrained_model, zoom
from utils import log_eval

from .REINFORCE_w_symSB import train_one_batch
from .REINFORCE_w_symSB_MO import train_one_batch as train_one_batch_mo
from .eval import eval


if TYPE_CHECKING:
    from .agent import Agent
    from NN.actor import AM_Actor, AM_preActor


def train(rank: int, agent: 'Agent') -> None:
    option = agent.option
    logger = None if option.no_log else SummaryWriter(option.log_dir)

    if option.distributed:
        device = torch.device('cuda', rank)
        dist.init_process_group(
            backend='gloo' if platform.system() == 'Windows' else 'nccl',
            world_size=option.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        agent.actor.to(device)  # this will increase memory cost of original place.
        agent.pre_actor.to(device)

        if option.normalization == 'batch':
            agent.pre_actor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                agent.pre_actor
            ).to(device)

        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

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

        dist.barrier()
    else:
        device = option.device

        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if '_re_' in option.problem:
        tsp_solvers = {
            20: TspSolver(*pretrained_model['tsp20']).to_device(device),
        }
        other_for_env: dict = {'tsp_solvers': tsp_solvers}
    else:
        other_for_env = {}

    agent.train()

    # set or restore seed
    if option.load_path is None:
        torch.manual_seed(option.seed)
        random.seed(option.seed)
    else:
        agent.load()

    z = torch.tensor(option.z, device=device)

    steps = option.epoch_size // option.batch_size

    training_bar = tqdm(
        total=(option.epoch_end - option.epoch_start) * steps,
        disable=option.no_progress_bar or rank != 0,
        desc='training',
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    )

    if 'cvrp' in option.problem:
        if option.problem == 'cvrp':
            generate_datasets: Callable[..., torch.Tensor] = generate_datasets_cvrp
        else:
            generate_datasets = generate_datasets_cvrp_balance
    elif 'tsp' in option.problem:
        generate_datasets = generate_datasets_tsp
    else:
        raise NotImplementedError

    if option.first_eval_once and not option.no_eval_in_train:
        # if rank == 0: all sub-process need to call eval, otherwise program will get stuck because of SyncBatchNorm
        agent.eval()
        eval_rewards, infeasible, action, addtion_rewards, time_used = eval(
            rank, agent, init_dist=False, other_for_env=other_for_env
        )
        agent.train()

        if rank == 0:
            log_eval(
                logger,
                agent,
                eval_rewards,
                infeasible,
                action,
                addtion_rewards,
                option.epoch_start - 1,
            )

    if option.fine_tune:
        assert option.val_dataset is not None
        val_problem = torch.tensor(
            np.load(option.val_dataset)[option.val_range[0] : option.val_range[1]]
        ).repeat(option.epoch_size, 1, 1)
    for epoch in range(option.epoch_start, option.epoch_end):
        if not option.fine_tune:
            epoch_problem = generate_datasets(
                option.epoch_size,
                option.customer_size,
                vehicle_num=option.vehicle_num,
                vehicle_capacity=option.vehicle_capacity,
            )
        else:
            epoch_problem = val_problem
        if option.distributed:
            train_sampler: torch.utils.data.distributed.DistributedSampler = (
                torch.utils.data.distributed.DistributedSampler(
                    cast(Dataset, epoch_problem), shuffle=False
                )
            )
            train_dataloader = DataLoader(
                cast(Dataset, epoch_problem),
                batch_size=option.batch_size // option.world_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=train_sampler,
            )
        else:
            train_dataloader = DataLoader(
                cast(Dataset, epoch_problem),
                batch_size=option.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        batch_problem: torch.Tensor
        for step, batch_problem in enumerate(train_dataloader):
            batch_problem = batch_problem.to(device)

            if option.zoom_on:
                batch_problem, zoom_ratio = zoom(batch_problem)
            else:
                zoom_ratio = None

            if option.is_mo_problem:
                train_one_batch_mo(
                    rank,
                    agent,
                    batch_problem,
                    agent.other_for_actor,
                    other_for_env,
                    z,
                    epoch * steps + step,
                    logger,
                    zoom_ratio,
                )
            else:
                train_one_batch(
                    rank,
                    agent,
                    batch_problem,
                    agent.other_for_actor,
                    other_for_env,
                    epoch * steps + step,
                    logger,
                    zoom_ratio,
                )

            training_bar.update()

        agent.lr_scheduler.step()

        # save new model after one epoch
        if (
            rank == 0
            and not option.no_save
            and (
                (option.save_per_epochs != 0 and epoch % option.save_per_epochs == 0)
                or epoch == option.epoch_end - 1
            )
        ):
            agent.save(epoch)

        if not option.no_eval_in_train:

            torch.cuda.empty_cache()

            # if rank == 0: all sub-process need to call eval, otherwise program will get stuck because of SyncBatchNorm
            agent.eval()
            eval_rewards, infeasible, action, addtion_rewards, time_used = eval(
                rank, agent, init_dist=False, other_for_env=other_for_env
            )
            agent.train()

            # if option.problem == 'cvrp':
            #    bl_vals = eval(rank, agent, baseline)
            #    baseline.update(eval_rewards, bl_vals, rank == 0)

            if rank == 0:
                log_eval(
                    logger,
                    agent,
                    eval_rewards,
                    infeasible,
                    action,
                    addtion_rewards,
                    epoch,
                )

        torch.cuda.empty_cache()

        if option.distributed:
            dist.barrier()

    training_bar.close()
