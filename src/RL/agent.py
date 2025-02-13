import os
import random
from typing import Callable
from functools import partial
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from options import Option

from NN.actor import (
    AM_Actor,
    AM_preActor,
    AM_CVRP_Actor,
    AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor,
    AM_TSP_2D_LimitVeh_Actor,
    AM_MO_Actor,
)
from problem.cvrp_balance import cal_reward_MO, cal_reward_MO_1D
from problem.cvrp import cal_reward as cal_reward_cvrp
from problem.tsp import cal_reward_MO_refine as cal_reward_MO_refine_tsp

from utils import get_inner_model, log_eval

from .train import train
from .eval import eval


class Agent:
    pre_actor: nn.Module
    actor: AM_Actor
    env: Callable[..., torch.Tensor]

    def __init__(self, option: Option) -> None:
        self.option = option

        basic_problem = ''
        if 'cvrp' in option.problem:
            basic_problem = 'cvrp'
        elif 'tsp' in option.problem:
            basic_problem = 'tsp'

        self.pre_actor = AM_preActor(
            option.embedding_dim,
            option.feed_forward_dim,
            option.n_heads_enc,
            option.n_blocks_graph,
            option.normalization,
            basic_problem,
        )
        if option.problem == 'cvrp':
            self.actor = AM_CVRP_Actor(option.n_heads_dec, option.embedding_dim)
        elif option.problem == 'cvrp_max':
            self.actor = AM_CVRP_2D_LimitVeh_NoMultiTrip_Actor(
                option.n_heads_dec,
                option.n_heads_veh,
                option.embedding_dim,
                not option.no_vehicle_encoder,
            )
        elif option.problem == 'tsp_max':
            self.actor = AM_TSP_2D_LimitVeh_Actor(
                option.n_heads_dec,
                option.n_heads_veh,
                option.embedding_dim,
                not option.no_vehicle_encoder,
            )
        elif option.problem[:3] == 'mo_':
            self.actor = AM_MO_Actor(
                option.preference_num,
                option.n_heads_dec,
                option.n_heads_veh,
                option.embedding_dim,
                not option.no_vehicle_encoder,
                basic_problem,
                False if option.problem == 'mo_cvrp_baseline' else True,
            )
        else:
            raise NotImplementedError

        if (
            option.use_cuda and not option.distributed
        ):  # notice 'not option.distributed' condition: if not exist will cause DDP memory unbalance!!
            self.actor.to(option.device)
            self.pre_actor.to(option.device)

        if option.problem[:3] == 'mo_':
            if '_re_' in option.problem:
                self.env = cal_reward_MO_refine_tsp
                self.other_for_actor = {
                    'vehicle_num': option.vehicle_num,
                    'will_re_tsp': True,
                }
            else:
                if option.problem != 'mo_cvrp_baseline':
                    self.env = cal_reward_MO
                else:
                    self.env = cal_reward_MO_1D
                self.other_for_actor = {'vehicle_num': option.vehicle_num}
        elif option.problem == 'cvrp':
            self.env = cal_reward_cvrp
            self.other_for_actor = {}
        elif '_max' in option.problem:
            self.env = partial(cal_reward_MO, only_max=True)
            self.other_for_actor = {'vehicle_num': option.vehicle_num}
        else:
            raise NotImplementedError

        if not option.eval_only:
            self.optimizer = torch.optim.Adam(
                [
                    {
                        'params': self.actor.parameters(),
                        'lr': option.lr_model,
                        'weight_decay': option.weight_decay,
                    },
                    {
                        'params': self.pre_actor.parameters(),
                        'lr': option.lr_model,
                        'weight_decay': option.weight_decay,
                    },
                ]
            )

            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, option.lr_decay
            )
            if option.epoch_start > 0:
                for _ in range(0, option.epoch_start):
                    self.lr_scheduler.step()

    def train(self) -> None:
        self.actor.train()
        self.pre_actor.train()

    def eval(self) -> None:
        self.actor.eval()
        self.pre_actor.eval()

    def save(self, epoch: int) -> None:
        print(' Saving model and state...', end='')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'pre_actor': get_inner_model(self.pre_actor).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'random_state': random.getstate(),
            },
            os.path.join(self.option.save_dir, 'epoch-{}.pt'.format(epoch)),
        )
        print('done.')

    def load(self) -> None:
        if self.option.load_path is not None:
            print(' [*] Loading data from {}...'.format(self.option.load_path), end='')
            load_data = torch.load(self.option.load_path, map_location='cpu')

            get_inner_model(self.actor).load_state_dict(load_data['actor'])
            get_inner_model(self.pre_actor).load_state_dict(load_data['pre_actor'])

            if not self.option.eval_only and not self.option.fine_tune:
                self.optimizer.load_state_dict(load_data['optimizer'])

                torch.set_rng_state(load_data['rng_state'])
                if self.option.use_cuda:
                    torch.cuda.set_rng_state(load_data['cuda_rng_state'])
                random.setstate(load_data['random_state'])

            torch.cuda.empty_cache()

            print('done.')

    def start_train(self) -> None:
        if self.option.distributed:
            mp.spawn(train, args=(self,), nprocs=self.option.world_size)
        else:
            train(0, self)

    def start_eval(self) -> None:
        self.eval()
        self.load()

        if self.option.distributed:
            ret = mp.Manager().Queue()
            mp.spawn(eval, args=(self, ret), nprocs=self.option.world_size)

            rewards, infeasible, action, addtion_rewards, time_used = ret.get()
            del ret
        else:
            rewards, infeasible, action, addtion_rewards, time_used = eval(0, self)

        logger = None if self.option.no_log else SummaryWriter(self.option.log_dir)
        log_eval(
            logger,
            self,
            rewards,
            infeasible,
            action,
            addtion_rewards,
            None,
            self.option.save_dir if not self.option.no_save else None,
        )
