from typing import Optional, Tuple, Union
import torch
import os

from ..solver import Solver
from .my_TSPEnv import my_TSPEnv as Env
from .TSPModel import TSPModel as Model


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TspSolver(Solver):  # modified from TSPTester
    def __init__(
        self,
        model_load_path: str = 'saved_tsp20_model',
        model_load_epoch: int = 510,
        use_cuda: bool = False,
        cuda_device_num: int = 0,
        aug_factor: int = 1,
    ) -> None:
        # save arguments
        self.problem_size = None
        model_load_path = f'{FILE_DIR_PATH}/result/' + model_load_path

        self.model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
        }

        tester_params = {
            'use_cuda': use_cuda,
            'cuda_device_num': cuda_device_num,
            'model_load': {
                'path': model_load_path,  # directory path of pre-trained model and log files saved.
                'epoch': model_load_epoch,  # epoch version of pre-trained model to laod.
            },
            'test_episodes': 100 * 1000,
            'test_batch_size': 10000,
            'augmentation_enable': True,
            'aug_factor': aug_factor,
            'aug_batch_size': 1000,
        }

        if tester_params['augmentation_enable']:
            tester_params['test_batch_size'] = tester_params['aug_batch_size']

        self.tester_params = tester_params

        # cuda
        if use_cuda:
            # torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            # torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        # self.env = Env(**self.env_params)
        self.model = Model(**self.model_params).to(device)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)  # type: ignore
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def to_device(self, device: Union[str, torch.device]) -> 'TspSolver':
        device = torch.device(device)
        self.model = self.model.to(device)
        self.device = device
        return self

    def solve(
        self,
        problems: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_type: str = 'argmax',
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # modified from TSPTester._test_one_batch
        '''
        problems: (batch_size, problem_size, 2)

        mask: (batch_size, problem_size), mask[:, 0] must all be False.

        eval_type: argmax or softmax

        return: [(batch_size,), (batch_size, problem_size)], not always start at 0, no same node.
        '''
        if problems.size(1) == 1:
            return (
                torch.zeros(problems.size(0), device=problems.device),
                torch.zeros((problems.size(0), 1), device=problems.device),
            )

        if self.device.type != 'cpu':
            torch.cuda.set_device(self.device.index)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        assert eval_type in ('softmax', 'argmax')
        self.model.model_params['eval_type'] = eval_type

        batch_size, problem_size, _ = problems.size()

        if problem_size != self.problem_size:
            self.env_params = {
                'problem_size': problem_size,
                'pomo_size': problem_size,
            }
            self.env = Env(**self.env_params)

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        if mask is not None:
            assert torch.all(mask[:, 0] == False)

            mask_for_encoder = mask.repeat(aug_factor, 1)
            mask_for_decoder = mask_for_encoder.unsqueeze(1).repeat(
                1, self.env_params['pomo_size'], 1
            )
            extra_ninf_mask_for_encoder = torch.zeros(
                size=(aug_factor * batch_size, problem_size)
            )
            extra_ninf_mask_for_decoder = torch.zeros(
                size=(
                    aug_factor * batch_size,
                    self.env_params['pomo_size'],
                    problem_size,
                )
            )
            extra_ninf_mask_for_encoder[mask_for_encoder] = float('-inf')
            extra_ninf_mask_for_decoder[mask_for_decoder] = float('-inf')
        else:
            extra_ninf_mask_for_encoder = None

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(problems, aug_factor)  # type: ignore
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, extra_ninf_mask_for_encoder)

        if mask is not None:
            self.env.step_state.ninf_mask = extra_ninf_mask_for_decoder

        # trace route
        route_list = []

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (augmentation*batch, pomo)
            state, reward, done = self.env.step(selected)

            # trace route
            route_list.append(selected)

        # trace route
        route = torch.stack(
            route_list, dim=-1
        )  # (augmentation*batch, pomo, problem_size)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, max_pomo_reward_index = aug_reward.max(
            dim=2
        )  # get best results from pomo
        # shape: (augmentation, batch)

        # trace route
        max_pomo_route = route[
            torch.arange(aug_factor * batch_size), max_pomo_reward_index.reshape(-1), :
        ]
        max_pomo_route = max_pomo_route.reshape(
            aug_factor, batch_size, -1
        )  # (augmentation, batch, problem_size)

        max_aug_pomo_reward, max_aug_pomo_reward_index = max_pomo_reward.max(
            dim=0
        )  # get best results from augmentation
        # shape: (batch,)

        # trace route
        max_aug_pomo_route = max_pomo_route[
            max_aug_pomo_reward_index.reshape(-1), torch.arange(batch_size), :
        ]

        torch.set_default_tensor_type(torch.FloatTensor)

        return -max_aug_pomo_reward, max_aug_pomo_route
