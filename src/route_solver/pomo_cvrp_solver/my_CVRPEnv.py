import torch

from .CVRPEnv import CVRPEnv
from .CVRProblemDef import augment_xy_data_by_8_fold


class my_CVRPEnv(CVRPEnv):
    def load_problems(self, problems: torch.Tensor, aug_factor: int = 1) -> None:
        '''
        problems: (batch_size, problem_size, 3)
        '''
        batch_size = problems.size(0)

        self.batch_size = batch_size

        depot_xy = problems[:, :1, :2]  # (batch_size, 1, 2)
        node_xy = problems[:, 1:, :2]  # (batch_size, customer_number, 2)
        node_demand = problems[:, 1:, 2]  # (batch_size, customer_number)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size
        )

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
