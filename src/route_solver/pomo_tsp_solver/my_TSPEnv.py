import torch

from .TSPEnv import TSPEnv
from .TSProblemDef import augment_xy_data_by_8_fold


class my_TSPEnv(TSPEnv):
    def load_problems(self, problems: torch.Tensor, aug_factor: int = 1) -> None:
        '''
        problems: (batch_size, problem_size, 2)
        '''
        self.batch_size = problems.size(0)

        self.problems = problems[:, :, :2]

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch_size, problem_size, 2) ``
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size
        )
