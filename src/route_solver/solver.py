import abc
from typing import Optional, Tuple, Union
import torch


class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        problems: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_type: str = 'argmax',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        problems: (batch_size, problem_size, feature_num)

        mask: (batch_size, problem_size), mask[:, 0] must all be False.

        eval_type: argmax or softmax

        return: [length(batch_size,), routes(batch_size, node_indexes)]
        '''

    @abc.abstractmethod
    def to_device(self, device: Union[str, torch.device]) -> 'Solver':
        pass
