from typing import TYPE_CHECKING, Optional
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import clip_grad_norms

if TYPE_CHECKING:
    from .agent import Agent


def train_one_batch(
    rank: int,
    agent: 'Agent',
    batch_problem: torch.Tensor,
    other_for_actor: dict,
    other_for_env: dict,
    log_step: int,
    logger: Optional[SummaryWriter] = None,
) -> None:
    enc_problems = agent.pre_actor(batch_problem, **other_for_actor)
    actions, log_prob, other = agent.actor(
        batch_problem, *enc_problems, **other_for_actor
    )
    reward = agent.env(batch_problem, actions, **other_for_env)

    loss = -(reward.detach().view(-1) * log_prob.view(-1)).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    #clip_grad_norms(agent.optimizer.param_groups, agent.option.max_grad_norm)
    agent.optimizer.step()

    if logger is not None and rank == 0 and log_step % agent.option.log_step == 0:
        logger.add_scalar('training/loss', loss.item(), log_step)
        logger.add_scalar('training/reward', reward.mean().item(), log_step)
