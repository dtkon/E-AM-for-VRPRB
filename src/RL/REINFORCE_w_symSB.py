from typing import TYPE_CHECKING, Optional
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import augment, clip_grad_norms

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
    zoom_ratio: Optional[torch.Tensor] = None,
) -> None:
    batch_size, problem_size, _ = batch_problem.size()

    N_aug = agent.option.N_aug

    problems = augment(batch_problem, N_aug)

    if zoom_ratio is not None:
        zoom_ratio = zoom_ratio.unsqueeze(1).repeat(N_aug, 1)

    enc_problems = agent.pre_actor(problems, **other_for_actor)
    actions, log_prob, other = agent.actor(problems, *enc_problems, **other_for_actor)

    reward = agent.env(problems, actions, **other_for_env)

    if other is None or agent.option.no_penalty:
        reward_logging = reward.clone()
    else:
        reward_logging = reward.clone()
        reward = set_penalty(reward, other)

    if zoom_ratio is not None:
        reward_logging *= zoom_ratio

    reward = reward.view(N_aug, batch_size, -1).permute(1, 0, 2).reshape(batch_size, -1)
    log_prob = (
        log_prob.view(N_aug, batch_size, -1).permute(1, 0, 2).reshape(batch_size, -1)
    )

    advantage = reward - reward.mean(dim=1).view(-1, 1)
    loss = -(advantage.detach() * log_prob).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    # clip_grad_norms(agent.optimizer.param_groups, agent.option.max_grad_norm)
    agent.optimizer.step()

    if logger is not None and rank == 0 and log_step % agent.option.log_step == 0:
        logger.add_scalar('training/loss', loss.item(), log_step)
        logger.add_scalar('training/reward', reward_logging.mean().item(), log_step)
        if other is not None:
            logger.add_scalar(
                'training/Encounter multi-trip', other.sum().item(), log_step
            )


def set_penalty(reward: torch.Tensor, should_punish: torch.Tensor) -> torch.Tensor:
    punish_value = reward.abs().max()
    punish_matrix = torch.zeros_like(reward)
    punish_matrix[should_punish] = punish_value
    return reward - punish_matrix
