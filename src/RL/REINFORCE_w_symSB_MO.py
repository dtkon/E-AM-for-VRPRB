from typing import TYPE_CHECKING, Optional
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import augment, clip_grad_norms
from .REINFORCE_w_symSB import set_penalty

if TYPE_CHECKING:
    from .agent import Agent


def train_one_batch(
    rank: int,
    agent: 'Agent',
    batch_problem: torch.Tensor,
    other_for_actor: dict,
    other_for_env: dict,
    z: torch.Tensor,
    log_step: int,
    logger: Optional[SummaryWriter] = None,
    zoom_ratio: Optional[torch.Tensor] = None,
) -> None:
    batch_size, problem_size, _ = batch_problem.size()

    N_aug = agent.option.N_aug

    rand_pref = torch.rand(
        (agent.option.N_pref, agent.option.preference_num), device=batch_problem.device
    )
    rand_pref = rand_pref / torch.sum(rand_pref, dim=1, keepdim=True)

    if not agent.option.no_fixed_pref:
        fixed_pref = torch.eye(agent.option.preference_num, device=batch_problem.device)
        rand_pref = torch.cat((fixed_pref, rand_pref))

    if (so := agent.option.so_mode) != -1:
        rand_pref = rand_pref[so : so + 1]

    problems = augment(batch_problem, N_aug)

    if zoom_ratio is not None:
        zoom_ratio = zoom_ratio.unsqueeze(1).repeat(N_aug, 1)

    # enc_problems = agent.pre_actor(problems, **other_for_actor)
    # backward in loop, so pre_actor call should in loop.

    log_loss = 0.0
    log_text = ''
    for pref in rand_pref:
        enc_problems = agent.pre_actor(problems, **other_for_actor)

        actions, log_prob, other = agent.actor(
            pref, problems, *enc_problems, **other_for_actor
        )
        reward = agent.env(problems, actions, **other_for_env)

        if other is None or agent.option.no_penalty:
            reward_logging = reward.clone()
        else:
            reward_logging = reward.clone()
            reward = set_penalty(reward, other)

        if zoom_ratio is not None:
            reward_logging *= zoom_ratio

        if agent.option.mo_reward_type == 'tch':
            # reward was negative, here we set it to positive to calculate TCH
            mo_reward: torch.Tensor = -((pref * (-reward - z)).max(dim=1)[0])
        elif agent.option.mo_reward_type == 'ws':
            mo_reward = (pref * reward).sum(dim=1)

        mo_reward = (
            mo_reward.view(N_aug, batch_size, -1)
            .permute(1, 0, 2)
            .reshape(batch_size, -1)
        )
        log_prob = (
            log_prob.view(N_aug, batch_size, -1)
            .permute(1, 0, 2)
            .reshape(batch_size, -1)
        )

        advantage = mo_reward - mo_reward.mean(dim=1).view(-1, 1)
        # loss += -(advantage.detach() * log_prob).mean()
        loss = -(advantage.detach() * log_prob).mean()
        log_loss += loss.item()

        agent.optimizer.zero_grad()
        loss.backward()
        # clip_grad_norms(agent.optimizer.param_groups, agent.option.max_grad_norm)
        agent.optimizer.step()

        if logger is not None and rank == 0 and log_step % agent.option.log_step == 0:
            encounter_multi_trip = ''
            if other is not None:
                encounter_multi_trip = f'Encounter multi-trip: {other.sum().item()}'
            log_text += f'pref: {pref.cpu()}. reward: {reward_logging[:,0].mean().item():.4f} + {reward_logging[:,1].mean().item():.4f} = {reward_logging.sum(1).mean().item():.4f}. loss: {loss.item()}. {encounter_multi_trip}\n\n'

    # loss /= agent.option.N_pref
    log_loss /= agent.option.N_pref

    if logger is not None and rank == 0 and log_step % agent.option.log_step == 0:
        logger.add_text('training/step', log_text, log_step)
        logger.add_scalar('training/loss', log_loss, log_step)

    # agent.optimizer.zero_grad()
    # loss.backward()
    # clip_grad_norms(agent.optimizer.param_groups, 1.0)
    # agent.optimizer.step()
