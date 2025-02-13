from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import os, json, copy, math
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import hvwfg, geatpy
import itertools
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from RL.agent import Agent


def cal_hv(
    rewards: torch.Tensor,
    ref_point: Optional[List[float]] = None,
    infeasibles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    rewards: (val_size, need_num, reward_dim)

    infeasibles: (val_size, need_num)

    return: (val_size,)
    '''
    if ref_point is not None:
        ref = np.array(ref_point)

    hv_s = []

    if infeasibles is None:
        infeasibles_: Union[List[None], torch.Tensor] = [None] * rewards.size(0)
    else:
        infeasibles_ = infeasibles

    for reward, infeasible in zip(rewards, infeasibles_):
        if infeasible is not None and not infeasible.all():
            reward = reward[~infeasible]

        obj: np.ndarray = (-reward.cpu()).numpy()
        if ref_point is not None:
            hv = hvwfg.wfg(obj.astype(float), ref.astype(float))
            hv_ratio = hv / (ref[0] * ref[1])
        else:
            hv_ratio = geatpy.indicator.HV(obj)
        hv_s.append(hv_ratio)

    return torch.tensor(hv_s, device=rewards.device)


def log_eval(
    logger: Optional[SummaryWriter],
    agent: 'Agent',
    rewards: torch.Tensor,
    infeasible: Optional[torch.Tensor] = None,
    action: Optional[torch.Tensor] = None,
    addition_rewards: Optional[torch.Tensor] = None,
    epoch: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> None:
    if agent.option.is_mo_problem:
        _log_eval_MO(
            logger,
            agent,
            rewards,
            infeasible,
            action,
            addition_rewards,
            epoch,
            save_dir,
        )
    else:
        _log_eval_SO(logger, rewards, infeasible, epoch, save_dir)


def _log_eval_SO(
    logger: Optional[SummaryWriter],
    rewards: torch.Tensor,
    infeasible: Optional[torch.Tensor] = None,
    epoch: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> None:
    '''
    rewards: (val_size, reward_dim)

    infeasible: (val_size, N_aug, sample_times)
    '''
    if logger is not None:
        logger.add_scalar('evaluating/avg_obj', (-rewards).mean().cpu().item(), epoch)

        if infeasible is not None:
            logger.add_scalar(
                'evaluating/infeasible_rate_numerator', infeasible.sum().item(), epoch
            )
            logger.add_scalar(
                'evaluating/infeasible_rate_denominator', infeasible.numel(), epoch
            )
            logger.add_scalar(
                'evaluating/solve_fail_rate_numerator',
                infeasible.all(-1).all(-1).sum().item(),
                epoch,
            )
            logger.add_scalar(
                'evaluating/solve_fail_rate_denominator',
                (infeasible.size(0) * infeasible.size(1)),
                epoch,
            )

    if save_dir is not None:
        with open(os.path.join(save_dir, 'eval.txt'), 'a') as f:
            f.write(f'{(-rewards).mean().cpu().item()}\n\n')

        with open(os.path.join(save_dir, 'objv.json'), 'w') as f:
            json.dump((-rewards).cpu().tolist(), f)


def _log_eval_MO(
    logger: Optional[SummaryWriter],
    agent: 'Agent',
    rewards: torch.Tensor,
    infeasible: Optional[torch.Tensor] = None,
    action: Optional[torch.Tensor] = None,
    addition_rewards: Optional[torch.Tensor] = None,
    epoch: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> None:
    '''
    rewards: (val_size, pref_num or need_num, reward_dim)

    infeasible: (val_size, pref_num, N_aug, sample_times) or (val_size, need_num)
    '''
    option = agent.option

    if (not math.isnan(option.first_zoom_ratio)) and (option.ref_point is not None):
        option.ref_point = [x * option.first_zoom_ratio for x in option.ref_point]

    if infeasible is not None and infeasible.dim() == 4:
        ins_pref_infeasible = infeasible.all(-1).all(-1)  # (val_size, pref_num)
        hv_s = cal_hv(rewards, option.ref_point, ins_pref_infeasible)
    else:
        hv_s = cal_hv(rewards, option.ref_point)

    if infeasible is not None:
        rewards_with_nan = rewards.clone()

        if infeasible.dim() == 4:  # ws, not pareto
            rewards_with_nan[ins_pref_infeasible] = torch.nan
            avg_obj: np.ndarray = (
                ((-rewards_with_nan).nanmean(0)).cpu().numpy()
            )  # (pref_num, reward_dim)
        elif infeasible.dim() == 2:  # pareto
            rewards_with_nan[infeasible] = torch.nan
            obj_sort_i = (
                torch.argsort((-rewards)[:, :, 0]).unsqueeze(-1).repeat(1, 1, 2)
            )
            obj_sort = (-rewards).gather(1, obj_sort_i)
            avg_obj = obj_sort.nanmean(0).cpu().numpy()  # (need_num, reward_dim)

            if (
                addition_rewards is not None
            ):  # no infeasible available for addition_rewards
                assert option.eval_type == 'pareto_sample'

                addition_best: np.ndarray = (
                    ((-addition_rewards).min(1)[0].mean(0)).cpu().numpy()
                )
                addition_avg: np.ndarray = (
                    ((-addition_rewards).mean(1).mean(0)).cpu().numpy()
                )

                addition_hv_s = cal_hv(addition_rewards, option.ref_point)
        else:
            raise NotImplementedError

        avg_best: np.ndarray = ((-rewards).min(1)[0].mean(0)).cpu().numpy()
        avg_avg: np.ndarray = ((-rewards_with_nan).nanmean(1).mean(0)).cpu().numpy()

    else:
        avg_obj = (-rewards).mean(0).cpu().numpy()  # (pref_num, reward_dim)

        avg_best = ((-rewards).min(1)[0].mean(0)).cpu().numpy()
        avg_avg = ((-rewards).mean(1).mean(0)).cpu().numpy()

    ax: plt.Axes
    s_ax: plt.Axes
    fig, ax = plt.subplots()

    ax.scatter(avg_obj[:, 0], avg_obj[:, 1])

    single_fig_ax: List[Tuple[plt.Figure, plt.Axes]] = []
    for i in range(option.pareto_log_num):
        s_fig, s_ax = plt.subplots()
        s_ax.scatter((-rewards.cpu())[i, :, 0], (-rewards.cpu())[i, :, 1])
        single_fig_ax.append((s_fig, s_ax))

    i_100 = avg_obj.shape[0] - 1
    i_75 = int(0.75 * i_100)
    i_50 = int(0.5 * i_100)
    i_25 = int(0.25 * i_100)

    if logger is not None:
        if rewards.size(1) > 1:
            logger.add_scalar('evaluating/HV', hv_s.mean().item(), epoch)

            logger.add_figure('evaluating/pareto_front', fig, epoch)

            for i in range(option.pareto_log_num):
                logger.add_figure(
                    f'evaluating/pareto_front_{option.val_range[0]+i}',
                    single_fig_ax[i][0],
                    epoch,
                )

            logger.add_text(
                'evaluating/avg_obj',
                f'avg_best: {avg_best.tolist()}\n\n'
                + f'pref: 0.00, 1.00. reward: {-avg_obj[0,0]:.4f} + {-avg_obj[0,1]:.4f} = {-avg_obj[0,0]-avg_obj[0,1]:.4f}.\n\n'
                + f'pref: 0.25, 0.75. reward: {-avg_obj[i_25,0]:.4f} + {-avg_obj[i_25,1]:.4f} = {-avg_obj[i_25,0]-avg_obj[i_25,1]:.4f}.\n\n'
                + f'pref: 0.50, 0.50. reward: {-avg_obj[i_50,0]:.4f} + {-avg_obj[i_50,1]:.4f} = {-avg_obj[i_50,0]-avg_obj[i_50,1]:.4f}.\n\n'
                + f'pref: 0.75, 0.25. reward: {-avg_obj[i_75,0]:.4f} + {-avg_obj[i_75,1]:.4f} = {-avg_obj[i_75,0]-avg_obj[i_75,1]:.4f}.\n\n'
                + f'pref: 1.00, 0.00. reward: {-avg_obj[i_100,0]:.4f} + {-avg_obj[i_100,1]:.4f} = {-avg_obj[i_100,0]-avg_obj[i_100,1]:.4f}.\n\n',
                epoch,
            )

            if infeasible is not None:
                logger.add_scalar(
                    'evaluating/infeasible_rate_numerator',
                    infeasible.sum().item(),
                    epoch,
                )
                logger.add_scalar(
                    'evaluating/infeasible_rate_denominator', infeasible.numel(), epoch
                )

                if infeasible.dim() == 4:
                    logger.add_scalar(
                        'evaluating/solve_fail_rate_numerator',
                        infeasible.all(-1).all(-1).sum().item(),
                        epoch,
                    )
                    logger.add_scalar(
                        'evaluating/solve_fail_rate_denominator',
                        (infeasible.size(0) * infeasible.size(1)),
                        epoch,
                    )
        else:
            logger.add_scalar('evaluating/avg_best_0', avg_best[0], epoch)
            logger.add_scalar('evaluating/avg_best_1', avg_best[1], epoch)

            logger.add_text(
                'evaluating/avg_obj', f'avg_best: {avg_best.tolist()}', epoch
            )

    if save_dir is not None:
        with open(os.path.join(save_dir, 'objv.json'), 'w') as f:
            json.dump((-rewards).cpu().tolist(), f)

        if addition_rewards is not None:
            addition = (
                f'sample_hv:{addition_hv_s.mean().item()}\n'
                + f'sample_avg_best: {addition_best.tolist()}\n'
                + f'sample_avg_avg: {addition_avg.tolist()}\n'
            )
            with open(os.path.join(save_dir, 'objv_sample.json'), 'w') as f:
                json.dump((-addition_rewards).cpu().tolist(), f)
        else:
            addition = ''

        with open(os.path.join(save_dir, 'eval.txt'), 'a') as f:
            f.write(
                f'hv:{hv_s.mean().item()}\n'
                + f'avg_best: {avg_best.tolist()}\n'
                + f'avg_avg: {avg_avg.tolist()}\n'
                + addition
                + 'avg_obj:\n'
                + f'pref: 0.00, 1.00. reward: {-avg_obj[0,0]:.4f} + {-avg_obj[0,1]:.4f} = {-avg_obj[0,0]-avg_obj[0,1]:.4f}.\n'
                + f'pref: 0.25, 0.75. reward: {-avg_obj[i_25,0]:.4f} + {-avg_obj[i_25,1]:.4f} = {-avg_obj[i_25,0]-avg_obj[i_25,1]:.4f}.\n'
                + f'pref: 0.50, 0.50. reward: {-avg_obj[i_50,0]:.4f} + {-avg_obj[i_50,1]:.4f} = {-avg_obj[i_50,0]-avg_obj[i_50,1]:.4f}.\n'
                + f'pref: 0.75, 0.25. reward: {-avg_obj[i_75,0]:.4f} + {-avg_obj[i_75,1]:.4f} = {-avg_obj[i_75,0]-avg_obj[i_75,1]:.4f}.\n'
                + f'pref: 1.00, 0.00. reward: {-avg_obj[i_100,0]:.4f} + {-avg_obj[i_100,1]:.4f} = {-avg_obj[i_100,0]-avg_obj[i_100,1]:.4f}.\n\n',
            )

        collide_num = 1
        base_pic_name = 'pareto_front'
        pic_name = base_pic_name + '.png'
        while os.path.isfile(os.path.join(save_dir, pic_name)):
            pic_name = base_pic_name + '-' + str(collide_num) + '.png'
            collide_num += 1
        fig.savefig(os.path.join(save_dir, pic_name))

        for i in range(option.pareto_log_num):
            collide_num = 1
            base_pic_name = f'pareto_front_{option.val_range[0]+i}'
            pic_name = base_pic_name + '.png'
            while os.path.isfile(os.path.join(save_dir, pic_name)):
                pic_name = base_pic_name + '-' + str(collide_num) + '.png'
                collide_num += 1
            single_fig_ax[i][0].savefig(os.path.join(save_dir, pic_name))

        # save action
        if action is not None:
            with open(os.path.join(save_dir, 'action.json'), 'w') as f:
                json.dump(route_remove_duplicates(action), f)

    plt.close('all')


def route_remove_duplicates(action: torch.Tensor) -> List[List[List[List[int]]]]:
    '''
    action: (batch_size, need_num, vehicle_num, action_length)
    '''
    result = [
        [
            [[node for node, _ in itertools.groupby(a_vel.tolist())] for a_vel in a_sol]
            for a_sol in a_ins
        ]
        for a_ins in action
    ]
    return result


def create_action_array_from_list(
    action_list: List[List[List[List[int]]]],
) -> np.ndarray:
    action_list = copy.deepcopy(action_list)
    max_length = 0
    for a_ins in action_list:
        for a_sol in a_ins:
            for a_vel in a_sol:
                if len(a_vel) > max_length:
                    max_length = len(a_vel)
    for a_ins in action_list:
        for a_sol in a_ins:
            for a_vel in a_sol:
                a_vel.extend([0] * (max_length - len(a_vel)))
    return np.array(action_list)
