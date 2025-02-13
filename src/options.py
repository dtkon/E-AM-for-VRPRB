import os
import sys
import time
import argparse
import re
from typing import List, Optional, Sequence
import torch

DEBUG_DDP = False


class Option(argparse.Namespace):
    # Problem
    problem: str
    customer_size: int
    vehicle_num: int
    vehicle_capacity: int
    preference_num: int
    mo_reward_type: str
    so_mode: int

    # Model
    embedding_dim: int
    feed_forward_dim: int
    n_heads_enc: int
    n_heads_dec: int
    n_heads_veh: int
    n_blocks_graph: int
    normalization: str

    # Model experimental
    no_vehicle_encoder: bool

    # Training
    batch_size: int
    epoch_size: int
    epoch_start: int
    epoch_end: int
    N_aug: int
    N_pref: int
    lr_model: float
    lr_decay: float
    weight_decay: float
    # max_grad_norm: float
    no_eval_in_train: bool
    z: List[float]
    no_fixed_pref: bool
    no_penalty: bool
    fine_tune: bool

    # Evaluate
    eval_only: bool
    val_range: List[int]
    val_batch_size: int
    val_dataset: Optional[str]
    val_customer_size: int
    val_vehicle_num: int
    quick_eval: bool
    eval_type: str
    val_N_aug: int
    sample_times: int
    max_parallel: int
    pareto_log_num: int
    ref_point: Optional[List[float]]

    # Misc
    seed: int
    no_cuda: bool
    no_DDP: bool
    no_save: bool
    no_log: bool
    no_write: bool
    run_name: str
    model_save_dir: str
    log_dir: str
    save_per_epochs: int
    log_step: int
    load_path: Optional[str]
    resume: Optional[str]
    first_eval_once: bool
    no_progress_bar: bool
    DDP_port_offset: int
    zoom: str

    # Add later
    use_cuda: bool
    distributed: bool
    world_size: int
    save_dir: str
    device: torch.device
    is_mo_problem: bool
    zoom_on: bool
    first_zoom_ratio: float


no_provide = object()


def get_options(args: Optional[Sequence[str]] = None) -> Option:
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Routing Problem with Reinforcement Learning"
    )

    # Problem
    parser.add_argument(
        '--problem',
        default='mo_cvrp',
        choices=(
            'cvrp',
            'cvrp_max',
            'mo_cvrp',
            'mo_cvrp_re_tsp',
            'tsp_max',
            'mo_tsp',
            'mo_tsp_re_tsp',
            'mo_cvrp_baseline',
        ),
    )
    parser.add_argument('--customer_size', type=int, default=20)
    parser.add_argument('--vehicle_num', default=no_provide, type=int)
    parser.add_argument('--vehicle_capacity', default=-1, type=int)
    parser.add_argument('--preference_num', default=2, type=int)
    parser.add_argument(
        '--mo_reward_type',
        default='tch',
        choices=('tch', 'ws'),
        help='Weighted-Tchebycheff (Weighted-TCH) Aggregation or Weighted-Sum Aggregation',
    )
    parser.add_argument('--so_mode', default=-1, type=int)

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--feed_forward_dim', type=int, default=512)
    parser.add_argument('--n_heads_enc', type=int, default=8)
    parser.add_argument('--n_heads_dec', type=int, default=8)
    parser.add_argument('--n_heads_veh', type=int, default=8)
    parser.add_argument('--n_blocks_graph', type=int, default=3)

    parser.add_argument(
        '--normalization', default='batch', choices=('layer', 'batch', 'instance')
    )

    # Model experimental
    parser.add_argument('--no_vehicle_encoder', action='store_true')

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_size', type=int, default=128000)
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--epoch_end', type=int, default=200)
    parser.add_argument('--N_aug', type=int, default=8)
    parser.add_argument('--N_pref', type=int, default=2)
    parser.add_argument('--lr_model', type=float, default=1e-4)
    parser.add_argument(
        '--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch'
    )
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument(
    #    '--max_grad_norm',
    #    type=float,
    #    default=1.0,
    #    help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)',
    # )
    parser.add_argument('--no_eval_in_train', action='store_true')
    parser.add_argument('--z', nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument('--no_fixed_pref', action='store_true')
    parser.add_argument('--no_penalty', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')

    # Evaluate
    parser.add_argument(
        '--eval_only', action='store_true', help='Set this value to only evaluate model'
    )
    parser.add_argument(
        '--val_range',
        type=int,
        nargs=2,
        default=[0, 200],
        help='Range of instances used for reporting validation performance',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=200,
        help="Batch size to use during evaluation",
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        default=no_provide,
        help='Dataset file to use for validation',
    )
    parser.add_argument('--val_customer_size', default=no_provide, type=int)
    parser.add_argument('--val_vehicle_num', default=no_provide, type=int)
    parser.add_argument('--quick_eval', action='store_true')
    parser.add_argument(
        '--eval_type',
        type=str,
        default='greedy_aug',
        choices=('greedy', 'sample', 'greedy_aug', 'pareto', 'pareto_sample'),
    )  # pareto_sample is both pareto and sample
    parser.add_argument('--val_N_aug', type=int, default=8)
    parser.add_argument('--sample_times', type=int, default=128)
    parser.add_argument('--max_parallel', type=int, default=sys.maxsize)
    parser.add_argument('--pareto_log_num', type=int, default=5)
    parser.add_argument(
        '--ref_point',
        type=float,
        nargs=2,
        default=None,
        help='No set for auto ref point, and set 0 0 for no ref point',
    )

    # Misc
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_DDP', action='store_true', help='Disable DDP')
    parser.add_argument('--no_save', action='store_true', help='Disable model saving')
    parser.add_argument('--no_log', action='store_true', help='Disable tensorboard log')
    parser.add_argument(
        '--no_write',
        action='store_true',
        help='Disable model saving and tensorboard log',
    )
    parser.add_argument('--run_name', default='', help='Name to identify the run')
    parser.add_argument(
        '--model_save_dir',
        default='saved_model',
        help='Directory to write output models to',
    )
    parser.add_argument('--log_dir', default='log')
    parser.add_argument(
        '--save_per_epochs',
        type=int,
        default=1,
        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints',
    )
    parser.add_argument(
        '--log_step',
        type=int,
        default=10,
        help='log info every log_step gradient steps',
    )
    parser.add_argument(
        '--load_path', help='Path to load model parameters and optimizer state from'
    )
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--first_eval_once', action='store_true')
    parser.add_argument(
        '--no_progress_bar', action='store_true', help='Disable progress bar'
    )
    parser.add_argument('--DDP_port_offset', type=int, default=0)
    parser.add_argument(
        '--zoom',
        choices=('on', 'off', 'auto'),
        default='auto',
        help='zoom on, off or auto when fine tune or eval instances',
    )

    opts = Option()
    parser.parse_args(args, namespace=opts)

    if opts.fine_tune:
        opts.val_batch_size = 1
        assert opts.val_range[1] - opts.val_range[0] == 1

    if not opts.eval_only:
        assert (
            opts.epoch_size % opts.batch_size == 0
        ), "Epoch size must be integer multiple of batch size!"
    assert (
        opts.val_range[1] - opts.val_range[0]
    ) % opts.val_batch_size == 0, (
        "Validation size must be integer multiple of validation batch size!"
    )
    assert opts.epoch_size >= opts.batch_size
    assert (opts.val_range[1] - opts.val_range[0]) >= opts.val_batch_size

    if opts.problem[:3] == 'mo_':
        opts.is_mo_problem = True
    else:
        opts.is_mo_problem = False

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.world_size = torch.cuda.device_count()
    opts.distributed = DEBUG_DDP or (
        opts.use_cuda and (opts.world_size > 1) and (not opts.no_DDP)
    )

    if opts.distributed:
        assert opts.batch_size % opts.world_size == 0
        assert opts.val_batch_size % opts.world_size == 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(4869 + opts.DDP_port_offset)

    if opts.run_name == '':
        opts.run_name = time.strftime("%Y%m%dT%H%M%S")
    else:
        opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))

    if opts.resume is not None:
        assert opts.load_path is None
        assert opts.epoch_start == 0
        opts.run_name = os.path.split(os.path.split(opts.resume)[0])[1]
        opts.epoch_start = (
            int(os.path.splitext(os.path.split(opts.resume)[1])[0].split("-")[1]) + 1
        )
        opts.load_path = opts.resume

    if opts.no_write:
        opts.no_save = True
        opts.no_log = True

    if opts.vehicle_num is no_provide:
        if opts.customer_size == 20:
            opts.vehicle_num = 4
        elif opts.customer_size == 50:
            opts.vehicle_num = 7
        elif opts.customer_size == 100:
            opts.vehicle_num = 11
        elif opts.customer_size == 200:
            opts.vehicle_num = 14
        elif opts.customer_size == 500:
            opts.vehicle_num = 24
        elif opts.customer_size == 1000:
            opts.vehicle_num = 32
        else:
            raise NotImplementedError

    if opts.val_customer_size is no_provide:
        opts.val_customer_size = opts.customer_size
    if opts.val_vehicle_num is no_provide:
        opts.val_vehicle_num = opts.vehicle_num

    if opts.val_dataset is no_provide:
        opts.val_dataset = None
        if opts.problem == 'cvrp':
            opts.val_dataset = f'datasets/cvrp{opts.val_customer_size}.npy'
        elif 'cvrp' in opts.problem:
            opts.val_dataset = (
                f'datasets/cvrp{opts.val_customer_size}-v{opts.val_vehicle_num}.npy'
            )
        elif 'tsp' in opts.problem:
            opts.val_dataset = f'datasets/tsp{opts.val_customer_size}.npy'
    elif opts.val_dataset == 'random':
        opts.val_dataset = None

    # if opts.z is no_provide:
    #    if opts.customer_size == 20:
    #        opts.z = [6.0, 6.0 / opts.vehicle_num]
    #    elif opts.customer_size == 50:
    #        opts.z = [10.0, 10.0 / opts.vehicle_num]
    #    elif opts.customer_size == 100:
    #        opts.z = [15.0, 15.0 / opts.vehicle_num]
    #    else:
    #        opts.z = [0.0, 0.0]

    # https://github.com/marmotlab/PAN-CAS/blob/main/MOCVRP/POMO/test_mocvrp_n100.py
    ref_cvrp = {
        20: [30.0, 3.0],  # cvrp(11.0, 4.0)
        50: [50.0, 3.0],  # cvrp(25.1, 5.2)
        75: [65.0, 3.0],
        100: [80.0, 3.0],  # cvrp(48.4, 6.4)
        120: [96.0, 3.0],
        150: [120.0, 3.0],
        199: [180.0, 3.0],
        200: [180.0, 3.0],  # cvrp(89.7, 8.8)
        500: [400.0, 3.0],  # cvrp(208.9, 11.7)
        1000: [800.0, 3.0],  # cvrp(402.7, 16.5)
    }
    ref_cvrp_old = {
        20: [30.0, 8.0],  # cvrp(11.0, 4.0)
        50: [50.0, 10.0],  # cvrp(25.1, 5.2)
        100: [80.0, 15.0],  # cvrp(48.4, 6.4)
        200: [180.0, 20.0],  # cvrp(89.7, 8.8)
        500: [400.0, 30.0],  # cvrp(208.9, 11.7)
        1000: [800.0, 40.0],  # cvrp(402.7, 16.5)
    }
    ref_tsp = {
        20: [30.0, 30.0],  # tsp(10.0, 3.8)
        50: [50.0, 50.0],  # tsp(22.2, 5.3)
        100: [80.0, 80.0],  # tsp(40.6, 8.0)
        200: [140.0, 140.0],  # tsp(70.0, 11.6)
        500: [320.0, 320.0],  # tsp(158.9, 16.0)
        1000: [540.0, 540.0],  # tsp(273.3, 21.5)
    }

    if opts.ref_point is None:
        if 'cvrp' in opts.problem:
            opts.ref_point = ref_cvrp[opts.customer_size]
        elif 'tsp' in opts.problem:
            opts.ref_point = ref_tsp[opts.customer_size]
        else:
            raise NotImplementedError
    else:
        if all((x == 0 for x in opts.ref_point)):
            opts.ref_point = None

    if opts.pareto_log_num > opts.val_range[1] - opts.val_range[0]:
        opts.pareto_log_num = opts.val_range[1] - opts.val_range[0]

        if opts.val_range[1] - opts.val_range[0] == 1:
            opts.pareto_log_num = 0

    if opts.zoom == 'on':
        opts.zoom_on = True
    elif opts.zoom == 'off':
        opts.zoom_on = False
    elif opts.zoom == 'auto':
        assert opts.val_dataset is not None
        if re.search(r'p\d+ E\d+-\d+[ec]\.npy', opts.val_dataset) is not None:
            opts.zoom_on = True
        else:
            opts.zoom_on = False
    else:
        raise NotImplementedError

    opts.log_dir = os.path.join(
        opts.log_dir,
        "{}_{}-v{}".format(opts.problem, opts.customer_size, opts.vehicle_num),
        opts.run_name,
    )

    opts.save_dir = os.path.join(
        opts.model_save_dir,
        "{}_{}-v{}".format(opts.problem, opts.customer_size, opts.vehicle_num),
        opts.run_name,
    )

    return opts
