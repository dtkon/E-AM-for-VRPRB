import os, re, time, argparse
from typing import List, Optional, Sequence


class Option(argparse.Namespace):
    load_file: str
    problem_index: int
    vehicle_num: int
    capacity: float

    population_size: int
    N_gen: int
    Iter: int
    LS_iter: int
    tournament_tau: float

    seed: int
    run_name: str
    save_dir: str
    save_per_gen: int
    no_save: bool
    no_display: bool
    no_log: bool
    ref_point: Optional[List[float]]
    resume_population: Optional[str]

    resume_gen: int
    save_path: str


no_provide = object()


def get_option(args: Optional[Sequence[str]] = None) -> Option:
    parser = argparse.ArgumentParser(description='CMMA for VRPRB')

    parser.add_argument('--load_file', required=True)
    parser.add_argument('--problem_index', type=int, default=0)
    parser.add_argument('--vehicle_num', type=int, default=no_provide)
    parser.add_argument('--capacity', type=float, default=1.0)

    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--N_gen', type=int, default=10000)
    parser.add_argument('--Iter', type=int, default=40)
    parser.add_argument('--LS_iter', type=int, default=300)
    parser.add_argument('--tournament_tau', type=float, default=0.25)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_name', default='')
    parser.add_argument('--save_dir', default='result')
    parser.add_argument('--save_per_gen', type=int, default=100)
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--ref_point', nargs=2, type=float, default=None)
    parser.add_argument('--resume_population')

    option = Option()
    parser.parse_args(args, namespace=option)

    if option.vehicle_num is no_provide:
        try:
            option.vehicle_num = int(
                re.search('-v(\\d+)\\.', option.load_file).group(1)  # type: ignore
            )
        except AttributeError as ae:
            if 'NoneType' in ae.args[0]:
                print(
                    'No vehicle number in load file\'s name. Expected file name example: cvrp20-v4.npy'
                )
                exit()
            else:
                raise ae

    if option.resume_population is None:
        if option.run_name == '':
            option.run_name = 'p{}_{}'.format(
                option.problem_index, time.strftime("%Y%m%dT%H%M%S")
            )
        else:
            option.run_name = 'p{}_{}_{}'.format(
                option.problem_index, option.run_name, time.strftime("%Y%m%dT%H%M%S")
            )
        option.resume_gen = 0
    else:
        option.run_name = os.path.split(os.path.split(option.resume_population)[0])[1]
        try:
            option.resume_gen = int(
                re.search('-(\\d+)\\.', option.resume_population).group(1)  # type: ignore
            )
        except AttributeError as ae:
            if 'NoneType' in ae.args[0]:
                print(
                    'No generation number in resume population file\'s name. Expected file name example: population-100.json'
                )
                exit()
            else:
                raise ae

    option.save_path = os.path.join(
        option.save_dir,
        os.path.splitext(os.path.basename(option.load_file))[0],
        option.run_name,
    )

    if not option.no_save:
        os.makedirs(option.save_path, exist_ok=True)

    return option
