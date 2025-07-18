import os, json, time
from typing import Any, Dict

import numpy as np
import hvwfg, geatpy

from option import get_option, Option
from split_tour import get_distance, get_max_distance
from evo import CMMA


def run(option: Option) -> None:
    if not option.no_save:
        with open(os.path.join(option.save_path, 'option.json'), 'w') as f:
            json.dump(vars(option), f, indent=True)
        if not option.no_log:
            log_f = open(os.path.join(option.save_path, 'log.txt'), 'w')

    if option.resume_population is not None:
        initial_P = json.load(open(option.resume_population))
        N_gen = option.N_gen - option.resume_gen
    else:
        initial_P = None
        N_gen = option.N_gen

    problem = np.load(option.load_file)[option.problem_index]

    for i, P in enumerate(
        CMMA(
            problem,
            option.vehicle_num,
            option.capacity,
            option.population_size,
            N_gen,
            option.Iter,
            option.LS_iter,
            option.tournament_tau,
            initial_P,
            option.save_path,
        ),
        option.resume_gen + 1,
    ):
        s = f'[{time.strftime("%m/%d %H:%M:%S")}] {option.load_file}[{option.problem_index}] gen {i} done.'
        if not option.no_display:
            print(s)
        if not option.no_save and not option.no_log:
            log_f.write(s + '\n')
            log_f.flush()

        if (i % option.save_per_gen == 0) or (i == N_gen):
            with open(os.path.join(option.save_path, f'population-{i}.json'), 'w') as f:
                json.dump(P, f)

            obj_value: Dict[str, Any] = {'obj': [], 'ref': option.ref_point}
            for solution in P:
                obj_value['obj'].append(
                    (
                        get_distance(problem, solution),
                        get_max_distance(problem, solution),
                    )
                )
            obj_array = np.array(obj_value['obj'])
            if option.ref_point is not None:
                hv = hvwfg.wfg(obj_array, np.array(option.ref_point))
                hv_ratio = hv / (option.ref_point[0] * option.ref_point[1])
                obj_value['hv'] = hv_ratio
            else:
                obj_value['hv'] = geatpy.indicator.HV(obj_array)

            if not option.no_display:
                s = f'[{time.strftime("%m/%d %H:%M:%S")}] {option.load_file}[{option.problem_index}] gen {i} HV: {obj_value["hv"]}'
                print(s)
            if not option.no_save and not option.no_log:
                log_f.write(s + '\n')

            with open(os.path.join(option.save_path, f'value-{i}.json'), 'w') as f:
                json.dump(obj_value, f)

    if not option.no_save and not option.no_log:
        log_f.close()


if __name__ == "__main__":
    run(get_option())
