import json, os, os.path as op
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import hvwfg, geatpy


def read_result_dir(
    result_dir: str, epoch: Optional[int] = None
) -> List[Dict[str, Any]]:
    results = []

    ins_dirs = [op.join(result_dir, x) for x in os.listdir(result_dir)]
    for ins_dir in ins_dirs:
        if epoch is None:
            value_files = filter(lambda f: 'value-' in f, os.listdir(ins_dir))
            sorted_value_files = sorted(
                value_files, key=lambda f: int(f[f.index('-') + 1 : f.index('.')])
            )
            last_value_file = op.join(ins_dir, sorted_value_files[-1])
        else:
            last_value_file = op.join(ins_dir, f'value-{epoch}.json')

        try:
            result = json.load(open(last_value_file))
            results.append(result)
        except FileNotFoundError:
            print(last_value_file)

    return results


def plt_avg_pareto(results: List[Dict[str, Any]], only_print_min: bool = False) -> None:
    obj_sorts = []
    for result in results:
        obj_nd = np.array(result['obj'])
        obj_sort_i = np.argsort(obj_nd[:, 0])
        obj_sort = obj_nd[obj_sort_i]
        obj_sorts.append(obj_sort)
    obj_stack = np.stack(obj_sorts, 0)
    obj_avg = obj_stack.mean(0)

    if only_print_min:
        return

    plt.scatter(obj_avg[:, 0], obj_avg[:, 1])
    plt.show()


def plt_pareto(obj: List[List[float]]) -> None:
    obj_nd = np.array(obj)
    plt.scatter(obj_nd[:, 0], obj_nd[:, 1])
    plt.show()


def reduce_results(results: List[Dict[str, Any]]):
    bests_list = []
    avgs_list = []
    for result in results:
        obj_nd = np.array(result['obj'])
        best = obj_nd.min(0)
        avg = obj_nd.mean(0)
        bests_list.append(best)
        avgs_list.append(avg)
    bests = np.stack(bests_list, 0)
    avgs = np.stack(avgs_list, 0)

    return bests.mean(0), avgs.mean(0)


if __name__ == '__main__':
    results = read_result_dir(
        r'D:\Workspace\Project\VRPRB-result\cmma-result\cvrp' + f'20-v4', 10000
    )
    plt_pareto(results[0]['obj'])
    exit()

    inss = [(20, 4), (20, 5), (50, 7), (50, 9), (100, 11), (100, 14)]

    for cus, veh in inss:
        results = read_result_dir(
            r'D:\Workspace\Project\VRPRB-result\cmma-result\cvrp' + f'{cus}-v{veh}', 5000
        )

        if cus == 20:
            ref_point = (30.0, 3.0)
        elif cus == 50:
            ref_point = (50.0, 3.0)
        elif cus == 100:
            ref_point = (80.0, 3.0)
        else:
            raise NotImplementedError

        hvs = []
        for result in results:
            obj_nd = np.array(result['obj'])
            hv = hvwfg.wfg(obj_nd, np.array(ref_point))
            hv_ratio = hv / (ref_point[0] * ref_point[1])
            hvs.append(hv_ratio)
        print(sum(hvs) / len(hvs))

        print(reduce_results(results))
