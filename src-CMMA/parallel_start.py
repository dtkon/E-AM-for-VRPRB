from multiprocessing import Pool
import itertools
import os, time
from typing import List

from main import run
from option import get_option

# https://www.jianshu.com/p/2c44b66d9ff6

problems = [
    'datasets/cvrp20-v4.npy',
    'datasets/cvrp20-v5.npy',
    'datasets/cvrp50-v7.npy',
    'datasets/cvrp50-v9.npy',
    'datasets/cvrp100-v11.npy',
    'datasets/cvrp100-v14.npy',
]
problem_indexes = list(range(0, 200))

args = []
for p, i in itertools.product(problems, problem_indexes):
    if 'cvrp20-' in p:
        ref_point = ('30', '8')
    elif 'cvrp50-' in p:
        ref_point = ('50', '10')
    elif 'cvrp100-' in p:
        ref_point = ('80', '15')
    else:
        raise NotImplementedError

    if 'cvrp100-' in p:
        population_size = 50
    else:
        population_size = 100

    arg = [
        '--load_file',
        str(p),
        '--problem_index',
        str(i),
        '--ref_point',
        *ref_point,
        '--population_size',
        str(population_size),
    ]
    if i != 0:
        arg.append('--no_log')

    args.append(arg)


def work(args: List[str]) -> List[str]:
    try:
        run(get_option(args))
        return args
    except Exception as e:
        raise RuntimeError(args) from e


def success_handler(args: List[str]) -> None:
    if not os.path.isdir('result'):
        os.mkdir('result')
    with open('result/paral_log.txt', 'a') as f:
        f.write(f'[{time.strftime("%m/%d %H:%M:%S")}] {args} done.\n')


def error_handler(e: Exception) -> None:
    if not os.path.isdir('result'):
        os.mkdir('result')
    with open('result/paral_log_error.txt', 'a') as f:
        f.write(f'[{time.strftime("%m/%d %H:%M:%S")}]\n')
        f.write("-->{}<--".format(e.__cause__) + '\n\n')


def main():
    po = Pool(2)

    for arg in args:
        po.apply_async(
            work, (arg,), callback=success_handler, error_callback=error_handler
        )

    po.close()
    po.join()


if __name__ == '__main__':
    main()
