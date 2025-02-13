from typing import Dict, Tuple

from .solver import Solver
from .utils import *

from .pomo_tsp_solver import TspSolver
from .pomo_cvrp_solver import CvrpSolver

pretrained_model: Dict[str, Tuple[str, int]] = {
    'tsp20': ('saved_tsp20_model', 510),
    'tsp50': ('saved_tsp50_model', 1000),
    'tsp75': ('saved_my_tsp75_model', 1500),
    'tsp100': ('saved_tsp100_model2_longTrain', 3100),
    'cvrp20': ('saved_my_CVRP20_model', 510),
    'cvrp50': ('saved_my_CVRP50_model', 1000),
    'cvrp75': ('saved_my_CVRP75_model', 1500),
    'cvrp100': ('saved_CVRP100_model', 30500),
}
