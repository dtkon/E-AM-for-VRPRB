import os
import pprint
import json
import random
import torch

from options import get_options
from RL.agent import Agent


def run():
    option = get_options()
    pprint.pprint(vars(option))

    # Set the random seed to initialize neural networks
    torch.manual_seed(option.seed)
    random.seed(option.seed)

    if not option.no_save:
        os.makedirs(option.save_dir, exist_ok=True)
        # Save arguments so exact configuration can always be found
        with open(os.path.join(option.save_dir, "option.json"), 'w') as f:
            json.dump(vars(option), f, indent=True)

    option.first_zoom_ratio = float('nan')

    # Set the device, not JSON serializable
    option.device = torch.device('cuda:0' if option.use_cuda else 'cpu')

    agent = Agent(option)

    if option.eval_only:
        agent.start_eval()
    else:
        agent.start_train()


if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run()
