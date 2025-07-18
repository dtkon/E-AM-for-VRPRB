# PDP-NCS

This repo implements our paper: Deep Reinforcement Learning for the Vehicle Routing Problem with Route Balancing

## Dependencies
* Python>=3.8
* PyTorch>=1.7
* hvwfg
* geatpy
* tqdm

## Usage

### Training

20 nodes:
```bash
CUDA_VISIBLE_DEVICES=0 python ./src/main.py --problem mo_cvrp --customer_size 20 --batch_size 512
```

50 nodes:
```bash
CUDA_VISIBLE_DEVICES=1 python ./src/main.py --problem mo_cvrp --customer_size 50 --batch_size 256
```

100 nodes:
```bash
CUDA_VISIBLE_DEVICES=0,1 python ./src/main.py --problem mo_cvrp --customer_size 100 --batch_size 128
```

Large scale:
```bash
python ./src/main.py --problem mo_cvrp_re_tsp --customer_size 100 --batch_size 90 --epoch_size 128070
```

### Inference

```bash
--eval_only 
--load_path '{add model to load here}'
--val_range 0 200 
--val_batch_size 200
--val_dataset '{add dataset here}' 
--eval_type pareto
--sample_times 128
```

#### Examples

For inference 200 VRPRB instances with 100 customers and 11 vehicles:

```bash
CUDA_VISIBLE_DEVICES=0 python ./src/main.py --eval_only --no_log --problem mo_cvrp --customer_size 100 --vehicle_num 11 --val_range 0 200 --val_batch_size 100 --model_save_dir "../drl-result" --load_path "/root/saved_model/mo_cvrp_100-v11/20240726T181058/epoch-199.pt" --eval_type pareto_sample --sample_times 32
```

For inference 20 VRPRB instances with 1000 customers and 40 vehicles:

```bash
python ./src/main.py --eval_only --no_log --problem mo_cvrp_re_tsp --customer_size 1000 --vehicle_num 40 --val_range 0 20 --val_batch_size 4 --eval_type pareto --sample_times 0 --load_path "/root/saved_model/mo_cvrp_re_tsp_100-v11/20241225T022930/epoch-110.pt" --max_parallel 2
```

Run ```python ./src/main.py -h``` for detailed help on the meaning of each argument.

Datasets for validation and pre-trained model can be found in [release](https://github.com/dtkon/E-AM-for-VRPRB/releases) of this repos.
