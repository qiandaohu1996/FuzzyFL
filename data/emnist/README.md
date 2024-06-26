# EMNIST Dataset

## Introduction

Split EMNIST dataset among `n_clients` as follows:
1.  classes are grouped into `n_components`
2.  for each group `c`, samples are partitioned across clients using dirichlet distribution

Inspired by the split in [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440)

## Instructions

### Base usage

For basic usage, `run generate_data.py` with a choice of the following arguments:

- ```--n_tasks```: number of tasks/clients, written as integer
- ```--alpha```: parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;
  default=``0.4``
- ```--n_components```: number of mixture components, written as integer; default=``-1``
- ```--s_frac```: fraction of the dataset to be used; default=``0.2``
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--val_frac```: fraction of validation set (from train set); default: ``0.0``
- ```--test_tasks_frac```: fraction of test tasks; default=``0.0``
- ```--seed``` : seed to be used before random sampling of data; default=``12345``

### Additional options

We als o provide some additional options to split the dataset

- ```--pathological_split```: if selected, the dataset will be split as in
  [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629);
  i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes.
- ```--n_shards```: number of shards given to each client/task;
  ignored if `--pathological_split` is not used;
  default=`2`
- ```--val_frac```: fraction of validation set (from train set); default=`0.0`

## Paper Experiments

In order to generate the data split for Table 2 (Full client participation), run

```
python generate_data.py \
    --n_tasks 100 \
    --n_components -1 \
    --alpha 0.4 \
    --s_frac 0.2 \
    --tr_frac 0.8 \
    --val_frac 0.2 \
    --seed 12345
```

In order to generate the data split for Table 3 (Unseen clients), run


emnist

python generate_data.py   --n_tasks 100   --n_components 3  --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2  --val_frac 0.2  --seed 12345

python generate_data.py   --n_tasks 100   --n_components 3  --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2  --val_frac 0.2  --seed 23456

python generate_data.py   --n_tasks 100   --n_components 3  --alpha 0.2   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2  --val_frac 0.2  --seed 23456

python generate_data.py   --n_tasks 50   --n_components 3  --alpha 0.8   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2  --val_frac 0.2  --seed 12345


emnist_c5

python generate_data.py   --n_tasks 25   --n_components 5  --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --n_components 5  --alpha 0.2   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --n_components 5  --alpha 0.6   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --n_components 5  --alpha 0.8   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345

emnist_c10

python generate_data.py   --n_tasks 150   --n_components 10  --alpha 0.1   --s_frac 0.15   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 150   --n_components 10  --alpha 0.2   --s_frac 0.15   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 150   --n_components 10  --alpha 0.3   --s_frac 0.15   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 150   --n_components 10  --alpha 0.4   --s_frac 0.15   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 150   --n_components 10  --alpha 0.5   --s_frac 0.15   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 150   --n_components 10  --alpha 0.6   --s_frac 0.15   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345

emnist_n200_c10

python generate_data.py   --n_tasks 200   --n_components 10  --alpha 0.3   --s_frac 0.2   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 23456




emnist_pathologic_cl5

python generate_data.py   --n_tasks 100   --pathological_split --n_shards 5 --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --pathological_split --n_shards 5 --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 23456

emnist_pathologic_cl10

python generate_data.py   --n_tasks 100   --pathological_split --n_shards 10 --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --pathological_split --n_shards 10 --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 23456

emnist_pathologic_cl20

python generate_data.py   --n_tasks 100   --pathological_split --n_shards 20 --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --pathological_split --n_shards 20 --alpha 0.4   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2 --val_frac 0.2  --seed 23456


emnist_alpha0.6

python generate_data.py   --n_tasks 100   --n_components 3  --alpha 0.6   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2  --val_frac 0.2  --seed 12345
python generate_data.py   --n_tasks 100   --n_components 3  --alpha 0.6   --s_frac 0.1   --tr_frac 0.8   --test_tasks_frac 0.2  --val_frac 0.2  --seed 23456


