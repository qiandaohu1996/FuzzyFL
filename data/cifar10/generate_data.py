"""
Download CIFAR-10 dataset, and splits it among clients
"""
import os
import argparse
import pickle

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset

from sklearn.model_selection import train_test_split

from utils import split_dataset_by_labels, pathological_non_iid_split


ALPHA = .4
N_CLASSES = 10
N_COMPONENTS = 3
SEED = 12345
RAW_DATA_PATH = "raw_data/"
<<<<<<< HEAD
PATH = "all_data/"
=======
PATH = "all_data0/"
>>>>>>> 4088e3c (.)


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks', help='number of tasks/clients;', type=int, required=True)
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
        '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
        'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
        'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components', help='number of components/clusters;', type=int, default=N_COMPONENTS)
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;',
        type=float,
        default=ALPHA
    )
    parser.add_argument(
        '--s_frac', help='fraction of the dataset to be used; default: 1.0;', type=float, default=1.0)
    parser.add_argument(
        '--tr_frac', help='fraction in training set; default: 0.8;', type=float, default=0.8)
    parser.add_argument(
        '--val_frac', help='fraction of validation set (from train set); default: 0.0;', type=float, default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed', help='seed for the random processes;', type=int, default=SEED)

    return parser.parse_args()


def count_labels(dataset, clients_indices):
    clients_label_counts = []

    for client_indices in clients_indices:
        # Initialize a label count dictionary for each client
        # assuming CIFAR10
        label_counts = {i: 0 for i in range(10)}

        for idx in client_indices:
            label = dataset[idx][1]
            label_counts[label] += 1

        clients_label_counts.append(label_counts)

    return clients_label_counts


def clients_label_count():
    transform = Compose([ToTensor(), Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset =\
        ConcatDataset([
            CIFAR10(root=RAW_DATA_PATH, download=True,
                    train=True, transform=transform),
            CIFAR10(root=RAW_DATA_PATH, download=False,
                    train=False, transform=transform)
        ])

    n_tasks = 20
    n_components = 3
    alpha = 0.4
    s_frac = 0.2
    clients_indices = \
        split_dataset_by_labels(
            dataset=dataset,
            n_classes=N_CLASSES,
            n_clients=n_tasks,
            n_clusters=n_components,
            alpha=alpha,
            frac=s_frac,
            seed=1234
        )

    # print(" clients_indices ", clients_indices)
    clients_label_counts = count_labels(dataset, clients_indices)
    # for client_id, label_counts in enumerate(clients_label_counts):
    #     print(f"Client {client_id} label counts: {label_counts}")

    return clients_label_counts


def main():
    args = parse_args()

    transform = Compose([ToTensor(), Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset =\
        ConcatDataset([
            CIFAR10(root=RAW_DATA_PATH, download=True,
                    train=True, transform=transform),
            CIFAR10(root=RAW_DATA_PATH, download=False,
                    train=False, transform=transform)
        ])

    if args.pathological_split:
        clients_indices =\
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed
            )

    count_labels(dataset, clients_indices)

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(
                clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
        # print("train_clients_indices  ", train_clients_indices)
        # print("test_clients_indices  ", test_clients_indices)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []
    print(PATH)
    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            if len(indices) == 0:
                continue

            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)
            # print(f"{mode}, task_{client_id}\n")
            train_indices, test_indices =\
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )
            # print("train_indices \n", train_indices)
            # print("test_indices \n", test_indices)
            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1. - args.val_frac,
                        random_state=args.seed
                    )
                # print("val_indices\n ", val_indices)
                save_data(val_indices, os.path.join(client_path, "val.pkl"))

            # print("train_indices \n", train_indices)
            # print("test_indices \n", test_indices)

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))


if __name__ == "__main__":
    main()
