import random
import time
import sys
import numpy as np


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i:group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index:index])
        current_index = index

    return res


def split_dataset_by_labels(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    
    
    # 打开文件以写入模式，如果文件不存在则创建
    log_file = open('mylog88.txt', 'w')

    # 将标准输出流重定向到文件
    sys.stdout = log_file
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    print("all_labels  ", all_labels)
    rng.shuffle(all_labels)
    print("all_labels  ", all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)
    print("clusters_labels")
    print(clusters_labels)

    label2cluster = {}                    # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    print("label2cluster")
    print(label2cluster)
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)
    # print("clusters")
    # print(clusters)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)                    # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        print("weights ",weights)
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)
    # Check if any client has zero samples
    print("clients_counts")
    print(clients_counts)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    poor_clients = np.where(np.array([len(indices) for indices in clients_indices]) < 10)[0]
    print("len(poor_clients)")
    print(len(poor_clients))
    rich_clients = np.where(np.array([len(indices) for indices in clients_indices]) > 200)[0]
    print("len(rich_clients)")
    print(len(rich_clients))
    samples_to_transfer = 5  # or any other number you want

    # Transfer samples from rich to poor clients
    for poor_client in poor_clients:
        while len(clients_indices[poor_client]) < 10:  # or any other threshold you want
            for rich_client in rich_clients:
                if len(clients_indices[rich_client]) > 200 + samples_to_transfer:  # keep the rich client rich
                    for _ in range(samples_to_transfer):
                        clients_indices[poor_client].append(clients_indices[rich_client].pop())
                    break
                
    clients_cluster_counts = np.zeros((n_clients, n_clusters), dtype=int)
    for client_id, indices in enumerate(clients_indices):
        for idx in indices:
            _, label = dataset[idx]
            cluster_id = label2cluster[label]
            clients_cluster_counts[client_id, cluster_id] += 1
    for i in range(len(clients_indices)):
            print(f"len(client[{i}]):", len(clients_indices[i]))
    # Print cluster counts for each client
    for client_id in range(n_clients):
        print(f"Client {client_id} has samples in clusters: {clients_cluster_counts[client_id]}")
    log_file.close()
        
    return clients_indices

def pathological_non_iid_split2(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices

 
def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    Split classification dataset among `n_clients` in a pathological non-i.i.d manner.
    :param dataset: Dataset to be split.
    :param n_classes: Number of classes in `dataset`.
    :param n_clients: Number of clients.
    :param n_classes_per_client: Number of classes per client.
    :param frac: Fraction of dataset to use.
    :param seed: Random seed.
    :return: List of subgroups, each subgroup is a list of indices.
    """
    log_file = open('pathological.txt', 'w')

    # 将标准输出流重定向到文件
    sys.stdout = log_file
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # Get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    rng.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    # New part: Count label occurrences for each client
    clients_label_counts = [{k: 0 for k in range(n_classes)} for _ in range(n_clients)]
    for client_id, indices in enumerate(clients_indices):
        for idx in indices:
            _, label = dataset[idx]
            clients_label_counts[client_id][label] += 1
    for client_label_counts in clients_label_counts:
        labels_to_remove = [label for label, count in client_label_counts.items() if count == 0]
        for label in labels_to_remove:
            del client_label_counts[label]
    # Print label counts for each client
    # for client_id in range(n_clients):
    #     print(f" {client_id} : {clients_label_counts[client_id]}")
    # log_file.close()

    return clients_indices

# Helper functions like iid_divide need to be defined
