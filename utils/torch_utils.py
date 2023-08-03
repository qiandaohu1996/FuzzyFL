from typing import List
import warnings
import torch
from torch import nn
from torch import no_grad
from copy import deepcopy
# import torch.nn as nn
from utils.my_profiler import calc_exec_time
import torch.multiprocessing as mp

calc_time = True


@no_grad()
def average_learners(
    learners, target_learner, weights=None, average_params=True, average_gradients=False
):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    if learner_id == 0:
                        target_state_dict[key].data = (
                            weights[learner_id] * state_dict[key].data
                        )
                    else:
                        target_state_dict[key].data += (
                            weights[learner_id] * state_dict[key].data
                        )

                if average_gradients:
                    if state_dict[key].grad is not None:
                        if learner_id == 0:
                            target_state_dict[key].grad.data = (
                                weights[learner_id] * state_dict[key].grad
                            )
                        else:
                            target_state_dict[key].grad.data += (
                                weights[learner_id] * state_dict[key].grad
                            )
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data


@no_grad()
# @calc_exec_time(calc_time=calc_time)
def fuzzy_average_cluster_model(
    client_models,
    cluster_models,
    membership_mat,  # use membership matrix instead of weights
    fuzzy_m,  # fuzzy parameter m
    clients_weights,
    average_params=True,
    average_gradients=False,
):
    clients_weights = clients_weights.to(membership_mat.device)
    # print("membership_mat",membership_mat[2:5])

    n_clusters = len(cluster_models)

    for cluster_id in range(n_clusters):
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            target_state_dict[key].data.zero_()
            if target_state_dict[key].data.dtype == torch.float32:
                for client_id, model in enumerate(client_models):
                    state_dict = model.state_dict(keep_vars=True)
                    membership_val = (
                        membership_mat[client_id][cluster_id]
                        * clients_weights[client_id]
                    ) ** fuzzy_m
                    target_state_dict[key].data += membership_val * state_dict[key].data.clone()
            else:
                for client_id, model in enumerate(client_models):
                    state_dict = model.state_dict()
                    target_state_dict[key].data += state_dict[key].data.clone()

    for cluster_id in range(n_clusters):
        total_membership_val = torch.sum(
            (membership_mat[:, cluster_id] * clients_weights) ** fuzzy_m
        )
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data /= total_membership_val


@no_grad()
# @calc_exec_time(calc_time=calc_time)
def fuzzy_average_cluster_model1(
    client_models,
    cluster_models,
    membership_mat,  # use membership matrix instead of weights
    fuzzy_m,  # fuzzy parameter m
    clients_weights,
    average_params=True,
    average_gradients=False,
):
    clients_weights = clients_weights.to(membership_mat.device)
    # print("membership_mat",membership_mat[2:5])

    n_clusters = len(cluster_models)
    client_state_dicts = [model.state_dict() for model in client_models]
    for cluster_id in range(n_clusters):
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)

        for key in target_state_dict:
            target_state_dict[key].data.zero_()
            client_key_data = torch.stack([state_dict[key].data.clone() for state_dict in client_state_dicts])
            print("client_key_data", client_key_data.shape)
            print("membership_mat[:, cluster_id]", membership_mat[:, cluster_id].shape)

            if target_state_dict[key].data.dtype == torch.float32:
                membership_vals = (membership_mat[:, cluster_id] * clients_weights) ** fuzzy_m
                print("membership_vals shape ", membership_vals)
                expanded_membership_mat = membership_vals
                target_state_dict[key].data = torch.matmul(expanded_membership_mat, client_key_data)
            else:
                target_state_dict[key].data = torch.sum(client_key_data, dim=0)

    for cluster_id in range(n_clusters):
        total_membership_val = torch.sum(
            (membership_mat[:, cluster_id] * clients_weights) ** fuzzy_m
        )
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data /= total_membership_val


@no_grad()
def fuzzy_average_cluster_model_single(cluster_id, client_state_dicts, cluster_state_dicts, membership_mat, fuzzy_m,
                                       clients_weights, results_queue):
    target_state_dict = cluster_state_dicts[cluster_id].copy()
    for key in target_state_dict:
        if target_state_dict[key].dtype == torch.float32:
            target_state_dict[key].fill_(0)
            for client_id, state_dict in enumerate(client_state_dicts):
                membership_val = (membership_mat[client_id][cluster_id] * clients_weights[client_id]) ** fuzzy_m
                target_state_dict[key].data += membership_val * state_dict[key].data
        else:
            for state_dict in client_state_dicts:
                target_state_dict[key].data += state_dict[key].data
    total_membership_val = torch.sum((membership_mat[:, cluster_id] * clients_weights) ** fuzzy_m)
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data /= total_membership_val
    results_queue.put((cluster_id, target_state_dict))


def fuzzy_average_cluster_model_parallel(client_models, cluster_models, membership_mat, fuzzy_m, clients_weights):
    clients_weights = clients_weights.to(membership_mat.device)
    n_clusters = len(cluster_models)
    processes = []
    results_queue = mp.Queue()

    client_state_dicts = [m.state_dict().detach() for m in client_models]
    cluster_state_dicts = [m.state_dict().detach() for m in cluster_models]
    for cluster_id in range(n_clusters):
        p = mp.Process(
            target=fuzzy_average_cluster_model_single,
            args=(cluster_id, client_state_dicts, cluster_state_dicts, membership_mat, fuzzy_m, clients_weights, results_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Load the updated state_dicts back into the models
    while not results_queue.empty():
        cluster_id, target_state_dict = results_queue.get()
        cluster_models[cluster_id].load_state_dict(target_state_dict)


@no_grad()
def fuzzy_average_client_model_parallel(
    cluster_models,
    client_models,
    membership_mat,
    average_params=True,
    average_gradients=False,
):
    n_clients = len(client_models)
    results_queue = mp.Queue()
    processes = []
    client_state_dicts = [m.state_dict().detach() for m in client_models]
    cluster_state_dicts = [m.state_dict().detach() for m in cluster_models]

    for client_id in range(n_clients):
        p = mp.Process(
            target=fuzzy_average_client_model_single,
            args=(client_id, cluster_state_dicts, client_state_dicts, membership_mat, results_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not results_queue.empty():
        client_id, target_state_dict = results_queue.get()
        client_models[client_id].load_state_dict(target_state_dict)


def fuzzy_average_client_model_single(
    client_id, cluster_state_dicts, client_state_dicts, membership_mat, results_queue
):
    target_state_dict = client_state_dicts[client_id].clone()
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.zero_()
            for cluster_id, state_dict in enumerate(cluster_state_dicts):
                membership_val = membership_mat[client_id][cluster_id]
                target_state_dict[key].data += membership_val * state_dict[key].detach().clone()
        else:
            target_state_dict[key].data.zero_()
            for state_dict in cluster_state_dicts:
                target_state_dict[key].data += state_dict[key].detach().clone()
    results_queue.put((client_id, target_state_dict))


@no_grad()
# @calc_exec_time(calc_time=calc_time)
def fuzzy_average_client_model(
    cluster_models, client_models, membership_mat, average_params=True, average_gradients=False
):

    n_clients = len(client_models)
    for client_id in range(n_clients):
        target_state_dict = client_models[client_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data.zero_()
                for cluster_id, cluster_model in enumerate(cluster_models):
                    state_dict = cluster_model.state_dict(keep_vars=True)
                    membership_val = membership_mat[client_id][cluster_id]
                    target_state_dict[key].data += membership_val * state_dict[key].data
            else:
                target_state_dict[key].data.zero_()
                for model in cluster_models:
                    state_dict = model.state_dict()
                    target_state_dict[key].data += state_dict[key].data


@no_grad()
def fuzzy_average_client_model_vec(
    cluster_models, client_models, membership_mat, average_params=True, average_gradients=False
):

    n_clients = len(client_models)
    n_clusters = len(cluster_models)

    # Convert membership matrix into tensor
    # membership_tensor = torch.Tensor(membership_mat)  # shape: [n_clients, n_clusters]

    # Iterate over each key in the state dict
    for key in client_models[0].state_dict():
        averaged_weights = torch.empty([n_clients] + list(cluster_models[0].state_dict()[key].shape),
                                        device = membership_mat.device)
        # print(type(client_models[0]))
        # print(type(client_models))
        # print(type(client_models.state_dict(keep_vars=True)))
        for client_id in range(n_clients):
            cluster_tensors = torch.stack([cluster.state_dict()[key] for cluster in cluster_models])
            if cluster_tensors.dtype == torch.float32:
                membership_tensor = membership_mat[client_id].view(n_clusters, *([1] * len(cluster_tensors.shape[1:])))
                averaged_weights[client_id] = (membership_tensor * cluster_tensors).sum(dim=0)
        for client_id, client_model in enumerate(client_models):
            client_state_dict = client_model.state_dict(keep_vars=True)
            if client_state_dict[key].data.dtype == torch.float32:
                client_state_dict[key].data = averaged_weights[client_id]
                client_model.load_state_dict(client_state_dict)


@no_grad()
@calc_exec_time(calc_time=calc_time)
def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]
    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data = (1 - alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data
            # print(f"key: {key}, target_state_dict[key].data: {target_state_dict[key].data},
            # source_state_dict[key].data: {source_state_dict[key].data}")


@no_grad()
def apfl_partial_average(personal_learner, global_learner, alpha):
    source_state_dict = global_learner.model.state_dict()
    target_state_dict = personal_learner.model.state_dict()

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data = (alpha * target_state_dict[key].data+ (1 - alpha) * source_state_dict[key].data)


@no_grad()
def agfl_partial_average(personal_learner, cluster_learners, alpha):
    target_state_dict = personal_learner.model.state_dict()
    source_state_dicts = [learner.model.state_dict() for learner in cluster_learners]
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.zero_()
            for cluster_id, source_state_dict in enumerate(source_state_dicts):
                target_state_dict[key].data += alpha[cluster_id + 1] * source_state_dict[key].data
                


def differentiate_learner(target, reference_state_dict, coeff=1.0):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)
    with no_grad:
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].grad = coeff * (
                    target_state_dict[key].data.clone()
                    - reference_state_dict[key].data.clone()
                )


def init_nn(nn, val=0.0):
    for para in nn.parameters():
        para.data.fill_(val)
    return nn


def get_param_tensor(model):
    """
    get `model` parameters as a unique flattened tensor
    :return: torch.tensor

    """
    return torch.cat([param.data.view(-1) for param in model.parameters() if param.requires_grad])


def get_param_list(models):
    """
    get `models` parameters as a unique flattened tensor
    :return: torch.tensor
    """
    param_list = torch.stack([torch.cat([param.flatten() for param in model.parameters() if param.requires_grad])for model in models])

    return param_list


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def compute_model_diff(model1: nn.Module, model2: nn.Module):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    diff_state_dict = {}

    for param_name in state_dict1:
        diff_state_dict[param_name] = state_dict1[param_name] - state_dict2[param_name]
    diff_model = deepcopy(model1)
    diff_model.load_state_dict(diff_state_dict)

    return diff_model


def compute_model_distances_FemnistCNN(model1: nn.Module, models: List[nn.Module]):
    # Get the device of the first model
    device = next(model1.parameters()).device

    # Ensure the model is on the correct device
    distances = []
    total_norm = 0.0
    for model2 in models:
        # Ensure the model is on the correct device
        model = compute_model_diff(model1 , model2)
        params = list(model.parameters())
        if params:
            param_vector = torch.cat([p.view(-1) for p in params])  # 展平为向量
            norm = param_vector.norm().item()  # 计算范数
            param_count = param_vector.numel()  # 计算参数数量
            norm_per_param = norm / param_count if param_count != 0 else 0  # 防止除以0
            # print(f"features. Norm : {norm_per_param}")
            total_norm += norm_per_param

        distances.append(total_norm)

    return torch.tensor(distances, device=device)


def compute_model_distances_mobileNetV2(model1: nn.Module, models: List[nn.Module]):

    device = next(model1.parameters()).device
    distances = []
    for model2 in models:
        model = compute_model_diff(model1 , model2)
        features_count = len(model.features)
        total_norm = 0
        for i in range(features_count):
            module = model.features[i]
            params = list(module.parameters())
            if params:  # 检查参数是否存在
                param_vector = torch.cat([p.view(-1) for p in params])  # 展平为向量
                norm = param_vector.norm().item()  # 计算范数
                param_count = param_vector.numel()  # 计算参数数量
                norm_per_param = norm / param_count if param_count != 0 else 0  # 防止除以0
                # print(f"features.{i} Norm per Parameter: {norm_per_param}")
                total_norm += norm_per_param

        module = model.classifier[1]
        params = list(module.parameters())
        param_vector = torch.cat([p.view(-1) for p in params])  # 展平为向量
        norm = param_vector.norm().item()  # 计算范数
        param_count = param_vector.numel()  # 计算参数数量
        norm_per_param = norm / param_count
        # print(f"classifier.1 Norm per Parameter: {norm_per_param}")
        total_norm += norm_per_param

        distances.append(total_norm)
    return torch.tensor(distances, device=device)


def simplex_projection(v, s=1):
    r"""
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, r"Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = -float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w

    # def trainable_params(src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
    # ) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    #     parameters = []
    #     keys = []
    #     if isinstance(src, OrderedDict):
    #         for name, param in src.items():
    #             if param.requires_grad:
    #                 parameters.append(param)
    #                 keys.append(name)
    #     elif isinstance(src, torch.nn.Module):
    #         for name, param in src.state_dict(keep_vars=True).items():
    #             if param.requires_grad:
    #                 parameters.append(param)
    #                 keys.append(name)

    #     if requires_name:
    #         return keys, parameters
    #     else:
    #         return parameters
