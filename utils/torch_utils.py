from typing import List
import warnings
import torch
from torch import nn
from torch import no_grad
from copy import deepcopy
from .constants import *
from torchvision.models import mobilenet 


from .models import *

# import torch.nn as nn
from utils.my_profiler import calc_exec_time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

calc_time = True
seed=666
torch.manual_seed(seed)

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
        weights = (1 / n_learners) * \
            torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    if learner_id == 0:
                        target_state_dict[key].data =  weights[learner_id] * state_dict[key].data
                    else:
                        target_state_dict[key].data.add_(weights[learner_id] * state_dict[key].data)

                if average_gradients:
                    if state_dict[key].grad is not None:
                        if learner_id == 0:
                            target_state_dict[key].grad.data = weights[learner_id] * state_dict[key].grad
                        else:
                            target_state_dict[key].grad.data.add_(weights[learner_id] * state_dict[key].grad)
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
                target_state_dict[key].data.add_(state_dict[key].data)


@no_grad()
# @calc_exec_time(calc_time=calc_time)
def fuzzy_average_cluster_model1(
    client_models,
    cluster_models,
    membership_mat,  # use membership matrix instead of weights
    fuzzy_m,  # fuzzy parameter m
    clients_weights,
    top,
    average_params=True,
    average_gradients=False,
):
    clients_weights = clients_weights.to(membership_mat.device)
    # print("membership_mat",membership_mat[2:5])
    n_clients = len(client_models)
    n_clusters = len(cluster_models)
    topk_indices=torch.zeros(n_clients,top) 
    if n_clusters==top:
        new_membership_mat=membership_mat
    else:
        new_membership_mat, topk_indices = select_top_k_cluster(
        membership_mat, top)
        # print("topk_indices ", topk_indices[:5])
    for cluster_id in range(n_clusters):
        target_state_dict = cluster_models[cluster_id].state_dict(
            keep_vars=True)
        for key in target_state_dict:
            target_state_dict[key].data.zero_()
            if target_state_dict[key].data.dtype == torch.float32:
                for client_id, model in enumerate(client_models):
                    if n_clusters==top or cluster_id in topk_indices[client_id]:
                    # if cluster_id in topk_indices[client_id]:
                        state_dict = model.state_dict(keep_vars=True)
                        membership_val = (
                            new_membership_mat[client_id][cluster_id]
                            * clients_weights[client_id]
                        ) ** fuzzy_m
                        target_state_dict[key].data.add_(
                            membership_val * state_dict[key].data
                        )
            else:
                for client_id, model in enumerate(client_models):
                    if n_clusters==top or cluster_id in topk_indices[client_id]:
                    # if cluster_id in topk_indices[client_id]:
                        state_dict = client_models[client_id].state_dict()
                        target_state_dict[key].data.add_(state_dict[key].data)

    for cluster_id in range(n_clusters):
        total_membership_val = torch.sum(
            (new_membership_mat[:, cluster_id] * clients_weights) ** fuzzy_m
        )
        # print("cluster ",cluster_id )
        if total_membership_val==0:
            torch.manual_seed(666) 
            if isinstance(cluster_models[cluster_id], FemnistCNN):
                print("total_membership_val=", total_membership_val)
                cluster_models[cluster_id] = FemnistCNN(num_classes=62).to(membership_mat.device)
            # if isinstance(cluster_models[cluster_id], mobilenet.MobileNetV2):
            #     weights = mobilenet.MobileNet_V2_Weights.DEFAULT
            #     cluster_models[cluster_id] = models.mobilenet_v2(weights=weights).to(membership_mat.device)
            #     cluster_models[cluster_id].classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes=10)
            continue
        target_state_dict = cluster_models[cluster_id].state_dict(
            keep_vars=True)
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data /= total_membership_val


@no_grad()
def fuzzy_average_cluster_model(
    client_models,
    cluster_models,
    membership_mat,  # use membership matrix instead of weights
    fuzzy_m,  # fuzzy parameter m
    clients_weights,
    top,
    average_params=True,
    average_gradients=False,
):

    clients_weights = clients_weights.to(membership_mat.device)
    n_clients = len(client_models)
    n_clusters = len(cluster_models)
    topk_indices = torch.zeros(n_clients, top) 

    if n_clusters == top:
        new_membership_mat = membership_mat
    else:
        new_membership_mat, topk_indices = select_top_k_cluster(membership_mat, top)

    def cluster_thread_logic(cluster_id):
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            target_state_dict[key].data.zero_()
            if target_state_dict[key].data.dtype == torch.float32:
                for client_id, model in enumerate(client_models):
                    if n_clusters == top or cluster_id in topk_indices[client_id]:
                        state_dict = model.state_dict(keep_vars=True)
                        membership_val = (
                            new_membership_mat[client_id][cluster_id]
                            * clients_weights[client_id]
                        ) ** fuzzy_m
                        target_state_dict[key].data.add_(
                            membership_val * state_dict[key].data
                        )
            else:
                for client_id, model in enumerate(client_models):
                    if n_clusters == top or cluster_id in topk_indices[client_id]:
                        state_dict = client_models[client_id].state_dict()
                        target_state_dict[key].data.add_(state_dict[key].data)

        total_membership_val = torch.sum(
            (new_membership_mat[:, cluster_id] * clients_weights) ** fuzzy_m
        )

        if total_membership_val == 0:
            torch.manual_seed(666) 
            if isinstance(cluster_models[cluster_id], FemnistCNN):
                print("total_membership_val=", total_membership_val)
                cluster_models[cluster_id] = FemnistCNN(num_classes=62).to(membership_mat.device)
        else:
            for key in target_state_dict:
                if target_state_dict[key].data.dtype == torch.float32:
                    target_state_dict[key].data /= total_membership_val

    max_threads = 20  # Adjust as needed

    # Use ThreadPoolExecutor to handle parallel processing
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(cluster_thread_logic, range(n_clusters))



@no_grad()
@calc_exec_time(calc_time=calc_time)
def fuzzy_average_client_model1(
    cluster_models,
    client_models,
    membership_mat,
    top,
    average_params=True,
    average_gradients=False,
):
    n_clients = len(client_models)
    n_clusters = len(cluster_models)
    if n_clusters==top:
        new_membership_mat=membership_mat
    else:
        new_membership_mat, topk_indices = select_top_k_cluster(
        membership_mat, top)

    for client_id in range(n_clients):
        target_state_dict = client_models[client_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            target_state_dict[key].data.zero_()
            if target_state_dict[key].data.dtype == torch.float32:
                if n_clusters == top:
                    iter_clusters = range(n_clusters)
                else:
                    iter_clusters = topk_indices[client_id]

                for cluster_id in iter_clusters:
                    state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
                    membership_val = new_membership_mat[client_id][cluster_id]
                    target_state_dict[key].data.add_(
                        membership_val * state_dict[key].data
                    )

            else:
                for model in cluster_models:
                    state_dict = model.state_dict()
                    target_state_dict[key].data.add_(state_dict[key].data)


@no_grad()
@calc_exec_time(calc_time=calc_time)
def fuzzy_average_client_model(
    cluster_models,
    client_models,
    membership_mat,
    top,
    average_params=True,
    average_gradients=False,
):

    n_clients = len(client_models)
    n_clusters = len(cluster_models)
    
    if n_clusters == top:
        new_membership_mat = membership_mat
    else:
        new_membership_mat, topk_indices = select_top_k_cluster(membership_mat, top)

    def client_thread_logic(client_id):
        target_state_dict = client_models[client_id].state_dict(keep_vars=True)
        for key in target_state_dict:
            target_state_dict[key].data.zero_()
            if target_state_dict[key].data.dtype == torch.float32:
                if n_clusters == top:
                    iter_clusters = range(n_clusters)
                else:
                    iter_clusters = topk_indices[client_id]

                for cluster_id in iter_clusters:
                    state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
                    membership_val = new_membership_mat[client_id][cluster_id]
                    target_state_dict[key].data.add_(
                        membership_val * state_dict[key].data
                    )

            else:
                for model in cluster_models:
                    state_dict = model.state_dict()
                    target_state_dict[key].data.add_(state_dict[key].data)

    max_threads = 20  # Adjust as needed

    # Use ThreadPoolExecutor to handle parallel processing
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(client_thread_logic, range(n_clients))


def select_top_k_cluster(membership_mat, k):
    # 获取每一行的前k个最大值及其索引
    topk_values, topk_indices = torch.topk(membership_mat, k, dim=1)

    new_membership_mat = torch.zeros_like(membership_mat)

    # 使用 scatter_ 方法填充这些最大值
    # row_indices = torch.arange(membership_mat.size(0)).unsqueeze(-1).to(topk_indices.device)
    new_membership_mat.scatter_(1, topk_indices, topk_values)
    # print("before new_membership_mat", new_membership_mat[:5])
    # 确保没有0值的行，避免NaN值出现
    row_sums = new_membership_mat.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] += 1e8

    # 归一化
    new_membership_mat /= row_sums
    # print("after new_membership_mat", new_membership_mat[:5])

    return new_membership_mat, topk_indices


def average_models(models, target_model, weights=None):
    """
    Compute the average of a list of models and store it into target_model.

    :param models: List of models to be averaged.
    :type models: List[nn.Module]
    :param target_model: Model where the averaged parameters will be stored.
    :type target_model: nn.Module
    :param weights: tensor of the same size as models, having values between 0 and 1, and summing to 1.
                    If None, uniform weights are used.
    :type weights: torch.Tensor

    """
    
    if weights is None:
        n_models = len(models)
        weights = (1 / n_models) * torch.ones(n_models, device=next(target_model.parameters()).device)
    else:
        weights = weights.to(next(target_model.parameters()).device)

    target_state_dict = target_model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for model_id, model in enumerate(models):
                state_dict = model.state_dict(keep_vars=True)

                if model_id == 0:
                    target_state_dict[key].data = weights[model_id] * state_dict[key].data
                else:
                    target_state_dict[key].data.add_(weights[model_id] * state_dict[key].data)
        else:
            # handle non-float parameters (e.g., batch normalization parameters)
            target_state_dict[key].data.fill_(0)
            for model_id, model in enumerate(models):
                state_dict = model.state_dict()
                target_state_dict[key].data.add_(state_dict[key].data)
 

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
                target_state_dict[key].data = (1 - alpha) * target_state_dict[
                    key
                ].data + alpha * source_state_dict[key].data
            # print(f"key: {key}, target_state_dict[key].data: {target_state_dict[key].data},
            # source_state_dict[key].data: {source_state_dict[key].data}")


@no_grad()
def apfl_partial_average(personal_learner, global_learner, alpha):
    source_state_dict = global_learner.model.state_dict()
    target_state_dict = personal_learner.model.state_dict()

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data = (
                alpha * target_state_dict[key].data
                + (1 - alpha) * source_state_dict[key].data
            )


@no_grad()
def agfl_partial_average(personal_learner, cluster_learners, alpha):
    target_state_dict = personal_learner.model.state_dict()
    source_state_dicts = [learner.model.state_dict()
                          for learner in cluster_learners]
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.zero_()
            for cluster_id, source_state_dict in enumerate(source_state_dicts):
                target_state_dict[key].data += (
                    alpha[cluster_id + 1] * source_state_dict[key].data
                )


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
    return torch.cat(
        [param.data.view(-1)
         for param in model.parameters() if param.requires_grad]
    )


def get_param_list(models):
    """
    get `models` parameters as a unique flattened tensor
    :return: torch.tensor
    """
    param_list = torch.stack(
        [
            torch.cat(
                [param.flatten()
                 for param in model.parameters() if param.requires_grad]
            )
            for model in models
        ]
    )

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
        diff_state_dict[param_name] = state_dict1[param_name] - \
            state_dict2[param_name]
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
        model = compute_model_diff(model1, model2)
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
        model = compute_model_diff(model1, model2)
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

