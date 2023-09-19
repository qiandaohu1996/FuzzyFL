from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch import no_grad

from client import Client
from utils.torch_utils import *
from utils.constants import *
from torchvision.models import MobileNetV2

from utils.models import FemnistCNN
from utils.datastore import *


class FuzzyClient(Client):
    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        idx,
        trans,
        initial_fuzzy_m,
        fuzzy_m_scheduler,
        logger,
        local_steps,
        tune_locally=False,
    ):
        super(FuzzyClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )

        assert self.n_learners == 1, "FuzzyClient only supports single learner."
        # self.only_with_global = False
        self.fuzzy_m = initial_fuzzy_m
        # utils.py line 484
        self.fuzzy_m_scheduler = fuzzy_m_scheduler
        self.trans = trans
        self.previous_membership_vec = None

    def init_membership_vector(self, n_clusters):
        membership_vector = torch.rand(n_clusters,)
        membership_vector = membership_vector / membership_vector.sum(dim=0, keepdim=True)
        return membership_vector

    @no_grad()
    # @calc_exec_time(calc_time=True)
    def update_membership_euclid(
        self,
        membership_mat,
        cluster_learners,
        client_learner,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum=None,
    ):
        # print("client_id ", client_id)
        # print("self ", self)
        n_clusters = len(cluster_learners)
        device = membership_mat.device

        eps = torch.tensor(1e-8, device=device)
        clamp_max = torch.tensor(1e8, device=device)
        p = float(2 / (fuzzy_m - 1))
        distances = torch.zeros(n_clusters, device=device)
        cluster_models = [learner.model for learner in cluster_learners]
        client_model = client_learner.model
        # print("client_model.shape ",client_model.parameters().data.shape)
        client_params = get_param_tensor(client_model).unsqueeze(0)  # shape: [1, dim]
        # print("client_params.shape ",client_params.shape)
        cluster_params = get_param_list(cluster_models)
        # print("cluster_params.shape ",cluster_params.shape)
        
        # self.torch_display(f"client_params[190:200] ", client_params[0, 190:200])
        # self.torch_display(f"client_params[490:500] ", client_params[490:500])
        for i in range(n_clusters):
            diff = client_params - cluster_params[i]
            distances[i] = torch.norm(diff)
            # print("cluster ",i)
            # print("cluster_model.shape ",cluster_models[i].parameters().data.shape)
            # self.torch_display("cluster_params[190:200] ",  cluster_params[i][190:200])
            # self.torch_display("cluster params[490:500] ",  cluster_params[i][490:500])
            
        # self.torch_display(f"distances ", distances)
        distances = distances - self.trans * distances.min()
        distances = torch.clamp(distances, eps, clamp_max)
        if client_id in [2,10]:
            self.torch_display(f"distances after -{self.trans}min  ", distances)
        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)
     
        if self.previous_membership_vec is not None:
            membership_mat[client_id, :] = self.previous_membership_vec * momentum + membership_mat[client_id, :] * (1 - momentum)
        self.previous_membership_vec = membership_mat[client_id].clone()
        # self.torch_display("membership_mat  ", membership_mat[client_id])
        
        torch.cuda.empty_cache()
        return membership_mat



    @no_grad()
    def update_membership_loss(
        self,
        membership_mat,
        cluster_learners,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum,
    ):
        # print("client ", client_id)
        # n_clusters = len(cluster_learners)
        device = membership_mat.device

        eps = torch.tensor(1e-10, device=device)
        clamp_max = torch.tensor(1e10, device=device)
        p = float(2 / (fuzzy_m - 1))

        all_losses = cluster_learners.gather_losses(self.val_iterator).to(device)
        # print("all_losses", all_losses[:, :2])
        mean_losses = all_losses.mean(dim=1)
        # self.torch_display("mean_losses ", mean_losses)
        distances = mean_losses - self.trans * mean_losses.min()
        # if client_id < 10:
        #     self.torch_display(f"mean_losses after -{self.trans}min  ", mean_losses)

        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)
        # print("mean_losses[j] > eps den ",den)
        # print("den sum ",den)
        # # self.torch_display("den ", den)

        if self.previous_membership_vec is not None:
            membership_mat[client_id] = self.previous_membership_vec * momentum + membership_mat[client_id] * (1 - momentum)
        # self.torch_display("membership_mat  ", membership_mat[client_id])
        
        self.previous_membership_vec = membership_mat[client_id]
        torch.cuda.empty_cache()
        return membership_mat

    @no_grad()
    def distances_to_membership(self, distances, p, eps, clamp_max):
        membership_vec = torch.zeros_like(distances)
        denom = (distances[:, None] / distances[None, :]) ** p
        denom = denom.sum(dim=1)
        denom = torch.clamp(denom, min=eps, max=clamp_max)
        membership_vec = 1 / denom
        return membership_vec

    @no_grad()
    def models_l2_distance(self, client_model, cluster_models):
        client_params = get_param_tensor(client_model)  # shape: [1, dim]
        cluster_params = get_param_list(cluster_models)  # shape: [k, dim]
        diffs = client_params.unsqueeze(0) - cluster_params
        distances = torch.norm(diffs, dim=1)
        return distances

    @no_grad()
    # @calc_exec_time(calc_time=True)
    def update_membership_level(
        self,
        membership_mat,
        cluster_learners,
        client_learner,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum=None,
    ):
        # print("client_id ", client_id)
        # print("self ", self)
        n_clusters = len(cluster_learners)
        device = membership_mat.device

        eps = torch.tensor(1e-10, device=device)
        clamp_max = torch.tensor(1e10, device=device)
        p = float(2 / (fuzzy_m - 1))
        cluster_models = [learner.model for learner in cluster_learners]
        client_model = client_learner.model
        distances = torch.zeros(n_clusters, device=device)

        if type(client_model) == FemnistCNN:
            distances = compute_model_distances_FemnistCNN(client_model, cluster_models)
        elif type(client_model) == MobileNetV2:
            distances = compute_model_distances_mobileNetV2(client_model, cluster_models)
        else:
            print("does not support the type of client_model")

        # self.torch_display("distances  ", distances)
        distances = distances - self.trans * distances.min()
        if client_id <10:
            self.torch_display(f"distances after -{self.trans}min  ", distances)

        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)

        if self.previous_membership_vec is not None:
            membership_mat[client_id] = self.previous_membership_vec * momentum + membership_mat[client_id] * (1 - momentum)

        self.previous_membership_vec = membership_mat[client_id].clone()

        return membership_mat

    @no_grad()
    def update_membership_grad(
        self,
        membership_mat,
        cluster_learners,
        client_learner,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum
    ):
        device = membership_mat.device
        eps = torch.tensor(1e-10, device=device)
        clamp_max = torch.tensor(1e10, device=device)
        p = float(2 / (fuzzy_m - 1))

        client_model = client_learner.model
        cluster_models = [learner.model for learner in cluster_learners]
        client_grad = torch.cat([param.grad.flatten() for param in client_learner.model.parameters() if param.grad is not None])
        # print(client_grad[:50])
        # Compute the parameter difference vector of the cluster models
        param_diffs = torch.stack([torch.cat([p1.data.flatten() - p2.data.flatten()
                                              for p1, p2 in zip(client_model.parameters(), cluster_model.parameters())])
                                   for cluster_model in cluster_models])
        # Compute the norm of the parameter difference vector for each cluster model
        grad_diffs = client_grad.unsqueeze(0) - param_diffs
        # print("grad_diffs ", grad_diffs.shape)
        distances = torch.tensor([torch.norm(grad_diff).item() for grad_diff in grad_diffs], device=device)
        # print("distances ", distances.shape)

        # Compute the cosine similarity between the gradient vector and the parameter difference vector
        # cos_similarities = F.cosine_similarity(client_grad.unsqueeze(0), param_diffs)
        distances = distances - self.trans * distances.min()
        # Convert cosine similarities to cosine distances
        # cos_distances = 1.0 - cos_similarities + eps
        # self.torch_display(f"distances - {self.trans} * distances.min ", distances)

        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)
        if self.previous_membership_vec is not None:
            membership_mat[client_id] = self.previous_membership_vec * momentum + membership_mat[client_id] * (1 - momentum)
        # self.torch_display("membership_mat  ", membership_mat[client_id])
        self.previous_membership_vec = deepcopy(membership_mat[client_id])
        return membership_mat

    @no_grad()
    def update_membership_graddot(self, membership_mat, cluster_learners, client_learner, client_id, global_fixed_m, fuzzy_m, momentum):

        device = membership_mat.device
        eps = torch.tensor(1e-10, device=device)
        clamp_max = torch.tensor(1e10, device=device)
        p = float(2 / (fuzzy_m - 1))

        client_model = client_learner.model
        cluster_models = [learner.model for learner in cluster_learners]

        client_grad = torch.cat([param.grad.flatten() for param in client_learner.model.parameters() if param.grad is not None])
        param_diffs = torch.stack([torch.cat([p1.data.flatten() - p2.data.flatten()
                                              for p1, p2 in zip(cluster_model.parameters(), client_model.parameters())])
                                   for cluster_model in cluster_models])
        # Compute the norm of the parameter difference vector for each cluster model
        distances = torch.matmul(client_grad.unsqueeze(0), param_diffs.t()).squeeze()
        # self.torch_display("distances", distances)

        distances = distances - self.trans * distances.min()

        self.torch_display(f"distances - {self.trans} * distances.min ", distances)
        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)
        if self.previous_membership_vec is not None:
            membership_mat[client_id] = self.previous_membership_vec * momentum + membership_mat[client_id] * (1 - momentum)
        # self.torch_display("membership_mat  ", membership_mat[client_id])
        self.previous_membership_vec = deepcopy(membership_mat[client_id])
        return membership_mat



    @no_grad()
    def update_membership_dot(
        self,
        membership_mat,
        cluster_models,
        client_model,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum,
    ):
        # print("client_id ", client_id)
        # print("self ", self)
        n_clusters = len(cluster_models)
        device = membership_mat.device

        eps = torch.tensor(1e-8, device=device)
        clamp_max = torch.tensor(1e10, device=device)
        p = float(2 / (fuzzy_m - 1))
        similarities = torch.zeros(n_clusters, device=device)

        cluster_params = get_param_list(cluster_models)
        client_param = get_param_tensor(client_model)

        similarities = torch.matmul(
            cluster_params, client_param.unsqueeze(-1)
        ).squeeze()
        print("similarities ", similarities)
        similarities = (similarities - self.trans * similarities.min()) + eps
        # self.torch_display(f"similarities after -{self.trans}min  ", similarities)

        for cluster_id in range(n_clusters):
            den = torch.zeros(1, device=device)
            for j in range(n_clusters):
                if similarities[j] > eps:
                    den += (similarities[cluster_id] / similarities[j]) ** p
                else:
                    den += 1 if cluster_id == j else 0
                den = torch.clamp(den, min=eps, max=clamp_max)
                membership_mat[cluster_id] = den
            membership_mat[client_id] = membership_mat[client_id] / membership_mat[client_id].sum()

        # self.torch_display("membership_mat  ", membership_mat[client_id])

        if self.previous_membership_vec is not None:
            membership_mat[
                client_id
            ] = self.previous_membership_vec * momentum + membership_mat[client_id] * (
                1 - momentum
            )
        # self.torch_display("membership_mat momentum  ", membership_mat[client_id])

        self.previous_membership_vec = (membership_mat[client_id]).clone()
        # self.torch_display("previous_membership_vec   ", self.previous_membership_vec)

        torch.cuda.empty_cache()

        return membership_mat

    def update_membership_softmax(
        self,
        membership_mat,
        cluster_params,
        client_params,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum=None,
    ):
        n_clusters = cluster_params.size(0)
        eps = 1e-10
        # if global_fixed_m is True:
        #     p = float(2 / (fuzzy_m - 1))
        # else:
        #     p = float(2 / (self.fuzzy_m - 1))
        distances = torch.zeros(n_clusters, device=client_params.device)

        # 计算每个客户端参数与所有聚类中心的距离
        diffs = client_params.unsqueeze(0) - cluster_params
        distances = torch.norm(diffs, dim=1)
        distances = (distances - self.trans * distances.min()) + eps

        # 计算 softmax 距离
        softmax_distances = torch.softmax(-distances, dim=0)

        membership_mat[client_id, :] = softmax_distances

        if self.previous_membership_vec is not None:
            membership_mat[client_id] = self.previous_membership_vec * momentum + membership_mat[client_id] * (1 - momentum)

        self.previous_membership_vec = deepcopy(membership_mat[client_id])

        torch.cuda.empty_cache()

        return membership_mat

    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    @no_grad()
    def update_membership_original(
        self,
        membership_mat,
        cluster_params,
        client_params,
        client_id,
        global_fixed_m,
        fuzzy_m,
    ):
        n_clusters = cluster_params.size(0)

        # self.fuzzy_m = np.ones((n_clusters,))/n_clusters
        eps = 1e-10
        if global_fixed_m is True:
            p = float(2 / (fuzzy_m - 1))
        else:
            p = float(2 / (self.fuzzy_m - 1))

        distances = torch.zeros(n_clusters, device=client_params.device)
        for i in range(n_clusters):
            diff = client_params - cluster_params[i]
            distances[i] = torch.norm(diff)

        # print("distances  ", [f"{d:.3f}" for d in distances])

        distances = (distances - self.trans * distances.min()) + eps
        # distances = (distances - self.trans * distances.min())/(distances.max()-  distances.min())
        # print(f"distances after -{self.trans}min ", [f"{d:.3f}" for d in distances])

        dens = []
        for cluster_id in range(n_clusters):
            den = 0.0
            for j in range(n_clusters):
                if cluster_id == j:
                    den += 1
                elif distances[j] < eps:
                    den
                else:
                    den += (distances[cluster_id] / distances[j]) ** p
                den = torch.clamp(den, max=1e8)

            dens.append(den)
            membership_mat[client_id, cluster_id] = 1.0 / den

        dens = torch.zeros(n_clusters, device="cuda:0")  # 初始化一个空张量

        for cluster_id in range(n_clusters):
            # 使用布尔掩码来区分距离大于eps的元素
            mask = distances > eps
            div = distances[cluster_id] / distances
            # 使用掩码将距离大于eps的元素替换为对应的计算值，否则为0
            div = torch.where(mask, div**p, torch.zeros_like(distances))
            # 对于距离等于0的元素，使用另一个布尔掩码来替换为1或0
            mask_zero = distances == 0
            div = torch.where(
                mask_zero,
                (cluster_id == torch.arange(n_clusters, device="cuda:0")).float(),
                div,
            )
            # 累加所有的值得到den
            den = div.sum()
            den = torch.clamp(den, max=1e8)

            dens[cluster_id] = den
            membership_mat[client_id, cluster_id] = 1.0 / den

        # formatted_dens = [f"{d:.3f}" for d in dens]
        dens = dens.cpu().tolist()
        rounded_dens = [round(d, 3) for d in dens]
        print("dens ", rounded_dens)

        # print("membership_mat client ", client_id, "  ", membership_mat[client_id])

        return membership_mat

    def torch_display(self, info, tensor):
        tensor = tensor.cpu().tolist()
        rounded_tensor = [round(d, 7) for d in tensor]
        print(info, rounded_tensor)

    @no_grad()
    def update_membership_cosine(
        self,
        membership_mat,
        cluster_models,
        client_model,
        client_id,
        global_fixed_m,
        fuzzy_m,
        momentum,
    ):
        # print("client_id ", client_id)
        n_clusters = len(cluster_models)

        device = membership_mat.device
        p = float(2 / (fuzzy_m - 1))
        eps = torch.tensor(1e-10, device=device)
        clamp_max = torch.tensor(1e10, device=device)

        distances = torch.zeros(n_clusters, device=device)

        cluster_params = get_param_list(cluster_models)
        client_params = get_param_tensor(client_model)

        distances = torch.zeros(n_clusters, device=device)

        cos_sim = F.cosine_similarity(client_params, cluster_params)
        distances = 1.0 - cos_sim
        # print("distances ", distances)

        distances = torch.clamp(distances, min=eps, max=clamp_max)

        distances = (distances - self.trans * distances.min()) + eps
        # self.torch_display(f"distances after -{self.trans}min  ", distances)

        for cluster_id in range(n_clusters):
            den = torch.zeros(1, device=device)
            for j in range(n_clusters):
                if cluster_id == j:
                    den += 1
                elif distances[j] > eps:
                    den += (distances[cluster_id] / distances[j]) ** p
                den = torch.clamp(den, min=eps, max=clamp_max)
                self.torch_display("den ", den)

            membership_mat[client_id, cluster_id] = 1.0 / den

        # self.torch_display("membership_mat  ", membership_mat[client_id])

        if self.previous_membership_vec is not None:
            membership_mat[
                client_id
            ] = self.previous_membership_vec * momentum + membership_mat[client_id] * (
                1 - momentum
            )
        # self.torch_display("membership_mat momentum ", membership_mat[client_id])

        torch.cuda.empty_cache()

        return membership_mat

    def update_global_membership_mat(
        self,
        membership_mat,
        cluster_params,
        client_params,
        client_id,
        global_fixed_m,
        fuzzy_m,
    ):
        n_clusters = cluster_params.size(0)
        comm_global_flag = False
        self.fuzzy_m = np.ones((n_clusters,)) / n_clusters
        # global_fixed_m=True
        if global_fixed_m is False:
            p = float(2 / (self.fuzzy_m - 1))
        else:
            p = float(2 / (fuzzy_m - 1))

        distances = torch.zeros(n_clusters, device=client_params.device)
        for i in range(n_clusters):
            diff = client_params - cluster_params[i]
            distances[i] = torch.norm(diff)

        # print("distances", distances)
        distances = distances - 0.5 * distances.min()
        # print("distances after min", distances)

        # membership_mat[client_id]=F.softmax(distances,dim=0).T

        for cluster_id in range(n_clusters):
            den = 0.0
            for j in range(n_clusters):
                den += (distances[cluster_id] / distances[j]) ** p
                if den > 1.0e6:
                    den = 1.0e6
            print("den ", den)
            membership_mat[client_id, cluster_id] = 1.0 / den
        if membership_mat[client_id, -1] > 0.95:
            comm_global_flag = True

        for cluster_id in range(n_clusters - 1):
            den = 0.0
            for j in range(n_clusters - 1):
                den += (distances[cluster_id] / distances[j]) ** p
                if den > 1.0e6:
                    den = 1.0e6
            print("den ", den)
            membership_mat[client_id, cluster_id] = 1.0 / den
        # print("membership_mat client ", client_id, "  ", membership_mat[client_id])

        return comm_global_flag
