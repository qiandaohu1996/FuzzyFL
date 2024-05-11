from torch import no_grad
from client.client import Client
from utils.torch_utils import *
from utils.constants import *


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
        self.fuzzy_m_scheduler = fuzzy_m_scheduler
        self.trans = trans
        self.previous_membership_vec = None

    def init_membership_vector(self, n_clusters):
        membership_vector = torch.rand(n_clusters,)
        membership_vector = membership_vector / membership_vector.sum(dim=0, keepdim=True)
        return membership_vector

    @no_grad()
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
        n_clusters = len(cluster_learners)
        device = membership_mat.device

        eps = torch.tensor(1e-8, device=device)
        clamp_max = torch.tensor(1e8, device=device)
        p = float(2 / (fuzzy_m - 1))
        distances = torch.zeros(n_clusters, device=device)
        cluster_models = [learner.model for learner in cluster_learners]
        client_model = client_learner.model
        client_params = get_param_tensor(client_model).unsqueeze(0)  # shape: [1, dim]
        cluster_params = get_param_list(cluster_models)
         
        for i in range(n_clusters):
            diff = client_params - cluster_params[i]
            distances[i] = torch.norm(diff)
        distances = distances - self.trans * distances.min()
        distances = torch.clamp(distances, eps, clamp_max)
        # if client_id in [2]:
        #     self.torch_display(f"distances after -{self.trans}min  ", distances)
        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)
     
        if self.previous_membership_vec is not None:
            membership_mat[client_id, :] = self.previous_membership_vec * momentum + membership_mat[client_id, :] * (1 - momentum)
        self.previous_membership_vec = membership_mat[client_id].clone()
        
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
        device = membership_mat.device

        eps = torch.tensor(1e-10, device=device)
        clamp_max = torch.tensor(1e10, device=device)
        p = float(2 / (fuzzy_m - 1))

        all_losses = cluster_learners.gather_losses(self.val_iterator).to(device)
        mean_losses = all_losses.mean(dim=1)
        distances = mean_losses - self.trans * mean_losses.min() 
        membership_mat[client_id, :] = self.distances_to_membership(distances, p, eps, clamp_max)

        if self.previous_membership_vec is not None:
            membership_mat[client_id] = self.previous_membership_vec * momentum + membership_mat[client_id] * (1 - momentum)
        
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
 

    def torch_display(self, info, tensor):
        tensor = tensor.cpu().tolist()
        rounded_tensor = [round(d, 7) for d in tensor]
        print(info, rounded_tensor)
 