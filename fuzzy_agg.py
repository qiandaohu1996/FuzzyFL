from copy import deepcopy

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from aggregator import Aggregator
from utils.my_profiler import calc_exec_time, segment_timing
from utils.torch_utils import *
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
# from utils.fuzzy_cluster import *
# from finch import FINCH

from learners.learner import *
from learners.learners_ensemble import *
from utils.fuzzy_utils import *


calc_time = True
SAMPLING_FLAG = False


class FuzzyGroupAggregator(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        comm_prob,
        single_batch_flag,
        measurement,
        fuzzy_m_momentum=0.8,
        n_clusters=5,
        top=3,
        pre_rounds=1,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(FuzzyGroupAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            comm_prob=comm_prob,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        self.single_batch_flag = single_batch_flag
        self.global_learner = self.global_learners_ensemble[0]
        self.membership_mat = None
        self.n_clusters = n_clusters
        self.pre_rounds = pre_rounds
        self.momentum = fuzzy_m_momentum
        self.fuzzy_m_scheduler = self.clients[0].fuzzy_m_scheduler
        self.fuzzy_m = self.clients[0].fuzzy_m
        self.measurement = measurement
        self.trans = self.clients[0].trans
        self.previous_membership_mat = None
        self.cluster_flag = False
        self.top = top
        self.write_logs()

    def init_membership_mat(self, n_clients, n_clusters):
        membership_mat = torch.rand(n_clients, n_clusters)
        membership_mat = membership_mat / \
            membership_mat.sum(dim=1, keepdim=True)

        print("init membership_mat[1:3]:", membership_mat[1:3])
        return membership_mat

    def pre_train(self):
        self.sample_clients()
        # print(self.sampled_clients)

        for client in self.sampled_clients:
            client.step(self.single_batch_flag)

        clients_learners = [
            client.learners_ensemble[0] for client in self.sampled_clients
        ]
        average_learners(
            clients_learners,
            self.global_learners_ensemble[0],
            weights=self.sampled_clients_weights,
        )

        self.update_clients(self.sampled_clients)

    @calc_exec_time(calc_time=calc_time)
    def pre_clusting(self):
        print("\n============start clustring==============")

        clients_updates = np.zeros((self.n_sampled_clients, self.model_dim))
        for client_id, client in enumerate(self.sampled_clients):
            clients_updates[client_id] = client.step_record_update(
                self.single_batch_flag
            )
        similarities = pairwise_distances(clients_updates, metric="cosine")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="precomputed", linkage="complete"
        )
        clustering.fit(similarities)
        self.clusters_indices = [
            np.argwhere(clustering.labels_ == i).flatten()
            for i in range(self.n_clusters)
        ]
        # print("clusters_indices ", self.clusters_indices)

        print("=============cluster completed===============")
        print("\ndivided into {} clusters:".format(self.n_clusters))
        for i, cluster in enumerate(self.clusters_indices):
            print(f"cluster {i}: {cluster}")

        cluster_weights = torch.zeros(self.n_clusters)

        learners = [
            deepcopy(self.sampled_clients[0].learners_ensemble[0])
            for _ in range(self.n_clusters)
        ]
        for learner in learners:
            learner.device = self.device

        self.cluster_learners = LearnersEnsemble(
            learners=learners, learners_weights=cluster_weights
        )

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_weights[cluster_id] = self.sampled_clients_weights[indices].sum()
            cluster_clients = [self.sampled_clients[i] for i in indices]

            average_learners(
                learners=[client.learners_ensemble[0]
                          for client in cluster_clients],
                target_learner=self.cluster_learners[cluster_id],
                weights=self.sampled_clients_weights[indices]
                / cluster_weights[cluster_id],
            )
        self.membership_mat = self.init_membership_mat(
            self.n_sampled_clients, self.n_clusters
        )
        self.membership_mat = self.membership_mat.to(self.device)

        # self.global_comm_clients=[self.clients[i] for i in self.global_comm_clients_indices]
        # self.update_clients(self.sample_clients)
        # self.sampling_rate=self.initial_sampling_rate
        # print("init membership_mat ",self.membership_mat[2:5])

    def train(self):
        # self.cluster_comm_clients_indices=list(range(self.n_clients))
        # self.global_comm_clients_indices=[]

        # self.cluster_comm_clients_weights=self.clients_weights
        # self.cluster_comm_clients_membership_mat=torch.empty((),device=self.device)
        self.sample_clients()
        
                
        def update_membership():
            print("fuzzy_m ", f"{self.fuzzy_m:.3f}")
            if self.measurement == "euclid":
                for client_id, client in enumerate(self.sampled_clients):
                    # print("client ", client_id)
                    client.update_membership_euclid(
                        client_learner=client_learners[client_id],
                        cluster_learners=self.cluster_learners,
                        membership_mat=self.membership_mat,
                        client_id=client_id,
                        global_fixed_m=True,
                        fuzzy_m=self.fuzzy_m,
                        momentum=self.momentum,
                    )
            elif self.measurement == "level":
                for client_id, client in enumerate(self.sampled_clients):
                    # print("client ", client_id)
                    client.update_membership_level(
                        client_learner=client_learners[client_id],
                        cluster_learners=self.cluster_learners,
                        membership_mat=self.membership_mat,
                        global_fixed_m=True,
                        client_id=client_id,
                        fuzzy_m=self.fuzzy_m,
                        momentum=self.momentum,
                    )
            elif self.measurement == "grad":
                for client_id, client in enumerate(self.sampled_clients):
                    client.update_membership_grad(
                        client_learner=client_learners[client_id],
                        cluster_learners=self.cluster_learners,
                        membership_mat=self.membership_mat,
                        global_fixed_m=True,
                        client_id=client_id,
                        fuzzy_m=self.fuzzy_m,
                        momentum=self.momentum,
                    )
            elif self.measurement == "loss":
                for client_id, client in enumerate(self.sampled_clients):
                    client.update_membership_loss(
                        cluster_learners=self.cluster_learners,
                        membership_mat=self.membership_mat,
                        global_fixed_m=True,
                        client_id=client_id,
                        fuzzy_m=self.fuzzy_m,
                        momentum=self.momentum,
                    )
            elif self.measurement == "graddot":
                for client_id, client in enumerate(self.sampled_clients):
                    client.update_membership_graddot(
                        client_learner=client_learners[client_id],
                        cluster_learners=self.cluster_learners,
                        membership_mat=self.membership_mat,
                        global_fixed_m=True,
                        client_id=client_id,
                        fuzzy_m=self.fuzzy_m,
                        momentum=self.momentum,
                    )
            else:
                assert self.measurement in list("euclid", "loss", "level", "grad", "graddot"), (f"measurement {self.measurement} is not valid!!!, \
                    the measurement must be one of 'euclid','loss' or 'level', 'grad', 'graddot'. ")
                # client.update_membership_dot(
                #     # client_model=client_models[client_id],
                #     cluster_models=cluster_models,
                #     client_model=client_models[client_id],
                #     membership_mat=self.membership_mat,
                #     global_fixed_m=True,
                #     client_id=client_id,
                #     fuzzy_m=self.fuzzy_m,
                #     momentum=self.momentum,
                # )
                # client.update_membership_cosine(
                #     # client_model=client_models[client_id],
                #     cluster_models=cluster_models,
                #     client_model=client_models[client_id],
                #     membership_mat=self.membership_mat,
                #     global_fixed_m=True,
                #     client_id=client_id,
                #     fuzzy_m=self.fuzzy_m,
                #     momentum=self.momentum,
                # )
            print("after updating membership: ", self.membership_mat[:2])
        
        cluster_models = [learner.model for learner in self.cluster_learners]
        client_learners = [client.learners_ensemble[0] for client in self.sampled_clients]
        # cluster_comm_clients_learners = [self.clients[i].learners_ensemble[0] for i in self.cluster_comm_clients_indices]

        client_models = [learner.model for learner in client_learners]
        # cluster_comm_client_models = [learner.model for learner in cluster_comm_clients_learners]
        # def worker_step(client):
        #     client.step(self.single_batch_flag) 
        with segment_timing("updating all clients' model"):
            # for client in  self.sampled_clients:
            for client in self.sampled_clients:
                client.step(self.single_batch_flag)
            # with ProcessPoolExecutor(max_workers=max_processes) as executor:
            #     executor.map(lambda client: client.step(self.single_batch_flag), self.sampled_clients)

            # max_processes = 3  # 可以根据需要设置 def worker(client):
            # with ProcessPoolExecutor(max_workers=max_processes) as executor:
            #     list(executor.map(worker_step, self.sampled_clients))
            # max_threads = 5  # Adjust as needed

            # Use ThreadPoolExecutor to handle parallel processing 

            # with ThreadPoolExecutor(max_workers=max_threads) as executor:
            #     executor.map(lambda client: client.step(self.single_batch_flag), self.sampled_clients)
            # print("membership_mat ",self.membership_mat[:5])
        with segment_timing("updating all clients' membership matrices "):
            update_membership()
            self.fuzzy_m_scheduler_step()

        with segment_timing("aggregate to get the cluster model "):
            fuzzy_average_cluster_model(
                client_models=client_models,
                cluster_models=cluster_models,
                membership_mat=self.membership_mat,
                fuzzy_m=self.fuzzy_m,
                top=self.top,
                clients_weights=self.sampled_clients_weights,
            )

        with segment_timing("aggregate to get the client model "):
            fuzzy_average_client_model(
                membership_mat=self.membership_mat,
                cluster_models=cluster_models,
                client_models=client_models,
                top=self.top,
            )

        # with segment_timing("aggregate to get the global model "):
        average_learners(
            learners=self.cluster_learners,
            target_learner=self.global_learners_ensemble[0],
        )

    def mix(self):
        if self.c_round < self.pre_rounds:
            self.pre_train()
            if self.c_round % self.log_freq == 0 or self.c_round == self.pre_rounds - 1:
                self.write_logs()
                self.write_local_test_logs()

        elif self.c_round == self.pre_rounds and self.cluster_flag is False:
            self.pre_clusting()
            self.cluster_flag = True
            self.c_round -= 1

        elif self.c_round >= self.pre_rounds:
            self.train()
            if self.c_round % self.log_freq == 0 or self.c_round == 199:
                self.write_logs()
                self.write_local_test_logs()

        if self.c_round % 50 == 0 or self.c_round == 199:
            print(self.membership_mat)

        self.c_round += 1

        # if self.c_round == 201:
        #     print("c_round==201")

    def update_clients(self, clients):
        for client in clients:
            for learner in client.learners_ensemble:
                copy_model(
                    target=learner.model, source=self.global_learners_ensemble[0].model
                )

    def update_cluster_comm_client_weights(self):
        self.cluster_comm_clients_weights = torch.tensor(
            [
                self.clients[i].n_train_samples
                for i in self.cluster_comm_clients_indices
            ],
            dtype=torch.float32,
        )
        self.cluster_comm_clients_weights = (
            self.cluster_comm_clients_weights / self.cluster_comm_clients_weights.sum()
        )

    def fuzzy_m_scheduler_step(self):
        self.fuzzy_m = self.fuzzy_m_scheduler.step(
            self.c_round - self.pre_rounds)
 

    # def distances_to_membership(self, distances, p, eps, clamp_max):
    #     distances = distances + eps
    #     # Add an extra dimension to get a shape of (8, 1, 3)
    #     distances = distances.unsqueeze(1)
    #     # Transpose the last two dimensions and divide
    #     denom = distances / distances.transpose(-2, -1)
    #     denom = (denom ** p).sum(dim=-1)
    #     denom = torch.clamp(denom, min=eps, max=clamp_max)
    #     membership_mat = 1 / denom
    #     return membership_mat

    # def model_l2_distance(self, client_models, cluster_models):
    #     """Compute the distance between multiple client_models and cluster_models."""
    #     # Convert each model's parameters into a 1-D vector
    #     client_params = torch.stack([self.flatten_model_params(model) for model in client_models])  # shape: [n, dim]
    #     cluster_params = torch.stack([self.flatten_model_params(model) for model in cluster_models])  # shape: [k, dim]

    #     # Compute the pairwise L2 distances
    #     distances = torch.cdist(client_params, cluster_params)  # shape: [n, k]

    #     return distances

    # def flatten_model_params(self, model):
    #     """Flatten all parameters of a model into a 1-D vector."""
    #     return torch.cat([param.data.view(-1) for param in model.parameters() if param.requires_grad])

    # def zero_diagonal(input):
    #     # Create a mask that is True for all diagonal elements
    #     mask = torch.eye(input.size(-1), device=input.device).bool()
    #     return input.masked_fill_(mask, 0)


"""
    if self.c_round % 5 == 0 :
        cluster_models.append(self.global_learners_ensemble[0].model)
        cluster_params = get_param_list(cluster_models)

        for client_id, client in enumerate(self.clients):
            client_params = get_param_tensor(client.learners_ensemble[0].model)
            comm_global_flag=client.update_global_membership_mat(
                client_params=client_params,
                cluster_params=cluster_params,
                membership_mat= self.membership_mat,
                global_fixed_m=True,
                fuzzy_m=self.fuzzy_m,
                client_id=client_id
                )
        if comm_global_flag:
            self.cluster_comm_clients_indices.remove(client_id)
            self.global_comm_clients_indices.append(client_id)

        print("cluster_comm_clients length",len(self.cluster_comm_clients_indices))
        print("global_comm_clients length",len(self.global_comm_clients_indices))
        self.update_cluster_comm_client_weights()
        self.cluster_comm_clients_membership_mat= self.membership_mat[self.cluster_comm_clients_indices, :]
    else:

    self.cluster_comm_clients=self.clients[:]
    print(cluster_params.size())
    self.cluster_comm_clients=[self.clients[i] for i in self.cluster_comm_clients_indices]
"""
