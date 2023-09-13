from copy import deepcopy

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from aggregator import Aggregator
from utils.my_profiler import calc_exec_time, segment_timing
from utils.torch_utils import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# from utils.fuzzy_cluster import *
# from finch import FINCH

from learners.learner import *
from learners.learners_ensemble import *
from utils.fuzzy_utils import *


calc_time = True
SAMPLING_FLAG = False
 

class FedSoftAggregator(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
        mu,
        n_clusters=3,
        tau=5,
        pre_rounds=1,
        sampling_rate=1.0,
        log_freq=10,
        sample_with_replacement=False,
        test_clients=None,
        verbose=1,
        seed=None,
    ):
        super(FedSoftAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            log_freq=log_freq,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )
        self.mu=mu
        self.cluster_weights_update_interval = tau
        self.pre_rounds=pre_rounds
        self.n_clusters=n_clusters
        print("self.clients_weights.shape",self.clients_weights.shape)
        self.aggregation_weights=self.clients_weights.unsqueeze(0).expand(n_clusters, -1)
        print("aggregation_weights", self.aggregation_weights.shape)
        self.cluster_flag=False
        self.sampled_clients_weights_for_clusters=[]
        self.write_logs()
        
    def mix(self):
        # Sample clients for this round
        if self.c_round < self.pre_rounds:
            self.sample_clients()
            self.pre_train()
            if self.c_round % self.log_freq == 0 or self.c_round == self.pre_rounds - 1:
                self.write_logs()
                self.write_local_test_logs()
        elif self.c_round == self.pre_rounds and self.cluster_flag is False:
            self.pre_clusting()
            self.cluster_flag = True
            self.c_round -= 1
            
        elif self.c_round >= self.pre_rounds:
            self.sample_clients()
            self.train()
            if self.c_round % self.log_freq == 0 or self.c_round == 199:
                self.write_logs()
                self.write_local_test_logs()
                
        self.c_round += 1
            
    def train(self):
        
        self.cluster_models = [learner.model for learner in self.cluster_learners]
        client_learners = [client.learners_ensemble[0] for client in self.sampled_clients]
        client_models = [learner.model for learner in client_learners]
        all_cluster_weights=  torch.zeros((self.n_sampled_clients,self.n_clusters),dtype=torch.float32,device=self.device)
        # If it's time to update cluster weights based on distances
        if self.c_round % self.cluster_weights_update_interval == 0:
            for client_id,client in enumerate(self.sampled_clients):
                all_cluster_weights[client_id] = client.update_cluster_weights(self.n_clusters)
            # Update cluster aggregation_weights based on received weights from clients
            self.update_aggregation_weights(all_cluster_weights)
        
        self.update_prox_clusters(self.cluster_models,all_cluster_weights)

        # Let each client perform their local updates 
        with segment_timing("updating all clients' model"):
            for client in self.sampled_clients:
                client.step(self.single_batch_flag)
                
        average_learners(
            learners=client_learners,
            target_learner=self.global_learners_ensemble[0],
        )
        self.aggregate_cluster_model(client_models,self.cluster_models,self.aggregation_weights)
        

    def aggregate_cluster_model(self,client_models,cluster_models,aggregation_weights):
        for cluster_id, cluster_model in enumerate(cluster_models):
            target_state_dict = cluster_model.state_dict(keep_vars=True)
            for key in target_state_dict:
                target_state_dict[key].data.zero_()
                if target_state_dict[key].data.dtype == torch.float32:
                    for client_id in range(len(client_models)):
                        client_state_dict = client_models[client_id].state_dict(keep_vars=True)
                        
                        # Use the cluster weight of the client for the current cluster as the importance weight
                        target_state_dict[key].data.add_(aggregation_weights[cluster_id][client_id] * client_state_dict[key].data)
                        
    
    def update_aggregation_weights(self, all_cluster_weights):
        """
        Compute aggregation weights based on received importance weights.
        """
        # 初始化
        self.aggregation_weights = torch.zeros((self.n_clusters, self.n_sampled_clients)).to(self.device)

        for cluster_id in range(self.n_clusters):
            # 计算分母
            denominator = sum(all_cluster_weights[client_id, cluster_id] * client.n_train_samples 
                            for client_id, client in enumerate(self.sampled_clients))

            for client_id, client in enumerate(self.sampled_clients):
                # 根据给定的公式计算聚合权重
                numerator = all_cluster_weights[client_id, cluster_id] * client.n_train_samples
                self.aggregation_weights[cluster_id, client_id] = numerator / denominator

        return self.aggregation_weights

    
    def sample_clients_for_clusters(self, clusters):
        """
        Sample clients for each cluster without repetition.
        """
        all_sampled_clients = set()
        for cluster_id in clusters:
            sampled_clients = set(self.rng.choices(
                population=self.clients,
                weights=self.clients_weights,
                k=self.n_clients_per_round,
            ))
            all_sampled_clients.update(sampled_clients)
            self.cluster_to_sampled_clients[cluster_id] = sampled_clients  # 为每个簇存储选择的客户端
            # self.sampled_clients_weights_for_clusters.append(torch.tensor(
            # [client.n_train_samples for client in sampled_clients],
            # dtype=torch.float32,
            # ))
            # 为每个簇存储选择的客户端
            # self.sampled_clients_weights_for_clusters[cluster_id] /= self.sampled_clients_weights.sum()
            
        self.sampled_clients=all_sampled_clients
        self.n_sampled_clients = len(all_sampled_clients)

        self.sampled_clients_weights = torch.tensor(
            [client.n_train_samples for client in all_sampled_clients],
            dtype=torch.float32,
        )
        self.sampled_clients_weights /= self.sampled_clients_weights.sum()


    def update_prox_clusters(self,cluster_models,all_cluster_weights):
        # Compute weighted sum of cluster models for proximal terms
        print("all_cluster_weights.shape",all_cluster_weights.shape)
        for client_id, client in enumerate(self.sampled_clients):
            client_weights = all_cluster_weights[client_id]  # 获取当前客户端的权重

            for learner in client.learners_ensemble:
                if callable(getattr(learner.optimizer, "set_proximal_params", None)):
                    # 设置近端参数
                    learner.optimizer.set_proximal_params(cluster_models, client_weights)
                    
    def update_clients(self):
        pass
    
    def pre_train(self):

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

        # self.update_clients(self.sampled_clients)
        
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

        # self.global_comm_clients=[self.clients[i] for i in self.global_comm_clients_indices]
        # self.update_clients(self.sample_clients)
        # self.sampling_rate=self.initial_sampling_rate
        # print("init membership_mat ",self.membership_mat[2:5])

