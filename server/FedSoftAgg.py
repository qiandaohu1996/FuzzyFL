from copy import deepcopy

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from server.Aggregator import Aggregator
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
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag,
        mu,
        n_clusters=3,
        tau=1,
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
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
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
        self.aggregation_weights=self.clients_weights.unsqueeze(0).expand(n_clusters, -1)
        self.cluster_flag=False
        self.sampled_clients_weights_for_clusters=[]

        self.write_logs()
        
    def mix(self):
        # Sample clients for this round
        if self.c_round < self.pre_rounds:
            self.pre_train()
        elif self.c_round == self.pre_rounds and self.cluster_flag is False:
            self.pre_clusting()
            self.cluster_flag = True
            for client in self.clients:
                client.learners_ensemble[0].optimizer.soft_sgd=True
        elif self.c_round > self.pre_rounds:
            self.train()
            
        if self.c_round % self.log_freq == 0 or self.c_round == 199:
            self.write_logs()
                
        self.c_round += 1
            
    def train(self):
        self.sample_clients()
        
        self.client_learners = [
            client.learners_ensemble[0] for client in self.sampled_clients
        ]
        client0_model=self.sampled_clients[0].learners_ensemble[0].model
        # for index, (param_name, param_tensor) in enumerate(client0_model.named_parameters()):
            # if index == 0:
                # print("param_tensor data", param_tensor.data[0])
                # print("param_tensor grad ", param_tensor.grad.data[0])
                # break
        cluster_models = [learner.model for learner in self.cluster_learners]
        all_cluster_weights=  torch.zeros((self.n_sampled_clients,self.n_clusters),dtype=torch.float32,device=self.device)
        
        # If it's time to update cluster weights based on distances
        if self.c_round % self.cluster_weights_update_interval == 0:
            for client_id,client in enumerate(self.sampled_clients):
                all_cluster_weights[client_id]=client.update_cluster_weights(self.cluster_learners)
            # Update cluster aggregation_weights based on received weights from clients
            self.update_aggregation_weights(all_cluster_weights)
            # print("all_cluster_weights ", all_cluster_weights[:1])
            # print("aggregation_weights ",self.aggregation_weights[:1])
            for cluster_id,cluster_learner in enumerate(self.cluster_learners):
                average_learners(self.client_learners,
                                 cluster_learner,
                                 self.aggregation_weights[cluster_id])
        with segment_timing("updating all clients' model"):
            for client in self.sampled_clients: 
                client.step(self.single_batch_flag)
            # cluster_model_params_list = [list(cluster_model.parameters()) for cluster_model in cluster_models]
            
            # self.update_prox_clusters(cluster_model_params_list)

        # Let each client perform their local updates 
                
        average_learners(
            learners=self.client_learners,
            target_learner=self.global_learners_ensemble[0],
            weights=self.sampled_clients_weights
        )
        self.update_clients()
        
    # def set_proximal_params(self, cluster_model_params_list, cluster_weights):
    #     # print("set_proximal_params begin")
    #     if len(cluster_model_params_list) == 0:
    #         raise ValueError("接收到一个空的簇模型列表")
        
    #     if len(cluster_model_params_list) != len(cluster_weights):
    #         raise ValueError("簇模型和簇权重的长度不匹配")
    #     for param_group in self.param_groups:
    #         for param_idx, param in enumerate(param_group['params']):
    #             param_state = self.state[param]
    #             # 对于每个参数，创建一个与所有簇模型参数关联的列表
    #             param_state['cluster_model_params_list'] = [torch.clone(cluster_model_params[param_idx].data) for cluster_model_params in cluster_model_params_list]
    #             param_state['cluster_weights'] = cluster_weights.clone()
                
        
    def update_prox_clusters(self,cluster_model_params_list):
        # Compute weighted sum of cluster models for proximal terms
        for client_id, client in enumerate(self.sampled_clients):
            for learner in client.learners_ensemble:
                if callable(getattr(learner.optimizer, "set_proximal_params", None)):
                    # 设置近端参数
                    learner.optimizer.set_proximal_params(cluster_model_params_list, client.cluster_weights)
                    
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
                numerator = all_cluster_weights[client_id, cluster_id] * client.n_train_samples
                self.aggregation_weights[cluster_id, client_id] = numerator / denominator

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
            self.cluster_to_sampled_clients[cluster_id] = sampled_clients   
            
        self.sampled_clients=all_sampled_clients
        self.n_sampled_clients = len(all_sampled_clients)

        self.sampled_clients_weights = torch.tensor(
            [client.n_train_samples for client in all_sampled_clients],
            dtype=torch.float32,
        )
        self.sampled_clients_weights /= self.sampled_clients_weights.sum()

    def pre_train(self):
        self.sample_clients()
        for client in self.sampled_clients: 
            client.step(self.single_batch_flag)
        client0_model=self.sampled_clients[0].learners_ensemble[0].model
        for index, (param_name, param_tensor) in enumerate(client0_model.named_parameters()):
            if index == 0:
                # print("param_tensor data", param_tensor.data[0])
                # print("param_tensor grad ", param_tensor.grad.data[0])
                break
        
        self.client_learners = [
            client.learners_ensemble[0] for client in self.sampled_clients
        ]
        average_learners(
            self.client_learners,
            self.global_learners_ensemble[0],
            weights=self.sampled_clients_weights,
        )
        # print("global_learners_ensemble ",(list(self.global_learners_ensemble[0].model.parameters())[0].data[0]))
        self.update_clients()
        
    def pre_clusting(self):
        print("\n============start clustring==============")
        self.sample_clients()
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

        self.cluster_learners = LearnersEnsemble(learners=learners, learners_weights=cluster_weights)

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
        self.client_learners = [
            client.learners_ensemble[0] for client in self.sampled_clients
        ]
        average_learners(
            learners=self.client_learners,
            target_learner=self.global_learners_ensemble[0],
            weights=self.sampled_clients_weights
        )
        self.update_clients()
            
    def update_clients(self):
        for client in self.sampled_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(
                    learner.model, self.global_learners_ensemble[learner_id].model
                )
 