import copy
import json
import random
import time
import numpy as np

import torch

from client.Client import Client
from utils.torch_utils import get_layer_param_tensor, get_param_list, get_param_tensor

class FuzzyDecentralizedClient(Client):
    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        idx,
        initial_fuzzy_m,
        comm_prob,
        logger,
        local_steps,
        seed,
        tune_locally=False,
    ):
        super(FuzzyDecentralizedClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
        self.round_neighbors=[]
        self.neighbors=set()
        self.comm_prob=comm_prob
        self.comm_flag=None
        self.comm_clients=[]
        self.intermediate_outputs={}
        self.fuzzy_m=initial_fuzzy_m
        self._lambda = 0.5
        rng_seed = seed + self.counter+self.n_train_samples  + self.idx + int(time.time())
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)
        
    def gather_sampled_client_ids(self):
        self.sampled_clients_ids=  [client.idx  for client in self.sampled_clients]
        
    def should_comm(self):
        self.comm_flag = self.rng.random() < self.comm_prob

    # def step(self, single_batch_flag=True):
    #     super.step(single_batch_flag)
            
    def get_intermediate_output(self):
        self.intermediate_output=self.learners_ensemble[0].model.intermediate_output
        return self.intermediate_output
    
    # calculate important weights of the client (self) and other clients, 然后选择累计重要性和大于90%的客户端 
             
    def euclidean_distance(self, param1, param2):
        return torch.norm(param1 - param2, p=2)
    
    def calc_important_weights(self, other_comm_clients):
        
        # device=self.learners_ensemble[0].device
        # eps = torch.tensor(1e-8, device=device)
        # clamp_max = torch.tensor(1e8, device=device)
        
        client_model = self.learners_ensemble[0].model
        other_clients_models= {client: client.learners_ensemble[0].model for client in other_comm_clients}
        distances = {}
        client_params = get_param_tensor(client_model)  # shape: [1, dim]
        
        # 以第一个卷积层为例
        # layer_name = 'output'
        # client_params = get_layer_param_tensor(client_model, layer_name)
        # print("client_params ",client_params.shape)
        # for client, other_model in other_clients_models.items():
        #     other_client_params = get_layer_param_tensor(other_model, layer_name)
        #     if other_client_params is not None:
        #         diff = client_params - other_client_params
        #         distances[client] = torch.norm(diff)

        for client,other_model in other_clients_models.items():
            other_client_params = get_param_tensor(other_model)
            diff = client_params - other_client_params
            distances[client] = torch.norm(diff)
            
        # print('distance', {k.idx:round(v.item(),2) for k,v in distances.items()})
        
        # threshold = 0.5 * min(distances.values())
        
        # for client in distances:
        #     distances[client]=torch.clamp(distances[client] - threshold, eps, clamp_max)
        total_fuzzy_distance = torch.sum(torch.exp(-self.fuzzy_m * torch.tensor(list(distances.values()))))
        # print('distance', {k.idx:round(v.item(),2) for k,v in distances.items()})

        # 计算重要性权重
        self.importance_weights = {client: torch.exp(-self.fuzzy_m * distance) / total_fuzzy_distance
                               for client, distance in distances.items()}
        # print('importance_clients', [k.idx  for k in self.importance_weights])
        # print('importance_weights', {k.idx:round(v.item(),2) for k,v in self.importance_weights.items()})
        # 选择累计重要性权重超过90%的客户端
        sorted_clients = sorted(self.importance_weights, key=self.importance_weights.get, reverse=True)
        cumulative_weight = 0.0
        self.round_neighbors = []
        for client in sorted_clients:
            cumulative_weight += self.importance_weights[client]
            self.round_neighbors.append(client)
            if cumulative_weight >= 0.9:
                break
        # print('self.round_neighbors', len(self.round_neighbors),  [client.idx for client in self.round_neighbors])
 
        self.neighbors.update(self.round_neighbors)
        total_weight_selected = sum([self.importance_weights[client] for client in self.round_neighbors])
        self.importance_weights = {client: self.importance_weights[client] / total_weight_selected
                            for client in self.round_neighbors}
        
    
    def local_aggregate(self, comm_neighbors):
        # self.renormalize_weight(comm_neighbors)
        # print('importance_weights2', {k.idx:round(v.item(),2) for k,v in self.importance_weights.items()})
        neighbor_model_params = average_model(comm_neighbors, self.importance_weights)
        client_model = self.learners_ensemble[0].model
        client_model_state_dict = client_model.state_dict()

        # 更新模型参数
        self._lambda = 1/(len(comm_neighbors)+1)
        for name, param in client_model.named_parameters():
            if name in neighbor_model_params:
                client_model_state_dict[name] = self._lambda * param.clone().detach() + (1 - self._lambda) * neighbor_model_params[name]

        # 将更新后的参数加载到模型中
        client_model.load_state_dict(client_model_state_dict)

def average_model(comm_neighbors, weights):
    # 初始化聚合模型
    traget_model = None

    # 遍历每个客户端模型，进行加权聚合
    for client in comm_neighbors: 
        model = client.learners_ensemble[0].model
        if traget_model is None:
            traget_model = {name: weights[client] * param.clone().detach() 
                            for name, param in model.named_parameters()}
        else:
            for name, param in model.named_parameters():
                traget_model[name] += weights[client] * param.clone().detach()

    return traget_model

