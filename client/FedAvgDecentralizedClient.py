

import random
import time

import numpy as np
from client.Client import Client


class FedAvgDecentralizedClient(Client):
    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        idx,
        logger,
        local_steps,
        seed,
        initial_fuzzy_m=1,
        comm_prob=1,
        tune_locally=False,
    ):
        super(FedAvgDecentralizedClient, self).__init__(
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
        self.fuzzy_m=initial_fuzzy_m
        self._lambda = 0.5
        rng_seed = seed + self.counter + self.n_train_samples  + self.idx + int(time.time())
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)
         
        
    def should_comm(self):
        self.comm_flag = self.rng.random() < self.comm_prob
   
    def local_aggregate(self, comm_neighbors):
        # self.renormalize_weight(comm_neighbors)
        # print('importance_weights2', {k.idx:round(v.item(),2) for k,v in self.importance_weights.items()})
        neighbor_model_params = self.average_model(comm_neighbors)
        client_model = self.learners_ensemble[0].model
        client_model_state_dict = client_model.state_dict()

        # 更新模型参数
        for name, param in client_model.named_parameters():
            if name in neighbor_model_params:
                client_model_state_dict[name] = self._lambda * param.clone().detach() + (1 - self._lambda) * neighbor_model_params[name]

        # 将更新后的参数加载到模型中
        client_model.load_state_dict(client_model_state_dict)

    def average_model(self, comm_neighbors, weights):
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

