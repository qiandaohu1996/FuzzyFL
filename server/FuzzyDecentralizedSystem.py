import copy

import pandas as pd
from server.Aggregator import Aggregator
from utils.torch_utils import average_learners, copy_model

class FuzzyDecentralizedSystem(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(FuzzyDecentralizedSystem, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )
        self.round_neighbors=[]
        self.pre_round=200
        self.pretrain=True
        columns = ['Round'] + [str(i) for i in range(40)]  # 20个客户端编号
        self.neighbors_df = pd.DataFrame(columns=columns)
    
    def mix(self):
        self.sample_clients()
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)
            client.should_comm()
        # print('sampled_clients ',)
        # print([client.idx for client in  self.sampled_clients])
        # print('client.comm_flag  ',)
        # print({client.idx:client.comm_flag for client in self.sampled_clients})
            
        if self.c_round > self.pre_round:
            self.pre_train=False
            
        comm_clients=self.gather_comm_clients()
        # print('comm_clients', [client.idx for client in comm_clients])
        self.update_clients(comm_clients,self.pretrain)
        
        if self.c_round % self.log_freq == 0:
            self.write_logs()
        
        round_data = [int(self.c_round)] + [None] * 40  # 初始化当前轮的数据
        for client in comm_clients:
            client_neighbors = [ne.idx for ne in client.round_neighbors]
            # print('client_neighbors', client_neighbors)
            round_data[int(client.idx + 1)] = client_neighbors  # 填充客户端的邻居数据
        # print('round_data', round_data)
        self.neighbors_df.loc[len(self.neighbors_df)] = round_data
        self.c_round += 1
        if self.c_round==200:
            neighborsfile=f'm{self.sampled_clients[0].fuzzy_m}_cp{self.sampled_clients[0].comm_prob}_neighbors.csv'
            print(neighborsfile)
            self.neighbors_df.to_csv(neighborsfile, index=False)
            
    
    def update_clients(self,comm_clients, pretrain):
        # print('comm_clients',[client.idx for client in comm_clients])
        
        for client in comm_clients:
            other_comm_clients=copy.copy(comm_clients)
            other_comm_clients.remove(client)
            # print('client', client.idx)
            # print('other_comm_clients',[client.idx for client in other_comm_clients])
            if pretrain:
                client.calc_important_weights(other_comm_clients)
                comm_neighbors=client.round_neighbors
            else:
                comm_neighbors= client.neighbors & comm_clients
            # print('comm_neighbors ',len(comm_neighbors),  [client.idx for client in comm_neighbors])
            client.local_aggregate(comm_neighbors)
            
    def gather_comm_clients(self):
        comm_clients =  [client  for client in self.sampled_clients if client.comm_flag]
        return comm_clients
    
    def get_other_intermediate_outputs(self,client,pretrain=True):
        intermediate_outputs={}
        round_clients=copy.copy(self.comm_clients)
        clients = set(round_clients)
        if pretrain:
            clients.remove(client)
        else:
            clients.remove(client) 
            clients= clients& set(self.comm_clients)
        for client in clients:
            intermediate_outputs[client]=client.get_intermediate_output()
        return intermediate_outputs
    
    
    def get_other_comm_clients(self,client,pretrain=True):
        round_clients=copy.copy(self.sampled_clients)
        clients = set(round_clients)
        if pretrain:
            clients.remove(client)
        else:
            clients.remove(client) 
            clients= clients & set(self.comm_clients)
        return clients
    

    def update_clients1(self,comm_clients,pretrain):
        
        for client in comm_clients:
            # print('client', client.idx)
            if pretrain:
                intermediate_outputs=self.get_other_intermediate_outputs(client,pretrain)
                client.calc_important_weights(intermediate_outputs)
                comm_neighbors=client.round_neighbors
            else:
                comm_neighbors=self.neighbors & comm_clients
            # print('comm_neighbors ',len(comm_neighbors),  [client.idx for client in comm_neighbors])
            client.local_aggregate(comm_neighbors)
            