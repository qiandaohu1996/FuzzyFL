import copy
from server import Aggregator
from utils.torch_utils import average_learners, copy_model


class FedAvgDecentralizedSystem(Aggregator):
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
        super(FedAvgDecentralizedSystem, self).__init__(
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
    
    def mix(self):
        self.sample_clients()
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)
            # client.should_comm()
        # print('client.comm_flag  ',)
        
        # print({client.idx:client.comm_flag for client in self.sampled_clients})
             
        # comm_clients=self.gather_comm_clients()
        
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [
                client.learners_ensemble[learner_id] for client in self.sampled_clients
            ]
            average_learners(learners, learner, weights=self.sampled_clients_weights)
        # print('comm_clients', [client.idx for client in comm_clients])
        self.update_clients()
        
        if self.c_round % self.log_freq == 0:
            self.write_logs()
        self.c_round += 1
        
    def update_clients(self):
        for client in self.sampled_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(
                    learner.model, self.global_learners_ensemble[learner_id].model
                )
    def gather_comm_clients(self):
        comm_clients =  [client  for client in self.sampled_clients if client.comm_flag]
        return comm_clients
    
    def get_other_comm_clients(self,client,pretrain=True):
        round_clients=copy.copy(self.sampled_clients)
        clients = set(round_clients)
        if pretrain:
            clients.remove(client)
        else:
            clients.remove(client) 
            clients= clients & set(self.comm_clients)
        return clients
    