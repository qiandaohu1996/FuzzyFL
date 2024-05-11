import os
import time
import random

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from client.Client import Client

from utils.torch_utils import *
from learners.learner import *
from learners.learners_ensemble import *
from utils.my_profiler import calc_exec_time
import torch
from tqdm import tqdm
from rich.progress import track

calc_time = True

class Aggregator(ABC):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag=True,
        sampling_rate=1.0,
        log_freq=10,
        sample_with_replacement=False,
        test_clients=None,
        verbose=1,
        seed=None,
        *args,
        **kwargs,
    ):
        rng_seed = seed if (seed is not None and seed >= 0) else int(time.time())
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.train_client_global_logger = train_client_global_logger
        self.test_client_global_logger = test_client_global_logger
        self.model_dim = self.global_learners_ensemble.model_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights = torch.tensor(
            [client.n_train_samples for client in self.clients], dtype=torch.float32
        )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = []
        self.n_sampled_clients = len(self.sampled_clients)
        self.sampled_clients_weights = None

        self.single_batch_flag = single_batch_flag
        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                # print('learner_id',learner_id)
                copy_model(
                    target=learner.model,
                    source=self.global_learners_ensemble[learner_id].model,
                )

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()

    def evaluate_clients(self,clients):
        total_train_loss,total_train_acc, total_test_loss, total_test_acc, total_train_samples,total_test_samples = .0, .0, .0, .0, 0, 0
        
        for client in clients: 
            train_loss, train_acc, test_loss, test_acc  = client.evaluate()
            n_train_samples= client.n_train_samples
            n_test_samples= client.n_test_samples
            total_train_loss += train_loss * n_train_samples
            total_train_acc += train_acc * n_train_samples
            total_test_loss += test_loss * n_test_samples
            total_test_acc += test_acc * n_test_samples
            total_train_samples += n_train_samples
            total_test_samples += n_test_samples
            
            if self.verbose>1:
                 print(f"Client {client.idx:<3} | Train Loss  {train_loss:.5f}  Train Acc {train_acc:.5f}  Train Samples {n_train_samples:<3} | Test Loss {test_loss:.5f}  Test Acc {test_acc:.5f}  Test Samples {n_test_samples:<3}")
        global_train_loss = total_train_loss / total_train_samples
        global_train_acc = total_train_acc / total_train_samples
        global_test_loss = total_test_loss / total_test_samples
        global_test_acc = total_test_acc / total_test_samples
            
        return global_train_loss,global_train_acc,global_test_loss,global_test_acc
    
    def evaluate(self):
        # train_loss, train_acc, test_loss, test_acc = self.evaluate_clients(self.sampled_clients)
        # self.train_client_local_logger.writerow([self.c_round, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{test_loss:.5f}", f"{test_acc:.5f}"])
        
        self.update_test_clients()
        # print(f'Epoch {self.c_round}')
        if self.verbose>1:
            print(f"Train Client")
            
        train_loss, train_acc, test_loss, test_acc = self.evaluate_clients(self.sampled_clients)
        self.train_client_global_logger.writerow([self.c_round, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{test_loss:.5f}", f"{test_acc:.5f}"])

        if self.verbose>1:
            print(f"Test Client")
            
        if self.verbose>0:
            print(f"Train Client Global | n_clients {len(self.sampled_clients)} | Train Loss {train_loss:.5f}  Train Acc {train_acc:.5f} | Test Loss {test_loss:.5f}  Test Acc {test_acc:.5f}")
            
        train_loss, train_acc, test_loss, test_acc = self.evaluate_clients(self.test_clients)
        self.test_client_global_logger.writerow([self.c_round, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{test_loss:.5f}", f"{test_acc:.5f}"])
        
        if self.verbose>0:
            print(f"Test Client Global  | n_clients {len(self.test_clients)} | Train Loss {train_loss:.5f}  Train Acc {train_acc:.5f} | Test Loss {test_loss:.5f}  Test Acc {test_acc:.5f}")
     
    def save_state(self, dir_path):
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ["train", learners_weights, self.clients],
            ["test", test_learners_weights, self.test_clients],
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
   
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ["train", learners_weights, self.clients],
            ["test", test_learners_weights, self.test_clients],
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]

    def sample_clients(self):
        if self.sampling_rate == 1:
            self.sampled_clients = self.clients
            self.sampled_clients_weights = self.clients_weights
            self.n_sampled_clients = self.n_clients
            return

        if self.sample_with_replacement:
            self.sampled_clients = self.rng.choices(
                population=self.clients,
                weights=self.clients_weights,
                k=self.n_clients_per_round,
            )
        else:
            self.sampled_clients = self.rng.sample(
                self.clients, k=self.n_clients_per_round
            )
        self.n_sampled_clients = len(self.sampled_clients)

        self.sampled_clients_weights = torch.tensor(
            [client.n_train_samples for client in self.sampled_clients],
            dtype=torch.float32,
        )
        self.sampled_clients_weights = (
            self.sampled_clients_weights / self.sampled_clients_weights.sum()
        )

 