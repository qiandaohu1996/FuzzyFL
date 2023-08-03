import os
import time
import random

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from client import Client

from utils.torch_utils import *
from utils.fuzzy_cluster import *

# from finch import FINCH

from learners.learner import *
from learners.learners_ensemble import *
from utils.my_profiler import calc_exec_time
import torch

calc_time = True


class Aggregator(ABC):
    r"""Base class for Aggregator. `Aggregator` dictates communications between clients

     Attributes
     ----------
     clients: List[Client]

     test_clients: List[Client]

     global_learners_ensemble: List[Learner]

     sampling_rate: proportion of clients used at each round; default is `1.`

     sample_with_replacement: is True, client are sampled with replacement; default is False

    self.n_clientss:

     n_learners:

     clients_weights:

     model_dim: dimension if the used model

     c_round: index of the current communication round

     log_freq:

     verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

     global_train_logger:

     global_test_logger:

     rng: random number generator

     np_rng: numpy random number generator

     Methods
     ----------
     __init__
     mix

     update_clients

     update_test_clients

     write_logs

     save_state

     load_state

    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
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
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger
        self.local_test_logger = local_test_logger
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
        if type(self) != GroupAPFL and type(self) != APFLAggregator:
            self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(
                    target=learner.model,
                    source=self.global_learners_ensemble[learner_id].model,
                )

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()

    def _write_logs(self, logger, clients, log_type):
        # print(f"len({log_type.lower()} clients)", len(clients))

        if len(clients) == 0:
            return
        total_loss, total_acc, total_samples = 0.0, 0.0, 0.0
        write_logs_func = getattr(Client, f"write_{log_type.lower()}_logs")

        for client_id, client in enumerate(clients):
            # print("client ", client_id)
            loss, acc = write_logs_func(client)
            if hasattr(clients, "n_train_samples"):
                n_samples = client.n_train_samples
            else:
                n_samples = client.n_test_samples

            total_loss += loss * n_samples
            total_acc += acc * n_samples
            total_samples += n_samples

            if self.verbose > 1:
                print(
                    f"Client {client_id}: {log_type} Loss: {loss:.5f} | {log_type} Acc: {acc * 100:.3f}%"
                )

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        logger.add_scalar(f"{log_type}/Loss", avg_loss, self.c_round)
        logger.add_scalar(f"{log_type}/Metric", avg_acc, self.c_round)

        if self.verbose > 0:
            print(
                f"{log_type} Loss: {avg_loss:.3f} | {log_type} Acc: {avg_acc * 100:.3f}%"
            )

    @calc_exec_time(calc_time=True)
    def write_logs(self):
        self.update_test_clients()
        self._write_logs(self.global_train_logger, self.sampled_clients, "Train")
        self._write_logs(self.global_test_logger, self.test_clients, "Test")

    def write_local_test_logs(self):
        self._write_logs(self.local_test_logger, self.sampled_clients, "Local_Test")
        if self.verbose > 0:
            print("-" * 40)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
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
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
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
        """
        sample a list of clients without repetition
        """

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


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally"""

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step(self.single_batch_flag)
        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_local_test_logs()

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    def mix(self):
        self.sample_clients()
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [
                client.learners_ensemble[learner_id] for client in self.sampled_clients
            ]
            average_learners(learners, learner, weights=self.sampled_clients_weights)

        self.update_clients()

        if self.c_round % self.log_freq == 0:
            self.write_logs()
            self.write_local_test_logs()
        self.c_round += 1

        torch.cuda.empty_cache()

    def update_clients(self):
        for client in self.sampled_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(
                    learner.model, self.global_learners_ensemble[learner_id].model
                )
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )
        torch.cuda.empty_cache()


class PersonalizedAggregator(CentralizedAggregator):
    r"""
    Clients do not synchronize there models, instead they only synchronize optimizers, when needed.

    """

    def update_clients(self):
        for client in self.sampled_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )


class APFLAggregator(Aggregator):
    r"""
    Implements
        `Adaptive Personalized Federated Learning`__(https://arxiv.org/abs/2003.13461)

    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
        sampling_rate=1,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(APFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )
        assert self.n_learners == 2, "APFL requires two learners"

    def mix(self):
        self.sample_clients()
        print(self.single_batch_flag)
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)
            learners = client.learners_ensemble

            if learners.adaptive_alpha:
                learners.update_alpha()
                apfl_partial_average(learners[0], learners[1], learners.alpha)

        if self.c_round % 5 == 0:
            average_learners(
                learners=[client.learners_ensemble[0] for client in self.clients],
                target_learner=self.global_learners_ensemble[0],
                weights=self.clients_weights,
            )
            # assign the updated model to all clients
            self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def write_logs(self):
        # 记录 本地训练精度 本地测试精度 平均值
        self._write_logs(self.global_train_logger, self.clients, "Train")
        self._write_logs(self.local_test_logger, self.clients, "Local_Test")

        if self.verbose > 0:
            print("-" * 50)

    def update_clients(self):
        for client in self.sampled_clients:
            copy_model(
                client.learners_ensemble[0].model,
                self.global_learners_ensemble[0].model,
            )


class AGFLAggregator(Aggregator):
    r"""
    Implements
        `Adaptive Personalized Federated Learning`__(https://arxiv.org/abs/2003.13461)

    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
        pre_rounds=50,
        comm_prob=0.2,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(AGFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            single_batch_flag=single_batch_flag,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )
        # assert self.n_learners == 2, "APFL requires two learners"
        self.pre_rounds=pre_rounds
        self.n_clusters=self.n_learners-1
        self.write_logs()
        
    def pre_train(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step(self.single_batch_flag)

        clients_learners = [client.learners_ensemble[0] for client in self.sampled_clients]
        average_learners(
            clients_learners,
            self.global_learners_ensemble[0],
            self.sampled_clients_weights,
        )

        for learner in clients_learners:
            copy_model(learner.model, self.global_learners_ensemble[0].model)
            
    @calc_exec_time(calc_time=calc_time)
    def pre_clusting(self, n_clusters):
        print("\n============start clustring==============")
        
        clients_updates = np.zeros((self.n_sampled_clients, self.model_dim))
        for client_id, client in enumerate(self.sampled_clients):
            clients_updates[client_id] = client.step_record_update(self.single_batch_flag)
        similarities = pairwise_distances(clients_updates, metric="euclidean")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="complete"
        )
        clustering.fit(similarities)
        self.clusters_indices = [
            np.argwhere(clustering.labels_ == i).flatten()
            for i in range(n_clusters)
        ]
        # print("clusters_indices ", self.clusters_indices)

        print("=============cluster completed===============")
        print("\ndivided into {} clusters:".format(n_clusters))
        for i, cluster in enumerate(self.clusters_indices):
            print(f"cluster {i}: {cluster}")

        cluster_weights = torch.zeros(n_clusters)

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_weights[cluster_id] = self.sampled_clients_weights[indices].sum()
            cluster_clients = [self.sampled_clients[i] for i in indices]

            average_learners(
                learners=[client.learners_ensemble[0]
                          for client in cluster_clients],
                target_learner=self.global_learners_ensemble[cluster_id+1],
                weights=self.sampled_clients_weights[indices]
                        / cluster_weights[cluster_id],
            )
            
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble[1:]):
                copy_model(learner.model, self.global_learners_ensemble[learner_id+1].model
                )

    def train(self):
        self.sample_clients()

        for clienr_id, client in enumerate(self.sampled_clients):
            print("client_id", clienr_id)
            client.step(self.single_batch_flag)
            client_learners = client.learners_ensemble

            if client_learners.adaptive_alpha:
                client_learners.update_alpha()
                agfl_partial_average(client_learners[0], client_learners[1:], client_learners.alpha)

        if self.c_round % 5 == 0:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_learners(learners, learner, weights=self.clients_weights) 
            
            for client in self.clients:
                average_learners(
                    learners=client.learners_ensemble[1:] ,
                    target_learner=client.learners_ensemble[0],
                    weights=self.clients_weights,
                )
        
        # for learner_id, learner in enumerate(self.global_learners_ensemble):
        #     learners = [client.learners_ensemble[learner_id] for client in self.clients]
        #     average_learners(learners, learner, weights=self.clients_weights)

        # assign the updated model to all clients
        # self.update_clients()

    def mix(self):
        if self.c_round < self.pre_rounds:
            self.pre_train()
            if self.c_round % self.log_freq == 0:
                self.write_logs()
                self.write_local_test_logs()

        elif self.c_round == self.pre_rounds:
            self.pre_clusting(n_clusters=self.n_clusters)

        else:
            self.train()
            if self.c_round % self.log_freq == 0 or self.c_round == 199:
                self.write_logs()
                self.write_local_test_logs()

        self.c_round += 1
 
        if self.c_round == 201:
            print("c_round==201")

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(
                    learner.model, self.global_learners_ensemble[learner_id].model
                )


class LoopLessLocalSGDAggregator(PersonalizedAggregator):
    """
    Implements L2SGD introduced in
    'Federated Learning of a Mixture of Global and Local Models'__. (https://arxiv.org/pdf/2002.05516.pdf)


    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
        comm_prob=0.2,
        penalty_parameter=0.1,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(LoopLessLocalSGDAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        self.comm_prob = comm_prob
        self.penalty_parameter = penalty_parameter

    @property
    def comm_prob(self):
        return self.__comm_prob

    @comm_prob.setter
    def comm_prob(self, comm_prob):
        self.__comm_prob = comm_prob

    def mix(self):
        comm_flag = self.np_rng.binomial(1, self.comm_prob, 1)

        if comm_flag:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [
                    client.learners_ensemble[learner_id] for client in self.clients
                ]
                average_learners(learners, learner, weights=self.clients_weights)

                partial_average(
                    learners=learners,
                    average_learner=learner,
                    alpha=self.penalty_parameter / self.comm_prob,
                )

        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
            self.write_local_test_logs()

        else:
            for client in self.clients:
                client.step(single_batch_flag=True)


class ClusteredAggregator(Aggregator):
    """
    Implements
     `Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints`.

     Follows implementation from https://github.com/felisat/clustered-federated-learning
    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        tol_1=0.4,
        tol_2=1.6,
        seed=None,
    ):
        super(ClusteredAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            single_batch_flag=single_batch_flag,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        assert (
            self.n_learners == 1
        ), "ClusteredAggregator only supports single learner clients."
        assert self.sampling_rate == 1.0, (
            f"`sampling_rate` is {sampling_rate}, should be {1.0},"
            f" ClusteredAggregator only supports full clients participation."
        )

        self.tol_1 = tol_1
        self.tol_2 = tol_2
        self.global_learners = self.global_learners_ensemble
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.n_clusters = 1

        # super().write_logs()

    def mix(self):
        self.sampled_clients = self.clients
        clients_updates = np.zeros((self.n_clients, self.n_learners, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step_record_update(
                self.single_batch_flag
            )

        similarities = np.zeros((self.n_learners, self.n_clients, self.n_clients))

        for learner_id in range(self.n_learners):
            similarities[learner_id] = pairwise_distances(
                clients_updates[:, learner_id, :], metric="cosine"
            )

        similarities = similarities.mean(axis=0)

        new_cluster_indices = []
        for indices in self.clusters_indices:
            max_update_norm = np.zeros(self.n_learners)
            mean_update_norm = np.zeros(self.n_learners)

            for learner_id in range(self.n_learners):
                max_update_norm[learner_id] = LA.norm(
                    clients_updates[indices], axis=1
                ).max()
                mean_update_norm[learner_id] = LA.norm(
                    np.mean(clients_updates[indices], axis=0)
                )

            max_update_norm = max_update_norm.mean()
            mean_update_norm = mean_update_norm.mean()

            print("mean_update_norm ", mean_update_norm)
            print("max_update_norm ", max_update_norm)
            if (
                mean_update_norm < self.tol_1
                and max_update_norm > self.tol_2
                and len(indices) > 2
            ):
                clustering = AgglomerativeClustering(
                    metric="precomputed", linkage="complete"
                )
                clustering.fit(similarities[indices][:, indices])
                cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
                cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
                new_cluster_indices += [cluster_1, cluster_2]
            else:
                new_cluster_indices += [indices]

        self.clusters_indices = new_cluster_indices

        self.n_clusters = len(self.clusters_indices)

        self.cluster_learners = [
            deepcopy(self.clients[0].learners_ensemble) for _ in range(self.n_clusters)
        ]

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]
            for learner_id in range(self.n_learners):
                average_learners(
                    learners=[
                        client.learners_ensemble[learner_id]
                        for client in cluster_clients
                    ],
                    target_learner=self.cluster_learners[cluster_id][learner_id],
                    weights=self.clients_weights[indices]
                    / self.clients_weights[indices].sum(),
                )

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)

        self.update_clients()

        if self.c_round % self.log_freq == 0:
            self.write_logs()
            self.write_local_test_logs()

        self.c_round += 1
        torch.cuda.empty_cache()

    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            for i in indices:
                for learner_id, learner in enumerate(self.clients[i].learners_ensemble):
                    copy_model(
                        target=learner.model,
                        source=self.cluster_learners[cluster_id][learner_id].model,
                    )

    # def write_logs(self):
    #         ## 记录 本地训练精度 本地测试精度 平均值
    #     self._write_logs(self.global_train_logger, self.clients, "Train")
    #     self._write_logs(self.local_test_logger, self.clients, "Local_Test")

    #     if self.verbose > 0:
    #         print("#" * 80)


class GroupAPFL(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        local_test_logger,
        single_batch_flag,
        comm_prob=1.0,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
        pre_rounds=50,
    ):
        super(GroupAPFL, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        # assert self.n_learners == 2, "GroupAPFL only supports 2 learner clients."
        assert self.sampling_rate == 1.0, (
            f"`sampling_rate` is {sampling_rate}, should be {1.0},"
            f" GroupAPFL only supports full clients participation before normal training."
        )

        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        print("clusters_indices:", self.clusters_indices)
        self.n_clusters = 1
        self.pre_rounds = pre_rounds
        self.alpha_list = np.zeros(self.n_clients)
        self.global_learner = self.global_learners_ensemble[0]
        self.single_batch_flag = single_batch_flag
        self.pre_write_logs()
        # self.alpha_list = [torch.tensor(client.alpha, device=self.device) for client in self.clients]
        self.comm_prob = comm_prob

        @property
        def comm_prob(self):
            return self.__comm_prob

        @comm_prob.setter
        def comm_prob(self, comm_prob):
            self.__comm_prob = comm_prob

    def pre_train(self):
        for client in self.clients:
            client.step(self.single_batch_flag)

        clients_learners = [
            self.clients[client_id].learners_ensemble[0]
            for client_id in range(self.n_clients)
        ]
        average_learners(
            clients_learners, self.global_learner, weights=self.clients_weights
        )
        self.pre_update_clients()
        self.pre_write_logs()

    def clustering(self, n_clusters):
        print("=============start clustering================")

        clients_updates = np.zeros((self.n_clients, self.model_dim))
        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step_record_update(
                self.single_batch_flag
            )

        self.n_clusters = n_clusters

        similarities = pairwise_distances(clients_updates, metric="cosine")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="precomputed", linkage="complete"
        )

        clustering.fit(similarities)

        self.clusters_indices = [
            np.argwhere(clustering.labels_ == i).flatten()
            for i in range(self.n_clusters)
        ]
        # self.n_clusters = len(self.clusters_indices)

        # self.n_cluster_clients = [len(indice)  for indice in self.clusters_indices]
        print("=============cluster completed===============")

        print("\ndivided into {} clusters:".format(self.n_clusters))
        for i, cluster in enumerate(self.clusters_indices):
            print(f"cluster {i}: {cluster}")

        learners = [
            deepcopy(self.clients[0].learners_ensemble[0])
            for _ in range(self.n_clusters)
        ]

        for learner in learners:
            # init_nn(learner.model)
            learner.device = self.device

        self.cluster_learners = LearnersEnsemble(
            learners=learners,
            learners_weights=torch.ones(self.n_clusters) / self.n_clusters,
        )

        self.c_round -= 1

    def train(self):
        cluster_weights = torch.zeros(self.n_clusters)

        for cluster_id, indices in enumerate(self.clusters_indices):
            print(f"cluster {cluster_id}")
            cluster_weights[cluster_id] = self.clients_weights[indices].sum()
            for i in indices:
                print(f"client {i}..")
                self.clients[i].svrg_step(
                    self.clients_weights[indices] / cluster_weights[cluster_id]
                )

        for cluster_id, indices in enumerate(self.clusters_indices):
            average_learners(
                learners=[
                    client.learners_ensemble[0] for client in self.clients[indices]
                ],
                target_learner=self.cluster_learners[cluster_id],
                weights=self.clients_weights[indices] / cluster_weights[cluster_id],
            )

        average_learners(
            learners=[client.learners_ensemble[1] for client in self.clients],
            target_learner=self.global_learner,
            weights=self.clients_weights,
        )

        self.update_clients()
        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def mix(self):
        """
        Pre-train the model for the specified number of pre_rounds, perform clustering, and then perform federated learning.
        """
        if self.c_round < self.pre_rounds:
            print(f"\n=======pretrain at {self.c_round} round=========")
            self.pre_train()

        elif self.n_clusters == 1:
            self.clustering()
        else:
            print(f"\n=========train at {self.c_round} round===========")
            self.train()
        self.c_round += 1

    def pre_write_type_logs(self, dataset_type):
        if dataset_type == "train":
            logger = self.global_train_logger
            clients = self.clients
        elif dataset_type == "test":
            logger = self.global_test_logger
            clients = self.test_clients

        if len(clients) > 0:
            global_loss = 0.0
            global_acc = 0.0
            total_n_samples = 0

            for client_id, client in enumerate(clients):
                loss, acc = client.pre_write_logs(dataset_type)

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")
                    print(
                        f"{dataset_type.capitalize()} Loss: {loss:.5f} | {dataset_type.capitalize()} Acc: {acc * 100:.3f}%"
                    )
                n_samples = getattr(client, "n_" + dataset_type + "_samples")

                global_loss += loss * n_samples
                global_acc += acc * n_samples
                total_n_samples += n_samples

            global_loss /= total_n_samples
            global_acc /= total_n_samples

            logger.add_scalar(
                f"{dataset_type.capitalize()}/Loss", global_loss, self.c_round
            )
            logger.add_scalar(
                f"{dataset_type.capitalize()}/Metric", global_acc, self.c_round
            )

            if self.verbose > 0:
                print("+" * 30)
                print(
                    f"{dataset_type.capitalize()} Loss: {global_loss:.3f} | \
                    {dataset_type.capitalize()} Acc: {global_acc * 100:.3f}%"
                )

    def pre_write_logs(self):
        self.update_test_clients()

        self.pre_write_type_logs("train")
        self.pre_write_type_logs("test")

        if self.verbose > 0:
            print("#" * 80)

    def write_logs(self):
        self.update_test_clients()
        global_logger, clients = self.global_test_logger, self.test_clients

        if len(self.clients) == 0:
            return
        global_test_loss = 0.0
        global_test_acc = 0.0
        total_n_test_samples = 0

        for client_id, client in enumerate(clients):
            test_loss, test_acc = client.write_test_logs()
            if self.verbose > 1:
                print("*" * 30)
                print(f"Client {client_id}..")
                print(
                    f"Train Loss: {test_loss:.3f} | Train Acc: {test_acc * 100:.3f}%| "
                )

            global_test_loss += test_loss * client.n_test_samples
            global_test_acc += test_acc * client.n_test_samples
            total_n_test_samples += client.n_test_samples
        global_test_loss /= total_n_test_samples
        global_test_acc /= total_n_test_samples

        global_logger.add_scalar("Global_Test/Loss", global_test_loss, self.c_round)
        global_logger.add_scalar("Global_Test/Metric", global_test_acc, self.c_round)
        if self.verbose > 0:
            print("+" * 50)
            print(
                f"Global Test Loss: {global_test_loss:.5f} | Test Acc: {global_test_acc * 100:.3f}% |"
            )
            print("+" * 50)

        global_logger, clients = self.global_train_logger, self.clients

        if len(self.clients) == 0:
            return

        cluster_train_loss = 0.0
        cluster_train_acc = 0.0
        global_train_loss = 0.0
        global_train_acc = 0.0

        total_n_samples = 0

        for client_id, client in enumerate(clients):
            train_loss, train_acc, train_loss2, train_acc2 = client.write_train_logs()
            if self.verbose > 1:
                print("*" * 30)
                print(f"Client {client_id}..")
                print(
                    f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%| "
                )

            cluster_train_loss += train_loss * client.n_train_samples
            cluster_train_acc += train_acc * client.n_train_samples
            global_train_loss += train_loss2 * client.n_train_samples
            global_train_acc += train_acc2 * client.n_train_samples

            total_n_samples += client.n_train_samples

        cluster_train_loss /= total_n_samples
        cluster_train_acc /= total_n_samples
        global_train_loss /= total_n_samples
        global_train_acc /= total_n_samples

        if self.verbose > 0:
            print("+" * 30)
            print(
                f"Cluster Train Loss: {cluster_train_loss:.3f} | Train Acc: {cluster_train_acc * 100:.3f}% |"
            )
            print(
                f"Global Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |"
            )
            print("+" * 50)

        global_logger.add_scalar("Cluster_Train/Loss", cluster_train_loss, self.c_round)
        global_logger.add_scalar(
            "Cluster_Train/Metric", cluster_train_acc, self.c_round
        )
        global_logger.add_scalar("Global_Train/Loss", global_train_loss, self.c_round)
        global_logger.add_scalar("Global_Train/Metric", global_train_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            for i in indices:
                learners = self.clients[i].learners_ensemble
                copy_model(learners[0].model, self.cluster_learners[cluster_id].model)
                copy_model(learners[1].model, self.global_learner.model)

    def pre_update_clients(self):
        for client in self.clients:
            copy_model(client.learners_ensemble[0].model, self.global_learner.model)

    def update_test_clients(self):
        for client in self.test_clients:
            copy_model(
                target=client.learners_ensemble[0].model,
                source=self.global_learner.model,
            )

            client.update_sample_weights()
            # client.update_learners_weights()


class AgnosticAggregator(CentralizedAggregator):
    """
    Implements
     `Agnostic Federated Learning`__(https://arxiv.org/pdf/1902.00146.pdf).

    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        lr_lambda,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(AgnosticAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        self.lr_lambda = lr_lambda

    def mix(self):
        self.sample_clients()

        clients_losses = []
        for client in self.sampled_clients:
            for _ in range(client.local_steps):
                client_losses = client.step(self.single_batch_flag)
                clients_losses.append(client_losses)

        clients_losses = torch.tensor(clients_losses)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]

            average_learners(
                learners=learners,
                target_learner=learner,
                weights=self.clients_weights,
                average_gradients=True,
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # update clients weights
        self.clients_weights += self.lr_lambda * clients_losses.mean(dim=1)
        self.clients_weights = simplex_projection(self.clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class FFLAggregator(CentralizedAggregator):
    """
    Implements q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        global_train_logger,
        global_test_logger,
        lr,
        q=1,
        sampling_rate=1.0,
        sample_with_replacement=True,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(FFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        self.q = q
        self.lr = lr
        assert (
            self.sample_with_replacement
        ), "FFLAggregator only support sample with replacement"

    def mix(self):
        self.sample_clients()

        hs = 0
        for client in self.sampled_clients:
            hs += client.step(lr=self.lr)

        # take account for the lr used inside optimizer
        hs /= self.lr * len(self.sampled_clients)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [
                client.learners_ensemble[learner_id] for client in self.sampled_clients
            ]
            average_learners(
                learners=learners,
                target_learner=learner,
                weights=hs * torch.ones(len(learners)),
                average_params=False,
                average_gradients=True,
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class DecentralizedAggregator(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        mixing_matrix,
        log_freq,
        global_train_logger,
        global_test_logger,
        sampling_rate=1.0,
        sample_with_replacement=True,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(DecentralizedAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        self.mixing_matrix = mixing_matrix
        assert (
            self.sampling_rate >= 1
        ), "partial sampling is not supported with DecentralizedAggregator"

    def update_clients(self):
        pass

    def mix(self):
        # update local models
        for client in self.clients:
            for _ in range(client.local_steps):
                client.step(self.single_batch_flag)

        # mix models
        mixing_matrix = torch.tensor(
            self.mixing_matrix.copy(), dtype=torch.float32, device=self.device
        )

        for learner_id, global_learner in enumerate(self.global_learners_ensemble):
            state_dicts = [
                client.learners_ensemble[learner_id].model.state_dict()
                for client in self.clients
            ]

            for key, param in global_learner.model.state_dict().items():
                shape_ = param.shape
                models_params = torch.zeros(
                    self.n_clients, int(np.prod(shape_)), device=self.device
                )

                for ii, sd in enumerate(state_dicts):
                    models_params[ii] = sd[key].view(1, -1)

                models_params = mixing_matrix @ models_params

                for ii, sd in enumerate(state_dicts):
                    sd[key] = models_params[ii].view(shape_)

            for client_id, client in enumerate(self.clients):
                client.learners_ensemble[learner_id].model.load_state_dict(
                    state_dicts[client_id]
                )
        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
