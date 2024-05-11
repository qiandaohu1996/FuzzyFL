
from copy import deepcopy
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import torch
from Aggregator import Aggregator
from learners.learners_ensemble import LearnersEnsemble
from utils.torch_utils import average_learners, copy_model

class GroupAPFL(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
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
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
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
            logger = self.train_client_global_logger
            clients = self.clients
        elif dataset_type == "test":
            logger = self.test_client_global_logger
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

            logger.add_scalar(f"{dataset_type.capitalize()}/Loss", global_loss, self.c_round
            )
            logger.add_scalar(f"{dataset_type.capitalize()}/Metric", global_acc, self.c_round)

            if self.verbose > 0:
                print("+" * 30)
                print(
                    f"{dataset_type.capitalize()} Loss: {global_loss:.3f} | \
                    {dataset_type.capitalize()} Acc: {global_acc * 100:.3f}%"
                )

    def write_logs(self):
        self.update_test_clients()
        global_logger, clients = self.test_client_global_logger, self.test_clients

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

        global_logger, clients = self.train_client_global_logger, self.clients

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