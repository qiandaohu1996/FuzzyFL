
from copy import deepcopy
import numpy as np
from sklearn.metrics import pairwise_distances
import torch

from server import Aggregator
from utils.torch_utils import average_learners, copy_model
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

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
        train_client_global_logger,
        test_client_global_logger,
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
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
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
    #     self._write_logs(self.train_client_global_logger, self.clients, "Train")

    #     if self.verbose > 0:
    #         print("#" * 80)





