import torch
from server import Aggregator
from utils.torch_utils import average_learners, copy_model


def _compute_scores(distances, i, n_clients, n_byzantine_clients):
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n_clients)
    ]
    _s = sorted(s)[: n_clients - n_byzantine_clients - 2]
    return sum(_s)

def multi_krum(distances, n_clients, n_byzantine_clients, n_aggregation_clients):
    if n_clients < 1:
        raise ValueError(f"Number of clients should be a positive integer. Got {n_clients}.")
    if n_aggregation_clients < 1 or n_aggregation_clients > n_clients:
        raise ValueError(f"Number of clients for aggregation should be >= 1 and <= {n_clients}. Got {n_aggregation_clients}.")
    if 2 * n_byzantine_clients + 2 > n_clients:
        raise ValueError(f"Too many Byzantine clients: 2 * {n_byzantine_clients} + 2 >= {n_clients}.")

    scores = [_compute_scores(distances, i, n_clients, n_byzantine_clients) for i in range(n_clients)]
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1])
    top_m_indices = [i for i, _ in sorted_scores[:n_aggregation_clients]]
    return top_m_indices

def pairwise_euclidean_distances(weights):
    weights_tensor = torch.stack(weights)
    distances = torch.nn.functional.pairwise_distance(weights_tensor, weights_tensor, p=2, keepdim=False)
    return distances

class KrumAggregator(Aggregator):
    
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag,
        byzantine_ratio,
        n_aggregation_clients,
        sampling_rate=1,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(KrumAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )
        self.n_clients = len(clients)
        self.n_byzantine_clients = int(byzantine_ratio * self.n_clients)
        self.n_aggregation_clients = n_aggregation_clients

    def mix(self):
        self.sample_clients()
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)

        distances = pairwise_euclidean_distances([client.get_model_parameters() for client in self.sampled_clients])
        top_m_indices = multi_krum(distances, self.n_clients, self.n_byzantine_clients, self.n_aggregation_clients)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            top_m_learners = [self.sampled_clients[i].learners_ensemble[learner_id] for i in top_m_indices]
            average_learners(top_m_learners, learner)

        self.update_clients()

        if self.c_round % self.log_freq == 0:
            self.evaluate()
        self.c_round += 1

    def update_clients(self):
        for client in self.sampled_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )
