
import torch
from server.CentralizedAgg import CentralizedAggregator
from utils.torch_utils import average_learners


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
        train_client_global_logger,
        test_client_global_logger,
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
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
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