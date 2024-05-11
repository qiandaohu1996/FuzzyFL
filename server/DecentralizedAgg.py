
import numpy as np
import torch
from server.Aggregator import Aggregator


class DecentralizedAggregator(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        mixing_matrix,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
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
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
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