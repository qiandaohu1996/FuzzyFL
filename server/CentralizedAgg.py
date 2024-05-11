
from Aggregator import Aggregator
from utils.torch_utils import average_learners, copy_model


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

