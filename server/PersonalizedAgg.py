
from server.CentralizedAgg import CentralizedAggregator

class PersonalizedAggregator(CentralizedAggregator):
    def update_clients(self):
        for client in self.sampled_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )