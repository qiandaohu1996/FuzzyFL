
from client.Client import Client


class AgnosticFLClient(Client):
    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally=False,
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1
        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)
        return losses

